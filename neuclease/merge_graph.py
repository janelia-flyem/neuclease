import os
import logging
import threading
from itertools import chain
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .util import Timer, uuids_match
from .rwlock import ReadWriteLock
from .dvid import fetch_repo_info, fetch_supervoxels_for_body, fetch_labels, fetch_complete_mappings, fetch_mutation_id, fetch_supervoxel_splits, fetch_supervoxel_splits_from_kafka
from .merge_table import MERGE_TABLE_DTYPE, load_mapping, load_merge_table, normalize_merge_table, apply_mapping_to_mergetable
from .focused.ingest import fetch_focused_decisions

_logger = logging.getLogger(__name__)


@contextmanager
def dummy_lock():
    """
    Provides a dummy object that supports the 'with' keyword.
    Useful for code that expects a lock, but you don't actually need a lock.
    """
    yield


class LabelmapMergeGraph:
    """
    Represents a volume-wide merge graph.
    The set of all possible edges are immutable, and initialized from a immutable merge table,
    but the edges for each body are extracted from the total set according to their
    dynamically-queried supervoxel members.
    """
        
    def __init__(self, table, primary_uuid=None, debug_export_dir=None, no_kafka=False):
        """
        Constructor.
        
        Args:
            table:
                Either a (pre-normalized) pd.DataFrame or numpy structured array,
                or a path to a .csv or .npy file from which one can be loaded.
                After loading must have the columns from MERGE_TABLE_DTYPE
            
            primary_uuid:
                The UUID which takes precedence for caching purposes,
                when updating body supervoxels in response to a request.
            
            debug_export_dir:
                Only used for debugging purposes! Leave as None for normal operation.
                Exports certain results tables during some operations.
                Significant performance impact if provided.
            
            no_kafka:
                Only used for unit-testing purposes, when no kafka server is available.
                Disables fetching of split supervoxel information entirely!
        """
        self.rwlock = ReadWriteLock()
        self.primary_uuid = primary_uuid
        self.debug_export_dir = debug_export_dir
        if debug_export_dir:
            os.makedirs(debug_export_dir, exist_ok=True)

        self.no_kafka = no_kafka

        if isinstance(table, str):
            self.merge_table_df = load_merge_table(table, normalize=True)
        elif isinstance(table, np.ndarray):
            self.merge_table_df = pd.DataFrame(table)
        elif isinstance(table, pd.DataFrame):
            self.merge_table_df = table
        else:
            assert False, f"Unsupported table type: {type(table)}"

        assert isinstance(self.merge_table_df, pd.DataFrame)
        assert list(self.merge_table_df.columns)[:9] == list(dict(MERGE_TABLE_DTYPE).keys())[:9]
        
        self._mapping_versions = {}
        
        # Supervoxels retrieved from DVID are cached in this member.
        self._sv_cache = {}
        
        # This lock protects the above cache
        self._sv_cache_main_lock = threading.Lock()
        
        # This dict holds a lock for each body, to avoid requesting supervoxels for the same body in parallel,
        # (but requesting supervoxels for different bodies in parallel is OK).
        self._sv_cache_key_locks = defaultdict(lambda: threading.Lock())


    def apply_mapping(self, mapping):
        if isinstance(mapping, str):
            mapping = load_mapping(mapping)
        apply_mapping_to_mergetable(self.merge_table_df, mapping)
        self.mapping = mapping


    def fetch_and_apply_mapping(self, server, uuid, instance):
        # For testing purposes, we have a special means of avoiding kafkas
        kafka_msgs = None
        if self.no_kafka:
            kafka_msgs = []
        mapping = fetch_complete_mappings(server, uuid, instance, include_retired=True, kafka_msgs=kafka_msgs)
        self.apply_mapping(mapping)


    def append_edges_for_focused_merges(self, server, uuid, focused_decisions_instance):
        """
        Read the proofreading focused merge decisions from a keyvalue
        instance (stored as individual JSON values),
        and append the resulting merge edges to the merge graph.
        
        Args:
            server, uuid, instance:
                For example, ('emdata3:8900', 'cc4c', 'segmentation_merged')
        
        Returns:
            The count of appended edges
        """
        repo_info = fetch_repo_info(server, uuid)
        if focused_decisions_instance not in repo_info["DataInstances"]:
            return 0
        
        focused_decisions = fetch_focused_decisions(server, uuid, focused_decisions_instance)
        focused_merges = focused_decisions.query('result == "merge" or result == "mergeLater"')
        focused_merges = focused_merges[["sv_a", "sv_b", "xa", "ya", "za", "xb", "yb", "zb"]]
        focused_merges.rename(inplace=True, columns={'sv_a': 'id_a', 'sv_b': 'id_b'})

        # These are manual merges: Give a great score.
        focused_merges['score'] = np.float32(0.01)
        
        # Ensure correct dtypes for concatenation
        for col, dtype in MERGE_TABLE_DTYPE:
            focused_merges[col] = focused_merges[col].astype(dtype, copy=False)
        
        focused_merges = focused_merges.loc[:, list(self.merge_table_df.columns)]
        self.merge_table_df = pd.concat((self.merge_table_df, focused_merges), ignore_index=True, copy=False)
        return len(focused_merges)


    def append_edges_for_split_supervoxels(self, instance_info, parent_sv_handling='unmap', read_from='kafka'):
        """
        Append edges to the merge table for the given split supervoxels (do not remove edges for their parents).
        
        Args:
            server, uuid, instance:
                Identifies a DVID labelmap instance from which new supervoxel IDs
                can be queried via points in the DVID labelmap volume.
            
            parent_sv_handling:
                One of the following:
                    - 'drop': Delete the edges that referred to the parent split IDs
                    - 'keep': Keep the edges that referred to the parent split IDs
                    - 'unmap': Keep the edges that referred to the parent split IDs, but reset their 'body' column to 0.
            
            read_from:
                What source to read split events from.  Either 'dvid' or 'kafka'.
                The 'dvid' option is faster (and doesn't balk if no Kafka server is found),
                but some of our older DVID servers have not recorded all of their splits
                to the internal mutation log, and so only kafka is a reliable source of split
                information in such cases.
            
        Returns:
            If any edges could not be preserved because the queried point in DVID does not seem to be a split child,
            a DataFrame of such edges is returned.
        """
        assert parent_sv_handling in ('keep', 'drop', 'unmap')
        assert read_from in ('dvid', 'kafka')
        
        if read_from == 'dvid':
            split_events = fetch_supervoxel_splits(*instance_info)
        elif read_from == 'kafka':
            split_events = fetch_supervoxel_splits_from_kafka(*instance_info)

        all_split_events = np.array(list(chain(*split_events.values())))
        if len(all_split_events) == 0:
            # No split events at all: Return empty dataframe
            bad_edges = self.merge_table_df.iloc[:0]
            return bad_edges
        
        # Each event is [mutid, old, remain, split]
        old_ids = all_split_events[:, 1]
        remain_ids = all_split_events[:, 2]
        split_ids = all_split_events[:, 3]

        # First extract relevant rows for faster queries below
        _parents = set(old_ids)
        children = set(remain_ids) | set(split_ids)
        parent_rows_df = self.merge_table_df.query('id_a in @_parents or id_b in @_parents').copy()
        assert parent_rows_df.columns[:2].tolist() == ['id_a', 'id_b']
        
        if parent_sv_handling == 'drop':
            self.merge_table_df = self.merge_table_df.drop(parent_rows_df.index)
        elif parent_sv_handling == 'unmap':
            self.merge_table_df.loc[parent_rows_df.index, 'body'] = np.uint64(0)

        with Timer(f"Appending {len(parent_rows_df)} edges with split supervoxel IDs", _logger):
            bad_edges = []
            update_rows = [parent_rows_df.iloc[:0]] # Init with empty df in case there are no updates

            with Timer("Fetching supervoxels from split edge coordinates", _logger):
                svs_a = fetch_labels(*instance_info, parent_rows_df[['za', 'ya', 'xa']].values, supervoxels=True)
                svs_b = fetch_labels(*instance_info, parent_rows_df[['zb', 'yb', 'xb']].values, supervoxels=True)
            
            for i in range(len(parent_rows_df)):
                sv_a = svs_a[i]
                sv_b = svs_b[i]

                # If either coordinate returns a non-sensical point, then
                # the provided split mapping does not match the currently stored labels.
                if sv_a not in children and sv_b not in children:
                    if sv_a not in children and sv_a != parent_rows_df.iloc[i, 0]:
                        bad_edges.append(('a', sv_a) + tuple(parent_rows_df.iloc[i]))
                    if sv_b not in children and sv_b != parent_rows_df.iloc[i, 1]:
                        bad_edges.append(('b', sv_b) + tuple(parent_rows_df.iloc[i]))
                else:
                    parent_rows_df.iloc[i, 0] = sv_a # id_a
                    parent_rows_df.iloc[i, 1] = sv_b # id_b
                    update_rows.append(parent_rows_df[i:i+1])

        update_table_df = pd.concat(update_rows, ignore_index=True)
        assert (update_table_df.columns == self.merge_table_df.columns).all()

        if bad_edges:
            bad_edges = pd.DataFrame(bad_edges, columns=['end', 'found_sv'] + list(update_table_df.columns))
        else:
            bad_edges = update_table_df.iloc[:0] # No bad edges: Empty DataFrame

        # Normalize the updates
        update_table_array = update_table_df.to_records(index=False)
        normalized_update = normalize_merge_table(update_table_array, False, None)
        normalized_update_df = pd.DataFrame(normalized_update, index=update_table_df.index)

        # Append the updates
        assert (normalized_update_df.columns == self.merge_table_df.columns).all()
        self.merge_table_df = pd.concat((self.merge_table_df, normalized_update_df), ignore_index=True, copy=False)

        return bad_edges


    def fetch_supervoxels_for_body(self, server, uuid, instance, body_id, logger=_logger):
        """
        Fetch the supervoxels for the given body from DVID.
        The results are cached internally, according to the body's current mutation id.
        
        Args:
            instance_info: tuple of (server, uuid, instance)
            body_id: uint64
        
        Returns:
            (mutation id, supervoxels)
        """
        mut_id = fetch_mutation_id(server, uuid, instance, body_id)

        key = (server, uuid, instance, body_id)
        key_lock = self.get_key_lock(*key)

        # Use a lock to avoid requesting the supervoxels from DVID in-parallel,
        # in case the user sends several requests at once for the same body,
        # which can happen if they click faster than dvid can respond.
        with key_lock:
            if key in self._sv_cache:
                cached_mut_id, supervoxels = self._sv_cache[key]
                if cached_mut_id == mut_id:
                    return mut_id, supervoxels # Cache is up-to-date
            
            with Timer() as timer:
                supervoxels = fetch_supervoxels_for_body(server, uuid, instance, body_id)
                supervoxels = np.asarray(supervoxels, np.uint64)
                supervoxels.sort()

            # Save in cache
            with self._sv_cache_main_lock:
                self._sv_cache[key] = (mut_id, supervoxels)

        logger.info(f"Retrieving supervoxel list from DVID took {timer.timedelta}")
        return mut_id, supervoxels


    def extract_rows(self, server, uuid, instance, body_id, logger=None):
        """
        Determine which supervoxels belong to the given body,
        and extract all edges involving those supervoxels (and only those supervoxels).
        """        
        body_id = np.uint64(body_id)
        if logger is None:
            logger = _logger
        mut_id, dvid_supervoxels = self.fetch_supervoxels_for_body(server, uuid, instance, body_id, logger)

        # Are we allowed to update the merge table 'body' column?
        permit_write = (self.primary_uuid is None or uuids_match(uuid, self.primary_uuid))

        # If we're permitted to write, then avoid running this function in parallel for the same computation.
        # (The first thread to enter will take a while to apply the body mapping, but the rest will be fast.)
        # Use a body-specific lock.  If we're not permitted to write, use a dummy lock (permit parallel computation).
        if permit_write:
            key_lock = self.get_key_lock(server, uuid, instance, body_id)
        else:
            key_lock = dummy_lock()

        with key_lock:
            svs_from_mapping = None
            with self.rwlock.context(write=False):
                try:
                    mapping_is_in_sync = (self._mapping_versions[body_id] == (server, uuid, instance, mut_id))
                except KeyError:
                    # This body doesn't exist in the mapping cache (yet)
                    svs_from_mapping = self.mapping[self.mapping == body_id]
                    mapping_is_in_sync = (set(svs_from_mapping) == set(dvid_supervoxels))
    
                # It's very fast to select rows based on the body_id,
                # so we prefer that if the mapping is already in sync with DVID.
                body_positions_orig = (self.merge_table_df['body'] == body_id).values.nonzero()[0]
                subset_df = self.merge_table_df.iloc[body_positions_orig]
    
                if mapping_is_in_sync:
                    return subset_df, dvid_supervoxels

            # If we get this far, our mapping is out-of-sync with DVID's agglomeration.
            # Query for the rows the slow way (via set membership).
            # 
            # Note:
            #    I tried speeding this up using proper index-based pandas selection:
            #        merge_table_df.loc[(supervoxels, supervoxels), 'body'] = body_id
            #    ...but that is MUCH worse for large selections, and only marginally
            #    faster for small selections.
            #    Using eval() seems to be the best option here.
            #    The worst body we've got takes ~30 seconds to extract.
    
            # Should we overwrite the body column for these rows?
            with self.rwlock.context(permit_write):
                # Must re-query the rows to change, since the table might have changed while the lock was released.
                body_positions_orig = (self.merge_table_df['body'] == body_id)
                
                # Likewise, must re-query the old mapping supervoxels
                svs_from_mapping = self.mapping[self.mapping == body_id]

                if self.debug_export_dir:
                    export_path = self.debug_export_dir + f"/body-{body_id}-table-before-sync.csv"
                    logger.info(f"Exporting {export_path}")
                    orig_rows = body_positions_orig.values.nonzero()[0] # can't use bool array with iloc
                    self.merge_table_df.iloc[orig_rows].to_csv(export_path, index=False)

                logger.info(f"Cached supervoxels (N={len(svs_from_mapping)}) don't match expected (N={len(dvid_supervoxels)}).")
                sv_set = set(dvid_supervoxels)
                subset_df = self.merge_table_df.query('id_a in @sv_set and id_b in @sv_set').copy()
                subset_df['body'] = body_id

                if permit_write:
                    logger.info(f"Overwriting cached mapping (erasing {body_positions_orig.sum()}, updating {len(subset_df)})")
                    # Before we overwrite, invalidate the mapping version for any body IDs we're about to overwrite
                    for prev_body in pd.unique(self.mapping.loc[dvid_supervoxels].fillna(0).astype(np.uint64)):
                        if prev_body in self._mapping_versions:
                            del self._mapping_versions[prev_body]
    
                    self._mapping_versions[body_id] = (server, uuid, instance, mut_id)

                    # Overwite old positions with zeros, overwrite new positions with the body_id
                    self.merge_table_df.loc[body_positions_orig, 'body'] = np.uint64(0)
                    self.merge_table_df.loc[subset_df.index, 'body'] = body_id

                    # Overwite old positions with zeros, overwrite new positions with the body_id,
                    # and add new positions if they don't exist in the mapping yet.
                    self.mapping[self.mapping == body_id] = np.uint64(0)
                    new_svs = pd.UInt64Index(dvid_supervoxels).difference(self.mapping.index)
                    if len(new_svs) > 0:
                        new_mapping_entries = pd.Series(index=np.fromiter(new_svs, dtype=np.uint64), data=body_id)
                        self.mapping = pd.concat((self.mapping, new_mapping_entries))
                    self.mapping.loc[dvid_supervoxels] = body_id
                    
                if self.debug_export_dir:
                    export_path = self.debug_export_dir + f"/body-{body_id}-table-after-sync.csv"
                    logger.info(f"Exporting {export_path}")
                    subset_df.to_csv(export_path, index=False)
                    assert set(pd.unique(subset_df[['id_a', 'id_b']].values.flat)) - sv_set == set(), \
                        "Our new subset includes supervoxels that DVID didn't want!"

                return subset_df, dvid_supervoxels

    
    def get_key_lock(self, dvid_server, uuid, instance, body_id):
        key = (dvid_server, uuid, instance, body_id)
        with self._sv_cache_main_lock:
            key_lock = self._sv_cache_key_locks[key]
        return key_lock

