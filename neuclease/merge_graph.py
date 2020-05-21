import os
import logging
import threading
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .util import Timer
from .dvid import fetch_repo_info, fetch_supervoxels, fetch_labels, fetch_complete_mappings, fetch_mutation_id, fetch_supervoxel_splits, fetch_supervoxel_splits_from_kafka
from .merge_table import MERGE_TABLE_DTYPE, load_mapping, load_merge_table, normalize_merge_table, apply_mapping_to_mergetable
from .focused.ingest import fetch_focused_decisions
from .adjacency import find_missing_adjacencies

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
        self.primary_uuid = None
        self.set_primary_uuid(primary_uuid)
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
        
        self._edge_cache = {}
        
        # This lock protects the above cache
        self._edge_cache_main_lock = threading.Lock()
        
        # This dict holds a lock for each body, to avoid requesting edges for the same body in parallel,
        # (but requesting edges for different bodies in parallel is OK).
        self._edge_cache_key_locks = defaultdict(lambda: threading.Lock())
        
        self.max_cache_len = 1000


    def set_primary_uuid(self, primary_uuid):
        _logger.info(f"Changing primary (cached) UUID from {self.primary_uuid} to {primary_uuid}")
        self.primary_uuid = primary_uuid


    def apply_mapping(self, mapping):
        if isinstance(mapping, str):
            mapping = load_mapping(mapping)
        apply_mapping_to_mergetable(self.merge_table_df, mapping)
        self.mapping = mapping


    def fetch_and_apply_mapping(self, server, uuid, instance, kafka_msgs=None):
        # For testing purposes, we have a special means of avoiding kafkas
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


    def append_edges_for_split_supervoxels(self, instance_info, parent_sv_handling='unmap', read_from='kafka', kafka_msgs=None):
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
        
        if read_from == 'kafka':
            split_events = fetch_supervoxel_splits_from_kafka(*instance_info, kafka_msgs=kafka_msgs)
        else:
            split_events = fetch_supervoxel_splits(*instance_info, 'dvid')

        # Drop 'type' field, keep only the int fields.
        all_split_events = []
        for events in split_events.values():
            for event in events:
                all_split_events.append(event[:-1])

        all_split_events = np.asarray(all_split_events, np.uint64)
        if len(all_split_events) == 0:
            all_split_events = np.zeros((0,4), np.uint64)
        
        assert all_split_events.shape[1] == 4
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


    def extract_edges(self, server, uuid, instance, body_id, *, session=None, logger=None):
        body_id = np.uint64(body_id)
        if logger is None:
            logger = _logger

        mutid = fetch_mutation_id(server, uuid, instance, body_id)
        key = (server, uuid, instance, body_id)
        key_lock = self.get_key_lock(*key)

        # Use a lock to avoid requesting the supervoxels from DVID in-parallel,
        # in case the user sends several requests at once for the same body,
        # which can happen if they click faster than dvid can respond.
        with key_lock:
            if key in self._edge_cache:
                cached_mutid, supervoxels, edges, scores = self._edge_cache[key]
                if cached_mutid == mutid:
                    logger.info("Returning cached edges")
                    return (mutid, supervoxels, edges, scores)

            logger.info("Edges not found in cache.  Extracting from merge graph.")
            dvid_supervoxels = fetch_supervoxels(server, uuid, instance, body_id, session=session)

            # It's very fast to select rows based on the body_id,
            # so we prefer that if the mapping is already in sync with DVID.
            svs_from_mapping = self.mapping[self.mapping == body_id].index
            mapping_is_in_sync = (set(svs_from_mapping) == set(dvid_supervoxels))

            if mapping_is_in_sync:
                subset_df = self.extract_premapped_rows(body_id)
            else:
                subset_df = self.extract_rows_by_sv(dvid_supervoxels)

            with Timer() as timer:                
                known_edges = subset_df[['id_a', 'id_b']].values
                extra_edges, orig_num_cc, final_num_cc, block_table = \
                    find_missing_adjacencies(server, uuid, instance, body_id, known_edges,
                                             svs=dvid_supervoxels, search_distance=10, connect_non_adjacent=True)
                extra_scores = np.zeros(len(extra_edges), np.float32)
            
            if orig_num_cc == 1:
                logger.info("Graph is contiguous")
            else:
                logger.info(f"Searched {len(block_table)} blocks for missing adjacencies.")
                if final_num_cc == 1:
                    logger.info(f"Finding missing adjacencies between {orig_num_cc} disjoint components took {timer.timedelta}")
                else:
                    logger.warning(f"Graph is not contiguous, but some missing adjacencies could not be found.")
                    logger.warning(f"Reducing {orig_num_cc} disjoint components into {final_num_cc} took {timer.timedelta}")
            
            edges = known_edges
            scores = subset_df['score'].values

            if len(extra_edges) > 0:
                edges = np.concatenate((edges, extra_edges))
                scores = np.concatenate((scores, extra_scores))

            # Cache before returning
            with self._edge_cache_main_lock:
                if key in self._edge_cache:
                    del self._edge_cache[key]
                if len(self._edge_cache) == self.max_cache_len:
                    first_key = next(iter(self._edge_cache.keys()))
                    del self._edge_cache[first_key]
                self._edge_cache[key] = (mutid, dvid_supervoxels, edges, scores)
        
        return (mutid, dvid_supervoxels, edges, scores)


    def extract_premapped_rows(self, body_id):
        body_positions_orig = (self.merge_table_df['body'] == body_id).values.nonzero()[0]
        subset_df = self.merge_table_df.iloc[body_positions_orig]
        return subset_df.copy()
    

    def extract_rows_by_sv(self, supervoxels):
        _sv_set = set(supervoxels)
        subset_df = self.merge_table_df.query('id_a in @_sv_set and id_b in @_sv_set')
        return subset_df.copy()


    def get_key_lock(self, dvid_server, uuid, instance, body_id):
        """
        Rather than using a single Lock to protect all bodies at once,
        we permit fine-grained locking for individual cached bodies.
        Each body's unique lock is produced by this function.
        """
        key = (dvid_server, uuid, instance, body_id)
        with self._edge_cache_main_lock:
            key_lock = self._edge_cache_key_locks[key]
        return key_lock
