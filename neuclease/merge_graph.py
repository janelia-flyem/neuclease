import os
import logging
import threading
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .util import Timer
from .rwlock import ReadWriteLock
from .dvid import fetch_supervoxels_for_body, fetch_label_for_coordinate, fetch_mappings, fetch_mutation_id
from .merge_table import load_edge_csv, load_mapping, load_merge_table, normalize_merge_table, apply_mapping_to_mergetable

_logger = logging.getLogger(__name__)

@contextmanager
def dummy_lock():
    """
    Provides a dummy object that supports the 'with' keyword.
    Useful for code that expects a lock, but you don't actually need a lock.
    """
    yield

def uuids_match(uuid1, uuid2):
    """
    Return True if the two uuids are the equivalent.
    
    >>> assert uuids_match('abcd', 'abcdef') == True
    >>> assert uuids_match('abc9', 'abcdef') == False
    """
    if len(uuid1) > len(uuid2):
        uuid1, uuid2 = uuid2, uuid1
    
    uuid2 = uuid2[:len(uuid1)]
    return (uuid2 == uuid1)

class LabelmapMergeGraph:
    """
    Represents a volume-wide merge graph.
    The set of all possible edges are immutable, and initialized from a immutable merge table,
    but the edges for each body are extracted from the total set according to their
    dynamically-queried supervoxel members.
    """
        
    def __init__(self, table_path, primary_uuid=None, debug_export_dir=None):
        self.rwlock = ReadWriteLock()
        self.primary_uuid = primary_uuid
        self.debug_export_dir = debug_export_dir
        if debug_export_dir:
            os.makedirs(debug_export_dir, exist_ok=True)
            
        self.merge_table_df = load_merge_table(table_path, normalize=True)
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
        

    def append_edges_for_split_supervoxels(self, split_mapping, server, uuid, instance, parent_sv_handling='unmap'):
        """
        Append edges to the merge table for the given split supervoxels (do not remove edges for their parents).
        
        Args:
            split_mapping:
                A mapping (2-column numpy array) of [[child, parent], [child,parent],...] or a path from which it can be loaded.
            
            server, uuid, instance:
                Identifies a DVID labelmap instance from which new supervoxel IDs
                can be queried via points in the DVID labelmap volume.
            
            parent_sv_handling:
                One of the following:
                    - 'drop': Delete the edges that referred to the parent split IDs
                    - 'keep': Keep the edges that referred to the parent split IDs
                    - 'unmap': Keep the edges that referred to the parent split IDs, but reset their 'body' column to 0.
        Returns:
            If any edges could not be preserved because the queried point in DVID does not seem to be a split child,
            a DataFrame of such edges is returned.
        """
        assert parent_sv_handling in ('keep', 'drop', 'unmap')
        if isinstance(split_mapping, str):
            split_mapping = load_edge_csv(split_mapping)

        assert np.unique(split_mapping[:,0]).shape[0] == split_mapping.shape[0], \
            "Split mapping should be given as [[child, parent],...], not [parent, child] (and every child should have only one parent)."

        # First extract relevant rows for faster queries below
        children = set(split_mapping[:,0])
        _parents = set(split_mapping[:,1])
        parent_rows_df = self.merge_table_df.query('id_a in @_parents or id_b in @_parents').copy()
        assert parent_rows_df.columns[:2].tolist() == ['id_a', 'id_b']
        
        if parent_sv_handling == 'drop':
            self.merge_table_df = self.merge_table_df.drop(parent_rows_df.index)
        elif parent_sv_handling == 'unmap':
            self.merge_table_df.loc[parent_rows_df.index, 'body'] = np.uint64(0)

        with Timer(f"Appending {len(parent_rows_df)} edges with split supervoxel IDs", _logger):
            bad_edges = []
            update_rows = [parent_rows_df.iloc[:0]] # Init with empty df in case there are no updates
            for i in range(len(parent_rows_df)):
                row = parent_rows_df.iloc[i:i+1]
                coord_a = row[['za', 'ya', 'xa']].values[0]
                coord_b = row[['zb', 'yb', 'xb']].values[0]
                
                if (coord_a == (0,0,0)).all() or (coord_b == (0,0,0)).all():
                    assert False, "FIXME"
                
                sv_a = fetch_label_for_coordinate(server, uuid, instance, coord_a, supervoxels=True)
                sv_b = fetch_label_for_coordinate(server, uuid, instance, coord_b, supervoxels=True)

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
        update_table_array = update_table_df.drop('body', axis=1).to_records(index=False)
        normalized_update = normalize_merge_table(update_table_array, False, None)
        normalized_update_df = pd.DataFrame(normalized_update, index=update_table_df.index)
        normalized_update_df['body'] = update_table_df['body'].values

        # Append the updates
        assert (normalized_update_df.columns == self.merge_table_df.columns).all()
        self.merge_table_df = pd.concat((self.merge_table_df, normalized_update_df), ignore_index=True, copy=False)

        return bad_edges


    def fetch_and_apply_mapping(self, server, uuid, labelmap_instance, split_mapping=None):
        if isinstance(split_mapping, str):
            split_mapping = load_edge_csv(split_mapping)

        split_sv_parents = []
        if split_mapping is not None:
            split_sv_parents = set(split_mapping[:,1])

        mapping = fetch_mappings(server, uuid, labelmap_instance, include_identities=True, retired_supervoxels=split_sv_parents)
        apply_mapping_to_mergetable(self.merge_table_df, mapping)


    def fetch_supervoxels_for_body(self, dvid_server, uuid, labelmap_instance, body_id, logger=_logger):
        """
        Fetch the supervoxels for the given body from DVID.
        The results are cached internally, according to the body's current mutation id.
        
        Returns:
            (mutation id, supervoxels)
        """
        mut_id = fetch_mutation_id(dvid_server, uuid, labelmap_instance, body_id)

        key = (dvid_server, uuid, labelmap_instance, body_id)
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
                supervoxels = fetch_supervoxels_for_body(dvid_server, uuid, labelmap_instance, body_id)
                supervoxels = np.asarray(supervoxels, np.uint64)
                supervoxels.sort()

            # Save in cache
            with self._sv_cache_main_lock:
                self._sv_cache[key] = (mut_id, supervoxels)

        logger.info(f"Retrieving supervoxel list from DVID took {timer.timedelta}")
        return mut_id, supervoxels


    def extract_rows(self, dvid_server, uuid, labelmap_instance, body_id, logger=None):
        """
        Determine which supervoxels belong to the given body,
        and extract all edges involving those supervoxels (and only those supervoxels).
        """
        body_id = np.uint64(body_id)
        if logger is None:
            logger = _logger
        
        mut_id, dvid_supervoxels = self.fetch_supervoxels_for_body(dvid_server, uuid, labelmap_instance, body_id, logger)

        # Are we allowed to update the merge table 'body' column?
        permit_write = (self.primary_uuid is None or uuids_match(uuid, self.primary_uuid))

        # If we're permitted to write, then avoid running this function in parallel for the same computation.
        # (The first thread to enter will take a while to apply the body mapping, but the rest will be fast.)
        # Use a body-specific lock.  If we're not permitted to write, use a dummy lock (permit parallel computation).
        if permit_write:
            key_lock = self.get_key_lock(dvid_server, uuid, labelmap_instance, body_id)
        else:
            key_lock = dummy_lock()

        with key_lock:
            with self.rwlock.context(write=False):
                try:
                    mapping_is_in_sync = (self._mapping_versions[body_id] == (dvid_server, uuid, labelmap_instance, mut_id))
                except KeyError:
                    mapping_is_in_sync = False
    
                # It's very fast to select rows based on the body_id,
                # so we prefer that if the mapping is already in sync with DVID.
                body_positions_orig = (self.merge_table_df['body'] == body_id).values.nonzero()[0]
                subset_df = self.merge_table_df.iloc[body_positions_orig]
                svs_from_table = np.unique(subset_df[['id_a', 'id_b']].values)
    
                if mapping_is_in_sync:
                    return subset_df, dvid_supervoxels
    
                # Maybe the mapping isn't in sync, but the supervoxels match anyway...
                if svs_from_table.shape == dvid_supervoxels.shape and (svs_from_table == dvid_supervoxels).all():
                    self._mapping_versions[body_id] = (dvid_server, uuid, labelmap_instance, mut_id)
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

                if self.debug_export_dir:
                    export_path = self.debug_export_dir + f"/body-{body_id}-table-before-sync.csv"
                    logger.info(f"Exporting {export_path}")
                    orig_rows = body_positions_orig.values.nonzero()[0] # can't use bool array with iloc
                    self.merge_table_df.iloc[orig_rows].to_csv(export_path, index=False)

                logger.info(f"Cached supervoxels (N={len(svs_from_table)}) don't match expected (N={len(dvid_supervoxels)}).  Updating cache.")
                sv_set = set(dvid_supervoxels)
                subset_df = self.merge_table_df.query('id_a in @sv_set and id_b in @sv_set').copy()
                subset_df['body'] = body_id
    
                if permit_write:
                    logger.info("Overwriting cached mapping")
                    # Before we overwrite, invalidate the mapping version for any body IDs we're about to overwrite
                    for prev_body in pd.unique(subset_df['body'].values):
                        if prev_body in self._mapping_versions:
                            del self._mapping_versions[prev_body]
    
                    self._mapping_versions[body_id] = (dvid_server, uuid, labelmap_instance, mut_id)
                    self.merge_table_df.loc[body_positions_orig.index, 'body'] = np.uint64(0)
                    self.merge_table_df.loc[subset_df.index, 'body'] = body_id

                if self.debug_export_dir:
                    export_path = self.debug_export_dir + f"/body-{body_id}-table-after-sync.csv"
                    logger.info(f"Exporting {export_path}")
                    subset_df.to_csv(export_path, index=False)
                    assert set(pd.unique(subset_df[['id_a', 'id_b']].values.flat)) - sv_set == set(), \
                        "Our new subset includes supervoxels that DVID didn't want!"


                return subset_df, dvid_supervoxels

    def get_key_lock(self, dvid_server, uuid, labelmap_instance, body_id):
        key = (dvid_server, uuid, labelmap_instance, body_id)
        with self._sv_cache_main_lock:
            key_lock = self._sv_cache_key_locks[key]
        return key_lock

