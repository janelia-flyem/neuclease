import logging
import threading
from functools import lru_cache

import numpy as np
import pandas as pd

from DVIDSparkServices.io_util.labelmap_utils import load_edge_csv

from .util import Timer
from .dvid import fetch_supervoxels_for_body, fetch_label_for_coordinate, fetch_mappings, fetch_mutation_id
from .merge_table import load_mapping, load_merge_table, apply_mapping_to_mergetable

_logger = logging.getLogger(__name__)

class LabelmapMergeGraph:
    """
    Represents a volume-wide merge graph.
    The set of all possible edges are immutable, and initialized from a immutable merge table,
    but the edges for each body are extracted from the total set according to their
    dynamically-queried supervoxel members.
    """
        
    def __init__(self, table_path, mapping=None, primary_uuid=None):
        self.lock = threading.Lock()
        self.primary_uuid = primary_uuid

        if isinstance(mapping, str):
            mapping = load_mapping(mapping)

        self.merge_table_df = load_merge_table(table_path, mapping, normalize=True)

        
    def append_edges_for_split_supervoxels(self, split_mapping, server, uuid, instance):
        if isinstance(split_mapping, str):
            split_mapping = load_edge_csv(split_mapping)

        assert np.unique(split_mapping[:,0]).shape[0] == split_mapping.shape[0], \
            "Split mapping should be given as [[child, parent],...], not [parent, child] (and every child should have only one parent)."

        # First extract relevant rows for faster queries below
        children = set(split_mapping[:,0])
        _parents = set(split_mapping[:,1])
        update_table_df = self.merge_table_df.query('id_a in @_parents or id_b in @_parents').copy()
        assert update_table_df.columns[:2].tolist() == ['id_a', 'id_b']

        with Timer(f"Appending {len(update_table_df)} edges with split supervoxel IDs", _logger):
            for i in range(len(update_table_df)):
                row = update_table_df.iloc[i:i+1]
                coord_a = row[['za', 'ya', 'xa']].values[0]
                coord_b = row[['zb', 'yb', 'xb']].values[0]
                
                if (coord_a == (0,0,0)).all() or (coord_b == (0,0,0)).all():
                    assert False, "FIXME"
                
                sv_a = fetch_label_for_coordinate(server, uuid, instance, coord_a, supervoxels=True)
                sv_b = fetch_label_for_coordinate(server, uuid, instance, coord_b, supervoxels=True)
                if sv_a not in children or sv_b not in children:
                    msg = f"The provided split mapping does not match the currently stored labels for row:\n {row}"
                    msg += f"Found labels: {sv_a} and {sv_b}"
                    raise RuntimeError(msg)
    
                update_table_df.iloc[i, 0] = sv_a # id_a
                update_table_df.iloc[i, 1] = sv_b # id_b
        
        self.merge_table_df = pd.concat(self.merge_table_df, update_table_df, ignore_index=True, copy=False)

    def fetch_and_apply_mapping(self, server, uuid, labelmap_instance):
        mapping = fetch_mappings(server, uuid, labelmap_instance, True)
        apply_mapping_to_mergetable(self.merge_table_df, mapping)

    @lru_cache(maxsize=1000)
    def fetch_supervoxels_for_body(self, dvid_server, uuid, labelmap_instance, body_id, mut_id, logger=_logger):
        """
        Fetch the supervoxels for the given body from DVID.
        The results are memoized via the @lru_cache decorator.
        
        The mut_id parameter is not used when fetching from DVID, but is listed as an argument
        to ensure a new LRU cache entry if the mutation ID has changed.
        
        Note: @lru_cache is threadsafe (https://bugs.python.org/issue28969)
        """
        with Timer() as timer:
            supervoxels = fetch_supervoxels_for_body(dvid_server, uuid, labelmap_instance, body_id)
            supervoxels = np.asarray(supervoxels, np.uint64)
            supervoxels.sort()

        logger.info(f"Retrieving supervoxel list from DVID took {timer.timedelta}")
        return supervoxels


    def extract_rows(self, dvid_server, uuid, labelmap_instance, body_id, logger=None):
        """
        Determine which supervoxels belong to the given body,
        and extract all edges involving those supervoxels (and only those supervoxels).
        """
        body_id = np.uint64(body_id)
        if logger is None:
            logger = _logger
        
        mut_id = fetch_mutation_id(dvid_server, uuid, labelmap_instance, body_id)
        dvid_supervoxels = self.fetch_supervoxels_for_body(dvid_server, uuid, labelmap_instance, body_id, mut_id, logger)

        with self.lock:
            # It's very fast to select rows based on the body_id,
            # so try that and see if the supervoxel set matches.
            # If it does, we can return immediately.
            body_positions_orig = (self.merge_table_df['body'] == body_id).values.nonzero()[0]
            subset_df = self.merge_table_df.iloc[body_positions_orig]
            svs_from_table = np.unique(subset_df[['id_a', 'id_b']].values)
            if svs_from_table.shape == dvid_supervoxels.shape and (svs_from_table == dvid_supervoxels).all():
                return subset_df, dvid_supervoxels
        
            logger.info(f"Cached supervoxels (N={len(svs_from_table)}) don't match expected (N={len(dvid_supervoxels)}).  Updating cache.")
            
            # Body doesn't match the desired supervoxels.
            # Extract the desired rows the slow way, by selecting all matching supervoxels
            #
            # Note:
            #    I tried speeding this up using proper index-based pandas selection:
            #        merge_table_df.loc[(supervoxels, supervoxels), 'body'] = body_id
            #    ...but that is MUCH worse for large selections, and only marginally
            #    faster for small selections.
            #    Using eval() seems to be the best option here.
            #    The worst body we've got still only takes ~2.5 seconds to extract.
            _sv_set = set(dvid_supervoxels)
            subset_positions = self.merge_table_df.eval('id_a in @_sv_set and id_b in @_sv_set').values
            subset_df = self.merge_table_df.iloc[subset_positions]
            
            if self.primary_uuid is None or uuid == self.primary_uuid:
                self.merge_table_df['body'].values[body_positions_orig] = 0
                self.merge_table_df['body'].values[subset_positions] = body_id
    
            return subset_df, dvid_supervoxels

