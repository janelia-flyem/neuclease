import logging
import threading
from functools import lru_cache

import numpy as np

from .util import Timer
from .dvid import fetch_supervoxels_for_body
from .merge_table import load_mapping, load_merge_table

_logger = logging.getLogger(__name__)

class LabelmapMergeGraph:
    """
    Represents a volume-wide merge graph.
    The set of all possible edges are immutable, and initialized from a immutable merge table,
    but the edges for each body are extracted from the total set according to their
    dynamically-queried supervoxel members.
    """
        
    def __init__(self, table_path, mapping_path=None, primary_uuid=None):
        self.lock = threading.Lock()
        self.primary_uuid = primary_uuid
        if mapping_path:
            self.mapping = load_mapping(mapping_path)
        else:
            self.mapping = None

        self.merge_table_df = load_merge_table(table_path, self.mapping, normalize=True)


    @lru_cache(maxsize=1000)
    def fetch_supervoxels_for_body(self, dvid_server, uuid, labelmap_instance, body_id, mut_id, logger):
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
        
        # FIXME: Actually fetch mutation ID for each body when that DVID endpoint is implemented...
        dvid_supervoxels = self.fetch_supervoxels_for_body(dvid_server, uuid, labelmap_instance, body_id, None, logger)

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

