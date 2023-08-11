import os
import logging
import threading
from socket import getfqdn
from textwrap import dedent
from functools import partial
from collections import defaultdict
from contextlib import contextmanager
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..util import Timer, uuids_match, perform_bigquery, compute_parallel
from ..dvid import (fetch_repo_info, find_repo_root, fetch_supervoxels, fetch_labels, fetch_mapping, fetch_complete_mappings,
                    fetch_mutation_id, fetch_supervoxel_splits, fetch_supervoxel_splits_from_kafka)
from ..focused.ingest import fetch_focused_decisions
from .merge_table import MERGE_TABLE_DTYPE, load_mapping, load_merge_table, normalize_merge_table, apply_mapping_to_mergetable
from .adjacency import find_missing_adjacencies

_logger = logging.getLogger(__name__)


@contextmanager
def dummy_lock():
    """
    Provides a dummy object that supports the 'with' keyword.
    Useful for code that expects a lock, but you don't actually need a lock.
    """
    yield


class LabelmapMergeGraphBase(ABC):
    def __init__(self, max_perbody_cache_len=100_000):
        # This stores the most recently extracted edge table for each body that's been accessed.
        # The body's current mutation ID is part of the cache key,
        # so it won't be used if the edge table is no longer valid for the body.
        self._perbody_edge_cache = {}

        # The maximum number of per-body edge tables to cache.
        # Cache is FIFO.
        self.max_perbody_cache_len = max_perbody_cache_len

        # This stores the 'missing' edges we found as a single table for ALL bodies.
        # Notes:
        #   - It is assumed that the total number of 'extra' edges will be relatively small.
        #   - We assume they're good, since they are usually the result of explicit merges,
        #     so we give them a very good score (cost 0.0).
        #   - In the current implementation, we don't obtain coordinates for these edges,
        #     so they can't be updated to handle supervoxel splits.
        #     In the case of missing supervoxels, we simply rely on find_missing_adjacencies()
        #     to re-locate the edge.
        self._extra_edge_cache = pd.DataFrame([], columns=['sv_a', 'sv_b'], dtype=np.uint64)

        # This lock protects the above caches
        self._edge_cache_main_lock = threading.Lock()

        # This dict holds a lock for each body, to avoid requesting edges for the same body in parallel,
        # (but requesting edges for different bodies in parallel is OK).
        self._edge_cache_key_locks = defaultdict(threading.Lock)

    def get_key_lock(self, repo_uuid, instance, body_id, mutid):
        """
        Rather than using a single Lock to protect all bodies at once,
        we permit fine-grained locking for individual cached bodies.
        Each body's unique lock is produced by this function.
        """
        key = (repo_uuid, instance, body_id, mutid)
        with self._edge_cache_main_lock:
            key_lock = self._edge_cache_key_locks[key]
        return key_lock

    def extract_edges(self, server, uuid, instance, body_id, find_missing=True, *, session=None, logger=None):
        with Timer("Extracting edges", logger):
            return self._extract_edges(server, uuid, instance, body_id, find_missing, session=session, logger=logger)

    def _extract_edges(self, server, uuid, instance, body_id, find_missing=True, *, session=None, logger=None):
        body_id = np.uint64(body_id)
        if logger is None:
            logger = _logger

        domain = server.split('://')[-1]
        server = server[:-len(domain)] + getfqdn(domain)
        repo_uuid = find_repo_root(server, uuid)

        # Mutation IDs are unique, even across UUIDs,
        # so we can warm the cache for bodies in ancestor nodes,
        # even if users are viewing descendent nodes.
        # If the body hasn't changed in the descendent, the cached version is valid.
        mutid = fetch_mutation_id(server, uuid, instance, body_id)

        key = (repo_uuid, instance, body_id, mutid)
        key_lock = self.get_key_lock(*key)

        # Use a lock to avoid requesting the supervoxels from DVID in-parallel,
        # in case the user sends several requests at once for the same body,
        # which can happen if they click faster than dvid can respond.
        with key_lock:
            if key in self._perbody_edge_cache:
                supervoxels, edges, scores = self._perbody_edge_cache[key]
                logger.info("Returning cached edges")
                return (mutid, supervoxels, edges, scores)

            logger.info("Edges not found in cache.  Extracting from merge graph.")
            dvid_supervoxels, subset_df = self._extract_stored_edges(server, uuid, instance, body_id, session, logger)

            cached_extra_edges = (
                self._extra_edge_cache.query(
                    'sv_a in @dvid_supervoxels and sv_b in @dvid_supervoxels'
                ).values
            )

            if not find_missing:
                orig_num_cc = 0
                extra_edges = np.zeros((0,2), dtype=np.uint64)
            else:
                with Timer() as timer:
                    known_edges = np.concatenate(
                        (subset_df[['id_a', 'id_b']].values,
                         cached_extra_edges),
                        axis=0
                    )
                    # We run this in a subprocess because otherwise the GIL seems to
                    # prevent parallel requests, perhaps in pd.Series.isin()
                    # Here, compute_parallel is just used to obtain a 1-process pool.
                    # If you're debugging, you'll want to set processes=0.
                    extra_edges, orig_num_cc, final_num_cc, block_table = \
                        compute_parallel(
                            find_missing_adjacencies,
                            [(
                                server, uuid, instance, body_id,
                                known_edges,
                                dvid_supervoxels,
                                10,
                                True
                            )],
                            starmap=True,
                            processes=1
                        )[0]

                self._store_extra_edges(extra_edges)

            if orig_num_cc == 1:
                logger.info("Graph is contiguous")
            elif find_missing:
                logger.info(f"Searched {len(block_table)} blocks for missing adjacencies.")
                if final_num_cc == 1:
                    logger.info(f"Finding missing adjacencies between {orig_num_cc} disjoint components took {timer.timedelta}")
                else:
                    logger.warning("Graph is not contiguous, but some missing adjacencies could not be found.")
                    logger.warning(f"Reducing {orig_num_cc} disjoint components into {final_num_cc} took {timer.timedelta}")
            else:
                logger.warning("Not looking for missing edges (if any)")

            extra_edges = np.concatenate((cached_extra_edges, extra_edges))
            extra_scores = np.zeros(len(extra_edges), np.float32)

            edges = np.concatenate((subset_df[['id_a', 'id_b']].values, extra_edges))
            scores = np.concatenate((subset_df['score'].values, extra_scores))

            # Cache before releasing key_lock, to ensure
            # that any waiting requests see the cached result.
            self._store_body_edges(key, dvid_supervoxels, edges, scores, logger)
        return (mutid, dvid_supervoxels, edges, scores)

    def _store_body_edges(self, key, dvid_supervoxels, edges, scores, logger):
        if self.max_perbody_cache_len == 0:
            return

        with self._edge_cache_main_lock:
            if key in self._perbody_edge_cache:
                del self._perbody_edge_cache[key]
            if len(self._perbody_edge_cache) == self.max_perbody_cache_len:
                first_key = next(iter(self._perbody_edge_cache.keys()))
                del self._perbody_edge_cache[first_key]
                logger.warning(f"Edge cache is full: Deleted an old entry: {first_key}")
            logger.warning(f"Caching entry: {key}")
            self._perbody_edge_cache[key] = (dvid_supervoxels, edges, scores)

    def _store_extra_edges(self, extra_edges):
        assert extra_edges.dtype == np.uint64
        extra_edges = pd.DataFrame(extra_edges, columns=['sv_a', 'sv_b'])
        with self._edge_cache_main_lock:
            self._extra_edge_cache = pd.concat(
                (self._extra_edge_cache, extra_edges),
                ignore_index=True
            )

    @abstractmethod
    def _extract_stored_edges(self, server, uuid, instance, body_id, session, logger):
        raise NotImplementedError()

    @classmethod
    def fetch_focused_merges(cls, server, uuid, focused_decisions_instance):
        """
        Read the proofreading focused merge decisions from a keyvalue
        instance (stored as individual JSON values).

        Args:
            server, uuid, instance:
                For example, ('emdata3:8900', 'cc4c', 'segmentation_merged')

        Returns:
            The focused edges, inferred from the focused proofreading decisions.
        """
        repo_info = fetch_repo_info(server, uuid)
        if focused_decisions_instance not in repo_info["DataInstances"]:
            return 0

        focused_decisions = fetch_focused_decisions(server, uuid, focused_decisions_instance)
        if len(focused_decisions) == 0 or 'result' not in focused_decisions.columns:
            return 0

        focused_merges = focused_decisions.query('result == "merge" or result == "mergeLater"')
        focused_merges = focused_merges[["sv_a", "sv_b", "xa", "ya", "za", "xb", "yb", "zb"]]
        focused_merges.rename(inplace=True, columns={'sv_a': 'id_a', 'sv_b': 'id_b'})

        # These are manual merges: Give a great score.
        focused_merges['score'] = np.float32(0.01)

        # Ensure correct dtypes for concatenation
        for col, dtype in MERGE_TABLE_DTYPE:
            focused_merges[col] = focused_merges[col].astype(dtype, copy=False)

        # This reindex isn't necessary, right?
        # cols = [k for k,v in MERGE_TABLE_DTYPE]
        # focused_merges = focused_merges.reindex(columns=cols)
        return focused_merges


class LabelmapMergeGraphBigQuery(LabelmapMergeGraphBase):
    def __init__(self, table=None, primary_uuid=None, max_perbody_cache_len=100_000, debug_export_dir=None):
        """
        Constructor.

        Args:
            table:
                The name of a table in BigQuery which contains all (or at least most)
                of the intra-body edges in the segmentation, with at least the following columns:

                    body_a, sv_a, sv_b, xa, ya, za, xb, yb, zb, cost

            primary_uuid:
                The UUID which takes precedence for caching purposes,
                when updating body supervoxels in response to a request.

            debug_export_dir:
                Only used for debugging purposes! Leave as None for normal operation.
                Exports certain results tables during some operations.
                Significant performance impact if provided.
        """
        super().__init__(max_perbody_cache_len)
        self.table = table
        table_uuid = table.split('-')[-1]
        if primary_uuid:
            assert uuids_match(table_uuid, primary_uuid), \
                f"Server primary_uuid ({primary_uuid}) doesn't match UUID in table name: ({table})"
        self.primary_uuid = primary_uuid or table_uuid
        self.debug_export_dir = debug_export_dir
        if debug_export_dir:
            os.makedirs(debug_export_dir, exist_ok=True)

    @classmethod
    def fetch_bq_edges(cls, server, uuid, instance, body_id, snapshot_table, snapshot_uuid, *, session=None, logger=None):
        curr_seg = (server, uuid, instance)
        snapshot_seg = (server, snapshot_uuid, instance)
        dvid_supervoxels = fetch_supervoxels(*curr_seg, body_id, session=session)
        snapshot_bodies = pd.unique(fetch_mapping(*snapshot_seg, dvid_supervoxels))

        q = dedent(f"""\
            select sv_a, sv_b, xa, ya, za, xb, yb, zb, cost
            from `{snapshot_table}`
            where body_a in ({', '.join(map(str, snapshot_bodies))})
        """)
        msg = f"Fetching edges for {len(snapshot_bodies)} body(s) from BigQuery snapshot table"
        with Timer(msg, logger):
            df = perform_bigquery(q)

        with Timer(f"Filtering edges for UUID {uuid}", logger):
            df['body_a'] = fetch_mapping(*curr_seg, df['sv_a'].values)
            df['body_b'] = fetch_mapping(*curr_seg, df['sv_b'].values)

            retired_a = (df['body_a'] == 0)
            retired_b = (df['body_b'] == 0)

            if retired_a.sum() > 0:
                df.loc[retired_a, 'sv_a'] = fetch_labels(*curr_seg, df.loc[retired_a, ['za', 'ya', 'xa']].values)
                df.loc[retired_a, 'body_a'] = fetch_mapping(*curr_seg, df.loc[retired_a, 'sv_a'].values)

            if retired_b.sum() > 0:
                df.loc[retired_b, 'sv_b'] = fetch_labels(*curr_seg, df.loc[retired_b, ['zb', 'yb', 'xb']].values)
                df.loc[retired_b, 'body_b'] = fetch_mapping(*curr_seg, df.loc[retired_b, 'sv_b'].values)

            edges = df.query('body_a == @body_id and body_b == @body_id').copy()

        return dvid_supervoxels, edges

    def _extract_stored_edges(self, server, uuid, instance, body_id, session, logger):
        dvid_supervoxels, edges = self.fetch_bq_edges(
            server, uuid, instance, body_id,
            self.table, self.primary_uuid,
            session=session, logger=logger
        )
        # Revert to the idiosyncratic column names used
        # by the other cleave server implementation.
        edges = edges.rename(columns={
            'sv_a': 'id_a',
            'sv_b': 'id_b',
            'cost': 'score',
            'body_a': 'body'
        })

        cols = [k for k,v in MERGE_TABLE_DTYPE]
        cols = [*cols, 'body']
        coltypes = dict(MERGE_TABLE_DTYPE)
        coltypes['body'] = np.uint64
        return dvid_supervoxels, edges[cols].astype(coltypes)


class LabelmapMergeGraphLocalTable(LabelmapMergeGraphBase):
    """
    Represents a volume-wide merge graph.
    The set of all possible edges are immutable, and initialized from a immutable merge table,
    but the edges for each body are extracted from the total set according to their
    dynamically-queried supervoxel members.
    """

    def __init__(self, table=None, primary_uuid=None, max_perbody_cache_len=100_000, debug_export_dir=None, no_kafka=False):
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
        super().__init__()
        self.primary_uuid = None
        self.set_primary_uuid(primary_uuid)
        self.debug_export_dir = debug_export_dir
        if debug_export_dir:
            os.makedirs(debug_export_dir, exist_ok=True)

        self.no_kafka = no_kafka

        if table is None:
            # Empty table -- allowed for debugging.
            self.merge_table_df = pd.DataFrame(np.zeros((0,), dtype=MERGE_TABLE_DTYPE))
            self.merge_table_df['body'] = np.uint64(0)
        elif isinstance(table, str):
            self.merge_table_df = load_merge_table(table, normalize=True)
        elif isinstance(table, np.ndarray):
            self.merge_table_df = pd.DataFrame(table)
        elif isinstance(table, pd.DataFrame):
            self.merge_table_df = table
        else:
            assert False, f"Unsupported table type: {type(table)}"

        assert isinstance(self.merge_table_df, pd.DataFrame)
        assert list(self.merge_table_df.columns)[:9] == list(dict(MERGE_TABLE_DTYPE).keys())[:9]

    def set_primary_uuid(self, primary_uuid):
        _logger.info(f"Changing primary (cached) UUID from {self.primary_uuid} to {primary_uuid}")
        self.primary_uuid = primary_uuid

    def _extract_stored_edges(self, server, uuid, instance, body_id, session, logger):
        dvid_supervoxels = fetch_supervoxels(server, uuid, instance, body_id, session=session)

        # It's very fast to select rows based on the body_id,
        # so we prefer that if the mapping is already in sync with DVID.
        svs_from_mapping = self.mapping[self.mapping == body_id].index
        mapping_is_in_sync = (set(svs_from_mapping) == set(dvid_supervoxels))

        if mapping_is_in_sync:
            subset_df = self.extract_premapped_rows(body_id)
        else:
            subset_df = self.extract_rows_by_sv(dvid_supervoxels)

        return dvid_supervoxels, subset_df

    def extract_premapped_rows(self, body_id):
        body_positions_orig = (self.merge_table_df['body'] == body_id).values.nonzero()[0]
        subset_df = self.merge_table_df.iloc[body_positions_orig]
        return subset_df.copy()

    def extract_rows_by_sv(self, supervoxels):
        _sv_set = set(supervoxels)  # noqa
        subset_df = self.merge_table_df.query('id_a in @_sv_set and id_b in @_sv_set')
        return subset_df.copy()

    def apply_mapping(self, mapping):
        if isinstance(mapping, str):
            mapping = load_mapping(mapping)
        apply_mapping_to_mergetable(self.merge_table_df, mapping)
        self.mapping = mapping

    def fetch_and_apply_mapping(self, server, uuid, instance, kafka_msgs=None):
        # For testing purposes, we have a special means of avoiding kafkas
        if self.no_kafka:
            kafka_msgs = []
        mapping = fetch_complete_mappings(server, uuid, instance, mutations=kafka_msgs)
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
        focused_merges = self.fetch_focused_merges(server, uuid, focused_decisions_instance)
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
        assert (update_table_df.columns == self.merge_table_df.columns).all(), \
            f"{update_table_df.columns.tolist()} != {self.merge_table_df.columns.tolist()}"

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
