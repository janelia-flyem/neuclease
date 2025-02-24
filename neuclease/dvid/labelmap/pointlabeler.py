import logging
from collections import namedtuple

from ...util import Timer
from .. import fetch_branch_nodes
from ._labelmap import (fetch_mapping, fetch_mappings, fetch_mutations,
                        fetch_labels_batched, fetch_bodies_for_many_points)

logger = logging.getLogger(__name__)
DvidSeg = namedtuple('DvidSeg', 'server uuid instance')


class PointLabeler:
    """
    Utility for labeling many points in a DataFrame with a 'body' column.
    Just a wrapper around fetch_bodies_for_many_points, but caches the labelmap
    mapping and mutations so they can be re-used.
    """

    def __init__(self, server, uuid, instance, mutations=None, mapping=None):
        self.dvidseg = DvidSeg(server, uuid, instance)
        self._mutations = mutations
        self._mapping = mapping
        self._last_mutation = None

    def update_bodies_for_points(self, point_df, batch_size=10_000, threads=0, processes=0):
        """
        Works in-place. Adds a 'body' and 'sv' columns to point_df.
        """
        if len(point_df) < 1_000_000 and 'sv' in point_df.columns:
            # This fast path is convenient for small point lists (especially testing)
            point_df['body'] = fetch_mapping(*self.dvidseg, point_df['sv'].values,
                                             batch_size=batch_size, threads=max(threads, processes))
        elif len(point_df) < 200_000:
            # This fast path is convenient for small point lists (especially testing)
            point_df['sv'] = fetch_labels_batched(*self.dvidseg, point_df[[*'zyx']].values, supervoxels=True,
                                                  batch_size=batch_size, threads=max(threads, processes))
            # This fast path is convenient for small point lists (especially testing)
            point_df['body'] = fetch_mapping(*self.dvidseg, point_df['sv'].values,
                                             batch_size=batch_size, threads=max(threads, processes))
        else:
            # This adds/updates 'sv' and 'body' columns to point_df
            fetch_bodies_for_many_points(*self.dvidseg, point_df, self.mutations, self.mapping,
                                         batch_size=batch_size, threads=threads, processes=processes)

    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = fetch_mappings(*self.dvidseg)
        return self._mapping

    @property
    def mutations(self):
        if self._mutations is None:
            msg = f"Fetching full mutation log for {self.dvidseg.instance} from {self.dvidseg.uuid} and ancestors"
            with Timer(msg, logger):
                self._mutations = fetch_mutations(*self.dvidseg)
        return self._mutations

    @property
    def last_mutation(self):
        if self._last_mutation:
            return self._last_mutation

        branch_nodes = fetch_branch_nodes(
            self.dvidseg.server,
            self.dvidseg.uuid,
            self.dvidseg.uuid,
            full_info=True
        )

        # By default, refer to the UUID log for timestamp
        last_mutation = {
            "uuid": self.dvidseg.uuid,
            "timestamp": branch_nodes.loc[self.dvidseg.uuid, 'Updated'],
            "mutid": 0,
        }

        # Look for the last mutation in the mutation log,
        # searching backwards in the DAG until a non-empty UUID is found.
        for uuid in branch_nodes.index[::-1]:
            muts = fetch_mutations(
                self.dvidseg.server,
                uuid,
                self.dvidseg.instance,
                dag_filter='leaf-only'
            )
            if len(muts):
                last_mutation = muts.iloc[-1].to_dict()
                break

        self._last_mutation = last_mutation
        return last_mutation
