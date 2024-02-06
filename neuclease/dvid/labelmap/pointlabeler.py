import logging
from collections import namedtuple

from ...util import Timer
from ._labelmap import fetch_mapping, fetch_mappings, fetch_mutations, fetch_labels_batched, fetch_bodies_for_many_points

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
            with Timer(f"Fetching full mutation log for {self.dvidseg.instance} from {self.dvidseg.uuid} and ancestors", logger):
                self._mutations = fetch_mutations(*self.dvidseg)
        return self._mutations
