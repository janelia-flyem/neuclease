import logging

from ._labelmap import fetch_mapping, fetch_mappings, fetch_mutations, fetch_bodies_for_many_points

logger = logging.getLogger(__name__)


class PointLabeler:
    """
    Utility for labeling many points in a DataFrame with a 'body' column.
    Just a wrapper around fetch_bodies_for_many_points, but caches the labelmap
    mapping and mutations so they can be re-used.
    """

    def __init__(self, server, uuid, instance, mutations=None, mapping=None):
        self.dvid_seg = (server, uuid, instance)
        self._mutations = mutations
        self._mapping = mapping

    def update_bodies_for_points(self, point_df, batch_size=10_000, threads=0, processes=0):
        """
        Works in-place. Adds a 'body' column to point_df, and also a
        """
        if len(point_df) < 1_000_000 and 'sv' in point_df.columns:
            # This fast path is convenient for small point lists (especially testing)
            point_df['body'] = fetch_mapping(*self.dvid_seg, point_df['sv'].values, batch_size=batch_size, threads=max(threads, processes))
        else:
            # This adds/updates 'sv' and 'body' columns to point_df
            fetch_bodies_for_many_points(*self.dvid_seg, point_df, self.mutations, self.mapping, batch_size=batch_size, threads=threads, processes=processes)

    def mapping(self):
        if self._mapping is None:
            self._mapping = fetch_mappings(*self.dvid_seg)
        return self._mapping

    def mutations(self):
        if self._mutations is None:
            self._mutations = fetch_mutations(*self.dvid_seg)
        return self._mutations
