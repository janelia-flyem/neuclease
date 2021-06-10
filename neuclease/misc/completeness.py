import logging

import numpy as np
import pandas as pd

from neuclease.dvid.annotation import body_synapse_counts

logger = logging.getLogger(__name__)


def completeness_forecast(labeled_point_df, partner_df, syn_counts_df=None, conf_threshold=0.0):
    """
    Produces a DataFrame listing all pairwise synapse connections,
    ordered according to the size of the smaller body in the pair.

    This represents a forecast of connectivity completeness, assuming bodies
    will be added to the traced set in order from large to small.
    Bodies are added to the traced set by either declaring them to have traced
    status, or by merging them into a body that already has traced status.

    The results are ordered by the body ranking of the smaller body in each pair,
    ensuring that connections in the table appear in the order in which they will
    become "complete" (i.e. both of their bodies are traced).

    See also: plot_connectivity_forecast(), below.

    Args:
        labeled_point_df:
            DataFrame at least with columns ['kind', 'conf', 'body'],
            indexed by point_id, which must uniquely identify each synapse point.
            This can be obtained via fetch_synapses_in_batches() followed by fetch_labels_batched().
            Alternatively, you can use load_gary_psds() and partner_table_to_synapse_table(),
            if you have a pickle file from Gary.

        partner_df:
            DataFrame with columns ['pre_id', 'post_id'].
            Also obtained via the above-mentioned functions.

        syn_counts_df:
            Optional. DataFrame with columns 'PreSyn', 'PostSyn' and (optionally) 'SynWeight'.
            If provided, specifes the synapse counts per body, and also each body's overall
            'weight' (size), which is used to rank the bodies from largest to smallest.
            The 'SynWeight' column determines the order in which you plan to trace the body set.
            If you don't provide this input, it will be computed from the first two arguments.

        conf_threshold:
            Optional.  If given, synapses will be pre-filtered to satisfy the given minimum confidence score.

    Returns:
        Two DataFrames:
            sorted_connection_df, sorted_bodies_df

        sorted_connection_df has output columns:
            'max_rank': 'body size ranking',
            'traced_tbar_frac': 'tbars on traced bodies',
            'minimally_connected_tbar_frac': 'traced tbars with a traced output',
            'traced_psd_frac': 'psds on traced bodies',
            'traced_conn_frac': 'fully traced connections'

        sorted_bodies_df is indexed by body ID, sorted by the body 'SynWeight'
    """
    point_df = labeled_point_df[['kind', 'conf', 'body']]
    assert point_df.index.name == 'point_id'

    if 0 in point_df['body']:
        logger.info("Filtering body 0 from synapses")
        # Filter points
        point_df = point_df.query('body != 0')

        # Filter pairs
        partner_df = partner_df.merge(point_df[[]], 'inner', left_on='pre_id', right_index=True)
        partner_df = partner_df.merge(point_df[[]], 'inner', left_on='post_id', right_index=True)

    if conf_threshold:
        logger.info(f"Filtering synapses for threshold {conf_threshold}")
        # Filter points
        point_df = point_df.query('conf >= @conf_threshold')

        # Filter pairs
        partner_df = partner_df.merge(point_df[[]], 'inner', left_on='pre_id', right_index=True)
        partner_df = partner_df.merge(point_df[[]], 'inner', left_on='post_id', right_index=True)

    # Append columns ['body_pre', 'body_post']
    logger.info("Appending body columns")
    conn_df = partner_df[['pre_id', 'post_id']]
    conn_df = conn_df.merge(point_df['body'], 'left', left_on='pre_id', right_index=True)
    conn_df = conn_df.merge(point_df['body'], 'left', left_on='post_id', right_index=True, suffixes=['_pre', '_post'])

    if syn_counts_df is None:
        logger.info("Computing per-body synapse table")
        syn_counts_df = body_synapse_counts(point_df)

    if 'SynWeight' not in syn_counts_df.columns:
        # The user didn't provide a preferred body weighting,
        # so calculate each body's SynWeight as its total inputs and outputs.
        # Each PSD counts as 1, each tbar counts as the number of partners it has.
        logger.info("Computing SynWeight column")
        output_counts = conn_df.groupby('body_pre')['pre_id'].size().rename('OutputPartners').rename_axis('body')
        syn_counts_df = syn_counts_df.merge(output_counts, 'left', left_index=True, right_index=True)
        syn_counts_df['OutputPartners'] = syn_counts_df['OutputPartners'].fillna(0.0).astype(int)
        syn_counts_df['SynWeight'] = syn_counts_df.eval('OutputPartners + PostSyn')

    # Rank the bodies from large to small
    logger.info("Ranking bodies")
    syn_counts_df = syn_counts_df.sort_values(['SynWeight'], ascending=False)
    syn_counts_df['rank'] = np.arange(1, 1+len(syn_counts_df), dtype=int)

    # Append columns for the rank of both bodies in each pair,
    # named ['body_pre_rank', 'body_post_rank']
    logger.info("Appending rank columns")
    conn_df = conn_df.merge(syn_counts_df['rank'].rename('body_pre_rank'), 'left', left_on='body_pre', right_index=True)
    conn_df = conn_df.merge(syn_counts_df['rank'].rename('body_post_rank'), 'left', left_on='body_post', right_index=True)

    # Determine the lesser and greater ranks (corresponding to the larger and smaller body).
    # This is just body_pre_rank and body_post_rank, but swapped as needed.
    ranks = np.sort(conn_df[['body_pre_rank', 'body_post_rank']].values, axis=1)
    conn_df['min_rank'] = ranks[:, 0]  # min rank -> larger body
    conn_df['max_rank'] = ranks[:, 1]  # max rank -> smaller body

    # Now order the connection pairs according to the SMALLER of the two bodies in the pair,
    # i.e. sort according to the body with greater rank (i.e. worse rank).
    # The idea is that if we were to trace all bodies in rank-order, the connection pairs
    # will be sorted in the order in which they are completed, i.e. with a traced body on
    # both the input side and the output side.
    #
    # Notice that when the table is sorted this way, then for every row in the table,
    # the preceding rows contain only lesser-ranked bodies (larger bodies) in both the
    # _pre and _post columns.  So, by the time we add a given max_rank body to our "traced set",
    # all of the preceding rows in the connection table have already been traced.
    logger.info("Sorting connections by body rank")
    conn_df = conn_df.sort_values(['max_rank', 'min_rank']).reset_index(drop=True)

    logger.info("Calculating cumulative columns (tbars/psds/connections)")

    # Moving down the list of connection pairs,
    # calculate the number of unique tbars we've seen so far.
    nondupes = ~(conn_df['pre_id'].duplicated())
    conn_df['num_tbars'] = nondupes.cumsum()

    # This indicates the number of unique tbars involved in connections thus far,
    # i.e. the number of tbars with at least one traced output partner.
    # It's not a particularly interesting statistic, but we happen to get it for free here.
    conn_df['minimally_connected_tbar_frac'] = conn_df['num_tbars'] / nondupes.sum()

    # As we move down the list, we're adding all of each body's tbars and psds
    # to our set of "traced" synapses in the sense that they belong to a traced body,
    # but don't necessarily have a traced partner (until we get further down the list).
    # To track the total number of captured tbars and psds at each row in the list,
    # we need to just keep track of the unique bodies mentioned on EITHER body_pre or body_post columns.
    # So, flatten those two columns into one giant list, in order, then merge synapse
    # counts onto the giant list, and compute cumulative sums
    # (but make sure each body only contributes once to each sum).
    flattened_bodies = pd.Series(conn_df[['body_pre', 'body_post']].values.reshape(-1), name='body').to_frame()
    flattened_bodies = flattened_bodies.merge(syn_counts_df, 'left', on='body')
    dupes = flattened_bodies['body'].duplicated()
    flattened_bodies.loc[dupes, 'PreSyn'] = 0
    flattened_bodies.loc[dupes, 'PostSyn'] = 0
    flattened_bodies['traced_tbars'] = flattened_bodies['PreSyn'].cumsum()
    flattened_bodies['traced_psds'] = flattened_bodies['PostSyn'].cumsum()

    # Reshape the above giant lists of cumulatively traced synapses
    # to fit it back into our connection table.
    conn_df['cumulative_traced_tbars'] = flattened_bodies['traced_tbars'].values.reshape(-1, 2).max(axis=1)
    conn_df['cumulative_traced_psds'] = flattened_bodies['traced_psds'].values.reshape(-1, 2).max(axis=1)

    conn_df['traced_tbar_frac'] = conn_df['cumulative_traced_tbars'] / syn_counts_df['PreSyn'].sum()
    conn_df['traced_psd_frac'] = conn_df['cumulative_traced_psds'] / syn_counts_df['PostSyn'].sum()

    # The cumulatively fully-traced connection count (i.e. connections for which both input
    # and output are in traced bodies) is simply the row number, due to the careful sorting
    # of the connection table as explained above.
    # Note: When visualizing, we'll usually want to put this in the Y-axis, not the X-axis.
    conn_df['traced_conn_count'] = 1 + conn_df.index
    conn_df['traced_conn_frac'] = conn_df['traced_conn_count'] / len(conn_df)

    return conn_df, syn_counts_df


def plot_connectivity_forecast(conn_df, max_rank=None, plotted_points=20_000):
    """
    Plot the curves of captured tbars, captured PSDs and captured dual-sided
    connections as bodies are traced/merged from large to small.

    Args:
        conn_df:
            connectivity completion dataframe, as returned by completeness_forecast()
        max_rank:
            Truncate the plot's X axis, stopping at the given rank.
        potted_points:
            The full input dataframe probably has too many points to plot at once.
            You can reduce the resolution of the plot by specifiying how many points
            you want to be shown in total, with this argument.
            You won't notice the difference at all when zoomed out, but if you zoom
            in on the plot you may notice that the X-axis is discontiguous.
    """
    import hvplot.pandas  # noqa

    _df = conn_df
    _df = _df[['max_rank',
               'traced_tbar_frac',
               # 'minimally_connected_tbar_frac',
               'traced_psd_frac',
               'traced_conn_frac']]

    # Zoom in on left-hand region
    if max_rank:
        _df = _df.query('max_rank <= @max_rank')

    _df = _df.drop_duplicates('max_rank', keep='last')

    _df = _df.rename(columns={
        'max_rank': 'body priority ranking',
        'traced_tbar_frac': 'tbars captured',
        # 'minimally_connected_tbar_frac': 'traced tbars with a traced output',
        'traced_psd_frac': 'psds captured',
        'traced_conn_frac': 'pairwise connections'
    })

    # Avoid plotting too many points
    step = max(1, len(_df) // plotted_points)
    _df = _df.iloc[::step]

    p = _df.hvplot(
            _df.columns[0],
            _df.columns[1:].tolist(),
            hover_cols=['tbars captured', 'psds captured', 'pairwise connections'],
            legend='bottom_right',
            ylabel='fraction',
            width=800,
            height=500)

    p.opts(
        title='connectivity after prioritized merging',
        fontsize={
            'title': 15,
            'labels': 14,
            'xticks': 10,
            'yticks': 10,
        }
    )
    # Hide annoying legend title
    # https://discourse.holoviz.org/t/removing-legend-title/1317/2
    p.get_dimension('Variable').label = ''
    return p
