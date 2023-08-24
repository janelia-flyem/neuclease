import re
import logging

import numpy as np
import pandas as pd

from neuclease.dvid.annotation import body_synapse_counts

logger = logging.getLogger(__name__)


def completeness_forecast(labeled_point_df, partner_df, syn_counts_df=None, body_annotations_df=None, *,
                          min_tbar_conf=0.0, min_psd_conf=0.0, roi=None,
                          sort_by='SynWeight', stop_at_rank=None, _syn_counts_only=False):
    """
    Produces a DataFrame listing all pairwise synapse connections,
    ordered according to the size of the smaller body in the pair.

    This represents a forecast of connectivity completeness, assuming bodies
    will be added to the "traced set" in some order (e.g. large to small).

    The results are ordered by the body ranking of the lesser (e.g. smaller) body
    in each pair, ensuring that connections in the table appear in the order in
    which they will become "complete" (i.e. both of their bodies are traced).

    Note:
        Conceptually, each body's synapses become part of the traced set during
        reconstruction by either declaring the body to have some "traced" status
        or by merging that body into another body which already has traced status.

    See also: plot_connectivity_forecast(), below.

    Args:
        labeled_point_df:
            DataFrame at least with columns ['kind', 'conf', 'body'],
            indexed by point_id, which must uniquely identify each synapse point.
            This can be obtained via fetch_synapses_in_batches() followed by fetch_labels_batched().
            Alternatively, you can use load_gary_psds() and partner_table_to_synapse_table(),
            if you have a pickle file from Gary.

        partner_df:
            DataFrame with columns ['pre_id', 'post_id'] and optionally ['body_pre', 'body_post'].
            If the body columns are provided, they need not be recalculated in this function.

        syn_counts_df:
            Optional. DataFrame with columns 'PreSyn', 'PostSyn' and (optionally) 'SynWeight'.
            If provided, specifes the synapse counts per body, and also each body's overall
            'weight' (size), which is used to rank the bodies from largest to smallest.
            The 'SynWeight' column determines the order in which you plan to trace the body set.
            If you don't provide this input, it will be computed from the first two arguments.

            Note:
                It is your responsibility to ensure that this pre-computed input is consistent
                with any filtering criteria you specified (e.g. roi, min_tbar_conf, min_psd_conf).

        body_annotations_df:
            Per-body annotation columns to append to the syn_counts_df.
            Useful for making more columns available to the sort_by argument.
            If syn_counts_df already contains matching columns, then
            body_annotations_df columns will override the columns in syn_counts_df.

        min_tbar_conf:
            Optional. Filter out tbars that don't meet this confidence threshold.

        min_psd_conf:
            Optional. Filter out psds that don't meet this confidence threshold.

        roi:
            String or list of strings. Filter out synapses that don't fall within one of the given ROIs.
            If this argument is used, then point_df must contain a column named 'roi'.

        sort_by:
            Column(s) in syn_counts_df to sort by in descending order.
            By default, SynWeight is used (which is calculated here if needed).
            This sort order determines the ordering of the connections in the result table.

        stop_at_rank:
            If you're only interested in the first N ranked bodies, then
            use this parameter to limit the size of the results,
            and also speed up the computation.

        _syn_counts_only:
            Internal use.

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
    # Validate and standardize arguments
    args = _sanitize_args(
        labeled_point_df, partner_df, syn_counts_df, body_annotations_df,
        min_tbar_conf, min_psd_conf, roi, sort_by, stop_at_rank)

    (labeled_point_df, partner_df, syn_counts_df, body_annotations_df,
        min_tbar_conf, min_psd_conf, roi, sort_by, stop_at_rank) = args

    point_df, partner_df = _filter_synapses(labeled_point_df, partner_df, min_tbar_conf, min_psd_conf, roi)
    conn_df = _body_conn_df(point_df, partner_df)
    syn_counts_df = _rank_syn_counts(point_df, conn_df, syn_counts_df, body_annotations_df, sort_by)
    if _syn_counts_only:
        return syn_counts_df

    conn_df = _completeness_forecast(conn_df, syn_counts_df, stop_at_rank)
    return conn_df, syn_counts_df


def ranked_synapse_counts(labeled_point_df, partner_df, *, body_annotations_df=None,
                          min_tbar_conf=0.0, min_psd_conf=0.0, roi=None,
                          sort_by='SynWeight'):
    """
    Calculate aggregate synapse weights and rankings for
    the synapses reported in labeled_point_df and partner_df.
    This is the same procedure as used in completeness_forecast,
    but without the final forecast ranking step.

    For argument descriptions, see completeness_forecast().
    """
    return completeness_forecast(
        labeled_point_df,
        partner_df,
        None,
        body_annotations_df,
        min_tbar_conf=min_tbar_conf,
        min_psd_conf=min_psd_conf,
        roi=roi,
        sort_by=sort_by,
        _syn_counts_only=True
    )


def _sanitize_args(labeled_point_df, partner_df, syn_counts_df, body_annotations_df,
                   min_tbar_conf, min_psd_conf, roi,
                   sort_by, stop_at_rank):
    """
    Check all arguments for validity and make minor changes if needed,
    such as subsetting columns or tweaking types.

    Since completeness_forecast() takes a very long time to run, it's convenient to
    have all arguments validated as the very first step.

    Returns all arguments, possibly with minor transformations.
    """
    if (roi or min_tbar_conf or min_psd_conf) and syn_counts_df is not None:
        msg = (
            "\n"
            "  You have specified filtering criteria (e.g. roi or conf), but you also provided a pre-computed syn_counts_df.\n"
            "  It is your responsibility to ensure that syn_counts_df was generated using the same filtering criteria.\n"
            "  To avoid this warning, omit syn_count_df and it will be computed for you."
        )
        logger.warning(msg)

    # Subset columns for labeled_point_df
    assert labeled_point_df.index.name == 'point_id'
    point_cols = ['kind', 'conf', 'body']
    if roi:
        assert 'roi' in labeled_point_df.columns

    if 'roi' in labeled_point_df.columns:
        point_cols = [*point_cols, 'roi']

    labeled_point_df = labeled_point_df[point_cols]

    partner_cols = [c for c in partner_df.columns
                    if c in ['pre_id', 'post_id', 'body_pre', 'body_post', 'roi']]
    partner_df = partner_df[partner_cols]

    assert (
        syn_counts_df is None or (
            isinstance(syn_counts_df, pd.DataFrame) and  # noqa
            syn_counts_df.index.name == 'body'
        )
    )
    assert (
        body_annotations_df is None or (
            isinstance(body_annotations_df, pd.DataFrame) and  # noqa
            body_annotations_df.index.name == 'body'
        )
    )

    if isinstance(sort_by, str):
        sort_by = [sort_by]

    # We can only sort using any columns the user provided,
    # plus SynWeight, which we can provide below.
    available_columns = {'PreSyn', 'PostSyn', 'SynWeight'}
    if body_annotations_df is not None:
        available_columns |= set(body_annotations_df.columns)
    if syn_counts_df is not None:
        available_columns |= set(syn_counts_df.columns)
        assert {*syn_counts_df.columns} >= {'PreSyn', 'PostSyn'}, \
            "synapse counts table needs columns for PreSyn and PostSyn"

    assert {*sort_by} <= available_columns, (
        "If you want to sort using non-standard columns, "
        "you have to provide them in syn_counts_df or body_annotations_df")

    return (labeled_point_df, partner_df, syn_counts_df, body_annotations_df,
            min_tbar_conf, min_psd_conf, roi,
            sort_by, stop_at_rank)


def _completeness_forecast(conn_df, syn_counts_df, stop_at_rank, _debug_cols=False):
    """
    Main implementation of completeness_forecast()

    Args:
        conn_df:
            Pre-filtered synpase connection table including body columns:
            ['pre_id', 'post_id', 'body_pre', 'body_post']
        syn_counts_df:
            Sorted synapse counts.
            Indexed by body, with at least columns ['PreSyn', 'PostSyn', 'rank']
        stop_at_rank:
            See description in completeness_forecast()

    Returns:
        Sorted conn_df, with added columns as described in completeness_forecast(),
        and truncated according to stop_at_rank.
    """
    full_conn_count = len(conn_df)

    # Filter out bodies above our rank of interest (too small)
    stop_at_rank = stop_at_rank or len(syn_counts_df)
    body_rank = syn_counts_df.query('rank <= @stop_at_rank')['rank']

    # Append columns for the rank of both bodies in each pair,
    # named ['body_pre_rank', 'body_post_rank']
    # Use an inner merge to simultaneously filter out high-ranking bodies (small bodies).
    logger.info("Appending rank columns")
    conn_df = conn_df.merge(body_rank.rename('body_pre_rank'), 'inner', left_on='body_pre', right_index=True)
    conn_df = conn_df.merge(body_rank.rename('body_post_rank'), 'inner', left_on='body_post', right_index=True)

    # Determine the lesser and greater ranks (corresponding to the larger and smaller body).
    # This is just body_pre_rank and body_post_rank, but swapped as needed.
    prepost_ranks = np.sort(conn_df[['body_pre_rank', 'body_post_rank']].values, axis=1)
    conn_df['min_rank'] = prepost_ranks[:, 0]  # min rank -> larger body
    conn_df['max_rank'] = prepost_ranks[:, 1]  # max rank -> smaller body

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
    conn_df = conn_df.sort_values(['max_rank', 'min_rank'], ignore_index=True)

    syn_counts_df.columns = [c.lower() for c in syn_counts_df.columns]
    syncols = [c for c in syn_counts_df.columns if c in ('presyn', 'postsyn', 'synweight')]
    logger.info(f"Calculating cumulative columns for: {', '.join([*syncols, 'conn'])}")

    if _debug_cols:
        # Moving down the list of connection pairs,
        # calculate the number of unique tbars we've seen so far.
        nondupes = ~(conn_df['pre_id'].duplicated())
        conn_df['num_tbars'] = nondupes.cumsum()

        # This indicates the number of unique tbars involved in connections thus far,
        # i.e. the number of tbars with at least one traced output partner.
        # It's not a particularly interesting statistic, but we happen to get it for free here.
        conn_df['minimally_connected_tbar_frac'] = conn_df['num_tbars'] / nondupes.sum()
        del nondupes

    # As we move down the list, we're adding all of each body's tbars and psds
    # to our set of "traced" synapses in the sense that they belong to a traced body,
    # but don't necessarily have a traced partner (until we get further down the list).
    # To track the total number of captured tbars and psds at each row in the list,
    # we need to just keep track of the unique bodies mentioned on EITHER body_pre or body_post columns.
    # So, flatten those two columns into one giant list, in order, then merge synapse
    # counts onto the giant list, and compute cumulative sums
    # (but make sure each body only contributes once to each sum).
    flattened_bodies = pd.DataFrame({'body': conn_df[['body_pre', 'body_post']].values.reshape(-1)})
    flattened_bodies = flattened_bodies.merge(syn_counts_df, 'left', on='body')
    dupes = flattened_bodies['body'].duplicated()
    for c in syncols:
        flattened_bodies.loc[dupes, c] = 0
        flattened_bodies[f'traced_{c}'] = flattened_bodies[c].cumsum()
    del dupes

    # Reshape the above giant lists of cumulatively traced synapses
    # to fit it back into our connection table.
    for c in syncols:
        conn_df[f'cumulative_traced_{c}'] = flattened_bodies[f'traced_{c}'].values.reshape(-1, 2).max(axis=1)
    del flattened_bodies

    for c in syncols:
        conn_df[f'traced_{c}_frac'] = conn_df[f'cumulative_traced_{c}'] / syn_counts_df[c].sum()

    # The cumulatively fully-traced connection count (i.e. connections for which both input
    # and output are in traced bodies) is simply the row number, due to the careful sorting
    # of the connection table as explained above.
    # Note: When visualizing, we'll usually want to put this in the Y-axis, not the X-axis.
    conn_df['cumulative_traced_conn'] = 1 + conn_df.index
    conn_df['traced_conn_frac'] = conn_df['cumulative_traced_conn'] / full_conn_count

    # It's useful to provide the max body rank
    logger.info("Adding body_max_rank and associated stats")
    conn_df['body_max_rank'] = conn_df['body_pre']
    post_is_max = conn_df.eval('body_post_rank > body_pre_rank')
    conn_df.loc[post_is_max, 'body_max_rank'] = conn_df.loc[post_is_max, 'body_post']
    del post_is_max

    # It's also useful to see the synapse stats of the body with max rank,
    # since the 'max rank' body is the body that's conceptually being "appended"
    # to the traced set.
    body_max_syn_counts = syn_counts_df.rename(
        columns={c: f'{c}_max_rank' for k in syncols})
    conn_df = conn_df.merge(body_max_syn_counts, 'left', left_on='body_max_rank', right_index=True)

    return conn_df


def _filter_synapses(point_df, partner_df, min_tbar_conf=0.0, min_psd_conf=0.0, roi=None):
    """
    Filter rows out of the given synapse point table (point_df) and pre-post
    connection table (partner_df) as specified by the other parameters.

    In addition to filtering out rows according to the input criteria,
    synapses will also be removed if their partner(s) were filtered out.
    So, if a PSD's tbar is filtered out, then the PSD will also be removed from
    these results, even if the PSD itself would have passed the given filters.

    Args:
        point_df:
            As explained in completeness_forecast()
        partner_df:
            As explained in completeness_forecast()
        min_tbar_conf:
            Minimum tbar confidence to include in the output.
        min_psd_conf:
            Minimum PSD confidence to include in the output.
        roi:
            ROI name or list of names.
            If this is provided, then point_df must contain a column for 'roi'.
            Rows which don't match one of the given ROI names will be excluded from the results.

    Returns:
        (point_df, partner_df)
        Same as the input, but excluding rows that were filtered out.
    """
    filters = []

    if 0 in point_df['body'].values:
        filters.append('body != 0')
    if min_tbar_conf:
        filters.append(f'kind != "PreSyn" or conf >= {min_tbar_conf}')
    if min_psd_conf:
        filters.append(f'kind != "PostSyn" or conf >= {min_psd_conf}')
    if roi:
        if isinstance(roi, str):
            roi = [roi]
        assert 'roi' in point_df.columns
        filters.append('roi in @roi')

    if filters:
        filters = [f'({f})' for f in filters]
        q = ' and '.join(filters)
        logger.info(f"Filtering points with: {q}")
        point_df = point_df.query(q)

    # Filter partner pairs (even if there were no specified filters),
    # in case the user's point_df and partner_df aren't perfectly matched
    # (e.g. if they did some pre-filtering of point_df.)
    logger.info("Filtering partners to exclude unlisted points")
    valid_pre = partner_df['pre_id'].isin(point_df.index)
    valid_post = partner_df['post_id'].isin(point_df.index)
    partner_df = partner_df.loc[valid_pre & valid_post].copy()

    # Also filter point list again, to toss out points which had no partner
    logger.info("Filtering points to exclude orphaned tbars/psds")
    valid_ids = pd.concat((partner_df['pre_id'].drop_duplicates().rename('point_id'),  # noqa
                           partner_df['post_id'].drop_duplicates().rename('point_id')),
                          ignore_index=True)
    point_df = point_df.loc[point_df.index.isin(valid_ids)]
    return point_df, partner_df


def _body_conn_df(point_df, partner_df):
    """
    Produce a point-wise connection table with additional columns for pre/post body IDs.
    If partner_df already contains the necessary columns, return them as-is.
    Otherwise, compute them.
    """
    body_cols = ['body_pre', 'body_post']
    if 'roi' in point_df.columns:
        body_cols = [*body_cols, 'roi']

    # Did the user already provide the {body_pre, body_post, roi} columns?
    if set(body_cols) <= {*partner_df.columns}:
        return partner_df[['pre_id', 'post_id', *body_cols]]

    return merge_body_partner_columns(point_df, partner_df)


def merge_body_partner_columns(point_df, partner_df):
    """
    Produce a point-wise connection table with additional columns for pre/post body IDs.

    Onto the given pre-to-post connection table given in partner_df,
    merge additional columns ['body_pre', 'body_post'] (and optionally 'roi'),
    obtained from the given point_df table.

    Returns:
        DataFrame with columns ['pre_id', 'post_id', 'body_pre', 'body_post', 'roi']

        ('roi' is included iff it is present in point_df)
    """
    post_merge_cols = ['body']
    if 'roi' in point_df.columns:
        # If the point_df includes 'roi', then we assign it to each connection according
        # to the PSD location (not the pre location), which is consistent with our general
        # conventions, e.g. in neuprint.
        post_merge_cols = ['body', 'roi']

    # Append columns ['body_pre', 'body_post']
    logger.info("Appending body columns")
    conn_df = partner_df[['pre_id', 'post_id']]
    conn_df = conn_df.merge(point_df['body'], 'left', left_on='pre_id', right_index=True)
    conn_df = conn_df.merge(point_df[post_merge_cols], 'left', left_on='post_id', right_index=True, suffixes=['_pre', '_post'])
    return conn_df


def _rank_syn_counts(point_df, conn_df, syn_counts_df=None, body_annotations_df=None, sort_by='SynWeight'):
    """
    - Generate a synapse counts table if one isn't provided.
    - Add a SynWeight column if necessary
    - Sort, and add a rank column
    """
    if syn_counts_df is None:
        logger.info("Computing per-body synapse table")
        syn_counts_df = body_synapse_counts(point_df)

    if body_annotations_df is not None:
        assert body_annotations_df.index.name == 'body'
        syn_counts_df = syn_counts_df.drop(columns=body_annotations_df.columns, errors='ignore')
        syn_counts_df = syn_counts_df.merge(body_annotations_df, 'left', on='body')

    # For categorical dtypes which include the empty string as a category,
    # we auto-convert NaN to "".
    # This way, NaN items inherit the same sort position as "".
    for col in syn_counts_df.columns:
        try:
            cat = syn_counts_df[col].dtype.categories
        except AttributeError:
            continue
        else:
            if '' in cat:
                syn_counts_df[col].fillna('', inplace=True)

    if 'SynWeight' in sort_by and 'SynWeight' not in syn_counts_df.columns:
        # The user didn't provide a preferred body weighting,
        # so calculate each body's SynWeight as its total inputs and outputs.
        # Each PSD counts as 1, each tbar counts as the number of partners it has.
        logger.info("Computing SynWeight column")

        # FIXME: Would value_counts() be faster than groupby()[].size()
        output_counts = conn_df.groupby('body_pre')['pre_id'].size().rename('OutputPartners').rename_axis('body')
        syn_counts_df = syn_counts_df.merge(output_counts, 'left', left_index=True, right_index=True)
        syn_counts_df['OutputPartners'] = syn_counts_df['OutputPartners'].fillna(0.0).astype(int)
        syn_counts_df['SynWeight'] = syn_counts_df.eval('OutputPartners + PostSyn')
        assert syn_counts_df['SynWeight'].sum() == 2 * syn_counts_df['PostSyn'].sum()

    if 'roi' in conn_df.columns:
        logger.info("Computing top roi columns")

        # It's much faster and more convenient to use 2-column value_counts()
        # than it is to use groupby().agg(pd.Series.mode)
        top_output_roi = (
            conn_df[['body_pre', 'roi']]
            .value_counts()
            .groupby('body_pre')
            .head(1)
            .reset_index(1)
            .rename_axis('body')
            .rename(columns={
                'roi': 'top_output_roi',
                0: 'top_output_roi_weight'
            })
        )
        syn_counts_df = syn_counts_df.merge(top_output_roi, 'left', left_index=True, right_index=True)
        syn_counts_df['top_output_roi_weight'] = syn_counts_df['top_output_roi_weight'].fillna(0.0).astype(int)

        top_input_roi = (
            conn_df[['body_post', 'roi']]
            .value_counts()
            .groupby('body_post')
            .head(1)
            .reset_index(1)
            .rename_axis('body')
            .rename(columns={
                'roi': 'top_input_roi',
                0: 'top_input_roi_weight'
            })
        )
        syn_counts_df = syn_counts_df.merge(top_input_roi, 'left', left_index=True, right_index=True)
        syn_counts_df['top_input_roi_weight'] = syn_counts_df['top_input_roi_weight'].fillna(0.0).astype(int)

    # Rank the bodies from large to small
    logger.info("Ranking bodies")
    syn_counts_df = syn_counts_df.sort_values(sort_by, ascending=False)
    syn_counts_df['rank'] = np.arange(1, 1+len(syn_counts_df), dtype=int)

    return syn_counts_df


def plot_connectivity_forecast(conn_df, max_rank=None, plotted_points=20_000, hover_cols=[], color_by_col=None,
                               title='connectivity after prioritized merging', export_path=None):
    """
    Plot the curves of captured tbars, captured PSDs and captured dual-sided
    connections as bodies are traced/merged from large to small.

    Note:
        Not all bodies are included in the resulting plot.
        We'll just be plotting the stats associated with the "max_rank" body,
        but some bodies never appear in the table as the "max_rank" body at all.
        If a body is "larger" (i.e. better ranked) than ALL bodies it conencts to,
        then it will never be the "max_rank" body and will therefore not be included
        in these plot curves.

    Args:
        conn_df:
            connectivity completion dataframe, as returned by completeness_forecast()
        max_rank:
            Truncate the plot's X axis, stopping at the given rank.
        plotted_points:
            The full input dataframe probably has too many points to plot at once.
            You can reduce the resolution of the plot by specifiying how many points
            you want to be shown in total, with this argument.
            You won't notice the difference at all when zoomed out, but if you zoom
            in on the plot you may notice that the X-axis is discontiguous.
        hover_cols:
            Columns to display in the hover text.
            Any '*_max_rank' columns should be referred to by their prefix only.
            For example, use ``hover_cols=['SynWeight']``, not ``hover_cols=['SynWeight_max_rank']``.
    """
    import holoviews as hv  # noqa
    import hvplot.pandas    # noqa
    from bokeh.plotting import figure, output_file, save as bokeh_save, output_notebook, show  # noqa

    # import holoviews as hv
    # hv.renderer('bokeh').theme = 'dark_minimal'

    assert not export_path or export_path.endswith('.html')

    if color_by_col and color_by_col not in hover_cols:
        hover_cols.append(color_by_col)

    _df = conn_df

    # conn_df will contain columns like 'SynWeight_max_rank',
    # but for simplicity we display that as 'SynWeight' in the hover text.
    _df = _df.rename(columns={f'{c}_max_rank': c for c in {*hover_cols}})

    show_cols = ['traced_presyn_frac',
                 # 'minimally_connected_tbar_frac',
                 'traced_synweight_frac',
                 'traced_postsyn_frac',
                 'traced_conn_frac']

    show_cols = [c for c in show_cols if c in _df.columns]

    # Zoom in on left-hand region
    if max_rank:
        _df = _df.query('max_rank <= @max_rank')

    _df = _df.drop_duplicates('max_rank', keep='last')

    renames = {
        'max_rank': 'body priority ranking',
        'traced_presyn_frac': 'tbars captured',
        # 'minimally_connected_tbar_frac': 'traced tbars with a traced output',
        'traced_synweight_frac': 'synweight captured',
        'traced_postsyn_frac': 'psds captured',
        'traced_conn_frac': 'pairwise connections'
    }
    renames = {k: v for k,v in renames.items() if k in _df.columns}
    _df = _df.rename(columns=renames)

    # Avoid plotting too many points
    step = max(1, len(_df) // plotted_points)
    _df = _df.iloc[::step]

    if color_by_col:
        p = _df.hvplot.scatter(
                'body priority ranking',  # noqa
                [renames[k] for k in show_cols],
                hover_cols=[*renames.values(), *hover_cols],
                legend='bottom_right',
                ylabel='fraction',
                by=color_by_col,
                width=800,
                height=500)
    else:
        p = _df.hvplot(
                'body priority ranking', # noqa
                [renames[k] for k in show_cols],
                hover_cols=[*renames.values(), *hover_cols],
                legend='bottom_right',
                ylabel='fraction',
                width=800,
                height=500)

        # Hide annoying legend title
        # https://discourse.holoviz.org/t/removing-legend-title/1317/2
        p.get_dimension('Variable').label = ''

    p.opts(
        title=title,
        fontsize={
            'title': 15,
            'labels': 14,
            'xticks': 10,
            'yticks': 10,
        }
    )

    # Render to bokeh so we can export html
    if export_path:
        output_file(filename=export_path, title=title)
        bokeh_save(hv.render(p))

    return p


def plot_categorized_connectivity_forecast(
        conn_df, category_col, max_rank=None, plotted_points=20_000, hover_cols=[],
        title='connectivity after prioritized merging', export_path=None, selection_link=None,
        secondary_line='synweight', secondary_categories=['Anchor', '0.5assign', ''],
        secondary_range=[0, 400]):
    """
    Plot the curves of captured tbars, captured PSDs and captured dual-sided
    connections as bodies are traced/merged from large to small.

    This function is similar to plot_connectivity_forecast() (above),
    but this plots the curves in several segments, according to a
    category provided in 'color_by_col'.

    Note:
        This function returns a bokeh plot, not holoviews.
        Display it in a notebook via bokeh.plotting.show().

    Note:
        Not all bodies are included in the resulting plot.
        We'll just be plotting the stats associated with the "max_rank" body,
        but some bodies never appear in the table as the "max_rank" body at all.
        If a body is "larger" (i.e. better ranked) than ALL bodies it connects to,
        then it will never be the "max_rank" body and will therefore not be included
        in these plot curves.

    Args:
        conn_df:
            connectivity completion dataframe, as returned by completeness_forecast()
        max_rank:
            Truncate the plot's X axis, stopping at the given rank.
        plotted_points:
            The full input dataframe probably has too many points to plot at once.
            You can reduce the resolution of the plot by specifiying how many points
            you want to be shown in total, with this argument.
            You won't notice the difference at all when zoomed out, but if you zoom
            in on the plot you may notice that the X-axis is discontiguous.
        hover_cols:
            Columns to display in the hover text.
            Any '*_max_rank' columns should be referred to by their prefix only.
            For example, use ``hover_cols=['SynWeight']``, not ``hover_cols=['SynWeight_max_rank']``.
    """
    from bokeh.plotting import figure, output_file, save as bokeh_save
    from bokeh.models import HoverTool, Range1d, LinearAxis
    from bokeh.palettes import Category20

    assert not export_path or export_path.endswith('.html')

    _df = conn_df

    # conn_df will contain columns like 'SynWeight_max_rank',
    # but for simplicity we display that as 'SynWeight' in the hover text.
    _df = _df.rename(columns={f'{c}_max_rank': c for c in {category_col, *hover_cols}})

    # Zoom in on left-hand region
    if max_rank:
        _df = _df.query('max_rank <= @max_rank')

    _df = _df.drop_duplicates('max_rank', keep='last')

    # Avoid plotting too many points
    step = max(1, len(_df) // plotted_points)
    _df = _df.iloc[::step]

    p = figure(align='center', height=500, width=800, title=title, y_range=(0, 1.0))
    p.title.text_font_size = '14pt'
    p.xaxis.axis_label = 'body rank'
    p.yaxis.axis_label = 'fraction of captured tbars, synweight, psds, and connectivity'
    p.yaxis.ticker = np.arange(0.0, 1.1, 0.1)

    if secondary_line:
        p.extra_y_ranges[secondary_line] = Range1d(*secondary_range)
        ax2 = LinearAxis(y_range_name=secondary_line, axis_label=secondary_line)
        ax2.axis_label_text_color = "navy"
        p.add_layout(ax2, 'right')

    dots = []
    for i, (cat, df) in list(enumerate(_df.groupby(category_col, observed=True)))[::-1]:
        p.line('max_rank', 'traced_presyn_frac', color=Category20[20][2 * (i % 10) + 1], line_width=5, source=df)
        if 'traced_synweight_frac' in df.columns:
            p.line('max_rank', 'traced_synweight_frac', color=Category20[20][2 * (i % 10)], line_width=5, source=df)
        p.line('max_rank', 'traced_postsyn_frac', color=Category20[20][2 * (i % 10) + 1], line_width=5, source=df)

        # Insert legend this time, too.
        p.line('max_rank', 'traced_conn_frac', color=Category20[20][2 * (i % 10)], line_width=5, source=df, legend_label=cat)

        if secondary_line and cat in secondary_categories:
            p.line('max_rank', secondary_line, color="navy", y_range_name=secondary_line, source=df)

        # if selection_link:
        #     # https://github.com/bokeh/bokeh/issues/10056#issuecomment-1308510074
        #     d1 = p.dot('max_rank', 'traced_presyn_frac', legend_label=cat, color=Category20[20][2 * (i % 10)], source=df)
        #     d2 = p.dot('max_rank', 'traced_postsyn_frac', color=Category20[20][2 * (i % 10) + 1], source=df)
        #     d3 = p.dot('max_rank', 'traced_conn_frac', color=Category20[20][2 * (i % 10)], source=df)
        #     dots.extend([d1, d2, d3])

    p.legend.location = "top_right"

    hover = HoverTool()
    hover.tooltips = [
        ("body rank", "@max_rank"),
        (category_col, f"@{category_col}"),
        ("tbars captured", "@traced_presyn_frac"),
    ]
    if 'traced_synweight_frac' in _df.columns:
        hover.tooltips.append(
            ("synweight captured", "@traced_synweight_frac")
        )

    hover.tooltips += [
        ("psds captured", "@traced_postsyn_frac"),
        ("connections captured", "@traced_conn_frac"),
    ]
    hover.tooltips += [("body", "@body_max_rank")]
    hover.tooltips += [(col, f"@{col}") for col in hover_cols]

    p.add_tools(hover)

    if selection_link:
        _add_link_taptool(p, selection_link, dots)

    if export_path:
        output_file(filename=export_path, title=title)
        bokeh_save(p)

    return p


def _add_link_taptool(bokeh_plot, template_link, dots):
    from bokeh.models import ColumnDataSource, OpenURL, TapTool, BoxSelectTool, HoverTool, TextAreaInput, CustomJS
    from neuclease.misc.neuroglancer import parse_nglink, format_nglink

    # Pre-generate a neuroglancer link to use if the user clicks on a point in the scatter plot.
    # Start with a generic link, then overwrite some settings.
    link_data = parse_nglink(template_link)

    # Note: In the template link, the segmentation layer MUST be the currently selected layer.
    selected_layer = link_data['selectedLayer']['layer']
    sel_i = [i for i, l in enumerate(link_data['layers']) if l['name'] == selected_layer][0]

    link_data['layers'][sel_i]['segmentQuery'] = "999999999"
    link_data['layers'][sel_i]['segments'] = ["9191919191"]
    template_link = format_nglink('https://clio-ng.janelia.org', link_data)

    # The OpenURL function will replace variables in the link (e.g. @body_max_rank)
    # with data from the ColumnDataSource (we added these variables to the CDS above).
    onclick_link = template_link.replace('999999999', '@body_max_rank')
    onclick_link = onclick_link.replace('9191919191', '@body_max_rank')

    # If the user clicks on a point, open the URL.
    # Docs: https://docs.bokeh.org/en/2.4.0/docs/user_guide/interaction/callbacks.html#openurl
    taptool = TapTool()
    taptool.callback = OpenURL(url=onclick_link)
    bokeh_plot.add_tools(taptool)

    # # Supposedly there's a way to get selection working for line glyphs,
    # # but I'm not sure I'm smart enough:
    # # https://github.com/bokeh/bokeh/issues/10056#issuecomment-1308510074
    # bokeh_plot.add_tools(BoxSelectTool(dimensions='width', renderers=dots))

    # # The JavaScript features below will make use of
    # # variables stored in the plot's "ColumnDataSource".
    # # The CDS already contains columns for the data in the scatter plot (e.g. px, py, color, etc.),
    # # but we need to sneak some extra stuff in there: body IDs and 3D coordinates (mean positions).
    # cds = bokeh_plot.select(type=ColumnDataSource)[0]
    # cds.selected.js_on_change(
    #     'indices',
    #     CustomJS(args={'cds': cds}, code="""\
    #         // Extract the body IDs from the column data that was stored in JavaScript.
    #         // The user's selected indices are passed via the callback object (cb_obj).
    #         const inds = cb_obj.line_indices;
    #         var bodies = [];
    #         for (let i = 0; i < inds.length; i++) {
    #             bodies.push(cds.data['body'][inds[i]]);
    #         }

    #         console.log("Trying..." + bodies.join(', '))

    #         // Open the template link, with the selected bodies
    #         var link = "TEMPLATE_LINK";
    #         link.replace("999999999", bodies.join(', '));
    #         link.replace("[\"9191919191\"]", bodies.map(String));
    #         window.open(link, '_blank').focus();
    #         console.log("Trying " + link)

    #         // If possible, also copy the body IDs to the user's clipboard.
    #         //
    #         // NOTE:
    #         //   Javascript doesn't allow writing to the clipboard unless:
    #         //   This page is served over https (so, this notebook won't work...)
    #         //   OR the page is hosted via a static local file://
    #         try {
    #             navigator.clipboard.writeText(bodies.join('\\n'));
    #         }
    #         catch (err) {
    #             console.error("Couldn't write body list to clipboard:", err)
    #         }
    # """.replace("TEMPLATE_LINK", template_link)))


def variable_width_hbar(df, bar_name, bar_width, value, color=None, stackcolors=None, *,
                        title=None, pad=None, width=800, height=800, vlim=None, vticker=None,
                        legend='bottom_right', flip_yaxis=False):
    """
    Create a horizontal bar chart whose bars have variable bases (widths).

    Args:
        df:
            DataFrame
        bar_name:
            Name of the column which contains the names of each bar
        bar_width:
            Name of the column which specifies the width of each bar, in arbitrary units.
        value:
            Name or names of the column(s) which specify the length of each bar, in arbitrary units.
            If a list of value columns is given, then a stacked bar chart will be created.
            The second value will be stacked on top of the first, etc.
        color:
            A column which contains the a list of colors to apply to each bar.
            Only used in non-stacked mode (when there is only one value column.)
        stackcolors:
            A column which contains the list of colors to apply to each stacked portion of every bar.
            That is, if you specified three value columns, then specify three stackcolors.
            Every stacked bar will be colored with the same stackcolors.
        title:
            Plot title.
        pad:
            Padding constant.  Expressed as a fraction of the figure's
            vertical span BEFORE the padding is inserted.
        width:
            Plot width, in screen units.
        height:
            Plot height, in screen units.
        vlim:
            The range of the value axis, i.e. the x-axis (since this is an HBar chart).
        vticker:
            Optionally provide explicit major tick locations.
        legend:
            Optionally specify the placement of the legend.
            (Only stacked bars will have a legend.)

    Returns:
        Bokeh figure

    Note:
        This is sometimes referred to as a cascade chart,
        although that seems to be a confusing term:
        https://stats.stackexchange.com/questions/159163

    TODO:
        This function doesn't have much to do with the rest of this file.
        It should be moved elsewhere.
    """
    from bokeh.palettes import Category10
    from bokeh.plotting import figure
    from bokeh.models import HoverTool
    # This is an HBar, so names will go on y-axis, values in x direction.
    df = df.copy()
    df['bar_name'] = df[bar_name]

    if isinstance(value, str):
        value = [value]

    def fix_name(col):
        return re.sub(r'[^0-9A-Za-z]', '_', col)
    normalized_value_cols = {v: fix_name(v) for v in value}
    df[[*normalized_value_cols.values()]] = df[[*normalized_value_cols.keys()]]

    if stackcolors is None:
        stackcolors = Category10[10] * ((len(value) + 9) // 10)

    if len(value) > 1:
        assert len(stackcolors) >= len(value)
        stackcolors = stackcolors[:len(value)]

    df['x0'] = 0
    for i, v in enumerate(value, start=1):
        df[f'x{i}'] = df[f'x{i-1}'] + df[value[i-1]]

    df['bar_width'] = df[bar_width]

    # For some reason, overriding the ticker with gigantic numbers doesn't work well,
    # so we scale the widths down to the 0-1 range.
    df['scaled_bar_width'] = df[bar_width] / df[bar_width].sum()

    if pad is None:
        df['pad'] = min(0.005, df['scaled_bar_width'].min())
    else:
        df['pad'] = pad

    df['padsum'] = df['pad'].cumsum().shift(1).fillna(0.0)

    df['y0'] = df['scaled_bar_width'].cumsum().shift(1).fillna(0.0) + df['padsum']
    df['y1'] = df['scaled_bar_width'].cumsum() + df['padsum']
    df['y'] = df.eval('(y0 + y1) / 2')

    if color is None:
        color = Category10[10] * ((len(value) + 9) // 10)

    if isinstance(color, str):
        df['_color'] = color
    else:
        df['_color'] = (list(color) * ((len(df) + len(color) - 1) // len(color)))[:len(df)]

    p = figure(
        title=title,
        width=width,
        height=height,
        tools="ypan,ybox_zoom,ywheel_pan,reset",
        active_drag='ybox_zoom',
        active_scroll='ywheel_pan',
    )
    if len(value) == 1:
        p.quad('x0', 'x1', 'y0', 'y1', color='_color', source=df)
    else:
        for i, (c, v) in enumerate(zip(stackcolors, value)):
            p.quad(f'x{i}', f'x{i+1}', 'y0', 'y1', color=c, source=df, legend_label=v)
        p.legend.location = legend

    hover = HoverTool()
    hover.tooltips = [
        (bar_name, "@bar_name"),
        (bar_width, "@bar_width"),
    ]
    for v, n in normalized_value_cols.items():
        hover.tooltips.append((v, f"@{n}"))

    p.add_tools(hover)

    if vlim is not None:
        p.x_range.start = vlim[0]
        p.x_range.end = vlim[1]

    if vticker is not None:
        p.xaxis.ticker = [0, 0.25, 0.5, 0.75, 1.0]

    p.yaxis.ticker = df['y']
    p.yaxis.major_label_overrides = dict(zip(df['y'], df[bar_name]))
    p.ygrid.grid_line_color = None
    p.y_range.start = df['y0'].min()
    p.y_range.end = df['y1'].max()

    if flip_yaxis:
        p.y_range.start, p.y_range.end = p.y_range.end, p.y_range.start

    return p
