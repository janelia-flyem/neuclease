from functools import partial
import numpy as np
import pandas as pd
import graph_tool as gt

from neuclease.util import Timer, compute_parallel


def construct_graph(weights):
    """
    Construct a single-layer graph from the given edge weights.

    The graph vertices will be numbered 0..N-1, corresponding to
    the sorted list of node IDs present in the index.

    Arg:
        weights:
            pd.Series, indexed by node *pairs* (e.g. body pairs).
            Values are edge weights, e.g. the number of
            synapses a pairwise connection between neurons.

    Returns:
        g, sorted_nodes
    """
    from dvidutils import LabelMapper
    assert weights.index.nlevels == 2, \
        "Please pass a series, indexed by e.g. [body_pre, body_post]"
    weights = weights.astype(np.int32)

    body_edges = weights.reset_index().iloc[:, :2].values.astype(np.uint64)
    sorted_bodies = np.sort(pd.unique(body_edges.reshape(-1)))

    vertexes = np.arange(len(sorted_bodies), dtype=np.uint32)
    vertex_mapper = LabelMapper(sorted_bodies.astype(np.uint64), vertexes)
    edges = vertex_mapper.apply(body_edges)

    g = gt.Graph(directed=True)
    g.add_vertex(np.uint32(len(vertexes)))
    g.add_edge_list(edges)
    g.ep["weight"] = g.new_edge_property("int")
    g.ep["weight"].a = weights.values

    return g, sorted_bodies


def construct_layered_graph(weight_series, layer_categories=None):
    """
    Given a set of weighted adjacency lists, construct a graph
    with weighted edges and a categorical edge property map
    which can be used as the categorical edge covariate in a
    'layered' stochastic block model inference.

    The 'weight series' should be a list of pd.Series objects,
    representing edge lists.  They should have a two-level index
    representing node pairs, and the value is a weight (e.g. strength).

    Furthermore, the names of the index levels should encode the node type,
    so that node IDs can be re-numbered into a single contiguous node ID space.
    """
    # Note: In this function, we make a somewhat arbitrary choice of terminology
    #       to distinguish between the input nodes and the vertex IDs we'll use
    #       in the constructed graph:
    #           - 'node' refers to input node IDs
    #           - 'vertex' refers to the gt.Graph vertex IDs
    node_to_vertex, total_vertices = _node_to_vertex_mappings(weight_series)
    edges, layers, weights = _edges_and_properties(weight_series, node_to_vertex, layer_categories)

    # Map edge tables to use vertex IDs
    g = gt.Graph(directed=True)
    g.add_vertex(np.uint32(total_vertices))
    g.add_edge_list(edges)
    g.ep["weight"] = g.new_edge_property("int")
    g.ep["weight"].a = weights
    g.ep["layer"] = g.new_edge_property("int")
    g.ep["layer"].a = layers

    return g, node_to_vertex


def _node_to_vertex_mappings(weight_series):
    """
    Given a list of pd.Series containing edge weights between nodes,
    Return a dict of mappings (one per node category) that can be used to
    convert original node IDs (which could be int, str, etc.) to
    non-overlapping vertex IDs to be used in a gt.Graph.

    Also return the total number of unique nodes across all categories,
    and return the default mapping of edge types (node ID pairs) to layer categories.
    """
    assert isinstance(weight_series[0], pd.Series), \
        "Please provide a list of pd.Series"

    # Determine node categories
    nodes = {}
    default_layer_categories = {}
    for i, weights in enumerate(weight_series):
        (left_type, left_nodes), (right_type, right_nodes) = _node_types_and_ids(weights)
        nodes.setdefault(left_type, []).append(left_nodes.unique())
        nodes.setdefault(right_type, []).append(right_nodes.unique())
        default_layer_categories[(left_type, right_type)] = i

    # Determine set of unique IDs in each node category,
    # and map them to a single global ID set.
    # Note:
    #   Each of these 'node-to-vertex maps' could have a different index dtype,
    #   but all will have int values.
    node_to_vertex = {}
    N = 0
    for node_type, values in nodes.items():
        s = np.sort(pd.unique(np.concatenate(values)))
        s = pd.Series(np.arange(N, N+len(s)), index=s, name='vertex').rename_axis(node_type)
        N += len(s)
        node_to_vertex[node_type] = s

    assert N == sum(map(len, node_to_vertex.values()))
    return node_to_vertex, N


def _edges_and_properties(weight_series, node_to_vertex, layer_categories=None):
    """
    Convert the given list of edge weight series into all the tables
    needed for the edges in the final gt.Graph,
    including vertex ID pairs, layer categorical IDs, and edge weights.

    Args:
        weight_series:
            list of pd.Series, representing edge weights
            Indexed by node pairs, weights in the values.

        node_to_vertex:
            A dict of pd.Series which map from original nodes to graph vertex ID.
            Original node ID is in the index, graph vertex ID is in the value.

        layer_categories:
            dict of edge type pairs -> layer ID
            If none are provided, then by default each unique
            pair of node types will be assigned to a unique layer.
            Note that in the default case, edges of type ('a', 'b')
            would be considered distinct from ('b', 'a').

            Example input:

                {
                    ('body', 'body'): 0,
                    ('body', 'lineage'): 1,
                    ('roi', 'body'): 2,
                    ('body', 'roi'): 2
                }
    Returns:
            edges, layers, weights
    """
    default_layer_categories = {}

    edges = []
    layers = []
    for weights in weight_series:
        (left_type, left_nodes), (right_type, right_nodes) = _node_types_and_ids(weights)
        left_vertexes = node_to_vertex[left_type].loc[left_nodes].values
        right_vertexes = node_to_vertex[right_type].loc[right_nodes].values

        if layer_categories:
            cat = layer_categories[(left_type, right_type)]
        else:
            # If no layer categories were explicitly provided,
            # we simply enumerate the edge types we encounter.
            max_cat = max(default_layer_categories.values(), default=-1)
            next_cat = max_cat + 1
            cat = default_layer_categories.setdefault((left_type, right_type), next_cat)

        e = np.array((left_vertexes, right_vertexes)).transpose()
        l = np.full(len(e), cat)  # noqa

        edges.append(e)
        layers.append(l)

    edges = np.concatenate(edges)
    layers = np.concatenate(layers)
    weights = np.concatenate([w.values for w in weight_series])

    return edges, layers, weights


def _node_types_and_ids(weights):
    """
    For the given pd.Series of edge weights,
    parse the index names to determine the 'node type'
    on the left side and right side, and return the complete
    list of node values from the index columns.
    The lists are not de-duplicated.
    """
    assert weights.index.nlevels == 2
    weights = weights.reset_index()
    left_name, right_name = weights.columns[:2]

    left_type = left_name.split('_')[0]
    right_type = right_name.split('_')[0]

    left_nodes = weights[left_name]
    right_nodes = weights[right_name]

    return (left_type, left_nodes), (right_type, right_nodes)


def extract_roi_counts(point_df, bodies):
    """
    Determine the number of pre/post synapses each body has in each ROI.
    Also determine the counts after "bilateralizing" the ROIs,
    i.e. combining left/right pairs of ROIs.

    Returns:
        Four pd.Series of counts:
        roi_pre, roi_post, biroi_pre, biroi_post
        Those are different combinations of pre/post and roi/bilateralized-roi
    """
    bodies  # for linter
    big_point_df = point_df.query('body in @bodies')
    roi_counts = big_point_df.groupby(['body', 'roi', 'kind']).size().rename('count')
    roi_counts = roi_counts[roi_counts > 0]
    roi_counts = roi_counts[roi_counts.index.get_level_values(1) != "<unspecified>"]
    roi_counts = roi_counts.reset_index().pivot(['body', 'roi'], 'kind', 'count')
    roi_counts.columns.name = ""
    roi_counts = roi_counts.reset_index()

    roi_pre = roi_counts.rename(columns={'roi': 'roipre'})
    roi_pre = roi_pre.set_index(['body', 'roipre'])['PreSyn'].rename('count').dropna().astype(int)
    roi_pre = roi_pre.loc[roi_pre > 0]

    roi_post = roi_counts.rename(columns={'roi': 'roipost'})
    roi_post = roi_post.set_index(['body', 'roipost'])['PostSyn'].rename('count').dropna().astype(int)
    roi_post = roi_post.loc[roi_post > 0]

    # Bilateralize: Combine counts for '(L)' rois and '(R)' rois.
    roi_counts['biroi'] = roi_counts['roi'].map(lambda s: s[:-3] if s[-3:] in ('(L)', '(R)') else s)
    biroi_counts = roi_counts.reset_index().groupby(['body', 'biroi'])[['PreSyn', 'PostSyn']].sum().reset_index()

    biroi_pre = biroi_counts.rename(columns={'biroi': 'biroipre'})
    biroi_pre = biroi_pre.set_index(['body', 'biroipre'])['PreSyn'].rename('count').dropna().astype(int)
    biroi_pre = biroi_pre.loc[biroi_pre > 0]

    biroi_post = biroi_counts.rename(columns={'biroi': 'biroipost'})
    biroi_post = biroi_post.set_index(['body', 'biroipost'])['PostSyn'].rename('count').dropna().astype(int)
    biroi_post = biroi_post.loc[biroi_post > 0]

    return roi_pre, roi_post, biroi_pre, biroi_post


def fetch_sanitized_body_annotations(server, uuid, cached_clio_ann=None, cached_dvid_ann=None, statuses=None):
    """
    Fetch body annotations from both clio and dvid.
    Combine them into a single table, and sanitize them to erase non-standard entries such as 'TBD', etc.

    Also, construct special columns for combined neuromere-side-hemilineage (roihemi) and a bilateralized version (biroihemi).

    Note: The resulting dataframe does not use the exact same values as either DVID or Clio.
    Note: Descending neurons are assigned a "hemilineage" of "brain".  Same for their "soma_neuromere".
    Note: Converts 'group' to a string
    """
    from neuclease.clio.api import fetch_json_annotations_all
    from neuclease.misc.vnc_statuses import fetch_vnc_statuses

    if cached_clio_ann is not None:
        clio_ann = cached_clio_ann.copy()
    else:
        with Timer("Fetching Clio annotations"):
            clio_ann = fetch_json_annotations_all('VNC')

    clio_ann = clio_ann.rename(columns={'bodyid': 'body'})
    clio_ann['body'] = clio_ann['body'].astype(np.uint64)

    if cached_dvid_ann is not None:
        dvid_ann = cached_dvid_ann.copy()
    else:
        with Timer("Fetching DVID annotations"):
            dvid_ann = fetch_vnc_statuses(server, uuid)

    dvid_ann = dvid_ann.reset_index().drop_duplicates('body')[['body', 'has_soma', 'is_cervical', 'status']]
    clio_ann = clio_ann.drop(columns='status').merge(dvid_ann, 'outer', on='body')

    if statuses:
        clio_ann = clio_ann.query('status in @statuses').copy()

    clio_ann['has_soma'].fillna(False, inplace=True)
    clio_ann['is_cervical'].fillna(False, inplace=True)
    clio_ann['soma_neuromere'].fillna("", inplace=True)
    clio_ann['hemilineage'].fillna("", inplace=True)

    # Convert groups to strings, even though they're originally integers.
    # This makes it easier to treat this feature like all the others for filtering purposes.
    clio_ann['group'] = clio_ann['group'].fillna("").astype(str)

    # The true set of hemilineage names has 'gaps', but that's not important here.
    # I think the following names aren't truly valid: 2B, 4A, 5A, 7A, 10A, 14B, 15A, 16A, 17B, 18A, 20B, 21B, 22B, 23A
    valid_hemilineages = {f'{i}{k}' for i in range(24) for k in 'AB'}  # noqa
    clio_ann.loc[clio_ann.query('hemilineage not in @valid_hemilineages').index, 'hemilineage'] = ""

    soma_neuromere = clio_ann['soma_neuromere']
    soma_neuromere = soma_neuromere.map(lambda s: 'ANm' if isinstance(s, str) and s.lower().startswith('anm') else s)
    soma_neuromere = soma_neuromere.map(lambda s: s if s in ('T1', 'T2', 'T3', 'ANm') else "")
    clio_ann['soma_neuromere'] = soma_neuromere.values

    clio_ann['soma_side'] = clio_ann['soma_side'].map(lambda s: s.lower() if isinstance(s, str) else s)
    clio_ann['soma_side'] = clio_ann['soma_side'].map(lambda s: {'rhs': 'R', 'lhs': 'L', 'midline': 'M'}.get(s, ''))

    descending = clio_ann.query('is_cervical and not has_soma')
    clio_ann.loc[descending.index, 'soma_neuromere'] = 'brain'
    clio_ann.loc[descending.index, 'hemilineage'] = 'brain'

    # By default, one-sided roihemi and bilateral biroihemi are the same...
    _idx = clio_ann.query('hemilineage != "" and soma_neuromere != ""').index
    clio_ann.loc[_idx, "biroihemi"] = clio_ann.loc[_idx, "soma_neuromere"] + '-' + clio_ann.loc[_idx, "hemilineage"]
    clio_ann.loc[_idx, "roihemi"] = clio_ann.loc[_idx, "soma_neuromere"] + '-' + clio_ann.loc[_idx, "hemilineage"]

    # ... but we overwrite the one-sided roihemi if a soma side is available.
    _idx = clio_ann.query('hemilineage != "" and soma_neuromere != "" and soma_side != ""').index
    clio_ann.loc[_idx, "roihemi"] = clio_ann.loc[_idx, "soma_neuromere"] + '(' + clio_ann.loc[_idx, "soma_side"] + ')' + '-' + clio_ann.loc[_idx, "hemilineage"]

    cols = ['body', 'has_soma', 'is_cervical', 'soma_side', 'soma_neuromere', 'hemilineage', 'roihemi', 'biroihemi', 'group', 'status']
    return clio_ann[cols].set_index('body').fillna("")


def blockmodel_clustering(strengths, metadata_series, constraint_series=[], initialization_series=[],
                          *, min_strength=0,
                          combine_metadata_layers=True,
                          min_blocks=None, max_blocks=None,
                          avoid_overfit=True):
    """
    Given a set of edges in 'strengths' and a set of metadata edges,
    construct a layered graph, whose base layer is determined by the strength edges
    and whose higher layer(s) are populated with the metadata edges.

    Then run gt.minimize_nested_blockmodel_dl() to find a partitioning.
    """
    with Timer("Preparing inputs"):
        g, node_to_vertex, pclabel_df, init_df, state_args, multilevel_mcmc_args = \
            prepare_blockmodel_clustering_input(
                strengths, metadata_series, constraint_series, initialization_series,
                min_strength=min_strength,
                combine_metadata_layers=combine_metadata_layers,
                min_blocks=min_blocks, max_blocks=max_blocks,
                avoid_overfit=avoid_overfit
            )

        if init_df is None:
            init_bs = None
        else:
            init_bs = [init_df['label'].values]

    with Timer(f"Inferring with strength cutoff {min_strength}"):
        # Use a separate process to execute the function,
        # to make it easier to kill if we change our mind.
        fn = partial(
            gt.minimize_nested_blockmodel_dl,
            # g,
            init_bs=init_bs,
            state_args=state_args,
            multilevel_mcmc_args=multilevel_mcmc_args,
        )
        nbs = compute_parallel(fn, [g], processes=1)[0]

    bodies = node_to_vertex['body'].index
    body_vertexes = node_to_vertex['body'].values

    df = pd.DataFrame({'body': bodies, 'block': nbs.get_bs()[0][body_vertexes]})
    return g, node_to_vertex, pclabel_df, init_df, nbs, df


def prepare_blockmodel_clustering_input(strengths, metadata_series, constraint_series=[], initialization_series=[],
                                        *, min_strength=0,
                                        combine_metadata_layers=True,
                                        min_blocks=None, max_blocks=None,
                                        avoid_overfit=True):

    # TODO:
    # Populate bfield to forbid left-left pairings?

    def filter_metadata(metadata_series, bodies):
        filtered_metadata = []
        for s in metadata_series:
            assert s.index.nlevels == 2
            s = s.loc[s.index.get_level_values(0).isin(bodies)]
            s = s.loc[s.index.get_level_values(1) != ""]
            s = s.loc[s > 0]
            filtered_metadata.append(s)
        return filtered_metadata

    def filter_body_labeling(series, bodies):
        filtered = []
        for s in series:
            assert s.index.nlevels == 1
            s = s.loc[s.index.isin(bodies)]
            filtered.append(s)
        return filtered

    def assign_layers(metadata_series):
        layers = {('body', 'body'): 0}
        for i, s in enumerate(metadata_series, start=1):
            assert s.index.nlevels == 2
            assert s.index.names[0] == 'body'
            assert '_' not in s.index.names[1], \
                ("The graph construction function interprets underscores in a speical way.\n"
                 "Avoid using undercores in your metadata index names")
            mtype = s.index.names[1]
            if combine_metadata_layers:
                layers[('body', mtype)] = 1
            else:
                layers[('body', mtype)] = i
        return layers

    with Timer("Filtering edges"):
        strong_conn = strengths[strengths > min_strength]
        bodies = sorted(pd.unique(strong_conn.reset_index().iloc[:, :2].values.reshape(-1)))
        metadata_series = filter_metadata(metadata_series, bodies)
        layers = assign_layers(metadata_series)

    with Timer("Constructing graph"):
        g, node_to_vertex = construct_layered_graph([strong_conn, *metadata_series], layers)

    with Timer("Assigning labelings"):
        constraint_series = filter_body_labeling(constraint_series, bodies)
        pclabel_df, pclabel = set_body_partition_labels(g, node_to_vertex, constraint_series, 'pclabel')

        initialization_series = filter_body_labeling(initialization_series, bodies)
        init_df, blabel = set_body_partition_labels(g, node_to_vertex, initialization_series, 'blabel')

    state_args = {
        "base_type": gt.LayeredBlockState,
        "ec": g.ep.layer,
        "recs": [g.ep.weight],
        "rec_types": ["discrete-geometric"],
        "deg_corr": True,
        "layers": True,
        "pclabel": pclabel
        # These are passed directly to minimize_nested_blockmodel_dl()
        #"b": blabel,
    }

    multilevel_mcmc_args = {
        "verbose": False,
        "entropy_args": {
            "dl": avoid_overfit,
            "exact": True
        },
    }
    if min_blocks:
        multilevel_mcmc_args['B_min'] = min_blocks
    if max_blocks:
        multilevel_mcmc_args['B_max'] = max_blocks

    return g, node_to_vertex, pclabel_df, init_df, state_args, multilevel_mcmc_args


def set_body_partition_labels(g, node_to_vertex, labeled_series, property_name='label'):
    if labeled_series is None or len(labeled_series) == 0:
        return None, None

    # Use the labeled_series to tag each body with a partition label.
    # Each body can be listed with multiple labels (in different series),
    # and we take the intersection of all of them.
    # We just merge on a column for each label set,
    # and then enumerate the unique column combinations.
    body_df = node_to_vertex['body'].to_frame()
    for s in labeled_series:
        body_df = body_df.merge(s, 'left', on='body')
    body_df = body_df.fillna("<blank>")
    label_names = [cs.name for cs in labeled_series]
    body_df['label'] = 1 + body_df.groupby(label_names).ngroup().astype(int)

    # Don't forget about metadata nodes!
    # They can't be clustered with bodies, but remain unconstrained within their own group.
    # We'll give them label partition label 0
    full_df = pd.DataFrame({'vertex': np.arange(g.num_vertices())})
    full_df = full_df.merge(body_df[['vertex', 'label']], 'left', 'vertex').fillna(0).astype(int)

    g.vp[property_name] = g.new_vertex_property("int", full_df['label'].values)
    return full_df, g.vp[property_name]
