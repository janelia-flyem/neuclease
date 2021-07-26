import numpy as np
import pandas as pd
import graph_tool.all as gt
from dvidutils import LabelMapper


def construct_graph(strengths):
    """
    Construct a single-layer graph from the given strengths.

    The graph vertices will be numbered 0..N-1, corresponding to
    the sorted list of bodies in the strengths.

    Arg:
        strengths:
            pd.Series, indexed by body *pairs*.
            Values are the number of synapses in the pairwise connection.

    Returns:
        g, sorted_bodies
    """
    assert strengths.index.nlevels == 2, \
        "Please pass a series, indexed by [body_pre, body_post]"
    strengths = strengths.astype(np.int32)

    body_edges = strengths.reset_index()[['body_pre', 'body_post']].values.astype(np.uint64)
    sorted_bodies = np.sort(pd.unique(body_edges.reshape(-1)))

    vertexes = np.arange(len(sorted_bodies), dtype=np.uint32)
    vertex_mapper = LabelMapper(sorted_bodies.astype(np.uint64), vertexes)
    edges = vertex_mapper.apply(body_edges)

    g = gt.Graph(directed=True)
    g.add_vertex(np.uint32(len(vertexes)))
    g.add_edge_list(edges)
    g.ep["strength"] = g.new_edge_property("int")
    g.ep["strength"].a = strengths.values

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
    # Note: Each of these 'vertex maps' could have a different dtype
    node_to_vertex = {}
    N = 0
    for node_type, values in nodes.items():
        s = np.sort(pd.unique(np.concatenate(values)))
        s = pd.Series(np.arange(N, N+len(s)), index=s, name='vertex').rename_axis(node_type)
        N += len(s)
        node_to_vertex[node_type] = s
    total_vertices = N

    if layer_categories is None:
        layer_categories = default_layer_categories

    # Map edge tables to use vertex IDs
    edges = []
    layers = []
    for weights in weight_series:
        (left_type, left_nodes), (right_type, right_nodes) = _node_types_and_ids(weights)
        left_vertexes = node_to_vertex[left_type].loc[left_nodes].values
        right_vertexes = node_to_vertex[right_type].loc[right_nodes].values

        cat = layer_categories[(left_type, right_type)]
        e = np.array((left_vertexes, right_vertexes)).transpose()
        l = np.full(len(e), cat)  # noqa

        edges.append(e)
        layers.append(l)

    edges = np.concatenate(edges)
    layers = np.concatenate(layers)
    weights = np.concatenate([w.values for w in weight_series])

    g = gt.Graph(directed=True)
    g.add_vertex(np.uint32(total_vertices))
    g.add_edge_list(edges)
    g.ep["weight"] = g.new_edge_property("int")
    g.ep["weight"].a = weights
    g.ep["layer"] = g.new_edge_property("int")
    g.ep["layer"].a = layers

    return g, node_to_vertex


def _node_types_and_ids(weights):
    assert weights.index.nlevels == 2
    weights = weights.reset_index()
    left_name, right_name = weights.columns[:2]

    left_type = left_name.split('_')[0]
    right_type = right_name.split('_')[0]

    left_nodes = weights[left_name]
    right_nodes = weights[right_name]

    return (left_type, left_nodes), (right_type, right_nodes)
