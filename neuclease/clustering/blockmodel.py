import numpy as np
import pandas as pd
import graph_tool.all as gt
from dvidutils import LabelMapper


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
    assert np.isinstance(weight_series[0], pd.Series), \
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
            Note that in that case, edges of type ('a', 'b')
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
