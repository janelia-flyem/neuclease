"""
Miscellaneous utility functions for dealing with graphs.

Warning:
    Some of these are made to work with graph_tool,
    some with networkx, and some with either.
"""

import warnings
from itertools import combinations
from collections import namedtuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import KDTree

from dvidutils import LabelMapper


def find_root(g, start=None):
    """
    Find the root node in a tree, given as a nx.DiGraph,
    tracing up the tree starting with the given start node.
    """
    if start is None:
        start = next(iter(g.nodes()))
    parents = [start]
    while parents:
        root = parents[0]
        parents = list(g.predecessors(parents[0]))
    return root


def toposorted_ancestors(g, n, reversed=False):
    """
    Find all ancestors of the node n from a DAG g (nx.DiGraph),
    i.e. the nodes from which n can be reached.
    The nodes are returned in topologically sorted order,
    from the graph root to the immediate parent of n.

    Warning:
        No attempt is made to ensure that g has no cycles.
        Running this on a non-DAG may result in an infinite loop.

    Args:
        g: nx.DiGraph
        n: starting node

    Returns:
        list
        Does not include n itself.

    Example:
        In [1]: g = nx.DiGraph()
           ...: g.add_edges_from([(1,2), (2,3), (3,4), (3,5), (4,6), (5,6), (2, 10), (10,11)])
           ...: toposorted_ancestors(g, 6)
        Out[1]: [1, 2, 3, 5, 4]
    """
    nodes = [n]
    ancestors = []
    while nodes:
        n = nodes.pop(0)
        parents = list(g.predecessors(n))
        ancestors.extend(parents)
        nodes.extend(parents)

    # Drop duplicates via dict insertion to preserve order
    ancestors = list({a: None for a in ancestors}.keys())

    if reversed:
        # ancestors is already in reverse-toposorted order.
        return ancestors
    else:
        return ancestors[::-1]


def tree_to_dict(tree, root, display_fn=str, *, _d=None):
    """
    Convert the given tree (nx.DiGraph) into a dict,
    suitable for display via the asciitree module.

    Args:
        tree:
            nx.DiGraph
        root:
            Where to start in the tree (ancestors of this node will be ignored)
        display_fn:
            Callback used to convert node values into strings, which are used as the dict keys.
        _d:
            Internal use only.
    """
    if _d is None:
        _d = {}
    d_desc = _d[display_fn(root)] = {}
    for n in tree.successors(root):
        tree_to_dict(tree, n, display_fn, _d=d_desc)
    return _d


_graph_tool_available = None
def graph_tool_available():
    """
    Return True if the graph_tool module is installed.
    """
    global _graph_tool_available
    
    # Just do this import check once.
    if _graph_tool_available is None:
        try:
            with warnings.catch_warnings():
                # Importing graph_tool results in warnings about duplicate C++/Python conversion functions.
                # Ignore those warnings
                warnings.filterwarnings("ignore", "to-Python converter")
                import graph_tool as gt
        
            _graph_tool_available = True
        except ImportError:
            _graph_tool_available = False
    return _graph_tool_available


def connected_components_nonconsecutive(edges, node_ids):
    """
    Run connected components on the graph encoded by 'edges' and node_ids.
    All nodes from edges must be present in node_ids.
    (Additional nodes are permitted, in which case each is assigned its own CC.)
    The node_ids do not need to be consecutive, and can have arbitrarily large values.

    For graphs with consecutively-valued nodes, connected_components() (below)
    will be faster because it avoids a relabeling step.
    
    Args:
        edges:
            ndarray, shape=(E,2), dtype np.uint32 or uint64
        
        node_ids:
            ndarray, shape=(N,), dtype np.uint32 or uint64

    Returns:
        ndarray, same shape as node_ids, labeled by component index from 0..C
    """
    assert node_ids.ndim == 1
    assert node_ids.dtype in (np.uint32, np.uint64)
    
    cons_node_ids = np.arange(len(node_ids), dtype=np.uint32)
    mapper = LabelMapper(node_ids, cons_node_ids)
    cons_edges = mapper.apply(edges)
    return connected_components(cons_edges, len(node_ids))


def connected_components(edges, num_nodes, _lib=None):
    """
    Run connected components on the graph encoded by 'edges' and num_nodes.
    The graph vertex IDs must be CONSECUTIVE.

    Args:
        edges:
            ndarray, shape=(E,2), dtype=np.uint32
        
        num_nodes:
            Integer, max_node+1.
            (Allows for graphs which contain nodes that are not referenced in 'edges'.)
        
        _lib:
            Do not use.  (Used for testing.)
    
    Returns:
        ndarray of shape (num_nodes,), labeled by component index from 0..C
    
    Note: Uses graph-tool if it's installed; otherwise uses networkx (slower).
    """
    if len(edges) == 0:
        # Corner case: No edges -- every node is a component.
        # Avoid errors in graph_tool with an empty edge list.
        return np.arange(num_nodes, dtype=np.uint32)

    if (graph_tool_available() or _lib == 'gt') and _lib != 'nx':
        import graph_tool as gt
        from graph_tool.topology import label_components
        g = gt.Graph(directed=False)
        g.add_vertex(num_nodes)
        g.add_edge_list(edges)
        cc_pmap, _hist = label_components(g)
        return cc_pmap.get_array()

    else:
        edges = np.asarray(edges, dtype=np.int64)

        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        g.add_edges_from(edges)

        cc_labels = np.zeros((num_nodes,), np.uint32)
        for i, component_set in enumerate(nx.connected_components(g)):
            cc_labels[np.array(list(component_set))] = i
        return cc_labels



class SparseNodeGraph:
    """
    Wrapper around gt.Graph() that permits arbitrarily large
    node IDs, which are internally mapped to consecutive node IDs.
    
    Ideally, we could just use g.add_edge_list(..., hashed=True),
    but that feature appears to be unstable.
    (In my build, at least, it tends to segfault).
    """
    
    def __init__(self, edge_array, directed=True):
        import graph_tool as gt
    
        assert edge_array.dtype in (np.uint32, np.uint64)
    
        self.node_ids = pd.unique(edge_array.reshape(-1))
        self.cons_node_ids = np.arange(len(self.node_ids), dtype=np.uint32)
    
        self.mapper = LabelMapper(self.node_ids, self.cons_node_ids)
        cons_edges = self.mapper.apply(edge_array)
    
        self.g = gt.Graph(directed=directed)
        self.g.add_edge_list(cons_edges, hashed=False)
        

    def _map_node(self, node):
        return self.mapper.apply(np.array([node], self.node_ids.dtype))[0]


    def add_edge_weights(self, weights):
        if weights is not None:
            assert len(weights) == self.g.num_edges()
            if np.issubdtype(weights.dtype, np.integer):
                assert weights.dtype.itemsize <= 4, "Can't handle 8-byte ints, sadly."
                self.g.new_edge_property("int", vals=weights)
            elif np.issubdtype(weights.dtype, np.floating):
                self.g.new_edge_property("float", vals=weights)
            else:
                raise AssertionError("Can't handle non-numeric weights")


    def get_out_neighbors(self, node):
        cons_node = self._map_node(node)
        cons_neighbors = self.g.get_out_neighbors(cons_node)
        return self.node_ids[cons_neighbors]
    

    def get_in_neighbors(self, node):
        cons_node = self._map_node(node)
        cons_neighbors = self.g.get_in_neighbors(cons_node)
        return self.node_ids[cons_neighbors]


def euclidean_mst(points, initial_radius=1.8, format='nx'):
    """
    Return a Euclidean MST for the given set of points.

    This implementation was adapted from neuprint-python's
    heal_skeleton function.

    It is based on KDTrees. It starts by grouping
    points into closely-clustered 'fragments' (low-hanging-fruit of
    easily merged points), and then performing nearest neighbor
    searches from each fragment to all other fragments.
    The initial grouping is determined by the 'initial_radius' setting.

    Note:
        This is NOT an ideal implementation, but it's fine for me.
        Apparently it should be possible to achieve this by computing
        the Delaunay Triangulation on the points first, and selecting
        the MST from within that edge set.  But I don't know the details,
        and I don't know if it is efficient for my most common use-case
        (a mostly-contiguous voxel mask), in which most points are
        within a known fixed distance from their nearest neighbor (1 or so).

    Args:
        points:
            array, (N, 3)
        initial radius:
            The implementation starts by loading all edges between
            neighbors within a certain radius.

            - If this radius is too large, then more edges than necessary
              will be loaded into the graph before the MST is computed. In
              the worst case, all points are within the radius, resulting
              in O(N^2) edges.

            - If this radius is too small, then the implementation will
              have to construct a high number of KDTrees, and search for
              nearest neighbors between them. In the worst case, the radius
              each point will be completely isolated, resulting in N KDTrees
              and O(N^2) neighbor searches.

            We typically deal with voxel masks with spatially contiguous components,
            so we use a default of 1.8.  (A cube diagonal is √3 == 1.732)
        format:
            Either 'nx' or 'pandas' or 'max-only' or 'max-or-initial'.

                - 'nx': nx.Graph
                - 'pandas': pd.DataFrame
                - 'max-only': Just the single greatest max distance (a scalar)
                - 'max-or-initial': Just the single greatest max distance, unless it
                  is less than the initial radius, in which case the initial radius
                  is returned. This avoids some computation in cases when you don't
                  care about small max distances.

    Returns:
        Depending on requested 'format', one of the following:
            - nx.Graph containing only the MST edges
            - pd.DataFrame containing only the MST edges
            - a scalar representing the max distance in the MST
    """
    assert format in ('nx', 'pandas', 'max-only', 'max-or-initial')
    initial_kd = KDTree(points)
    initial_edges = initial_kd.query_pairs(initial_radius, output_type='ndarray')
    frag_edges = _intercomponent_edges(points, initial_edges)

    # Skip the distance calc and MST if possible.
    if frag_edges:
        if format in ('max-only', 'max-or-initial'):
            return max(distance for _u, _v, distance in frag_edges)
    elif format == 'max-or-initial':
        return initial_radius

    # Construct graph
    g = nx.Graph()
    g.add_edges_from((u, v, {'distance': d}) for u, v, d in frag_edges)
    v = points[initial_edges[:, 1]] - points[initial_edges[:, 0]]
    initial_distances = np.linalg.norm(v, axis=1)
    g.add_edges_from((
        (u, v, {'distance': d})
        for (u, v), d in zip(initial_edges, initial_distances))
    )

    if format == 'nx':
        mst_edges = nx.minimum_spanning_edges(g, weight='distance', data=False)
        return g.edge_subgraph(mst_edges)

    mst_edges = nx.minimum_spanning_edges(g, weight='distance', data=True)
    if format in ('max-only', 'max-or-initial'):
        return max(d['distance'] for _u, _v, d in mst_edges)

    assert format == 'pandas'
    return pd.DataFrame(
        [(u, v, d['distance']) for u, v, d in mst_edges],
        columns=['u', 'v', 'distance']
    )


def _intercomponent_edges(points, initial_edges):
    """
    Helper for euclidean_mst().
    Computes the distance from each component (point cluster) to every
    other point cluster, according to the two closest points between clusters.
    """
    cc = connected_components(initial_edges, len(points))
    if cc.max() == 0:
        # There's just one component; no inter-component edges
        return []

    points_df = pd.DataFrame(points, columns=[*'xyz'])

    # Construct a KDTree for each 'fragment' (cluster of connected points)
    Fragment = namedtuple('Fragment', ['frag_node', 'df', 'kd'])
    fragments = []
    for frag_node in range(cc.max()+1):
        point_mask = (cc == frag_node)
        frag = Fragment(
            frag_node,
            points_df.iloc[point_mask],
            KDTree(points[point_mask])
        )
        fragments.append(frag)

    # Sort from big-to-small, so the calculations below use a
    # KD tree for the larger point set in every fragment pair.
    fragments = sorted(fragments, key=lambda frag: -len(frag.df))

    # We could use the full graph and connect all fragements
    # to their nearest neighbors within other fragments,
    # but it's faster to treat each whole fragment as
    # a single node and run MST on that quotient graph,
    # which is usually tiny.
    frag_graph = nx.Graph()
    for frag_a, frag_b in combinations(fragments, 2):
        frag_b_points = frag_b.df[[*'xyz']].values
        frag_b_distances, frag_a_indexes = frag_a.kd.query(frag_b_points)

        # Select the best point in our query (frag_b's points)
        frag_index_b = np.argmin(frag_b_distances)

        # Identify the the frag_a point it matched to.
        frag_index_a = frag_a_indexes[frag_index_b]

        # The frag_*_indexes are relative to each fragment's own point list.
        # Translate from frag-relative indexes to global points_df indexes
        global_idx_a = frag_a.df.index[frag_index_a]
        global_idx_b = frag_b.df.index[frag_index_b]
        dist_ab = frag_b_distances[frag_index_b]

        # Add edge from one fragment node to another,
        # but keep track of which fine-grained point
        # indexes were used to calculate distance.
        frag_graph.add_edge(
            frag_a.frag_node, frag_b.frag_node,
            point_idx_a=global_idx_a, point_idx_b=global_idx_b,
            distance=dist_ab
        )

    # Compute inter-fragment MST edges
    interfrag_edges = nx.minimum_spanning_edges(frag_graph, weight='distance', data=True)

    # Convert from coarse fragment IDs to fine-grained point IDs
    point_edges = [
        (attr['point_idx_a'], attr['point_idx_b'], attr['distance'])
        for _u, _v, attr in interfrag_edges
    ]
    return point_edges
