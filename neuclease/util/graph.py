"""
Miscellaneous utility functions for dealing with graphs.

Warning:
    Some of these are made to work with graph_tool,
    some with networkx, and some with either.
"""

import warnings
import numpy as np
import pandas as pd
import networkx as nx
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
                import graph_tool as gt #@UnusedImport
        
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
    
    
    
