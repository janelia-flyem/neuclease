import csv
import time
import logging
import warnings
import contextlib
from datetime import timedelta

import numpy as np
import pandas as pd
import networkx as nx

@contextlib.contextmanager
def Timer(msg=None, logger=None):
    if msg:
        logger = logger or logging.getLogger(__name__)
        logger.info(msg + '...')
    result = _TimerResult()
    yield result
    result.stop = time.time()
    if msg:
        logger.info(msg + f' took {result.timedelta}')

class _TimerResult(object):
    def __init__(self):
        self.start = time.time()
        self.stop = None

    @property
    def seconds(self):
        if self.stop is None:
            return time.time() - self.start
        else:
            return self.stop - self.start

    @property
    def timedelta(self):
        return timedelta(seconds=self.seconds)

def csv_has_header(csv_path):
    with open(csv_path, 'r') as csv_file:
        first_line = csv_file.readline()
        csv_file.seek(0)
        if ',' not in first_line:
            # csv.Sniffer doesn't work if there's only one column in the file
            try:
                int(first_line)
                has_header = False
            except:
                has_header = True
        else:
            has_header = csv.Sniffer().has_header(csv_file.read(1024))
            csv_file.seek(0)

    return has_header

def read_csv_col(csv_path, col=0, dtype=np.uint64):
    int(col) # must be an int
    header = None
    if csv_has_header:
        header = 0
    return pd.read_csv(csv_path, header=header, usecols=[col], names=['foo'], dtype=np.uint64)['foo']

_graph_tool_available = None
def graph_tool_available():
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

def connected_components(edges, num_nodes):
    """
    Run connected components on the graph encoded by 'edges' and num_nodes.
    The graph vertex IDs must be CONSECUTIVE.
    
    edges:
        ndarray, shape=(N,2), dtype=np.uint32
    
    num_nodes:
        Integer, max_node+1.
        (Allows for graphs which contain nodes that are not referenced in 'edges'.)
    
    Returns:
        ndarray of shape (num_nodes,), labeled by component index from 0..C
    
    Note: Uses graph-tool if it's installed; otherwise uses networkx (slower).
    """

    if graph_tool_available():
        import graph_tool as gt
        from graph_tool.topology import label_components
        g = gt.Graph(directed=False)
        g.add_vertex(num_nodes)
        g.add_edge_list(edges)
        cc_pmap, _hist = label_components(g)
        return cc_pmap.get_array()

    else:
        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        g.add_edges_from(edges)

        cc_labels = np.zeros((num_nodes,), np.uint32)
        for i, component_set in enumerate(nx.connected_components(g)):
            cc_labels[np.array(list(component_set))] = i
        return cc_labels
