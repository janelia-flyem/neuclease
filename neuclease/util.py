import csv
import time
import json
import logging
import warnings
import contextlib
from datetime import timedelta
from itertools import starmap

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


def uuids_match(uuid1, uuid2):
    """
    Return True if the two uuids are the equivalent.
    
    >>> assert uuids_match('abcd', 'abcdef') == True
    >>> assert uuids_match('abc9', 'abcdef') == False
    """
    assert uuid1 and uuid2, "Empty UUID"
    n = min(len(uuid1), len(uuid2))
    return (uuid1[:n] == uuid2[:n])


def chunkify_table(table, approx_chunk_len):
    """
    Generator.
    Break the given array into chunks of approximately the given size.
    
    FIXME: This leaves the last chunk with all 'leftovers' if the chunk
           size doesn't divide cleanly.  Would be better to more evenly
           distribute them.
    """
    total_len = len(table)
    num_chunks = max(1, total_len // approx_chunk_len)
    chunk_len = total_len // num_chunks

    partitions = list(range(0, chunk_len*num_chunks, chunk_len))
    if partitions[-1] < total_len:
        partitions.append( total_len )

    for (start, stop) in zip(partitions[:-1], partitions[1:]):
        yield table[start:stop]


def read_csv_header(csv_path):
    """
    Open the CSV file at the given path and return it's header column names as a list.
    If it has no header (as determined by csv.Sniffer), return None.
    """
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

        if not has_header:
            return None
    
        rows = iter(csv.reader(csv_file))
        header = next(rows)
        return header


def csv_has_header(csv_path):
    return (read_csv_header(csv_path) is not None)


def read_csv_col(csv_path, col=0, dtype=np.uint64):
    """
    Read a single column from a CSV file as a pd.Series.
    """
    int(col) # must be an int
    header_names = read_csv_header(csv_path)
    if header_names:
        header_row = 0
        names = [header_names[col]]
    else:
        header_row = None
        names = ['noname']

    s = pd.read_csv(csv_path, header=header_row, usecols=[col], names=names, dtype=dtype)[names[0]]
    
    if header_row is None:
        s.name = None
    return s


def extract_subvol(array, box):
    """
    Extract a subarray according to the given box.
    """
    return array[box_to_slicing(*box)]


def box_to_slicing(start, stop):
    """
    For the given bounding box (start, stop),
    return the corresponding slicing tuple.

    Example:
    
        >>> assert bb_to_slicing([1,2,3], [4,5,6]) == np.s_[1:4, 2:5, 3:6]
    """
    return tuple( starmap( slice, zip(start, stop) ) )


def round_coord(coord, grid_spacing, how):
    """
    Round the given coordinate up or down to the nearest grid position.
    """
    assert how in ('down', 'up')
    if how == 'down':
        return (coord // grid_spacing) * grid_spacing
    if how == 'up':
        return ((coord + grid_spacing - 1) // grid_spacing) * grid_spacing


def round_box(box, grid_spacing, how='out'):
    # FIXME: Better name would be align_box()
    """
    Expand/shrink the given box out/in to align it to a grid.

    box: (start, stop)
    grid_spacing: int or shape
    how: One of ['out', 'in', 'down', 'up'].
         Determines which direction the box corners are moved.
    """
    directions = { 'out':  ('down', 'up'),
                   'in':   ('up', 'down'),
                   'down': ('down', 'down'),
                   'up':   ('up', 'up') }

    box = np.asarray(box)
    assert how in directions.keys()
    return np.array( [ round_coord(box[0], grid_spacing, directions[how][0]),
                       round_coord(box[1], grid_spacing, directions[how][1]) ] )


def lexsort_inplace(columns):
    """
    Lexsort the columns of the given array, in-place.
    """
    assert columns.ndim == 2
    assert columns.flags['C_CONTIGUOUS']
    mem_view = memoryview(columns.reshape(-1))
    
    # Convert to 1D structured array for in-place sort
    dtype = [(str(i), columns.dtype) for i in range(columns.shape[1])]
    array_view = np.frombuffer(mem_view, dtype)
    array_view.sort()


def is_lexsorted(columns):
    """
    Given a 2d array, return True if the array is lexsorted,
    i.e. the first column is sorted, with ties being broken
    by the second column, and so on.
    """
    prev_rows = columns[:-1]
    next_rows = columns[1:]
    
    # Mark non-decreasing positions in every column
    nondecreasing = (next_rows >= prev_rows)
    if not nondecreasing[:,0].all():
        return False
    
    # Mark increasing positions in every column, but allow later columns to
    # inherit the status of earlier ones, if an earlier one was found to be increasing.
    increasing = (next_rows > prev_rows)
    np.logical_or.accumulate(increasing, axis=1, out=increasing)
    
    # Every column must be non-decreasing, except in places where
    #  an earlier column in the row is increasing.
    return (nondecreasing[:, 1:] | increasing[:,:-1]).all()


class NumpyConvertingEncoder(json.JSONEncoder):
    """
    Encoder that converts numpy arrays and scalars
    into their pure-python counterparts.
    
    (No attempt is made to preserve bit-width information.)
    
    Usage:
    
        >>> d = {"a": np.arange(3, dtype=np.uint32)}
        >>> json.dumps(d, cls=NumpyConvertingEncoder)
        '{"a": [0, 1, 2]}'
    """
    def default(self, o):
        if isinstance(o, (np.ndarray, np.number)):
            return o.tolist()
        return super().default(o)


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


def connected_components(edges, num_nodes, _lib=None):
    """
    Run connected components on the graph encoded by 'edges' and num_nodes.
    The graph vertex IDs must be CONSECUTIVE.
    
    edges:
        ndarray, shape=(N,2), dtype=np.uint32
    
    num_nodes:
        Integer, max_node+1.
        (Allows for graphs which contain nodes that are not referenced in 'edges'.)
    
    _lib:
        Do not use.  (Used for testing.)
    
    Returns:
        ndarray of shape (num_nodes,), labeled by component index from 0..C
    
    Note: Uses graph-tool if it's installed; otherwise uses networkx (slower).
    """

    if (graph_tool_available() or _lib == 'gt') and _lib != 'nx':
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
