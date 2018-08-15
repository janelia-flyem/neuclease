import os
import io
import sys
import csv
import time
import json
import vigra
import logging
import inspect
import warnings
import contextlib
from datetime import timedelta
from itertools import product, starmap
from skimage.util import view_as_blocks

import requests
from tqdm import tqdm

import numpy as np
import pandas as pd
import networkx as nx
from numba import jit

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


def fetch_file(url, output=None, chunksize=2**10, *, session=None):
    """
    Fetch a file from the given endpoint,
    and save it to bytes, a file object, or a file path.

    Args:
        url:
            Complete url to fetch from.
        
        output:
            If None, file is returned in-memory, as bytes.
            If str, it is interpreted as a path to which the file will be written.
            Otherwise, must be a file object to write the bytes to (e.g. a BytesIO object).
        
        chunksize:
            Data will be streamed in chunks, with the given chunk size.

    Returns:
        None, unless no output file object/path is provided,
        in which case the fetched bytes are returned.
    """
    session = session or requests.Session()
    with session.get(url, stream=True) as r:
        r.raise_for_status()

        if output is None:
            return r.content

        if isinstance(output, str):
            # Create a file on disk and write to it.
            with open(output, 'wb') as f:
                for chunk in r.iter_content(chunksize):
                    f.write(chunk)
        else:
            # output is a file object
            for chunk in r.iter_content(chunksize):
                output.write(chunk)


def ndrange(start, stop=None, step=None):
    """
    Generator.

    Like np.ndindex, but accepts start/stop/step instead of
    assuming that start is always (0,0,0) and step is (1,1,1).
    
    Example:
    
    >>> for index in ndrange((1,2,3), (10,20,30), step=(5,10,15)):
    ...     print(index)
    (1, 2, 3)
    (1, 2, 18)
    (1, 12, 3)
    (1, 12, 18)
    (6, 2, 3)
    (6, 2, 18)
    (6, 12, 3)
    (6, 12, 18)
    """
    if stop is None:
        stop = start
        start = (0,)*len(stop)

    if step is None:
        step = (1,)*len(stop)

    assert len(start) == len(stop) == len(step), \
        f"tuple lengths don't match: ndrange({start}, {stop}, {step})"

    yield from product(*starmap(range, zip(start, stop, step)))


def view_rows_as_records(table):
    """
    Return a 1D strucured-array view of the given 2D array,
    in which each row is converted to a strucutred element.
    
    The structured array fields will be named '0', '1', etc.
    """
    assert table.ndim == 2
    assert table.flags['C_CONTIGUOUS']
    mem_view = memoryview(table.reshape(-1))

    dtype = [(str(i), table.dtype) for i in range(table.shape[1])]
    array_view = np.frombuffer(mem_view, dtype)
    return array_view


def lexsort_inplace(table):
    """
    Lexsort the given 2D table of the given array, in-place.
    The table is sorted by the first column (table[:,0]),
    then the second, third, etc.
    
    Equivalent to:
    
        order = np.lexsort(table.transpose()[::-1])
        table = table[order]
    
    But should (in theory) be faster and use less RAM.
    
    WARNING:
        Tragically, this function seems to be much slower than the straightforward
        implementation shown above, so its only advantage is its reduced RAM requirements.
    """
    # Convert to 1D structured array for in-place sort
    array_view = view_rows_as_records(table)
    array_view.sort()


def lexsort_columns(table):
    """
    Lexsort the given 2D table of the given array, in-place.
    The table is sorted by the first column (table[:,0]),
    then the second, third, etc.
    """
    order = np.lexsort(table.transpose()[::-1])
    return table[order]


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


@jit(nopython=True)
def groupby_presorted(a, sorted_cols):
    """
    Given an array of data and some sorted reference columns to use for grouping,
    yield subarrays of the data, according to runs of identical rows in the reference columns.
    
    JIT-compiled with numba.
    For pre-sorted structured array input, this is much faster than pandas.DataFrame(a).groupby().
    
    Args:
        a: ND array, any dtype, shape (N,) or (N,...)
        sorted_cols: ND array, at least 2D, any dtype, shape (N,...),
                     not necessarily the same shape as 'a', except for the first dimension.
                     Must be pre-ordered so that identical rows are contiguous,
                     and therefore define the group boundaries.

    Note: The contents of 'a' need not be related in any way to sorted_cols.
          The sorted_cols array is just used to determine the split points,
          and the corresponding rows of 'a' are returned.

    Examples:
    
        a = np.array( [[0,0,0],
                       [1,0,0],
                       [2,1,0],
                       [3,1,1],
                       [4,2,1]] )

        # Group by second column
        groups = list(groupby_presorted(a, a[:,1:2]))
        assert (groups[0] == [[0,0,0], [1,0,0]]).all()
        assert (groups[1] == [[2,1,0], [3,1,1]]).all()
        assert (groups[2] == [[4,2,1]]).all()
    
        # Group by third column
        groups = list(groupby_presorted(a, a[:,2:3]))
        assert (groups[0] == [[0,0,0], [1,0,0], [2,1,0]]).all()
        assert (groups[1] == [[3,1,1], [4,2,1]]).all()

        # Group by external column
        col = np.array([10,10,40,40,40]).reshape(5,1) # must be at least 2D
        groups = list(groupby_presorted(a, col))
        assert (groups[0] == [[0,0,0], [1,0,0]]).all()
        assert (groups[1] == [[2,1,0], [3,1,1],[4,2,1]]).all()
        
    """
    assert sorted_cols.ndim >= 2
    assert sorted_cols.shape[0] == a.shape[0]

    if len(a) == 0:
        return

    start = 0
    row = sorted_cols[0]
    for stop in range(len(sorted_cols)):
        next_row = sorted_cols[stop]
        if (next_row != row).any():
            yield a[start:stop]
            start = stop
            row = next_row

    # Last group
    yield a[start:len(sorted_cols)]


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


def closest_approach(sv_vol, id_a, id_b):
    """
    Given a segmentation volume and two label IDs which it contains,
    Find the two coordinates within id_a and id_b, respectively,
    which mark the two objects' closest approach, i.e. where the objects
    come closest to touching, even if they don't actually touch.
    
    Returns (coord_a, coord_b, distance)
    """
    mask_a = (sv_vol == id_a)
    mask_b = (sv_vol == id_b)

    if not mask_a.any() or not mask_b.any():
        # If either object is not present, there is no closest approach
        return (-1,-1,-1), (-1,-1,-1), np.inf
    
    if id_a == id_b:
        # IDs are identical.  Choose an arbitrary point.
        first_point = tuple(np.transpose(mask_a.nonzero())[0])
        return first_point, first_point, 0.0
    
    # For all voxels, find the shortest vector toward id_b
    to_b_vectors = vigra.filters.vectorDistanceTransform(mask_b.astype(np.uint32))
    
    # Magnitude of those vectors == distance to id_b
    to_b_distances = np.linalg.norm(to_b_vectors, axis=-1)

    # We're only interested in the voxels within id_a;
    # everything else is infinite distance
    to_b_distances[~mask_a] = np.inf

    # Find the point within id_a with the smallest vector
    point_a = np.unravel_index(np.argmin(to_b_distances), to_b_distances.shape)

    # Its closest point id_b is indicated by the corresponding vector
    point_b = tuple((point_a + to_b_vectors[point_a]).astype(int))

    return (point_a, point_b, to_b_distances[point_a])


def upsample(orig_data, upsample_factor):
    """
    Upsample the given array by duplicating every
    voxel into the corresponding upsampled voxels.
    """
    orig_shape = np.array(orig_data.shape)
    upsampled_data = np.empty( orig_shape * upsample_factor, dtype=orig_data.dtype )
    v = view_as_blocks(upsampled_data, orig_data.ndim*(upsample_factor,))
    
    slicing = (Ellipsis,) + (None,)*orig_data.ndim
    v[:] = orig_data[slicing]
    return upsampled_data


def tqdm_proxy(iterable, *, logger=None, level=logging.INFO, **kwargs):
    """
    Useful as an (almost) drop-in replacement for tqdm which can be used
    in EITHER an interactive console OR a script that logs to file.
    
    Automatically detects whether or not sys.stdout is a file or a console,
    and configures tqdm accordingly.
    
    Example:

        for i in tqdm_proxy(range(1000)):
            # do some stuff
    """
    assert 'file' not in kwargs, \
        "There's no reason to use this function if you are providing your own output stream"
    
    _tqdm = tqdm
    _file = None
    
    try:
        import ipykernel.iostream
        from tqdm import tqdm_notebook
        if isinstance(sys.stdout, ipykernel.iostream.OutStream):
            _tqdm = tqdm_notebook
        _file = sys.stdout
    except ImportError:
        pass
    
    if not _file and os.isatty(sys.stdout.fileno()):
        _file = sys.stdout
    else:
        if logger is None:
            frame = inspect.stack()[1]
            modname = inspect.getmodulename(frame[1])
            if modname:
                logger = logging.getLogger(modname)
            else:
                logger = logging.getLogger("unknown")

        _file = TqdmToLogger(logger, level)

        if 'ncols' not in kwargs:
            kwargs['ncols'] = 100
        
        if 'miniters' not in kwargs:
            # Aim for 5% updates
            if 'total' in kwargs:
                kwargs['total'] = kwargs['total'] // 20

    kwargs['file'] = _file
    return _tqdm(iterable, **kwargs)


class TqdmToLogger(io.StringIO):
    """
    Output stream for tqdm which will output to logger module instead of stdout.
    Copied from:
    https://github.com/tqdm/tqdm/issues/313#issuecomment-267959111
    """
    logger = None
    level = logging.INFO
    buf = ''

    def __init__(self, logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)

