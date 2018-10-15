import os
import io
import sys
import copy
import time
import json
import vigra
import logging
import inspect
import warnings
import contextlib
from datetime import datetime, timedelta
from itertools import product, starmap
from collections.abc import Mapping

import requests
from tqdm import tqdm

# Disable the monitor thread entirely.
# It is more trouble than it's worth, especially when using tqdm_proxy, below.
tqdm.monitor_interval = 0

import numpy as np
import pandas as pd
import networkx as nx

from dvidutils import LabelMapper

from .view_as_blocks import view_as_blocks

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def Timer(msg=None, logger=None):
    """
    Simple context manager that acts as a wall-clock timer.
    
    Args:
        msg:
            Optional message to be logged at the start
            and stop of the timed period.
        logger:
            Which logger to write the message to.

    Example:
        >>> with Timer("Doing stuff") as timer:
        ...     # do stuff here
        >>>
        >>> print(timer.seconds)
        >>> print(timer.timedelta)
    """
    if msg:
        logger = logger or logging.getLogger(__name__)
        logger.info(msg + '...')
    result = _TimerResult()
    yield result
    result.stop = time.time()
    if msg:
        logger.info(msg + f' took {result.timedelta}')


class _TimerResult(object):
    """
    Helper class, yielded by the Timer context manager.
    """
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


def post_file(url, f, *, session=None):
    """
    Args:
        url:
            Complete url to which the file will be posted.
        f:
            The file to post.
            Either a path to a file, a (binary) file object,
            or a bytes object.
    """
    session = session or requests.Session()
    if isinstance(f, str):
        fname = f
        with open(fname, 'rb') as f:
            r = session.post(url, data=f)
    else:
        # Either bytes or a file object
        r = session.post(url, data=f)

    r.raise_for_status()


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


def write_json_list(objects, f):
    """
    Like json.dump(), but writes each item to its own line (no indentation).
    """
    assert isinstance(objects, list)

    def _impl(f):
        f.write('[\n')
        for s in objects[:-1]:
            json.dump(s, f)
            f.write(',\n')
        json.dump(objects[-1], f)
        f.write('\n]')

    if isinstance(f, str):
        with open(f, 'w') as fp:
            _impl(fp)
    else:
        _impl(f)


DEFAULT_TIMESTAMP = datetime.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
def parse_timestamp(ts, default=DEFAULT_TIMESTAMP):
    """
    Parse the given timestamp as a datetime object.
    If it is already a datetime object, it will be returned as-is.
    If it is None, then the given default timestamp will be returned.
    
    Acceptable formats are:

        2018-01-01             (date only)
        2018-01-01 00:00:00    (date and time)
        2018-01-01 00:00:00.0  (date and time with microseconds)
    
    """
    if ts is None:
        ts = copy.copy(default)

    if isinstance(ts, datetime):
        return ts

    if isinstance(ts, str):
        if len(ts) == len('2018-01-01'):
            ts = datetime.strptime(ts, '%Y-%m-%d')
        elif len(ts) == len('2018-01-01 00:00:00'):
            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        elif len(ts) >= len('2018-01-01 00:00:00.0'):
            frac = ts.split('.')[1]
            zero_pad = 6 - len(frac)
            ts += '0'*zero_pad
            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
        else:
            raise AssertionError("Bad timestamp format")

    return ts


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


def find_root(g, start):
    """
    Find the root node in a tree, given as a nx.DiGraph,
    tracing up the tree starting with the given start node.
    """
    parents = [start]
    while parents:
        root = parents[0]
        parents = list(g.predecessors(parents[0]))
    return root


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


def extract_labels_from_volume(points_df, volume, box_zyx=None, vol_scale=0, label_names=None):
    """
    Given a list of point coordinates and a label volume, assign a
    label to each point based on its position in the volume.
    
    Extracting values from an array in numpy is simple.
    In the simplest case, this is equivalent to:
    
        coords = points_df[['z', 'y', 'x']].values.transpose()
        points_df['label'] = volume[(*coords,)]

    But this function supports extra features:
    
    - Points outside the volume extents are handled gracefully (they remain unlabeled).
    - The volume can be offset from the origin (doesn't start at (0,0,0)).
    - The volume can be downscaled.
    - Both label values (ints) and label names are output, if the label names were specified.
    
    Args:
        points_df:
            DataFrame with at least columns ['x', 'y', 'z'].
            The points in this DataFrame should be provided at SCALE-0,
            regardless of vol_scale.
            This function appends two additional columns to the DataFrame, IN-PLACE.
        
        volume:
            3D ndarray of label voxels
        
        box_zyx:
            The (min,max) coordinates in which the volume resides in the point coordinate space.
            It is assumed that this box is provided at the same scale as vol_scale,
            (i.e. it is not necessarily given using scale-0 coordiantes).
        
        vol_scale:
            Specifies the scale at which volume (and box_zyx) were provided.
            The coordinates in points_df will be downscaled accordingly.
            
        label_names:
            Optional.  Specifies how label names map to label IDs.
            If provided, a new column 'label_name' will be appended to
            points_df in addition to the 'label' column.

            Must be either:
            - a mapping of `{ name : label }`, indicating each ROI's
              label ID in the output image, or
            - a list label names in which case the mapping is determined automatically
              by enumerating the labels in the given order (starting at 1).
    
    Returns:
        None.  Results are appended to the points_df as new column(s).
    """
    if box_zyx is None:
        box_zyx = np.array(([0]*volume.ndim, volume.shape))

    assert ((box_zyx[1] - box_zyx[0]) == volume.shape).all() 

    downsampled_coords_zyx = points_df[['z', 'y', 'x']] // (2**vol_scale)

    # Drop everything outside the combined_box
    min_z, min_y, min_x = box_zyx[0] #@UnusedVariable
    max_z, max_y, max_x = box_zyx[1] #@UnusedVariable
    q = 'z >= @min_z and y >= @min_y and x >= @min_x and z < @max_z and y < @max_y and x < @max_x'
    downsampled_coords_zyx.query(q, inplace=True)

    logger.info(f"Extracting {len(downsampled_coords_zyx)} ROI index values")
    points_df['label'] = 0
    downsampled_coords_zyx -= box_zyx[0]
    points_df.loc[downsampled_coords_zyx.index, 'label'] = volume[tuple(downsampled_coords_zyx.values.transpose())]

    if label_names is not None:
        if not isinstance(label_names, Mapping):
            label_names = { name: i for i, name in enumerate(label_names, start=1) }

        # Reverse the mapping
        label_names = { v:k for k,v in label_names.items() }
        
        if 0 not in label_names:
            label_names[0] = '<unspecified>'

        points_df['label_name'] = pd.Categorical( points_df['label'].map(label_names),
                                                  categories=label_names.values(),
                                                  ordered=False )


def unordered_duplicated(df, subset=None, keep='first'):
    """
    Like pd.DataFrame.duplicated(), but sorts each row first, so
    rows can be considered duplicates even if their values don't
    appear in the same order.
    """
    if subset is None:
        subset = list(df.columns)
    normalized_cols = np.sort(df[subset].values, axis=1)
    dupes = pd.DataFrame(normalized_cols).duplicated(keep=keep).values
    return pd.Series(dupes, index=df.index)


def drop_unordered_duplicates(df, subset=None, keep='first'):
    """
    Like pd.DataFrame.drop_duplicates(), but sorts each row first, so
    rows can be considered duplicates even if their values don't
    appear in the same order. 
    """
    dupes = unordered_duplicated(df, subset, keep)
    return df.loc[~dupes]


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

    # Special case for tqdm_proxy(range(...))
    if isinstance(iterable, range) and 'total' not in kwargs:
        kwargs['total'] = (iterable.stop - iterable.start) // iterable.step
    
    try:
        import ipykernel.iostream
        from tqdm import tqdm_notebook
        if isinstance(sys.stdout, ipykernel.iostream.OutStream):
            return tqdm_notebook(iterable, **kwargs)
    except ImportError:
        pass
    
    _tqdm = tqdm
    _file = None
    disable_monitor = False
    
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

        # The tqdm monitor thread messes up our 'miniters' setting, so disable it.
        disable_monitor = True

        if 'ncols' not in kwargs:
            kwargs['ncols'] = 100
        
        if 'miniters' not in kwargs:
            # Aim for 5% updates
            if 'total' in kwargs:
                kwargs['miniters'] = kwargs['total'] // 20
            elif hasattr(iterable, '__len__'):
                kwargs['miniters'] = len(iterable) // 20
                

    kwargs['file'] = _file
    bar = _tqdm(iterable, **kwargs)
    if disable_monitor:
        bar.monitor_interval = 0
    return bar


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



