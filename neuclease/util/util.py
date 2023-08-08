import os
import re
import io
import sys
import copy
import time
import math
import json
import vigra
import logging
import inspect
import copyreg
import contextlib
from textwrap import indent
from itertools import chain
from operator import itemgetter
from functools import partial, lru_cache
from multiprocessing import get_context
from multiprocessing.pool import ThreadPool
from datetime import datetime, timedelta
from itertools import product
from collections import OrderedDict
from collections.abc import Mapping, Iterable, Iterator, Sequence

import pytz
import ujson
import requests
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
from numba import jit

from .downsample_with_numba import downsample_binary_3d_suppress_zero
from .box import box_to_slicing, box_union, extract_subvol, box_intersection
from .view_as_blocks import view_as_blocks

# Disable the tqdm monitor thread entirely.
# It is more trouble than it's worth, especially when using tqdm_proxy, below.
tqdm.monitor_interval = 0

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def Timer(msg=None, logger=None, level=logging.INFO, log_start=True):
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
    result = _TimerResult()
    if msg:
        logger = logger or logging.getLogger(__name__)
        if log_start:
            logger.log(level, msg + '...')
    try:
        yield result
    except BaseException as ex:
        result.stop = time.time()
        if msg:
            logger.error(msg + f' failed due to {type(ex).__name__} after {result.timedelta}')
        raise
    else:
        result.stop = time.time()
        if msg:
            logger.log(level, msg + f' took {result.timedelta}')


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


def load_df(npy_path):
    return pd.DataFrame(np.load(npy_path, allow_pickle=True))


@contextlib.contextmanager
def switch_cwd(d, create=False):
    """
    Context manager.
    chdir into the given directory (creating it first if desired),
    and exit back to the original CWD after the context manager exits.
    """
    if create:
        os.makedirs(d, exist_ok=True)
    old_dir = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old_dir)


class ndrange:
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

    See also: ``ndindex_array()``
    """

    def __init__(self, start, stop=None, step=None):
        if stop is None:
            stop = start
            start = (0,) * len(stop)

        if step is None:
            step = (1,) * len(stop)

        assert len(start) == len(stop) == len(step), \
            f"tuple lengths don't match: ndrange({start}, {stop}, {step})"

        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        return product(*map(range, self.start, self.stop, self.step))

    def __repr__(self):
        return f"ndrange({self.start}, {self.stop}, {self.step})"

    def __len__(self):
        span = (np.array(self.stop) - self.start)
        step = np.array(self.step)
        return np.prod( (span + step-1) // step )


def ndrange_array(start, stop=None, step=1):
    """
    Like np.ndindex, but accepts start/stop/step instead of
    assuming that start is always (0,0,0) and step is (1,1,1),
    and returns an array instead of an iterator.

    Like np.arange, but with tuple arguments.
    (This implementation only supports integer dtypes, though.)
    """
    start = np.asarray(start)
    if stop is None:
        stop = start
        start = (0,) * len(stop)

    def ndindex(shape):
        """Like np.ndindex, but returns ndarray"""
        return np.indices(shape).reshape(len(shape), -1).transpose()

    shape = (stop - start + step - 1) // step
    return start + step * ndindex(shape)


def ndindex_array(*shape, dtype=np.int32):
    """
    Like np.ndindex, but returns an array.
    
    numpy has no convenience function for this, and won't any time soon.
    https://github.com/numpy/numpy/issues/1234#issuecomment-545990743
    
    Example:
    
        >>> ndindex_array(3,4)
        array([[0, 0],
               [0, 1],
               [0, 2],
               [0, 3],
               [1, 0],
               [1, 1],
               [1, 2],
               [1, 3],
               [2, 0],
               [2, 1],
               [2, 2],
               [2, 3]])
    """
    return np.indices(shape, dtype=dtype).reshape(len(shape), -1).transpose()


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
        if isinstance(o, (np.ndarray, np.number, np.bool_)):
            return o.tolist()
        return super().default(o)


def dump_json(obj, f=None, indent=2, convert_nans=False, unsplit_int_lists=False, nullval="NaN"):
    """
    Pretty-print the given object to json, either to a file or to a returned string.

    obj:
        Object to serialize to json.
        Permitted to contain numpy arrays.

    f:
        A file handle to write to, or a file path to create,
        or None, in which case the json is returned as a string.

    convert_nans:
        If True, replace NaN values with the provided nullval ("NaN" by default).
        Otherwise, the default python behavior is to write the word NaN
        (without quotes) into the json file, which is not compliant with
        the json standard.

    unsplit_int_lists:
         When pretty-printing, json splits lists of integers (e.g. [123, 456, 789])
         across several lines.  For short lists, this is undesirable.
         This option will "unsplit" those lists, putting them back on a single line.
         This is implemented as a post-processing step using text matching.
         Might not be fast for huge files.

    nullval:
        If convert_nans is True, then ``nullval`` is used to replace nan values.

    Returns:
        Nothing if f was provided, otherwise returns a string.
    """
    if convert_nans:
        obj = _convert_nans(obj, nullval)

    kwargs = dict(indent=indent,
                  allow_nan=not convert_nans,
                  cls=NumpyConvertingEncoder)

    if unsplit_int_lists:
        json_text = json.dumps(obj, **kwargs)
        json_text = unsplit_json_int_lists(json_text)

        if isinstance(f, str):
            with open(f, 'w') as f:
                f.write(json_text)
        elif f:
            f.write(json_text)
        else:
            return json_text
    else:
        if isinstance(f, str):
            with open(f, 'w') as f:
                json.dump(obj, f, **kwargs)
        elif f:
            json.dump(obj, f, **kwargs)
        else:
            return json.dumps(obj, **kwargs)


def convert_nans(o, nullval="NaN", _c=None):
    """
    Traverse the given collection-of-collections and
    replace all NaN values with the string "NaN".
    Also converts numpy arrays into lists.
    Intended for preprocessing objects before JSON serialization.
    """
    _c = _c or {}

    if isinstance(o, float) and math.isnan(o):
        return nullval
    elif isinstance(o, np.number):
        if np.isnan(o):
            return nullval
        return o.tolist()
    elif isinstance(o, (str, bytes)) or not isinstance(o, (Sequence, Mapping)):
        return o

    # Even though this function is meant mostly for JSON,
    # so we aren't likely to run into self-referencing
    # or cyclical object graphs, we handle that case by keeping
    # track of the objects we've already processed.
    if id(o) in _c:
        return _c[id(o)]

    if isinstance(o, np.ndarray):
        ret = []
        _c[id(o)] = ret
        ret.extend([convert_nans(x, nullval, _c) for x in o.tolist()])
    elif isinstance(o, Sequence):
        ret = []
        _c[id(o)] = ret
        ret.extend([convert_nans(x, nullval, _c) for x in o])
    elif isinstance(o, Mapping):
        ret = {}
        _c[id(o)] = ret
        ret.update({k: convert_nans(v, nullval, _c) for k,v in o.items()})
    else:
        raise RuntimeError(f"Can't handle {type(o)} object: {o}")

    return ret


# used in json_dump(), above
_convert_nans = convert_nans


def unsplit_json_int_lists(json_text):
    """
    When pretty-printing json data, it will split all lists across several lines.
    For small lists of integers (such as [x,y,z] points), that may not be desirable.
    This function "unsplits" all lists of integers and puts them back on a single line.

    Example:
        >>> s = '''\\
        ... {
        ...   "body": 123,
        ...   "supervoxel": 456,
        ...   "coord": [
        ...     123,
        ...     456,
        ...     789
        ...   ],
        ... }
        ... '''

        >>> u = unsplit_json_int_lists(s)
        >>> print(u)
        {
        "body": 123,
        "supervoxel": 456,
        "coord": [123,456, 781],
        }

    """
    json_text = re.sub(r'\[\s+(\d+),', r'[\1,', json_text)
    json_text = re.sub(r'\n\s*(\d+),', r' \1,', json_text)
    json_text = re.sub(r'\n\s*(\d+)\s*\]', r' \1]', json_text)
    return json_text


def write_json_list(objects, f):
    """
    Like json.dump(), but writes each item to its own line (no indentation).
    """
    assert isinstance(objects, list)

    def _impl(f):
        f.write('[\n')
        for s in objects[:-1]:
            ujson.dump(s, f)
            f.write(',\n')
        ujson.dump(objects[-1], f)
        f.write('\n]')

    if isinstance(f, str):
        with open(f, 'w') as fp:
            _impl(fp)
    else:
        _impl(f)


def gen_json_objects(f, batch_size=None, parse=True):
    """
    Generator.
    
    Given a file containing a JSON list-of-objects,
    parse the objects one-by-one and iterate over them.
    
    Args:
        f:
            A file containing a JSON document which must be a list-of-objects.
            Must be an actual on-disk file (or a path to one),
            becuase it will be memory-mapped and therefore must have a fileno(). 

        batch_size:
            If provided, the objects will be yielded in groups
            (lists) of the specified size.

        parse:
            If True, each json object will be parsed and yielded as a dict.
            Otherwise, the raw text of the object is returned.
    """
    m = np.memmap(f, mode='r')
    it = map(bytes, _gen_json_objects(m))
    
    if parse:
        it = map(ujson.loads, it)
        
    if batch_size is None:
        yield from it
    else:
        yield from iter_batches(it, batch_size)


@jit(nopython=True, nogil=True)
def _gen_json_objects(text_array):
    """
    Generator.
    
    Parse a JSON list-of-objects one at a time,
    without reading in the entire file at once.
    
    Each object is yielded and then discarded.
    
    Warnings:
        - The input MUST be valid JSON, and specifically must be a list-of-objects.
          Any other input results in undefined behavior and/or errors.
        - Strings containing curly braces are not supported.
          (The document must not contain any curly braces except for the ones
          defining actual JSON objects.)
        
    Args:
        text_array:
            A np.array (dtype == np.uint8) which, when interpreted as text,
            contains a list-of-dicts JSON document.
    
    Yields:
        Every object in the document, one at a time.
    """
    nest_level = 0
    cur_start = 0
    cur_stop = 0
    for i, c in enumerate(text_array):
        if c == b'{'[0]:
            if nest_level == 0:
                cur_start = i
            nest_level += 1
        if c == b'}'[0]:
            nest_level -= 1
            if nest_level == 0:
                cur_stop = i+1
                yield text_array[cur_start:cur_stop]


def iter_batches(it, batch_size):
    """
    Iterator.

    Consume the given iterator/iterable in batches and
    yield each batch as a list of items.

    The last batch might be smaller than the others,
    if there aren't enough items to fill it.

    If the given iterator supports the __len__ method,
    the returned batch iterator will, too.

    This iterator implements special (efficient) support
    for lists, numpy arrays, Series/DataFrames, and h5py.Datasets.
    """
    if hasattr(it, '__len__'):
        return _iter_batches_with_len(it, batch_size)
    else:
        return _iter_batches(it, batch_size)


class _iter_batches:
    def __init__(self, it, batch_size):
        assert batch_size >= 0
        self.base_iterator = it
        self.batch_size = self._validate_batch_size(it, batch_size)

    @classmethod
    def _validate_batch_size(cls, it, batch_size):
        """
        User is not allowed to provide a batch_size of 0 unless
        the iterable is empty anyway.
        """
        if batch_size > 0:
            return batch_size

        msg = "Can't use batch_size of 0 with a non-empty iterable"
        if hasattr(it, '__len__'):
            assert len(it) == 0, msg
            return 1

        try:
            next(it)
        except StopIteration:
            # Iterable was empty, so we won't
            # complain about the zero batch size.
            return 1

        raise RuntimeError(msg)

    def __iter__(self):
        return self._iter_batches(self.base_iterator, self.batch_size)

    def _iter_batches(self, it, batch_size):
        if isinstance(it, (pd.DataFrame, pd.Series)):
            for batch_start in range(0, len(it), batch_size):
                yield it.iloc[batch_start:batch_start+batch_size]
            return

        if isinstance(it, (list, np.ndarray, h5py.Dataset)):
            for batch_start in range(0, len(it), batch_size):
                yield it[batch_start:batch_start+batch_size]
            return

        if not isinstance(it, Iterator):
            assert isinstance(it, Iterable)
            it = iter(it)

        while True:
            batch = []
            try:
                for _ in range(batch_size):
                    batch.append(next(it))
            except StopIteration:
                return
            finally:
                if batch:
                    yield batch


class _iter_batches_with_len(_iter_batches):
    def __len__(self):
        return int(np.ceil(len(self.base_iterator) / self.batch_size))


def compute_parallel(func, iterable, chunksize=1, threads=0, processes=0, ordered=None,
                     leave_progress=False, total=None, initial=0, starmap=False, show_progress=None,
                     context=None, shutdown_delay=0.15, **pool_kwargs):
    """
    Use the given function to process the given iterable in a ThreadPool or process Pool,
    showing progress using tqdm.

    Args:
        func:
            The function to process each item with.

        iterable:
            The items to process.

        chunksize:
            Send items to the pool in chunks of this size.

        threads:
            If given, use a ThreadPool with this many threads.

        processes
            If given, use a multiprocessing Pool with this many processes.
            Note: When using a process pool, your function and iterable items must be pickleable.

        ordered:
            Must be either True, False, or None:

            - If True, process the items in order, and return results
              in the same order as provided in the input.

            - If False, process the items as quickly as possible,
              meaning that some results will be presented out-of-order,
              depending on how long they took to complete relative to the
             other items in the pool.

            - If None, then the items will be processed out-of-order,
              but the results will be reordered to correspond to the original
              input order before returning.

        total:
            Optional. Specify the total number of tasks, for progress reporting.
            Not necessary if your iterable defines __len__.

        initial:
            Optional. Specify a starting value for the progress bar.

        starmap:
            If True, each item should be a tuple, which will be unpacked into
            the arguments to the given function, like ``itertools.starmap()``.

        show_progress:
            If True, show a progress bar.
            By default, only show a progress bar if ``iterable`` has more than one element.

        context:
            In Python, process pools can be created via 'fork', 'spawn', or 'forkserver'.
            Spawn is more robust, but comes with certain requirements
            (such as your main code being shielded within a __main__ conditional).
            See the Python multiprocessing docs for details.

        pool_kwargs:
            keyword arguments to pass to the underlying Pool object,
            such as ``initializer`` or ``maxtasksperchild``.
    """
    assert not bool(threads) or not bool(processes), \
        "Specify either threads or processes, not both"
    assert context in (None, 'fork', 'spawn', 'forkserver')
    assert ordered in (True, False, None)
    reorder = (ordered is None)

    # Pick a pool implementation
    if threads:
        pool = ThreadPool(threads, **pool_kwargs)
    elif processes:
        pool = get_context(context).Pool(processes, **pool_kwargs)
    else:
        pool = _DummyPool()

    if total is None and hasattr(iterable, '__len__'):
        total = len(iterable)

    # Pick a map() implementation
    if not threads and not processes:
        f_map = map
    elif ordered:
        f_map = partial(pool.imap, chunksize=chunksize)
    else:
        f_map = partial(pool.imap_unordered, chunksize=chunksize)

    # If we'll need to reorder the results,
    # then pass an index into (and out of) the function,
    # which we'll use to sort the results afterwards.
    if reorder:
        iterable = enumerate(iterable)

    # By default we call the function directly,
    # but the 'reorder' or 'starmap' options require wrapper functions.
    if reorder and starmap:
        func = partial(_idx_passthrough_apply_star, func)
    elif reorder and not starmap:
        func = partial(_idx_passthrough, func)
    elif not reorder and starmap:
        func = partial(_apply_star, func)

    if show_progress is None:
        if hasattr(iterable, '__len__') and len(iterable) == 1:
            show_progress = False
        else:
            show_progress = True

    with pool:
        iter_results = f_map(func, iterable)
        results_progress = tqdm_proxy(iter_results, initial=initial, total=total, leave=leave_progress, disable=not show_progress)
        try:
            with results_progress:
                # Here's where the work is actually done, i.e. during iteration.
                results = []
                for item in results_progress:
                    results.append(item)
        except KeyboardInterrupt as ex:
            # If the user killed the job early, provide the results
            # that have completed so far via an exception attribute.
            if reorder:
                results.sort(key=itemgetter(0))
                results = [r for (_, r) in results]

            # IPython users can access the exception via sys.last_value
            raise KeyboardInterruptWithResults(results, total or '?') from ex
        finally:
            # I see hangs here from time to time during normal operation,
            # even when no exception is raised (I think).
            # I suspect this is either related to my use of 'fork' as a multiprocecssing mode,
            # (which is generally frowned upon), or perhaps it's a bug in multiprocessing itself.
            # In any case, I'll try to combat the issue via:
            #
            #   1. Fight possible race conditions with a slight delay after the
            #      last item completes.
            #
            #   2. I also tried calling pool.close() here, but it seemed to cause
            #      deadlocks when terminate() is called.
            #
            # For reference, here is an example traceback for a hanged pool:
            #
            #     Thread 2854017 (idle): "Dask-Default-Threads-2848426-2"
            #         poll (multiprocessing/popen_fork.py:28)
            #         wait (multiprocessing/popen_fork.py:48)
            #         join (multiprocessing/process.py:140)
            #         _terminate_pool (multiprocessing/pool.py:617)
            #         __call__ (multiprocessing/util.py:224)
            #         terminate (multiprocessing/pool.py:548)
            #         __exit__ (multiprocessing/pool.py:623)
            #         compute_parallel (neuclease/util/util.py:723)
            #         ...
            #
            # Relevant discussions that might be related to this deadlock issue:
            # - https://sefiks.com/2021/07/05/handling-hang-in-python-multiprocessing/
            # - https://bugs.python.org/issue33997 and PR https://github.com/python/cpython/pull/8009
            # - https://stackoverflow.com/questions/65620077
            #
            #
            # Note: I could call pool.close() here, but I think that creates a
            #        deadlock in terminate() so I don't do that anymore.
            if shutdown_delay:
                time.sleep(shutdown_delay)
            pool.terminate()

    if reorder:
        results.sort(key=itemgetter(0))
        results = [r for (_, r) in results]

    return results


class KeyboardInterruptWithResults(KeyboardInterrupt):
    def __init__(self, partial_results, total_items):
        super().__init__()
        self.partial_results = partial_results
        self.total_items = total_items

    def __str__(self):
        return f'{len(self.partial_results)}/{self.total_items} results completed (see sys.last_value)'


class _DummyPool:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def close(self):
        pass

    def terminate(self):
        pass


def _apply_star(func, arg):
    return func(*arg)


def _idx_passthrough(func, idx_arg):
    idx, arg = idx_arg
    return idx, func(arg)


def _idx_passthrough_apply_star(func, idx_arg):
    idx, arg = idx_arg
    return idx, func(*arg)


DEFAULT_TIMESTAMP = datetime.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')


def parse_timestamp(ts, default=DEFAULT_TIMESTAMP, default_timezone="US/Eastern"):
    """
    Parse the given timestamp as a datetime object.
    If it is already a datetime object, it will be returned as-is.
    If it is None, then the given default timestamp will be returned.

    If the timestamp is not yet "localized", it will be assigned a
    timezone according to the default_timezone argument.
    (That is, we assume the time in the string was recorded in the specified timezone.)
    Localized timestamps include a suffix to indicate the offset from UTC.
    See the examples below.

    Note:
        By POSIX timestamp conventions, the +/- sign of the timezone
        offset might be reversed of what you expected unless you're
        already familiar with this sort of thing.

    Example timestamps:

        2018-01-01             (date only)
        2018-01-01 00:00       (date and time)
        2018-01-01 00:00:00    (date and time with seconds)
        2018-01-01 00:00:00.0  (date and time with microseconds)

        2018-01-01 00:00-4:00  (date and time, localized with some US timezone offset)

    Returns:
        datetime

    """
    if ts is None:
        ts = copy.copy(default)

    if isinstance(ts, (datetime, pd.Timestamp)):
        return ts

    if isinstance(ts, str):
        ts = pd.Timestamp(ts)

    if ts.tzinfo is None and default_timezone is not None:
        ts = pd.Timestamp.tz_localize(ts, pytz.timezone(default_timezone))

    return ts


def closest_approach(sv_vol, id_a, id_b, check_present=True):
    """
    Given a segmentation volume and two label IDs which it contains,
    find the two coordinates within id_a and id_b, respectively,
    which mark the two objects' closest approach, i.e. where the objects
    come closest to touching, even if they don't actually touch.
    
    Returns (coord_a, coord_b, distance)
    """
    assert id_a != 0 and id_b != 0, \
        "Can't use label 0 as an object ID in closest_approach()"
    
    assert sv_vol.dtype not in (np.uint64, np.int64, np.int32), \
        f"Volume type {sv_vol.dtype} is not convertible to uint32 without precision loss"
    
    mask_a = (sv_vol == id_a)
    mask_b = (sv_vol == id_b)

    if check_present and (not mask_a.any() or not mask_b.any()):
        # If either object is not present, there is no closest approach
        return (-1,-1,-1), (-1,-1,-1), np.inf
    
    if id_a == id_b:
        # IDs are identical.  Choose an arbitrary point.
        first_point = tuple(np.transpose(mask_a.nonzero())[0])
        return first_point, first_point, 0.0

    return closest_approach_between_masks(mask_a, mask_b)


def closest_approach_between_masks(mask_a, mask_b):
    """
    Given two non-overlapping binary masks,
    find the two coordinates within mask_a and mask_b, respectively,
    which mark the two objects' closest approach, i.e. where the objects
    come closest to touching, even if they don't actually touch.

    FIXME:
        This uses vigra's vectorDistanceTransform(), which uses a
        lot of RAM and computes the distance at all points in the mask.
        For sparse enough masks, it might be much more efficient to convert
        the masks to lists of coordinates and then use KD-trees to find
        the closest approach.
    """
    # Avoid circular import
    from neuclease.util.segmentation import compute_nonzero_box

    # Wrapper function just for visibility to profilers
    def vectorDistanceTransform(mask):
        mask = mask.astype(np.uint32)
        mask = vigra.taggedView(mask, 'zyx'[-mask.ndim:])

        # vigra always returns the vectors (in the channel dimension)
        # in 'xyz' order, but we want zyx order!
        vdt = vigra.filters.vectorDistanceTransform(mask)
        vdt = vdt[..., ::-1]
        return vdt

    # Extract the minimal subvolume that captures both masks
    box_a = compute_nonzero_box(mask_a)
    box_b = compute_nonzero_box(mask_b)
    box_u = box_union(box_a, box_b)
    mask_a = extract_subvol(mask_a, box_u)
    mask_b = extract_subvol(mask_b, box_u)

    # For all voxels, find the shortest vector toward id_b
    to_b_vectors = vectorDistanceTransform(mask_b)

    # Magnitude of those vectors == distance to id_b
    to_b_distances = np.linalg.norm(to_b_vectors, axis=-1)

    # We're only interested in the voxels within id_a;
    # everything else is infinite distance
    to_b_distances[~mask_a] = np.inf

    # Find the point within id_a with the smallest vector
    point_a = np.unravel_index(np.argmin(to_b_distances), to_b_distances.shape)
    distance = to_b_distances[tuple(point_a)]

    # Its closest point id_b is indicated by the corresponding vector
    point_a = np.asarray(point_a, np.int32)
    point_b = (point_a + to_b_vectors[tuple(point_a)]).astype(np.int32)

    # Add the subvolume offset
    point_a = tuple(point_a + box_u[0])
    point_b = tuple(point_b + box_u[0])
    return (point_a, point_b, distance)


def approximate_closest_approach(vol, id_a, id_b, scale=1):
    """
    Like closest_approach(), but first downsamples the data (for speed).
    
    The returned coordinates may not be precisely what closest_approach would have returned,
    but they are still guaranteed to reside within the objects of interest.
    """
    mask_a = (vol == id_a)
    mask_b = (vol == id_b)

    if not mask_a.any() or not mask_b.any():
        return ((-1, -1, -1), (-1, -1, -1), np.inf)

    scaled_mask_a, _ = downsample_binary_3d_suppress_zero(mask_a, (2**scale))
    scaled_mask_b, _ = downsample_binary_3d_suppress_zero(mask_b, (2**scale))

    scaled_point_a, scaled_point_b, _ = closest_approach_between_masks(scaled_mask_a, scaled_mask_b)

    scaled_point_a = np.asarray(scaled_point_a)
    scaled_point_b = np.asarray(scaled_point_b)

    # Compute the full-res box that corresponds to the downsampled points
    point_box_a = np.array([scaled_point_a, 1+scaled_point_a]) * (2**scale)
    point_box_b = np.array([scaled_point_b, 1+scaled_point_b]) * (2**scale)
    
    point_box_a = box_intersection(point_box_a, [(0,0,0), vol.shape])
    point_box_b = box_intersection(point_box_b, [(0,0,0), vol.shape])

    # Select the first non-zero point in the full-res box
    point_a = np.transpose(extract_subvol(mask_a, point_box_a).nonzero())[0] + point_box_a[0]
    point_b = np.transpose(extract_subvol(mask_b, point_box_b).nonzero())[0] + point_box_b[0]

    distance = np.linalg.norm(point_b - point_a)
    return (tuple(point_a), tuple(point_b), distance)


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


def downsample_mask(mask, factor, method='or'):
    """
    Downsample a boolean mask by the given factor.
    """
    assert method in ('or', 'and')

    mask = np.asarray(mask)
    assert mask.ndim >= 1
    if not isinstance(factor, Iterable):
        factor = mask.ndim*(factor,)

    factor = np.asarray(factor)
    assert (factor >= 1).all(), f"Non-positive downsampling factor: {factor}"
    assert not any(mask.shape % factor), \
        "mask shape must be divisible by the downsampling factor"

    if (factor == 1).all():
        return mask

    mask = np.asarray(mask, order='C')
    v = view_as_blocks(mask, (*factor,))
    last_axes = (*range(v.ndim),)[-mask.ndim:]

    if method == 'or':
        f = np.logical_or.reduce
    if method == 'and':
        f = np.logical_and.reduce

    return f(v, axis=last_axes)


def extract_labels_from_volume(points_df, volume, box_zyx=None, vol_scale=0, label_names=None, name_col=None):
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
    - The volume can be provided in downscaled form, in which case the
      given points will be downscaled before sampling is performed.
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
            Optional.  Specifies how label IDs map to label names.
            If provided, a new column 'label_name' will be appended to
            points_df in addition to the 'label' column.

            Must be either:
            - a mapping of `{ label_id: name }` (or `{ name : label_id }`),
              indicating each label ID in the output image, or
            - a list label names in which case the mapping is determined automatically
              by enumerating the labels in the given order (starting at 1).

        name_col:
            Customize the name of the column which will be used to store the extracted
            label and label names. Otherwise, the results are stored in the columns
            'label' and 'label_name'.

    Returns:
        None.  Results are appended to the points_df as new column(s).
    """
    if box_zyx is None:
        box_zyx = np.array(([0]*volume.ndim, volume.shape))

    assert ((box_zyx[1] - box_zyx[0]) == volume.shape).all() 

    assert points_df.index.duplicated().sum() == 0, \
        "This function doesn't work if the input DataFrame's index has duplicate values."

    downsampled_coords_zyx = (points_df[['z', 'y', 'x']] // (2**vol_scale)).astype(np.int32)

    # Drop everything outside the combined_box
    min_z, min_y, min_x = box_zyx[0] #@UnusedVariable
    max_z, max_y, max_x = box_zyx[1] #@UnusedVariable
    dc = downsampled_coords_zyx
    downsampled_coords_zyx = dc.loc[   (dc['z'] >= min_z) & (dc['z'] < max_z)
                                     & (dc['y'] >= min_y) & (dc['y'] < max_y)
                                     & (dc['x'] >= min_x) & (dc['x'] < max_x) ]
    del dc

    logger.info(f"Extracting labels from volume at {len(downsampled_coords_zyx)} points")
    downsampled_coords_zyx -= box_zyx[0]

    points_df.drop(columns=['label', 'label_name'], errors='ignore', inplace=True)
    points_df['label'] = volume.dtype.type(0)
    points_df.loc[downsampled_coords_zyx.index, 'label'] = volume[tuple(downsampled_coords_zyx.values.transpose())]

    # If no names were supplied, we're done.
    if label_names is None:
        return

    if isinstance(label_names, Mapping):
        # We need a mapping of label_ids -> names.
        # If the user provided the reverse mapping,
        # then flip it.
        (k,v) = next(iter(label_names.items()))
        if isinstance(k, str):
            # Reverse the mapping
            label_names = { v:k for k,v in label_names.items() }
    else:
        label_names = dict(enumerate(label_names, start=1))

    name_set = ['<unspecified>', *label_names.values()]
    default_names = ['<unspecified>']*len(points_df)
    # FIXME: More than half of the runtime of this function is spent on this line!
    #        Is there some way to speed this up?
    points_df['label_name'] = pd.Categorical( default_names,
                                              categories=name_set,
                                              ordered=False )
    for label, name in label_names.items():
        rows = points_df['label'] == label
        points_df.loc[rows, 'label_name'] = name

    if name_col:
        points_df.drop(columns=[name_col, f'{name_col}_label'], errors='ignore', inplace=True)
        points_df.rename(inplace=True, columns={'label': f'{name_col}_label', 'label_name': name_col})


def compute_merges(orig_vol, agg_vol):
    """
    Given an original volume and another volume which was generated
    exclusively from merges of the original, recover the merge decisions
    that were made.  That is, give the list of merges in the original
    volume that could reconstruct the geometry of segments in the
    agglomerated volume.
    
    Args:
        orig_vol:
            label volume, original segmentation

        agg_vol:
            label volume, agglomerated segmentation
    
    Returns:
        dict: { agg_id: [orig_id, orig_id, ...] },
        where the original IDs present in each merge are listed from largest to smallest.
        Agglomerated segments that exactly match an original segment (no merges) are not
        included in the results. (All lists in the results have at least two items.)
    
    Notes:
      - This function does not make any attempt to handle splits gracefully.
        For correct results, the every segment in the original volume should
        be a subset of only one segment in the agglomerated volume.
    
      - The label IDs in the agglomerated volume need not be related
        in any way to the label IDs in the original.
    """
    # Compute the set of unique orig-agg pairs, and the size of each
    df = pd.DataFrame({'orig': orig_vol.reshape(-1), 'agg': agg_vol.reshape(-1)})
    paired_seg_voxels = df.groupby(['orig', 'agg']).size().rename('voxels')
    paired_seg_voxels = pd.DataFrame(paired_seg_voxels)

    # For each agg ID with more than one corresponding 'orig' ID,
    # Compute the list of merges that reconstruct the agg geometry
    merges = {}    
    for agg, g_df in paired_seg_voxels.groupby('agg'):
        if len(g_df) > 1:
            merged_orig = g_df.sort_values('voxels', ascending=False).index.get_level_values('orig')
            merges[agg] = merged_orig.tolist()

    return merges


def unordered_duplicated(df, subset=None, keep='first'):
    """
    Like pd.DataFrame.duplicated(), but sorts each row first, so
    rows can be considered duplicates even if their values don't
    appear in the same order.

    Example:
    
        >>> df = pd.DataFrame( [(1, 2, 0.0),
                                (2, 1, 0.1), # <-- duplicate a/b columns
                                (3, 4, 0.2)],
                              columns=['a', 'b', 'score'])

        >>> unordered_duplicated(df, ['a', 'b'])
        0    False
        1     True
        2    False
        dtype: bool    
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

    Example:

        >>> df = pd.DataFrame( [(1, 2, 0.0),
                                (2, 1, 0.1), # <-- duplicate a/b columns
                                (3, 4, 0.2)],
                              columns=['a', 'b', 'score'])

        >>> drop_unordered_duplicates(df, ['a', 'b'])
           a  b  score
        0  1  2    0.0
        2  3  4    0.2

    """
    dupes = unordered_duplicated(df, subset, keep)
    return df.loc[~dupes]


def fix_df_names(df):
    """
    Rename all columns of the given dataframe with programmer-friendly alternatives,
    i.e. lowercase and replace spaces with underscores.
    """
    return df.rename(columns={c: c.lower().replace(' ', '_').replace('-', '_') for c in df.columns})


def downgrade_nullable_dtypes(df, null_handling='error'):
    """
    Convert all nullable integer columns to use a standard numpy dtype.

    New versions of pandas introduced the concept of nullable integer types,
    which are not identical to numpy dtypes:

        https://pandas.pydata.org/docs/user_guide/integer_na.html

    This function will just convert them back to regular numpy dtypes.
    If a column contains null values, the behavior depends on null_handling parameter.

    Args:
        df:
            DataFrame, possibly containing column dtypes like 'Int64' (instead of 'int64')

        null_handling:
            Either 'error', 'skip', or 'makefloat'.

            If 'error':
                Attempt to convert ALL nullable columns, and let pandas raise an exception
                if one of your columns contains a null value and therefore can't be
                downgraded to a numpy integer dtype.

            If 'skip'
                Downgrade columns which don't contain null values, but skip the ones that do.

            If 'makefloat':
                Downgrade columns which contain nulls to np.float64, as pandas uses by default.
    Returns:
        DataFrame
    """
    assert null_handling in ('error', 'skip', 'makefloat')
    pd_to_np = {'boolean': 'bool'}
    for sign, size in product(['U', ''], [8, 16, 32, 64]):
        k = f'{sign}Int{size}'
        pd_to_np[k] = k.lower()

    new_dtypes = {}
    for col, dtype in df.dtypes.items():
        if null_handling == 'skip' and df[col].isnull().any():
            continue
        elif null_handling == 'makefloat' and str(dtype) in pd_to_np and df[col].isnull().any():
            if str(dtype) == "boolean":
                new_dtypes[col] = object
            else:
                new_dtypes[col] = np.float64
        else:
            new_dtypes[col] = pd_to_np.get(str(dtype), dtype)

    return df.astype(new_dtypes)


def swap_df_cols(df, prefixes=None, swap_rows=None, suffixes=['_a', '_b']):
    """
    Swap selected columns of a dataframe, specified as a list of prefixes and two suffixes.
    Operates IN-PLACE, but incurs a full copy internally of the selected columns.

    Args:
        df:
            Input dataframe, with columns to be swapped.

        prefixes:
            columns to swap, minus their suffixes.
            If not provided, all columns with corresponding suffixes will be swapped.

        swap_rows:
            Optional.
            Specify a subset of rows in the dataframe to apply the swap to.
            Should be a Series boolean values, or a list of index values. 
            If this is a Series, it must have the same index as the input dataframe.
            If not provided, all rows are swapped.

        suffixes:
            Used to identify the left/right columns of each swapped pair.

    Returns:
        None.  Operates IN-PLACE.

    Example:
        >>> df = pd.DataFrame(np.arange(12).reshape(3,4), columns=['x_a', 'x_b', 'y_a', 'y_b'])

        >>> df
           x_a  x_b  y_a  y_b
        0    0    1    2    3
        1    4    5    6    7
        2    8    9   10   11

        >>> swap_df_cols(df, None, [True, False, True])
           x_a  x_b  y_a  y_b
        0    1    0    3    2
        1    4    5    6    7
        2    9    8   11   10

    """
    suffixes = list(suffixes)
    assert len(suffixes) == 2

    if prefixes is None:
        prefixes = set()
        suffix_len = len(suffixes[0])
        assert suffix_len == len(suffixes[1]), "Suffixes are not the same length"
        for col in df.columns:
            prefix = col[:-suffix_len]
            if (prefix + suffixes[0] in df) and (prefix + suffixes[1] in df):
                prefixes.add(prefix)
        assert prefixes, "Could not find any column pairs with the given suffixes"

    if swap_rows is None:
        swap_rows = slice(None)
    else:
        assert swap_rows.dtype == np.bool

    all_cols = [p + s for p,s in product(prefixes, suffixes)]
    missing_cols = set(all_cols) - set(df.columns)
    assert not missing_cols, \
        f"The following columns do not exist in the input DataFrame: {list(missing_cols)}"

    orig_df = df[all_cols].copy()

    for prefix in prefixes:
        col_a = prefix + suffixes[0]
        col_b = prefix + suffixes[1]
        df.loc[swap_rows, col_a] = orig_df.loc[swap_rows, col_b]
        df.loc[swap_rows, col_b] = orig_df.loc[swap_rows, col_a]


def tqdm_proxy(iterable=None, *, logger=None, level=logging.INFO, **kwargs):
    """
    Useful as an (almost) drop-in replacement for ``tqdm`` which can be used
    in EITHER an interactive console OR a script that logs to file.

    Automatically detects whether or not sys.stdout is a file or a console,
    and configures tqdm accordingly.

    - If your code is running from an interactive console, this acts like plain ``tqdm``.
    - If your code is running from an ipython notebook, this acts like ``tqdm.notebook.tqdm``.
    - If your code is running from a batch script (i.e. printing to a log file, not the console),
      this code uses the supplied logger to periodically output a textual progress bar.
      If no logger is supplied, a logger is automatically created using the name of
      the calling module.

    Example:

        for i in tqdm_proxy(range(1000)):
            # do some stuff

    Note for JupyterLab users:

        If you get errors in this function, you need to run the following commands:

            conda install -c conda-forge ipywidgets
            jupyter nbextension enable --py widgetsnbextension
            jupyter labextension install @jupyter-widgets/jupyterlab-manager

        ...and then reload your jupyterlab session, and restart your kernel.
    """
    assert 'file' not in kwargs, \
        "There's no reason to use this function if you are providing your own output stream"

    # Special case for tqdm_proxy(range(...))
    if iterable is not None and isinstance(iterable, range) and 'total' not in kwargs:
        kwargs['total'] = (iterable.stop - iterable.start) // iterable.step

    try:
        import ipykernel.iostream
        from tqdm.notebook import tqdm as tqdm_notebook
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


@jit(nopython=True, nogil=True)
def encode_coords_to_uint64(coords):
    """
    Encode an array of (N,3) int32 into an array of (N,) uint64,
    giving 21 bits per coord.
    """
    assert coords.shape[1] == 3

    N = len(coords)
    encoded_coords = np.empty(N, np.uint64)

    for i in range(N):
        z, y, x = coords[i].astype(np.int64)

        # Mask to just 21 bits
        z &= ((1 << 21) - 1)
        y &= ((1 << 21) - 1)
        x &= ((1 << 21) - 1)

        encoded = 0
        encoded |= z << 42
        encoded |= y << 21
        encoded |= x
        encoded_coords[i] = encoded

    return encoded_coords


@jit(nopython=True, nogil=True)
def decode_coords_from_uint64(encoded_coords, signed=True):
    """
    The reciprocal to encoded_coords_to_uint64(), above.

    Args:
        encoded_coords:
            A 1D array of uint64
        signed:
            If True, then interprets each encoded Z,Y,X value as a 21-bit
            signed 2's complement value, and emit negative int32 values as needed.
    """
    N = len(encoded_coords)
    coords = np.empty((N,3), np.int32)

    for i in range(N):
        encoded = encoded_coords[i]
        z = np.int32((encoded >> 2*21) & 0x1F_FFFF)  # 21 bits
        y = np.int32((encoded >>   21) & 0x1F_FFFF)  # 21 bits
        x = np.int32((encoded >>    0) & 0x1F_FFFF)  # 21 bits

        if signed:
            # Check sign bits and extend if necessary
            if encoded & (1 << (3*21-1)):
                z |= np.int32(0xFFFF_FFFF << 21)

            if encoded & (1 << (21*2-1)):
                y |= np.int32(0xFFFF_FFFF << 21)

            if encoded & (1 << (21*1-1)):
                x |= np.int32(0xFFFF_FFFF << 21)

        coords[i] = (z,y,x)

    return coords


@jit(nopython=True, nogil=True)
def encode_coords_to_blockmajor_uint64(coords, signed=True):
    """
    Encode an array of (N,3) int32 into an array of (N,) uint64,
    but arrange the bit representations such that the block ID (Bz, By, Bx)
    occupies the most significant bits and the within-block point ID (bz, by, bx)
    occupies the least significant bits.

    The block shape is hard-coded as (64,64,64), and therefore 6 bits are
    allocated to each dimension of the within-block point ID.
    The bit fields of the encoding are shown below.

    0 Bz (15 bits)    By (15 bits)    Bx (15 bits)    bz (6) by (6) bx (6)
    - --------------- --------------- --------------- ------ ------ ------
    3 210987654321098 765432109876543 210987654321098 765432 109876 543210

    If the resulting integers were then sorted, they'd appear in
    block scan-order, and voxelwise scan-order within each block.

    Args:
        coords:
            ndarray of int32, shape (N,3)
            All values must be be in range(-2**20, 2**20), unless signed=False,
            in which case the acceptable range is (0, 2**21).

        signed:
            To properly preserve sort order of negative coordinates, the block IDs
            are stored in 'offset binary' form unless signed=False,
            in which case the block IDs are stored without modification.
            See above for the acceptable coordinate ranges in the signed and
            unsigned cases.
    Returns:
        1D array, uint64, length N
    """
    assert coords.shape[1] == 3

    N = len(coords)
    encoded_coords = np.empty(N, np.uint64)

    for i in range(N):
        z, y, x = coords[i].astype(np.int64)

        # Mask to just 21 bits
        z &= ((1 << 21) - 1)
        y &= ((1 << 21) - 1)
        x &= ((1 << 21) - 1)

        if signed:
            # Convert to 'offset binary', i.e. invert the MSB
            z ^= (1 << 20)
            y ^= (1 << 20)
            x ^= (1 << 20)

        # Blocks are 64px (2**6)
        Bz = z >> 6
        By = y >> 6
        Bx = x >> 6

        mask = (1 << 6) - 1
        bz = z & mask
        by = y & mask
        bx = x & mask

        encoded = np.int64(0)
        encoded |= Bz << (15*2 + 6*3)
        encoded |= By << (15*1 + 6*3)
        encoded |= Bx << (15*0 + 6*3)
        encoded |= bz << (6*2)
        encoded |= by << (6*1)
        encoded |= bx << (6*0)
        encoded_coords[i] = np.uint64(encoded)

    return encoded_coords


@jit(nopython=True, nogil=True)
def decode_coords_from_blockmajor_uint64(encoded_coords, signed=True):
    """
    Reciprocal of encode_coords_to_blockmajor_uint64().
    Not needed very often, but useful for testing.
    """
    N = len(encoded_coords)
    coords = np.empty((N,3), np.int32)

    for i in range(N):
        encoded = encoded_coords[i]
        Bz = encoded >> (15*2 + 6*3) & ((1 << 15) - 1)
        By = encoded >> (15*1 + 6*3) & ((1 << 15) - 1)
        Bx = encoded >> (15*0 + 6*3) & ((1 << 15) - 1)

        bz = encoded >> (6*2) & ((1 << 6) - 1)
        by = encoded >> (6*1) & ((1 << 6) - 1)
        bx = encoded >> (6*0) & ((1 << 6) - 1)

        z = np.int32((Bz << 6) | bz)
        y = np.int32((By << 6) | by)
        x = np.int32((Bx << 6) | bx)

        if signed:
            # Convert from 'offset binary', i.e. invert the MSB
            z ^= (1 << 20)
            y ^= (1 << 20)
            x ^= (1 << 20)

            # Check sign bits and extend if necessary
            if z & (1 << 20):
                z |= np.int32(0xFFFF_FFFF << 21)

            if y & (1 << 20):
                y |= np.int32(0xFFFF_FFFF << 21)

            if x & (1 << 20):
                x |= np.int32(0xFFFF_FFFF << 21)

        coords[i] = (z,y,x)

    return coords


def sort_blockmajor(df, inplace=False, ignore_index=False, show_blockmajor_id=False):
    """
    Sort the given dataframe with block coordinates (assuming 64px blocks)
    and then voxel coordinate within each block.

    (This function works even when some coordinates are negative.)

    Args:
        df:
            DataFrame containing at least columns 'zyx'

        inplace:
            If True, sort in place. Otherwise return a new DataFrame.

        show_blockmajor_id:
            If True, leave the 'blockmajor_id' column in the result,
            which is the value that was used to sort the data.
            Otherwise, delete that column before returning.

    Return:
        If inplace=False, return a new DataFrame.
        Otherwise, return None.
    """
    assert set(df.columns) >= {*'zyx'}, "DataFrame must contain zyx columns"
    if not inplace:
        df = df.copy()

    # TODO: Verify acceptable coordinate min/max for 'signed blockmajor' encoding,
    #       and possibly fall back to ordinary multi-column df.sort_values()
    #       if necessary.

    df['_blockmajor_id'] = encode_coords_to_blockmajor_uint64(df[[*'zyx']].values)
    df.sort_values('_blockmajor_id', inplace=True, ignore_index=ignore_index)

    if show_blockmajor_id:
        df = df.rename(columns={'_blockmajor_id': 'blockmajor_id'})
    else:
        del df['_blockmajor_id']

    if not inplace:
        return df


def mask_centroid(mask, as_int=False):
    """
    Compute the centroid of an ND mask.
    Requires N passes but not much RAM overhead.
    """
    # Use broadcasting tricks to avoid creating a full field of coordinates.
    # When implicitly broadcasted with the 'where' arg below,
    # the operation sums over all coordinates that belong to a non-zero voxel.
    mask = mask.astype(bool, copy=False)
    slicing = tuple(slice(None, s) for s in mask.shape)
    coords = np.ogrid[slicing]

    size = mask.sum()
    centroid = []
    for a in coords:
        c = np.add.reduce(a, axis=None, where=mask) / size
        centroid.append(c)

    centroid = np.array(centroid)
    if as_int:
        return centroid.astype(np.int32)
    else:
        return centroid


@lru_cache(maxsize=1)
def sphere_mask(radius):
    """
    Return the binary mask of a sphere.
    Resulting array is a cube with side 2R+1
    """
    r = radius
    cz, cy, cx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    distances_sq = cz**2 + cy**2 + cx**2
    mask = (distances_sq <= r**2)

    # The result will be cached, so don't let the caller overwrite it!
    mask.flags['WRITEABLE'] = False
    return mask


@lru_cache(maxsize=1)
def ellipsoid_mask_axis_aligned(rz, ry, rx):
    """
    Return the binary mask of an axis-aligned ellipsoid.
    Resulting array has dimensions (2*rz+1, 2*ry+1, 2*rx+1)
    """
    cz, cy, cx = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    k = (cz/rz)**2 + (cy/ry)**2 + (cx/rx)**2
    mask = (k <= 1)

    # The result will be cached, so don't let the caller overwrite it!
    mask.flags['WRITEABLE'] = False
    return mask


def place_sphere(vol, center_point, radius, label=1):
    """
    Place a sphere mask within the given volume
    """
    p = np.asarray(center_point)
    r = radius
    sphere_box = [p - r, p + r + 1]
    cropped_box = box_intersection([(0,0,0), vol.shape], sphere_box)
    m = extract_subvol(sphere_mask(radius), cropped_box - sphere_box[0])

    subvol = vol[box_to_slicing(*cropped_box)]
    subvol[:] = np.where(m, label, subvol)


def ellipsoid_mask(v0, v1, v2):
    """
    Return the binary mask of an ellipsoid with the given semi-axis vectors (the 3 radii),
    which should be orthogonal to each other but not necessarily axis-aligned.
    """
    det = np.linalg.det
    V = np.stack((v0, v1, v2))
    R = np.ceil(np.abs(V).max(axis=0))

    # Pad the bounding box since V is not axis-aligned.
    R *= np.sqrt(2)

    R = R.astype(int)
    shape = 2*R + 1

    # We'll be testing each coordinate for inside/outside ellipsoid
    zyx = ndrange_array(-R, R+1)

    # Re-use this array for each of the determinant calculations below.
    m = np.zeros((len(zyx), 3, 3), np.float32)

    # Parametric equation of an ellipse from its semi-axis vectors:
    # https://en.wikipedia.org/wiki/Ellipsoid#Parametric_representation
    #
    #   F(X) = 0 = det([X, v1, v2])**2 + det([v0, X, v2])**2 + det([v0, v1, X])**2
    #
    #     where X = (x,y,z)
    m[:, 0, :] = zyx
    m[:, (1, 2), :] = (v1, v2)
    d0 = det(m)

    m[:, 1, :] = zyx
    m[:, (0, 2), :] = (v0, v2)
    d1 = det(m)

    m[:, 2, :] = zyx
    m[:, (0, 1), :] = (v0, v1)
    d2 = det(m)

    del m

    F = d0**2 + d1**2 + d2**2 - det([v0, v1, v2])**2
    return (F <= 0).reshape(shape)


def fit_ellipsoid(mask):
    """
    Fit an ellipsoid to the given mask.
    Uses PCA to obtain the major axes of the mask points,
    then rescales the PCA vectors to obtain an ellipse
    that has the same volume as the given mask.

    Returns:
        size, center, radii_vec

            where radii_vec is given as three row vectors.
    """
    assert mask.ndim == 3
    points = np.array(mask.nonzero())
    size = points.shape[1]
    center = points.mean(axis=1)
    cov = np.cov(points, bias=True)

    # Notes:
    # - Eigenvalues of covariance matrix are PCA magnitudes
    # - Eigenvectors are in columns, and are normalized
    eigvals, eigvecs = np.linalg.eigh(cov)
    assert np.allclose(np.linalg.norm(eigvecs, axis=0), 1.0)

    # For tiny shapes or pathological cases,
    # we get tiny eigenvalues, even negative.
    # Set a floor of 0.25 to indicate a 0.5 radius (1-px diameter.)
    eigvals = np.maximum(0.25, eigvals)

    # The PCA radii are not scaled to be the
    # semi-major/minor radii of the ellipsoid unless we rescale them.
    # The volume of an ellipsoid with semi-radii a,b,c is 4πabc/3
    # so scale the radii up until the volume of the corresponding ellipsoid
    # matches the soma's actual volume.

    pca_vol = (4/3) * np.pi * np.prod(eigvals)
    s = size / pca_vol

    # Apply that factor to the PCA magnitudes to obtain better ellipsoid radii
    radii_mag = eigvals * np.power(s, 1/3)

    # Also scale up the eigenvectors
    radii_vec = radii_mag[None, :] * eigvecs

    # Transpose to row vectors, order large-to-small
    radii_vec = radii_vec.transpose()[::-1, :]

    return size, center, radii_vec


def upload_to_bucket(bucket, blob_name, blob_contents):
    """
    Upload a blob of data to the specified google storage bucket.
    """
    if isinstance(bucket, str):
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket)

    blob = bucket.blob(blob_name)
    blob.cache_control = 'public, no-store'
    blob.upload_from_string(blob_contents, content_type='application/json')
    return blob.public_url


def perform_bigquery(q, client=None, project='janelia-flyem', downgrade_nullables_via='skip'):
    """
    Send the given SQL query to BigQuery
    and return the results as a DataFrame.

    Args:
        q:
            SQL query string

        client:
            Optional. A bigquery Client object.

        project:
            GCP Project name

        downgrade_nullables_via:
            Annoyingly, newer versions of the bigquery Python library
            make use of the new pandas' nullable integer types[1],
            even when no null values are present in the results.
            Those types are not 100% backwards-compatible with the older
            numpy-based integer dtypes, even when no null values are present.
            In this function, we automatically "downgrade" nullable dtypes to
            their numpy integer equivalents when possible.
            See `downgrade_nullable_dtypes()` for details.
            To disable downgrading altogether, set `downgrade_nullables_via=False`.

    Returns:
        pd.DataFrame

    [1]: https://pandas.pydata.org/docs/user_guide/integer_na.html

    """
    from google.cloud import bigquery
    assert 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ

    if client is None:
        assert project in os.environ['GOOGLE_APPLICATION_CREDENTIALS'], \
            "Usually the credentials file name mentions the project name.  It looks like you have the wrong credentials loaded."
        client = bigquery.Client(project)

    # In theory, there are faster ways to download table data using parquet,
    # but bigquery keeps giving me errors when I try that.
    r = client.query(q).result()
    df = r.to_dataframe()
    if downgrade_nullables_via:
        df = downgrade_nullable_dtypes(df, downgrade_nullables_via)
    return df


def perform_bigquery_upload(df, full_table_name, client=None):
    """
    Upload the given DataFrame to BigQuery.
    If the table already exists, it will be overwritten.
    """
    from google.cloud import bigquery
    assert 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ

    project, dataset, table = full_table_name.split('.')

    if client is None:
        assert project in os.environ['GOOGLE_APPLICATION_CREDENTIALS'], \
            "Usually the credentials file name mentions the project name.  It looks like you have the wrong credentials loaded."
        client = bigquery.Client(project)
    else:
        assert client.project == project, \
            "Client project doesn't match the table's project name"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
    )
    job = client.load_table_from_dataframe(df, f'{dataset}.{table}', job_config=job_config)
    r = job.result()
    if job.state != "DONE":
        raise RuntimeError(f"Upload failed: job state is {job.state}")
    return r


def upload_chunked_table(df, full_table_name, gbucket_dir, gbucket_subdir, local_dir='/tmp'):
    """
    Export the given DataFrame to chunked parquet, upload it to a gbucket,
    and then load it as a table in BigQuery.

    Args:
        df: The DataFrame to upload.  The index is NOT uploaded.

    This is useful for tables which are too large to upload in one chunk
    via the Python API.
    """
    import pyarrow
    import subprocess

    assert gbucket_dir.startswith('gs://')

    project, dataset, table = full_table_name.split('.')

    if df is not None:
        os.makedirs(f"{local_dir}/{table}")
        pyarrow.parquet.write_table(pyarrow.Table.from_pandas(df), f"{local_dir}/{table}/*.parquet")

    subprocess.run(f"gsutil -q cp -r {local_dir}/{table} {gbucket_dir}/{gbucket_subdir}/", shell=True, check=True)
    subprocess.run(f"gsutil ls -d {gbucket_dir}/{gbucket_subdir}/{table}", shell=True, check=True)
    subprocess.run(f"bq load --source_format=PARQUET {dataset}.{table} {gbucket_dir}/{gbucket_subdir}/{table}/*.parquet", shell=True, check=True)


def perform_bigquery_and_write_table(q, full_table_name, client=None):
    """
    Perform a query in BigQuery and save the resulting table to a new table with the given name.
    If there is already a table with that name, it will be overwritten.
    """
    from google.cloud import bigquery
    assert 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ

    project, dataset, table = full_table_name.split('.')

    if client is None:
        assert project in os.environ['GOOGLE_APPLICATION_CREDENTIALS'], \
            "Usually the credentials file name mentions the project name.  It looks like you have the wrong credentials loaded."
        client = bigquery.Client(project)
    else:
        assert client.project == project, \
            "Client project doesn't match the table's project name"

    job_config = bigquery.QueryJobConfig(
        destination=full_table_name,
        write_disposition='WRITE_TRUNCATE'
    )
    job = client.query(q, job_config=job_config)
    r = job.result()
    if job.state != "DONE":
        raise RuntimeError(f"Upload failed: job state is {job.state}")
    return r


def extract_bigquery_table_to_gbucket(full_table_name, gbucket_dir, gbucket_subdir=None, client=None):
    """
    Extract a BigQuery table into a gbucket as a series of parquet files.
    """
    from google.cloud import bigquery
    assert 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ
    assert gbucket_dir.startswith('gs://')

    project, dataset, table = full_table_name.split('.')

    if client is None:
        assert project in os.environ['GOOGLE_APPLICATION_CREDENTIALS'], \
            "Usually the credentials file name mentions the project name.  It looks like you have the wrong credentials loaded."
        client = bigquery.Client(project)
    else:
        assert client.project == project, \
            "Client project doesn't match the table's project name"

    gbucket_subdir = gbucket_subdir or table

    job_config = bigquery.job.ExtractJobConfig()
    job_config.destination_format = 'parquet'
    job = client.extract_table(
        bigquery.DatasetReference(project, dataset).table(table),
        f"{gbucket_dir}/{gbucket_subdir}/*.parquet",
        location="US",
        job_config=job_config,
    )
    r = job.result()
    if job.state != "DONE":
        raise RuntimeError(f"Extract failed: job state is {job.state}")
    return r


def construct_query_to_convert_bq_ints(table, gcp_project='janelia-flyem', fields_only=False, client=None):
    """
    If you want to export an entire table from BigQuery,
    you first need to convert any UINT64 to int64, since the former
    is not supported by their parquet export.

    This function produces a query string that can be used to retrieve the
    entire contents of the table while casting all integers to int64 along the way.
    It retains any nested (struct) object structure in the results.

    Note:
        It is assumed that the table's first row is representative of the full table schema.
        If any nullable integer fields in the first row are missing, those fields won't be converted.

    Args:
        table:
            name of the BigQuery table for which the conversion query should be made,
            e.g. "fullbrain_seg_20220130_32fb16fb8f.rsg32_32fb16fb"
        gcp_project:
            name of the GCP project which owns the BigQuery table

    Returns:
        str
    """
    def _select_field(full_k, v):
        k = full_k.split('.')[-1]
        if isinstance(v, dict):
            members = (_select_field(f'{full_k}.{ki}', vi) for ki, vi in v.items())
            members = ',\n'.join(members)
            return indent(f"struct(\n{members}\n) as {k}", '  ')

        if v is None:
            return f'  {full_k} as {k}'
        elif isinstance(v, int):
            return f"  cast({full_k} as int64) as {k}"
        else:
            return f"  {full_k}"

    if client is None:
        from google.cloud import bigquery
        client = bigquery.Client(gcp_project)
    r = client.query(f"select * from `{table}` limit 1").result()
    example_row = dict(next(r))
    fields = ',\n'.join(_select_field(k,v) for k,v in example_row.items())
    if fields_only:
        return fields
    return (
        f"select\n"
        f"{fields}\n"
        f"from `{table}`"
    )


def display_sql(q):
    """
    Display syntax-highlighted SQL in a jupyter notebook using pygments.
    """
    from textwrap import dedent
    q = dedent(q)

    try:
        from IPython.display import HTML, Code, display
        from pygments import highlight
        from pygments.lexers import SqlLexer
        from pygments.formatters import HtmlFormatter
    except ImportError:
        print(q)
    else:
        formatter = HtmlFormatter(style='emacs')
        css = formatter.get_style_defs('.highlight')
        html = highlight(q, SqlLexer(), formatter)
        display(HTML(f"<style>{css}</style>{html}"))


def find_files(root_dir, file_exts=None, skip_exprs=None, file_exprs=None):
    """
    Utility for finding files that match a pattern within a directory tree,
    but skipping directories that match a different pattern.
    Skipped directories are not even searched, saving time.

    Args:
        root_dir:
            The root directory for the search

        file_exts:
            A file extension or list of extensions to search for.
            Cannot be used in conjunction with file_exprs.

        skip_exprs:
            A regular expression (or list of them) to specify which
            directories should be skipped entirely during the search.

            Note:
                The root_dir is always searched, even it if matches
                something in skip_exprs.

        file_exprs:
            A regular expression (or list of them) to specify which file names to search for.
            Cannot be used in conjunction with file_exts.

    Returns:
        list of matching file paths

    Note:
        Only files are searched for. Directories will not be returned,
        even if their names happen to match one of the given file_exprs.

    Example:
        Search for all .json files in an N5 directory hierarchy,
        but skip the block directories such as 's0', 's1', etc.
        Also, skip the 'v1' directories, just for the sake of this example.

        ..code-block:: ipython

            In [1]: root_dir = '/nrs/flyem/render/n5/Z0720_07m_BR/render/Sec32'
               ...: find_files( root_dir, '.json', ['s[0-9]+', 'v1'])
            Out[1]:
            ['/nrs/flyem/render/n5/Z0720_07m_BR/render/Sec32/attributes.json',
             '/nrs/flyem/render/n5/Z0720_07m_BR/render/Sec32/v2_acquire_trimmed_sp1_adaptive___20210315_093643/attributes.json',
             '/nrs/flyem/render/n5/Z0720_07m_BR/render/Sec32/v2_acquire_trimmed_sp1_adaptive___20210409_161756/attributes.json',
             '/nrs/flyem/render/n5/Z0720_07m_BR/render/Sec32/v2_acquire_trimmed_sp1_adaptive___20210409_162015/attributes.json',
             '/nrs/flyem/render/n5/Z0720_07m_BR/render/Sec32/v2_acquire_trimmed_sp1_adaptive___20210409_165800/attributes.json',
             '/nrs/flyem/render/n5/Z0720_07m_BR/render/Sec32/v3_acquire_trimmed_sp1_adaptive___20210419_204640/attributes.json']
    """
    assert not file_exts or not file_exprs, \
        "Please specify file extensions or whole file patterns, not both."

    file_exts = file_exts or []
    skip_exprs = skip_exprs or []
    file_exprs = file_exprs or []

    if isinstance(skip_exprs, str):
        skip_exprs = [skip_exprs]
    if isinstance(file_exts, str):
        file_exts = [file_exts]
    if isinstance(file_exprs, str):
        file_exprs = [file_exprs]

    if file_exts:
        # Strip leading '.'
        file_exts = map(lambda e: e[1:] if e.startswith('.') else e, file_exts)

        # Handle double-extensions like '.tar.gz' properly
        file_exts = map(lambda e: e.replace('.', '\\.'), file_exts)

        # Convert file extensions -> file expressions (regex)
        file_exprs = map(lambda e: f".*\\.{e}", file_exts)

    # Combine and compile expression lists
    file_expr = '|'.join(f"({e})" for e in file_exprs)
    file_rgx = re.compile(file_expr)

    skip_expr = '|'.join(f"({e})" for e in skip_exprs)
    skip_rgx = re.compile(skip_expr)

    def _find_files(parent_dir):
        logger.debug("Searching %s", parent_dir)

        try:
            # Get only the parent directory contents (not subdir contents),
            # i.e. just one iteration of os.walk()
            _, subdirs, files = next(os.walk(parent_dir))
        except StopIteration:
            return []

        files = sorted(files)
        subdirs = sorted(subdirs)

        # Matching files
        if file_expr:
            files = filter(lambda f: file_rgx.fullmatch(f), files)
        files = map(lambda f: f"{parent_dir}/{f}", files)

        # Exclude skipped directories
        if skip_expr:
            subdirs = filter(lambda d: not skip_rgx.fullmatch(d), subdirs)
        subdirs = map(lambda d: f"{parent_dir}/{d}", subdirs)

        # Recurse
        subdir_filesets = map(_find_files, subdirs)

        # Concatenate
        return chain(files, *subdir_filesets)

    return list(_find_files(root_dir))


def convert_nested_ordered_dict(x):
    """
    Perform a deep copy of the given object, but convert
    all internal OrderedDicts to plain dicts along the way.

    Args:
        x: Any pickleable object

    Returns:
        A copy of the input, in which all OrderedDicts contained
        anywhere in the input (as iterable items or attributes, etc.)
        have been converted to plain dicts.
    """
    # Temporarily install a custom pickling function
    # (used by deepcopy) to convert OrderedDict to dict.
    orig_pickler = copyreg.dispatch_table.get(OrderedDict, None)
    copyreg.pickle(
        OrderedDict,
        lambda d: (dict, ([*d.items()],))
    )
    try:
        return copy.deepcopy(x)
    finally:
        # Restore the original OrderedDict pickling function (if any)
        del copyreg.dispatch_table[OrderedDict]
        if orig_pickler:
            copyreg.dispatch_table[OrderedDict] = orig_pickler
