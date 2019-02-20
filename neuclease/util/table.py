"""
Various utility functions for dealing with 2D numpy arrays (and 1D numpy structured arrays).
"""
import numpy as np
from numba import jit


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
        table[:] = table[order]
    
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
    Lexsort the given 2D table of the given array.
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


@jit(nopython=True, nogil=True)
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


@jit(nopython=True, nogil=True)
def groupby_spans_presorted(sorted_cols):
    """
    Similar to groupby_presorted(), but yields only the (start, stop)
    indexes of the contiguous groups, (not the group subarrays themselves).
    """
    assert sorted_cols.ndim >= 2
    if len(sorted_cols) == 0:
        return

    start = 0
    row = sorted_cols[0]
    for stop in range(len(sorted_cols)):
        next_row = sorted_cols[stop]
        if (next_row != row).any():
            yield (start, stop)
            start = stop
            row = next_row

    # Last group
    yield (start, len(sorted_cols))


def group_sums_presorted(a, sorted_cols):
    """
    Similar to a.groupby(sorted_cols).sum()

    Args:
        a: Columns to aggregate

        sorted_cols: Columns to group by
    
    Returns:
        (group_ids, group_sums)
    """
    assert a.ndim >= 2
    assert sorted_cols.ndim >= 2
    assert a.shape[0] == sorted_cols.shape[0]

    # Two passes: first to get len
    @jit(nopython=True, nogil=True)
    def count_groups():
        num_groups = 0
        for _ in groupby_presorted(a, sorted_cols):
            num_groups += 1
        return num_groups

    num_groups = count_groups()
    print(f"Aggregating {num_groups} groups")
    
    groups_shape = (num_groups,) + sorted_cols.shape[1:]
    groups = np.zeros(groups_shape, dtype=sorted_cols.dtype)

    results_shape = (num_groups,) + a.shape[1:]
    agg_results = np.zeros(results_shape, dtype=a.dtype)
    
    @jit(nopython=True, nogil=True)
    def _agg(a, sorted_cols, groups, agg_results):
        pos = 0
        for i, group_rows in enumerate(groupby_presorted(a, sorted_cols)):
            groups[i] = sorted_cols[pos]
            pos += len(group_rows)
            agg_results[i] = group_rows.sum(0) # axis 0
        return (groups, agg_results)

    return _agg(a, sorted_cols, groups, agg_results)

