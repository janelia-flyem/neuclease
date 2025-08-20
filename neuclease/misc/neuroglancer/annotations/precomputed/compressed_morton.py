import numpy as np
from numba import njit, guvectorize


def compressed_morton_code(grid_coord_c_order, grid_shape_c_order):
    """
    Return the compressed morton code[1] for a grid coordinate
    that resides in a grid with the given shape.

    [1]: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md#compressed-morton-code

    This implementation uses numba.guvectorize().
    Multiple coordinates and/or grid shapes can be provided simultaneously,
    and they will be broadcasted together to produce an output of the appropriate shape.

    Important:
    
        This function expects the grid coordinate to be in C order,
        i.e. the fastest-changing stride is listed last.
        For typical 3D arrays, C-order indexing is [z, y, x],
        so pass your grid coordinates here in [z, y, x] order, too.

    Examples:

        In [10]: compressed_morton_code(
            ...:     [1, 5, 3],
            ...:     [3, 16, 5]
            ...: )
        Out[10]: array(143, dtype=uint64)

        In [11]: compressed_morton_code(
            ...:     [[1, 5, 3], [2, 3, 4]],
            ...:     [3, 16, 5]
            ...: )
        Out[11]: array([143, 114], dtype=uint64)

        In [12]: compressed_morton_code(
            ...:     [1, 5, 3],
            ...:     [[3, 16, 5], [6, 32, 10]]
            ...: )
        Out[12]: array([143, 143], dtype=uint64)

        In [13]: compressed_morton_code(
            ...:     [[1, 5, 3], [2, 3, 4]],
            ...:     [[3, 16, 5], [6, 32, 10]]
            ...: )
        Out[13]: array([143, 114], dtype=uint64)
    """
    grid_coord = np.asarray(grid_coord_c_order, np.uint64)
    grid_shape = np.asarray(grid_shape_c_order, np.uint64)
    if (grid_coord >= grid_shape).any():
        raise ValueError(f'grid_coord contains out-of-bounds values relative to grid_shape')

    axis_bits = np.ceil(np.log2(grid_shape)).astype(np.int8)
    output_shape = np.broadcast_shapes(grid_shape.shape, grid_coord.shape)[:-1]
    output_code = np.zeros(output_shape, dtype=np.uint64)
    __compressed_morton_code(grid_coord, axis_bits, output_code)
    return output_code


@guvectorize('(d),(d)->()', nopython=True)
def __compressed_morton_code(grid_coord, axis_bits, result):
    D = len(axis_bits)
    curr_axis_pos = np.zeros(D, dtype=np.uint64)
    curr_axis = 0
    output_pos = np.uint64(0)
    output_code = np.uint64(0)

    for _ in range(D * 64):
        curr_axis = (curr_axis - 1) % D
        if curr_axis_pos[curr_axis] >= axis_bits[curr_axis]:
            continue

        input_pos = curr_axis_pos[curr_axis]
        output_code |= ((grid_coord[curr_axis] >> input_pos) & np.uint64(1)) << output_pos
        output_pos += np.uint64(1)
        curr_axis_pos[curr_axis] += 1

    # With guvectorize, we treat a scalar output as if it had a single element.
    # https://numba.readthedocs.io/en/stable/user/vectorize.html#scalar-return-values
    result[0] = output_code


@njit
def compressed_morton_code_no_broadcast(grid_coord_c_order, grid_shape_c_order):
    """
    Same as compressed_morton_code(), but for when both grid_coord
    and grid_shape are known to have ndim 1 or 2 and both arguments
    have dtype np.uint64, which allows us to easily predict the output
    shape without needing to call np.broadcast_shapes().

    Therefore this whole function can be wrapped with @njit,
    and can be called from within other jit-compiled functions.
    """
    grid_coord = grid_coord_c_order
    grid_shape = grid_shape_c_order

    axis_bits = np.ceil(np.log2(grid_shape)).astype(np.int8)
    if grid_coord.ndim == 1:
        output_code = np.zeros(1, dtype=np.uint64)
        __compressed_morton_code(grid_coord, axis_bits, output_code)
        return output_code[0]
    else:
        output_code = np.zeros(len(grid_coord), dtype=np.uint64)
        __compressed_morton_code(grid_coord, axis_bits, output_code)
        return output_code


def compressed_morton_decode(morton_code, grid_shape_c_order):
    morton_code = np.asarray(morton_code, np.uint64)
    grid_shape = np.asarray(grid_shape_c_order, np.uint64)

    D = grid_shape.shape[-1]
    axis_bits = np.ceil(np.log2(grid_shape)).astype(np.int8)
    output_shape = np.broadcast_shapes(morton_code.shape, grid_shape.shape[:-1]) + (D,)
    output_grid_coord = np.zeros(output_shape, dtype=np.uint64)
    __compressed_morton_decode(morton_code, axis_bits, output_grid_coord)
    return output_grid_coord


@guvectorize('(),(d)->(d)', nopython=True)
def __compressed_morton_decode(morton_code, axis_bits, grid_coord_c_order):
    D = len(axis_bits)
    curr_axis_pos = np.zeros(D, dtype=np.uint64)
    curr_axis = 0
    
    input_pos = np.uint64(0)
    grid_coord_c_order[:] = np.uint64(0)

    for _ in range(D * 64):
        curr_axis = (curr_axis - 1) % D
        if curr_axis_pos[curr_axis] >= axis_bits[curr_axis]:
            continue

        output_pos = curr_axis_pos[curr_axis]
        grid_coord_c_order[curr_axis] |= ((morton_code >> input_pos) & np.uint64(1)) << output_pos
        input_pos += np.uint64(1)
        curr_axis_pos[curr_axis] += 1
