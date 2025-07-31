import logging
from itertools import chain
from typing import NamedTuple

import numpy as np
from numba import njit, guvectorize
from numba.typed import List

from ._write_buffers import _write_buffers
from ._util import _encode_uint64_series, _geometry_cols, _ndindex_array

logger = logging.getLogger(__name__)

GridSpec = NamedTuple("GridSpec", [('chunk_shapes', np.ndarray), ('grid_shapes', np.ndarray)])


def _write_annotations_by_spatial_chunk(df, coord_space, annotation_type, bounds, num_levels, target_chunk_limit, output_dir, write_sharded):
    """
    Write the annotations to the spatial index.
    Currently, we only support a single spatial grid level,
    resulting in a single annotation list.

    Returns:
        JSON metadata to write into the 'spatial' key of the info file.
    """
    df, gridspec = _assign_spatial_chunks(
        df,
        coord_space,
        annotation_type,
        bounds,
        num_levels,
        target_chunk_limit
    )
    metadata = __write_assigned_annotations_by_spatial_chunk(
        df,
        gridspec,
        output_dir,
        write_sharded
    )
    return metadata


def _assign_spatial_chunks(df, coord_space, annotation_type, bounds, num_levels, target_chunk_limit: int):
    """
    Assign each annotation to a spatial grid cell.
    If an annotation intersects multiple grid cells, we duplicate
    its row so we can assign it to all of the intersecting cells.
    """
    geometry_cols = _geometry_cols(coord_space.names, annotation_type)
    df = df[[*chain(*geometry_cols), 'ann_buf', 'id_buf']].copy(deep=False)
    gridspec = _define_spatial_grids(bounds, coord_space, num_levels)
    level_annotation_counts = _compute_target_annotations_per_level(len(df), gridspec, target_chunk_limit)

    logger.info("Shuffling annotations before assigning spatial grid levels")
    df = df.sample(frac=1.0)

    logger.info("Assigning spatial grid chunks")
    df['level'] = np.repeat(range(num_levels), level_annotation_counts.astype(int)).astype(np.uint64)
    if annotation_type == 'point':
        df = _assign_point_spatial_chunks(df, coord_space.names, bounds, gridspec)
    elif annotation_type == 'axis_aligned_bounding_box':
        df = _assign_axis_aligned_bounding_box_spatial_chunks(df, geometry_cols, bounds, gridspec)
    else:
        raise NotImplementedError(f"Spatial indexing for {annotation_type} annotations is not implemented")

    logger.info("Done assigning spatial grid chunks")
    return df, gridspec


def _assign_point_spatial_chunks(df, coord_names, bounds, gridspec):
    grid_indices = (df[[*coord_names]] - bounds[0]) // gridspec.chunk_shapes[df['level']]
    grid_indices = grid_indices.astype(int)

    # Make sure annotations at the exact upper bound get valid grid coordinates.
    grid_indices = np.minimum(grid_indices, gridspec.grid_shapes[df['level']] - 1)

    df['chunk_code'] = compressed_morton_code(grid_indices, gridspec.grid_shapes[df['level']])
    return df


def _assign_axis_aligned_bounding_box_spatial_chunks(df, geometry_cols, bounds, gridspec):
    boxes = df[[*geometry_cols[0], *geometry_cols[1]]].to_numpy().reshape(len(df), 2, -1)

    # Ensure start < end
    swap_mask = (boxes[:, 0, :] > boxes[:, 1, :])[:, None, :]
    swap_mask = np.concatenate([swap_mask, swap_mask], axis=1)
    boxes[swap_mask] = boxes[:, ::-1, :][swap_mask]

    # Offset to start at lower bound
    boxes = boxes - bounds[0]

    grid_boxes = np.zeros_like(boxes, np.uint64)
    grid_boxes[:, 0] = np.floor(boxes[:, 0] / gridspec.chunk_shapes[df['level']]).astype(np.uint64)
    grid_boxes[:, 1] = np.ceil(boxes[:, 1] / gridspec.chunk_shapes[df['level']]).astype(np.uint64)
    df[f'chunk_code'] = grid_box_codes(grid_boxes, gridspec.grid_shapes[df['level']])

    # Duplicate the annotations which span multiple chunks.
    df = df.explode('chunk_code')
    return df


@njit
def grid_box_codes(grid_boxes, grid_shapes):
    lists = List()
    for grid_box, grid_shape in zip(grid_boxes, grid_shapes):
        grid_indices = _ndindex_array(grid_box[1] - grid_box[0]) + grid_box[0]
        codes = _compressed_morton_code_pairwise(grid_indices, grid_shape)
        lists.append(List(codes))
    return lists


def _define_spatial_grids(bounds, coord_space, num_levels: int) -> GridSpec:
    """
    Compute suitable chunk shapes and grid shapes for each level 
    of the spatial index, following the guidelines from the spec[1]:

        Typically the grid_shape for level 0 should be a vector of all 1
        (with chunk_size equal to upper_bound - lower_bound), and each component
        of chunk_size of each successively level should be either equal to, or half of,
        the corresponding component of the prior level chunk_size, whichever results
        in a more spatially isotropic chunk.

    [1]: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#spatial-index

    Args:
        bounds:
            np.ndarray, shape (2, D)
            lower and upper bounds of the union of all annotations
        num_levels:
            The number of spatial index levels. Must be at least 1.

    Returns:
        chunk_shapes, grid_shapes
    """
    # Level 0 chunk shape and grid shape -- just one chunk.
    bounds = np.asarray(bounds, np.float64)

    # We want roughly isotropic chunks in physical units, so we'll multiply
    # by the coordinate scales and then divide the scales out at the end.
    chunk_shape = (bounds[1] - bounds[0]) * coord_space.scales
    grid_shape = np.ones_like(chunk_shape, dtype=np.uint64)

    chunk_shapes = [chunk_shape]
    grid_shapes = [grid_shape]

    for level in range(1, num_levels):
        chunk_shape = chunk_shape.copy()
        grid_shape = grid_shape.copy()

        max_dim = np.argmax(chunk_shape)
        target_width = chunk_shape[max_dim] / 2

        for dim, dim_width in enumerate(chunk_shape):
            if dim == max_dim:
                # Always split across the widest dimension.
                chunk_shape[dim] = target_width
                grid_shape[dim] *= 2
            elif (dim_width / target_width) > 1.5:
                # Split across this dimension to make it more isotropic.
                chunk_shape[dim] = dim_width / 2
                grid_shape[dim] *= 2
            else:
                # Splitting would make it less isotropic,
                # so leave this dimension unsplit.
                chunk_shape[dim] = dim_width
                grid_shape[dim] *= 1

        chunk_shapes.append(chunk_shape)
        grid_shapes.append(grid_shape)

    return GridSpec(
        # Convert from physical units back to coordinate units.
        chunk_shapes=np.array(chunk_shapes, dtype=np.float64) / coord_space.scales,
        grid_shapes=np.array(grid_shapes, dtype=np.uint64)
    )


def _compute_target_annotations_per_level(num_annotations, gridspec, target_chunk_limit: int):
    """
    Compute the TOTAL number of annotations at each level of the spatial index.
    The target_chunk_limit is how many annotations we aim to place in each chunk
    (regardless of the level).
    
    Since the spatial annotations are not distributed uniformly in space,
    we will likely end up undershooting and overshooting the target for various
    chunks within a level.

    Furthermore, since the number of annotations passed in here is based on the
    table BEFORE duplicating annotations which span multiple chunks, the number
    of annotations at each level will eventually be more than what is returned here,
    after the appropriate duplications.

    Returns:
        np.ndarray, shape (num_levels,)
    """
    num_levels = len(gridspec.grid_shapes)
    chunk_counts_by_level = np.prod(gridspec.grid_shapes, axis=1)
    annotation_counts = chunk_counts_by_level * target_chunk_limit
    
    # Clamp to total number of annotations remaining after earlier levels
    for level in range(num_levels - 1):
        annotation_counts[level] = min(
            annotation_counts[level],
            num_annotations - sum(annotation_counts[:level])
        )

    # Last level gets all remaining annotations, if any.
    annotation_counts[-1] = num_annotations - sum(annotation_counts[:-1])

    return annotation_counts


def __write_assigned_annotations_by_spatial_chunk(df, gridspec, output_dir, write_sharded):
    """
    Write the spatial index, given a dataframe in which the 'level'
    and grid chunk codes for each annotation have already been assigned.

    Returns:
        JSON metadata to write into the 'spatial' key of the info file.
    """
    bufs_by_grid = (
        df[['level', 'chunk_code', 'id_buf', 'ann_buf']]
        .groupby(['level', 'chunk_code'], sort=False)
        .agg({'id_buf': ['count', b''.join], 'ann_buf': b''.join})
    )
    logger.info(f"Combining annotation and ID buffers for spatial index")
    bufs_by_grid.columns = ['count', 'id_buf', 'ann_buf']
    bufs_by_grid['count_buf'] = _encode_uint64_series(bufs_by_grid['count'])
    bufs_by_grid['combined_buf'] = bufs_by_grid[['count_buf', 'ann_buf', 'id_buf']].sum(axis=1)

    bufs_by_grid = bufs_by_grid.reset_index()

    metadata = []
    for level, level_bufs in bufs_by_grid.groupby('level'):
        logger.info(f"Writing annotations to 'by_spatial_level_{level}' index")

        if write_sharded:
            # Sharded key is the compressed morton code of the grid coordinate.
            level_bufs.index = level_bufs['chunk_code']
        else:
            # Unsharded key is string of the grid coordinate, e.g. '0_0_0'
            grid_coords = reverse_morton_code(level_bufs['chunk_code'].to_numpy(), gridspec.grid_shapes[level])
            level_bufs.index = [map('_'.join, grid_coords.astype(str))]

        level_metadata = _write_buffers(
            level_bufs['combined_buf'],
            output_dir,
            f"by_spatial_level_{level}",
            write_sharded
        )
        level_metadata['chunk_size'] = gridspec.chunk_shapes[level].tolist()
        level_metadata['grid_shape'] = gridspec.grid_shapes[level].tolist()
        level_metadata['limit'] = int(level_bufs['count'].max())
        metadata.append(level_metadata)

    return metadata


def compressed_morton_code(grid_coord, grid_shape):
    """
    Return the compressed morton code[1] for a grid coordinate
    that resides in a grid with the given shape.

    [1]: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md#compressed-morton-code

    This implementation uses numba.guvectorize().
    Multiple coordinates and/or grid shapes can be provided simultaneously,
    and they will be broadcasted together to produce an output of the appropriate shape.

    Examples:

        In [10]: compressed_morton_code(
            ...:     [1,5,3],
            ...:     [3, 16, 5]
            ...: )
        Out[10]: array(143, dtype=uint64)

        In [11]: compressed_morton_code(
            ...:     [[1,5,3], [2, 3, 4]],
            ...:     [3, 16, 5]
            ...: )
        Out[11]: array([143, 114], dtype=uint64)

        In [12]: compressed_morton_code(
            ...:     [1,5,3],
            ...:     [[3, 16, 5], [6, 32, 10]]
            ...: )
        Out[12]: array([143, 143], dtype=uint64)

        In [13]: compressed_morton_code(
            ...:     [[1,5,3], [2, 3, 4]],
            ...:     [[3, 16, 5], [6, 32, 10]]
            ...: )
        Out[13]: array([143, 114], dtype=uint64)
    """
    grid_coord = np.asarray(grid_coord, np.uint64)
    grid_shape = np.asarray(grid_shape, np.uint64)
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
def _compressed_morton_code_pairwise(grid_coord, grid_shape):
    """
    Same as compressed_morton_code(), but for when both grid_coord
    is known to have shape (N,D) and both arguments have dtype np.uint64,
    and therefore this whole function can be wrapped with @njit.
    This function can be called from within other jit-compiled functions.
    """
    axis_bits = np.ceil(np.log2(grid_shape)).astype(np.int8)
    output_code = np.zeros(len(grid_coord), dtype=np.uint64)
    __compressed_morton_code(grid_coord, axis_bits, output_code)
    return output_code


def reverse_morton_code(morton_code, grid_shape):
    morton_code = np.asarray(morton_code, np.uint64)
    grid_shape = np.asarray(grid_shape, np.uint64)

    D = grid_shape.shape[-1]
    axis_bits = np.ceil(np.log2(grid_shape)).astype(np.int8)
    output_shape = np.broadcast_shapes(morton_code.shape, grid_shape.shape[:-1]) + (D,)
    output_grid_coord = np.zeros(output_shape, dtype=np.uint64)
    __reverse_morton_code(morton_code, axis_bits, output_grid_coord)
    return output_grid_coord


@guvectorize('(),(d)->(d)', nopython=True)
def __reverse_morton_code(morton_code, axis_bits, grid_coord):
    D = len(axis_bits)
    curr_axis_pos = np.zeros(D, dtype=np.uint64)
    curr_axis = 0
    
    input_pos = np.uint64(0)
    grid_coord[:] = np.uint64(0)

    for _ in range(D * 64):
        curr_axis = (curr_axis - 1) % D
        if curr_axis_pos[curr_axis] >= axis_bits[curr_axis]:
            continue

        output_pos = curr_axis_pos[curr_axis]
        grid_coord[curr_axis] |= ((morton_code >> input_pos) & np.uint64(1)) << output_pos
        input_pos += np.uint64(1)
        curr_axis_pos[curr_axis] += 1
