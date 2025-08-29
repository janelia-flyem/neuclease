import logging
from itertools import chain
from typing import NamedTuple

import numpy as np
from numba import njit
from numba.typed import List

from .compressed_morton import compressed_morton_code, compressed_morton_decode, compressed_morton_code_no_broadcast
from ._write_buffers import _write_buffers
from ._util import _encode_uint64_series, _geometry_cols, _ndindex_array, TableHandle

logger = logging.getLogger(__name__)

GridSpec = NamedTuple("GridSpec", [('chunk_shapes', np.ndarray), ('grid_shapes', np.ndarray)])


def _write_annotations_by_spatial_chunk(
        df_handle: TableHandle,
        coord_space,
        annotation_type,
        bounds,
        num_levels,
        target_chunk_limit,
        shuffle_before_assigning_spatial_levels,
        output_dir,
        write_sharded
):
    """
    Write the annotations to the spatial index.

    Args:
        df_handle:
            TableHandle.  The handle's reference will be unset before this function returns.
            The enclosed DataFrame must have columns ['id_buf', 'ann_buf', *geometry_cols].
            Internally, the data will be copied during processing and again
            during writing, incurring significant RAM usage for large datasets.
            The caller can save some RAM by deleting their own reference to the input
            after constructing the TableHandle (but before calling this function).

        coord_space:
            CoordinateSpace.
            The coordinate space of the annotations.

        annotation_type:
            Literal['point', 'axis_aligned_bounding_box', 'ellipsoid', 'line']
            The type of annotation to export.

        bounds:
            np.ndarray, shape (2, D)
            Lower and upper bounds of the union of all annotations.
            The bounds are in coordinate units.

        num_levels:
            The number of spatial index levels. Must be at least 1.

        target_chunk_limit:
            The maximum number of annotations to place in each chunk.
            (The same target is used for all levels.)
            Since the spatial annotations are not distributed uniformly in space,
            we will likely end up undershooting and overshooting the target for various
            chunks within a level.
            The final maximum number of annotations per chunk we end up with at each
            level will be emitted in the the 'limit' setting of the metadata for each level.

            Note:
                Instead of specifying a valid limit here, you can disable subsampling in neuroglancer
                by setting this to the special value of 0.  This is only valid when num_levels=1.

        shuffle_before_assigning_spatial_levels:
            Whether to shuffle the annotations before assigning spatial levels.
            If False, the annotations will be assigned to spatial levels in the order
            they appear in the input dataframe, with earlier annotations assigned to
            coarser spatial levels.
            By default, we shuffle the annotations to avoid any bias in the spatial
            assignment, which is what the neuroglancer spec recommends.

        output_dir:
            Directory to write the annotations to.
            Subdirectories for each level of the spatial index will be created in output_dir,
            named 'by_spatial_level_<level>'.

        write_sharded:
            Whether to write the annotations in sharded format.

    Returns:
        JSON metadata to write into the 'spatial' key of the info file.
    """
    df_handle, gridspec = _assign_spatial_chunks(
        df_handle,
        coord_space,
        annotation_type,
        bounds,
        num_levels,
        target_chunk_limit,
        shuffle_before_assigning_spatial_levels
    )

    metadata = _write_assigned_annotations_by_spatial_chunk(
        df_handle,
        gridspec,
        (target_chunk_limit == 0),
        output_dir,
        write_sharded
    )
    return metadata


def _assign_spatial_chunks(
    df_handle: TableHandle,
    coord_space,
    annotation_type,
    bounds,
    num_levels,
    target_chunk_limit,
    shuffle_before_assigning_spatial_levels
):
    """
    Assign each annotation to a spatial grid cell.
    If an annotation intersects multiple grid cells, we duplicate
    its row so we can assign it to all of the intersecting cells.

    Args:
        df_handle:
            TableHandle.  The handle's reference will be unset before this function returns.
        ...

    Returns:
        df, gridspec

        - df is a shuffled copy of the input df with all columns removed except
          'ann_buf' and 'id_buf', and with additional columns for 'level' and 'chunk_code'.
          Some rows from the original dataframe may be duplicated if those annotations
          span across multiple chunks (at the level we selected them to reside in).
        - gridspec: chunk_shapes and grid_shapes.  See _define_spatial_grids() for details.
    """
    df = df_handle.df
    df_handle.df = None

    geometry_cols = _geometry_cols(coord_space.names, annotation_type)
    df = df[[*chain(*geometry_cols), 'ann_buf', 'id_buf']].copy(deep=False)
    gridspec = _define_spatial_grids(bounds, coord_space, num_levels)
    level_annotation_counts = _compute_target_annotations_per_level(len(df), gridspec, target_chunk_limit)

    if shuffle_before_assigning_spatial_levels:
        logger.info("Shuffling annotations before assigning spatial grid levels")
        df = df.sample(frac=1.0)

    logger.info("Assigning spatial grid chunks")
    df['level'] = np.repeat(
        range(num_levels),
        level_annotation_counts.astype(int)
    ).astype(np.uint64)
    
    match annotation_type:
        case 'point':
            df = _assign_spatial_chunks_for_points(df, geometry_cols, bounds, gridspec)
        case 'axis_aligned_bounding_box':
            df = _assign_spatial_chunks_for_axis_aligned_bounding_boxes(df, geometry_cols, bounds, gridspec)
        case 'ellipsoid':
            df = _assign_spatial_chunks_for_ellipsoids(df, geometry_cols, bounds, gridspec)
        case 'line':
            df = _assign_spatial_chunks_for_lines(df, geometry_cols, bounds, gridspec)
        case _:
            raise NotImplementedError(f"Spatial indexing for {annotation_type} annotations is not implemented")

    logger.info("Done assigning spatial grid chunks")
    return TableHandle(df), gridspec


def _define_spatial_grids(bounds, coord_space, num_levels: int) -> GridSpec:
    """
    Compute suitable chunk shapes and grid shapes for each level 
    of the spatial index, following the guidelines from the spec[1]:

        > Typically the grid_shape for level 0 should be a vector of all 1
        > (with chunk_size equal to upper_bound - lower_bound), and each component
        > of chunk_size of each successively level should be either equal to, or half of,
        > the corresponding component of the prior level chunk_size, whichever results
        > in a more spatially isotropic chunk.

    [1]: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#spatial-index

    Args:
        bounds:
            np.ndarray, shape (2, D)
            lower and upper bounds of the union of all annotations

        coord_space:
            Needed to aim for roughly isotropic chunks in physical units.

        num_levels:
            The number of spatial index levels. Must be at least 1.

    Returns:
        GridSpec(chunk_shapes, grid_shapes)

        - chunk_shapes is the array (for N levels) of the size of each
          grid cell at the corresponding level, in coordinate units.
        - grid_shapes is the array (for N levels) of the number of grid cells
          along each dimension at the corresponding level.

        For instance, level 0 consists of a single chunk encompassing the entire
        volume occupied by the annotations, so its chunk_shape is the entire bounds
        (offset by the lower bound) and its grid_shape is [1,1,...].
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

    # Convert from physical units back to coordinate units.
    chunk_shapes = np.array(chunk_shapes, dtype=np.float64) / coord_space.scales
    grid_shapes = np.array(grid_shapes, dtype=np.uint64)

    return GridSpec(chunk_shapes, grid_shapes)


def _compute_target_annotations_per_level(num_annotations, gridspec, target_chunk_limit: int):
    """
    Compute the TOTAL number of annotations at each level of the spatial index.
    The target_chunk_limit is how many annotations we aim to place in each chunk
    (regardless of the level).
    
    Since the spatial annotations are not necessarily distributed uniformly in space,
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

    if target_chunk_limit != 0:
        annotation_counts = chunk_counts_by_level * target_chunk_limit
    else:
        assert num_levels == 1, \
            "The special target_chunk_limit of 0 is only permitted when num_spatial_levels=1"
        assert chunk_counts_by_level.tolist() == [1]
        annotation_counts = np.array([num_annotations])
    
    # Clamp to total number of annotations remaining after earlier levels
    for level in range(num_levels - 1):
        annotation_counts[level] = min(
            annotation_counts[level],
            num_annotations - sum(annotation_counts[:level])
        )

    # Last level gets all remaining annotations, if any.
    annotation_counts[-1] = num_annotations - sum(annotation_counts[:-1])

    return annotation_counts


def _assign_spatial_chunks_for_points(df, geometry_cols, bounds, gridspec):
    coord_names = geometry_cols[0]
    grid_indices = (df[[*coord_names]] - bounds[0]) // gridspec.chunk_shapes[df['level']]
    grid_indices = grid_indices.astype(np.int32)

    # Make sure annotations at the exact upper bound get valid grid coordinates.
    grid_indices = np.minimum(grid_indices, gridspec.grid_shapes[df['level']].astype(np.int32) - 1)

    # Switch to C order before computing compressed morton code.
    df['chunk_code'] = compressed_morton_code(
        grid_indices.to_numpy()[:, ::-1],
        gridspec.grid_shapes[df['level']][:, ::-1]
    )
    
    # FIXME: remove this debug code or put it somewhere else.
    # import pyarrow.feather as feather
    # df[[f'grid_{c}' for c in coord_names]] = grid_indices
    # df[[f'{c}a' for c in coord_names]] =  grid_indices * gridspec.chunk_shapes[df['level']] + bounds[0]
    # df[[f'{c}b' for c in coord_names]] = (grid_indices + 1) * gridspec.chunk_shapes[df['level']] + bounds[0]
    # feather.write_feather(df.drop(columns=['id_buf', 'ann_buf']), '/tmp/points.feather')
    
    return df[['level', 'chunk_code', 'id_buf', 'ann_buf']]


def _assign_spatial_chunks_for_axis_aligned_bounding_boxes(df, geometry_cols, bounds, gridspec):
    boxes = df[[*geometry_cols[0], *geometry_cols[1]]].to_numpy().reshape(len(df), 2, -1)

    # Ensure start < end
    swap_mask = (boxes[:, 0, :] > boxes[:, 1, :])[:, None, :]
    swap_mask = np.concatenate([swap_mask, swap_mask], axis=1)
    boxes[swap_mask] = boxes[:, ::-1, :][swap_mask]

    chunk_shapes = gridspec.chunk_shapes[df['level']]

    # FIXME: would be faster to compute the spans in _grid_box_codes()
    grid_spans = np.zeros_like(boxes, np.uint64)
    grid_spans[:, 0] = np.floor((boxes[:, 0] - bounds[0]) / chunk_shapes).astype(np.uint64)
    grid_spans[:, 1] = np.ceil((boxes[:, 1] - bounds[0]) / chunk_shapes).astype(np.uint64)

    df[f'chunk_code'] = _box_grid_codes(grid_spans, gridspec.grid_shapes[df['level']])

    # Duplicate the annotations which span multiple chunks.
    df = df[['level', 'chunk_code', 'id_buf', 'ann_buf']].explode('chunk_code')
    return df


@njit
def _box_grid_codes(grid_spans, grid_shapes):
    lists = List()
    for grid_span, grid_shape in zip(grid_spans, grid_shapes):
        grid_indices = grid_span[0] + _ndindex_array(grid_span[1] - grid_span[0])

        # Switch to C order before computing compressed morton code.
        codes = compressed_morton_code_no_broadcast(grid_indices[:, ::-1], grid_shape[::-1])
        lists.append(List(codes))
    return lists


def _assign_spatial_chunks_for_ellipsoids(df, geometry_cols, bounds, gridspec):
    centroids = df[geometry_cols[0]].to_numpy()
    radii = df[geometry_cols[1]].to_numpy()

    boxes = np.concatenate([centroids - radii, centroids + radii], axis=1).reshape(len(df), 2, -1)
    boxes = boxes - bounds[0]

    df[f'chunk_code'] = _ellipsoid_grid_codes(
        centroids,
        radii,
        df['level'].to_numpy(),
        bounds[0],
        gridspec.grid_shapes,
        gridspec.chunk_shapes
    )

    # Duplicate the annotations which span multiple chunks.
    df = df[['level', 'chunk_code', 'id_buf', 'ann_buf']].explode('chunk_code')
    return df


@njit
def _ellipsoid_grid_codes(centroids, radii, levels, grid_origin, grid_shapes, chunk_shapes):
    D = len(grid_shapes[0])
    lists = List()
    for centroid, radius, level in zip(centroids, radii, levels):
        grid_shape = grid_shapes[level]
        chunk_shape = chunk_shapes[level]

        grid_span = np.zeros((2, D), np.uint64)
        grid_span[0] = np.floor((centroid - radius - grid_origin) / chunk_shape)
        grid_span[1] = np.ceil((centroid + radius - grid_origin) / chunk_shape)

        grid_indices = grid_span[0] + _ndindex_array(grid_span[1] - grid_span[0])
        codes = List()
        for grid_index in grid_indices:
            if _ellipsoid_chunk_overlap(centroid, radius, grid_origin, chunk_shape, grid_index):
                # Switch to C order before computing compressed morton code.
                code = compressed_morton_code_no_broadcast(grid_index[::-1], grid_shape[::-1])
                codes.append(code)
        lists.append(codes)
    return lists


@njit
def _ellipsoid_chunk_overlap(center, radii, grid_origin, cell_shape, grid_index):
    """
    Ported from the C++ implementation[1] by jbms, except that we just return
    a boolean indicating whether the ellipsoid and cell have any overlap (True)
    or are completely disjoint (False).

    [1]: https://github.com/google/neuroglancer/pull/522#issuecomment-1940516294
    """
    rank = len(center)
    min_sum = 0.0

    for i in range(rank):
        cell_size = cell_shape[i]
        cell_start = grid_index[i] * cell_size + grid_origin[i]
        cell_end = cell_start + cell_size
        center_pos = center[i]
        
        start_dist = abs(cell_start - center_pos)
        end_dist = abs(cell_end - center_pos)
        
        if center_pos >= cell_start and center_pos <= cell_end:
            min_distance = 0.0
        else:
            min_distance = min(start_dist, end_dist)
        
        min_sum += min_distance**2 / radii[i]**2
    
    return min_sum <= 1.0


def _assign_spatial_chunks_for_lines(df, geometry_cols, bounds, gridspec):
    endpoints = df[[*geometry_cols[0], *geometry_cols[1]]].to_numpy().reshape(len(df), 2, -1)

    # Ensure start < end
    swap_mask = (endpoints[:, 0, :] > endpoints[:, 1, :])[:, None, :]
    swap_mask = np.concatenate([swap_mask, swap_mask], axis=1)
    endpoints[swap_mask] = endpoints[:, ::-1, :][swap_mask]

    df[f'chunk_code'] = _line_grid_codes(
        endpoints,
        df['level'].to_numpy(),
        bounds[0],
        gridspec.grid_shapes,
        gridspec.chunk_shapes
    )

    # Duplicate the annotations which span multiple chunks.
    df = df[['level', 'chunk_code', 'id_buf', 'ann_buf']].explode('chunk_code')
    return df


@njit
def _line_grid_codes(endpoints, levels, grid_origin, grid_shapes, chunk_shapes):
    D = len(grid_shapes[0])
    lists = List()
    for (point_a, point_b), level in zip(endpoints, levels):
        grid_shape = grid_shapes[level]
        chunk_shape = chunk_shapes[level]

        grid_span = np.zeros((2, D), np.uint64)
        grid_span[0] = np.floor((point_a - grid_origin) / chunk_shape)
        grid_span[1] = np.ceil((point_b - grid_origin) / chunk_shape)

        codes = List()
        grid_indices = grid_span[0] + _ndindex_array(grid_span[1] - grid_span[0])
        for grid_index in grid_indices:
            if _line_chunk_overlap(point_a, point_b, grid_origin, chunk_shape, grid_index):
                # Switch to C order before computing compressed morton code.
                code = compressed_morton_code_no_broadcast(grid_index[::-1], grid_shape[::-1])
                codes.append(code)
        lists.append(codes)
    return lists


@njit
def _line_chunk_overlap(point_a, point_b, grid_origin, cell_shape, grid_index):
    """
    Ported from the C++ implementation[1] by jbms.
    Returns True if the line intersects the cell, False otherwise.

    [1]: https://github.com/google/neuroglancer/pull/522#issuecomment-1940516294
    """
    rank = len(point_a)
    min_t = 0.0
    max_t = 1.0
    
    for i in range(rank):
        a = point_a[i]
        b = point_b[i]
        line_lower = min(a, b)
        line_upper = max(a, b)
        box_lower = grid_origin[i] + cell_shape[i] * grid_index[i]
        box_upper = box_lower + cell_shape[i]
        
        line_range = line_upper - line_lower
        
        if box_lower > line_lower:
            if line_range == 0.0:
                # Line is a point, check if it's outside the box
                if line_lower < box_lower:
                    return False
            else:
                t = (box_lower - line_lower) / line_range
                if t > 1:
                    return False
                min_t = max(min_t, t)
        
        if box_upper < line_upper:
            if line_range == 0.0:
                # Line is a point, check if it's outside the box
                if line_lower > box_upper:
                    return False
            else:
                t = (box_upper - line_lower) / line_range
                if t < 0:
                    return False
                max_t = min(max_t, t)
    
    return max_t >= min_t


def _write_assigned_annotations_by_spatial_chunk(df_handle, gridspec, disable_subsampling, output_dir, write_sharded):
    """
    Write the spatial index, given a dataframe in which the 'level'
    and grid chunk codes for each annotation have already been assigned.

    Args:
        df_handle:
            TableHandle.  The handle's reference will be unset before this function returns.
        gridspec:
            GridSpec object defining the spatial index.
        disable_subsampling:
            Whether to disable subsampling by seeting "limit" to 1 in the info file.
            (See inline comments.)
        output_dir:
            Directory to write the annotations to.
            Subdirectories will be created for each level of the spatial index.
        write_sharded:
            Whether to write the annotations in sharded format.

    Returns:
        JSON metadata to write into the 'spatial' key of the info file.
    """
    logger.info(f"Concatenating annotations by spatial chunk")
    bufs_by_grid = (
        df_handle.df[['level', 'chunk_code', 'id_buf', 'ann_buf']]
        .groupby(['level', 'chunk_code'], sort=False)
        .agg({'id_buf': ['count', b''.join], 'ann_buf': b''.join})
    )

    # We're done with the original input; delete it to save
    # RAM before writing (which takes a lot of RAM).
    df_handle.df = None

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
            grid_coords = compressed_morton_decode(
                level_bufs['chunk_code'].to_numpy(),
                gridspec.grid_shapes[level]
            )
            level_bufs.index = [map('_'.join, grid_coords.astype(str))]

        level_metadata = _write_buffers(
            level_bufs['combined_buf'],
            output_dir,
            f"by_spatial_level_{level}",
            write_sharded
        )
        level_metadata['chunk_size'] = gridspec.chunk_shapes[level].tolist()
        level_metadata['grid_shape'] = gridspec.grid_shapes[level].tolist()

        if disable_subsampling:
            # To be honest, I don't completely understand why this
            # disables subsampling, but according to jbms[1]:
            #
            #   > Neuroglancer "subsamples" by showing only a prefix of the list of
            #   > annotations according to the spacing setting.  If you set "limit" to 1 in
            #   > the info file, you won't get subsampling by default.
            #
            # [1]: https://github.com/google/neuroglancer/issues/227#issuecomment-651944575
            level_metadata['limit'] = 1
        else:
            level_metadata['limit'] = int(level_bufs['count'].max())
        metadata.append(level_metadata)

    return metadata
