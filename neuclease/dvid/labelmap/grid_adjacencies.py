"""
Defines extract_grid_adjacencies()
"""
import logging
from functools import partial
import numpy as np
import pandas as pd

from neuclease.util import Grid, align_box, view_as_blocks, compute_parallel, ndindex_array, box_intersection
from neuclease.dvid import fetch_volume_box, fetch_labelmap_voxels_chunkwise, fetch_raw

logger = logging.getLogger(__name__)

# The lookup table is huge (several GB), so if we were to
# pass it to subprocesses via a partial it would be insanely slow.
# As a hacky workaround, we store it in this global before
# forking the process during compute_parallel().
GRID_TO_BASE = None


def extract_grid_adjacencies(server, root_uuid, instance, grid, grid_to_base_lookup, lookup_min_grid_sv=0, full_box=None, processes=32):
    """
    Find pairs of adjacent 'grid' supervoxels which come from the same 'base' supervoxel.

    Args:
        server:
            dvid server
        root_uuid:
            The UUID in which the supervoxels where ingested,
            for which the grid_to_base_lookup will be valid.
        instance:
            labelmap instance name
            Note:
                Supervoxel IDs MUST fit within uint32 and should (ideally)
                be occupy a near-contiguous ID space starting near-ish to 0
        grid:
            neuclease.util.Grid
            Should describe the grid upon which the 'gridded' supervoxels were defined.
            Non-zero offset IS permitted.
            Hints:
                - In the maleCNS brain, use Grid((2048, 2048, 2048)), with no offset.
                - In the maleCNS VNC, use Grid((2048, 2048, 2048), (-256, -256, -256))
        grid_to_base_lookup:
            np.array, mapping from DVID SV to Michal's "base" ID (non-gridded supervoxels)
        lookup_min_grid_sv:
            The 0th item in grid_to_base_lookup must correspond to SV 0,
            but the lookup can begin with the minimum gridded SV in position 1.
            Use this argument to specify the gridded SV ID which corresponds to position 1
            in grid_to_base_lookup.
        full_box:
            The bounding box to analyze. By default, analyze the entire segmentation volume.
            Hint: The boundary between male CNS VNC starts at Z=45056
    """
    global GRID_TO_BASE
    try:
        GRID_TO_BASE = grid_to_base_lookup

        root_seg = server, root_uuid, instance
        grid = Grid.as_grid(grid)

        if full_box is None:
            full_box = fetch_volume_box(*root_seg)

        full_box = np.asarray(full_box)
        z_size = 8 * np.prod(full_box[1, (1, 2)]) / 1e9
        y_size = 8 * np.prod(full_box[1, (0, 2)]) / 1e9
        x_size = 8 * np.prod(full_box[1, (0, 1)]) / 1e9
        logger.info(f"Size of Z-tiles, Y-tiles, X-tiles (in GB): {z_size}, {y_size}, {x_size}")

        # Fetch scale-7 to use as a mask to limit the extent of the computation.
        # Since there might be an offset, we have to fetch an aligned (padded) region,
        # downsample its non-zero mask, rescale the downsampled boxes, then chop off
        # the padding from those rescaled boxes.
        aligned_box = align_box(full_box, grid, 'out')
        mask_scale = 7
        mask = fetch_labelmap_voxels_chunkwise(*root_seg, aligned_box // 2**mask_scale, scale=mask_scale, threads=8) != 0

        # Determine the list of grid2k chunks which need to be analyzed
        gridded_mask = view_as_blocks(mask, tuple(grid.block_shape // 2**mask_scale))
        nonempty_mask_grid = gridded_mask.any(axis=(3,4,5)).nonzero()
        nonempty_left_corners = grid.block_shape * np.transpose(nonempty_mask_grid)
        nonempty_left_corners += aligned_box[0]
        nonempty_right_corners = nonempty_left_corners + grid.block_shape
        nonempty_grid_boxes = np.concatenate(
            (
                nonempty_left_corners[:, None, :],
                nonempty_right_corners[:, None, :]
            ),
            axis=1
        )
        nonempty_grid_boxes = box_intersection(nonempty_grid_boxes, full_box)

        fn = partial(_extract_grid_adjacencies, root_seg, lookup_min_grid_sv)
        all_edges = compute_parallel(fn, nonempty_grid_boxes, processes=processes, leave_progress=True)
        all_edges = pd.concat(all_edges, ignore_index=True)
        return all_edges
    finally:
        GRID_TO_BASE = None


def _extract_grid_adjacencies(root_seg, lookup_min_grid_sv, box):
    """
    Extract the adjacencies from this grid square to the one just "below" it along all three dimensions (Z,Y,X).
    Only adjacencies between "gridded" supervoxels that belong to the same "base" supervoxel will be extracted.
    The output DataFrame has columns for the 'left' and 'right' coordinates (on opposite sides of the edge).
    """
    grid_to_base_lookup = GRID_TO_BASE

    chunk_edges = []
    for axis, c in enumerate(box[0]):
        box_left, box_right = box.copy(), box.copy()
        box_left[:, axis] = (c - 1, c)
        box_right[:, axis] = (c, c + 1)

        # I'm using fetch_raw because otherwise I have to fetch an entire 64px slab,
        # which would be too much data from DVID to transfer in a single request,
        # and I'd need to batch it, and yuck.  With /raw, dvid chunks the data.
        # The root node is supervoxels, so this is okay.
        tile_left = fetch_raw(*root_seg, box_left, dtype=np.uint64)
        tile_right = fetch_raw(*root_seg, box_right, dtype=np.uint64)

        # Our supervoxels fit within 32 bit
        tile_left = tile_left.astype(np.uint32)
        tile_right = tile_right.astype(np.uint32)

        # We're only interested in edges between SVs with the same base ID.
        pairs_df = pd.DataFrame({'sv_left': tile_left.ravel(), 'sv_right': tile_right.ravel()})
        pairs_df[['zr', 'yr', 'xr']] = box_right[0] + ndindex_array(*tile_right.shape)

        left_grid_svs = pairs_df['sv_left'].values
        left_lookup_input = np.where(left_grid_svs, left_grid_svs - lookup_min_grid_sv + 1, 0)

        right_grid_svs = pairs_df['sv_right'].values
        right_lookup_input = np.where(right_grid_svs, right_grid_svs - lookup_min_grid_sv + 1, 0)

        pairs_df['base_left'] = grid_to_base_lookup[left_lookup_input]
        pairs_df['base_right'] = grid_to_base_lookup[right_lookup_input]

        pairs_df = pairs_df.query('base_left == base_right and base_left != 0')
        pairs_df = pairs_df[['sv_left', 'sv_right', 'zr', 'yr', 'xr']]

        # Note:
        #     It's tempting to simply use the centroid as the edge coordinate,
        #     but that doesn't work in cases where the edge voxels don't include the centroid.
        #     For instance, consider the case where the base supervoxel contains a hole on
        #     one or both sides of the grid boundary which happens to contain the edge centroid.
        #     Here, we compute the centroid, and then find the closest location to it among the
        #     edge locations that actually exist in the data.
        centroids = pairs_df.groupby(['sv_left', 'sv_right'])[['zr', 'yr', 'xr']].mean().astype(np.int32).reset_index()
        centroids = centroids.rename(columns={'zr': 'zc', 'yr': 'yc', 'xr': 'xc'})

        pairs_df = pairs_df.merge(centroids, 'left', on=['sv_left', 'sv_right'])
        r = pairs_df[['zr', 'yr', 'xr']].values
        c = pairs_df[['zc', 'yc', 'xc']].values
        pairs_df['dist'] = np.linalg.norm(r - c, axis=1)
        selections = pairs_df.groupby(['sv_left', 'sv_right'])['dist'].idxmin()

        edges = pairs_df.loc[selections]
        edges[['zl', 'yl', 'xl']] = edges[['zr', 'yr', 'xr']] - (box_right[0] - box_left[0])
        edges = edges[['sv_left', 'sv_right', 'xl', 'yl', 'zl', 'xr', 'yr', 'zr']]

        chunk_edges.append(edges)
    return pd.concat(chunk_edges, ignore_index=True)

