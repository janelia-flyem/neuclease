import logging

import numpy as np
import pandas as pd

from skimage.util import view_as_blocks
from skimage.graph import MCP_Geometric
import vigra
from vigra.analysis import labelMultiArrayWithBackground

from neuprint import SynapseCriteria as SC, fetch_synapses
from neuclease.util import tqdm_proxy, Timer, box_to_slicing
from neuclease.dvid.labelmap import fetch_labelmap_specificblocks, fetch_seg_around_point

logger = logging.getLogger(__name__)


def measure_tbar_mito_distances(seg_src, mito_src, body,
                                initial_radius_s0=512, max_radius_s0=1024, radius_step_s0=128,
                                scale=0, mito_min_size_s0=10_000, npclient=None, tbars=None):
    # Fetch tbars
    if tbars is None:
        tbars = fetch_synapses(body, SC(rois='FB', type='pre', primary_only=True), client=npclient)

    tbars['mito-distance'] = np.inf
    tbars['done'] = False
    tbars['mito-x'] = 0
    tbars['mito-y'] = 0
    tbars['mito-z'] = 0

    for row in tqdm_proxy(tbars.itertuples(), total=len(tbars)):
        with Timer("Processing point", logger):
            p = (row.z, row.y, row.x)
            radius = initial_radius_s0
            mito_distance, mito_point = _measure_tbar_mito_distances(seg_src, mito_src, body, p, tbars, radius, scale, mito_min_size_s0)
            tbars.loc[row.Index, 'mito-distance'] = mito_distance
            tbars.loc[row.Index, 'done'] = True
            tbars.loc[row.Index, ['mito-z', 'mito-y', 'mito-x']] = mito_point

    return tbars


def _measure_tbar_mito_distances(seg_src, mito_src, body, main_tbar_point_s0, tbar_points_s0,
                                 radius_s0, scale, mito_min_size_s0, mito_scale_offset=1):
    p = np.asarray(main_tbar_point_s0) // (2**scale)
    mito_min_size = mito_min_size_s0 // ((2**scale)**3)
    radius = radius_s0 // (2**scale)

    body_mask, mask_box, body_block_corners = _fetch_body_mask(seg_src, p, radius, scale, body)
    body_mito_mask = _fetch_mito_mask(mito_src, body_mask, body_block_corners, scale, mito_min_size, mito_scale_offset)

    p_local = p - mask_box[0]
    distances, mito_points_local = _calc_distances(body_mask, body_mito_mask, [p_local])

    mito_points = mito_points_local + mask_box[0]
    return (2**scale)*distances[0], (2**scale)*mito_points[0]


def _fetch_body_mask(seg_src, p, radius, scale, body):
    """
    Fetch a mask for the given body around the given point, with the given radius.
    Only the connected component that covers the given point will be returned.
    If the component doesn't extend out to the given radius in all dimensions,
    then the returned subvolume may be smaller than the requested radius would seem to imply.
    """
    with Timer("Fetching body segmentation", logger):
        seg, seg_box, p_local, _ = fetch_seg_around_point(*seg_src, p, radius, scale, body)

    # Due to downsampling effects, it's possible that the
    # tbar fell off its body in the downsampled image.
    # Force it to have the correct label.
    seg[(*p_local,)] = body

    # Compute mito CC within body mask
    body_mask = (seg == body).view(np.uint8)
    del seg

    # Find the connected component that contains our point of interest
    # amd limit our analysis to those voxels.
    body_mask = vigra.taggedView(body_mask, 'zyx')
    with Timer("Computing body CC", logger):
        body_cc = labelMultiArrayWithBackground(body_mask)

    # Update mask
    body_mask = (body_cc == body_cc[(*p_local,)])
    body_block_mask = view_as_blocks(body_mask, (64,64,64)).any(axis=(3,4,5))
    body_block_corners = seg_box[0] + (64 * np.argwhere(body_block_mask))
    mask_box = (body_block_corners.min(axis=0),
                body_block_corners.max(axis=0) + 64)

    local_box = mask_box - seg_box[0]
    body_mask = body_mask[box_to_slicing(*local_box)]

    return body_mask, mask_box, body_block_corners


def _fetch_mito_mask(mito_src, body_mask, body_block_corners, scale, mito_min_size, mito_scale_offset):
    assert scale - mito_scale_offset >= 0, \
        "FIXME: need to upsample the mito seg if using scale 0.  Not implemented yet."

    with Timer("Fetching mito mask", logger):
        mito_seg = fetch_labelmap_specificblocks(*mito_src, body_block_corners, scale - mito_scale_offset, threads=4)

    mito_mask = np.array([0,1,1,1,0], np.uint8)[mito_seg]  # mito mask class 4 means "empty"

    body_mito_mask = np.where(body_mask, mito_mask, 0)
    body_mito_mask = vigra.taggedView(body_mito_mask, 'zyx')
    with Timer("Computing mito CC", logger):
        body_mito_cc = labelMultiArrayWithBackground(body_mito_mask)

    # Erase small mitos from body_mito_mask
    mito_sizes = np.bincount(body_mito_cc.reshape(-1))
    mito_sizes[0] = 0
    body_mito_mask = (mito_sizes > mito_min_size)[body_mito_cc]
    return body_mito_mask


def _calc_distances(body_mask, body_mito_mask, local_points_zyx):
    """
    Calculate the distances from a set of mito points to a set of tbar points,
    restricting paths to the given body mask.

    Uses skimage.graph.MCP_Geometric to perform the graph search.
    """
    if body_mito_mask.sum() == 0:
        return np.inf, (0,0,0)

    # MCP uses float64, so we may as well use that now and avoid copies
    body_costs = np.where(body_mask, 1.0, np.inf)

    # Insanely, MCP source code copies to fortran order internally,
    # so let's pass in fortran order to start with.
    # That shaves ~10-20% from the initialization time.
    body_costs = body_costs.transpose()
    body_mito_mask = body_mito_mask.transpose()
    local_points_xyz = np.asarray(local_points_zyx)[:, ::-1]

    with Timer("Initializing MCP", logger):
        mcp = MCP_Geometric(body_costs)

    with Timer("Finding costs", logger):
        distance_vol, _ = mcp.find_costs(np.argwhere(body_mito_mask), local_points_xyz)

    point_distances = distance_vol[(*local_points_xyz.transpose(),)]

    with Timer("Finding tracebacks", logger):
        # The necessary path traceback data is retained
        # in MCP internally after find_costs() above.
        p_mito_xyz = np.zeros((len(local_points_xyz), 3), np.int16)

        for i, (p_xyz, d) in enumerate(zip(local_points_xyz, point_distances)):
            if d != np.inf:
                p_mito_xyz[i] = mcp.traceback(p_xyz)[0]

    # Translate back to C-order
    p_mito_zyx = p_mito_xyz[:, ::-1]
    return point_distances, p_mito_zyx


if __name__ == "__main__":
    from neuprint import Client
    from neuclease import configure_default_logging
    configure_default_logging()

    c = Client('neuprint.janelia.org', 'hemibrain:v1.1')
    body = 519046655
    tbars = fetch_synapses(body, SC(rois='FB', type='pre', primary_only=True))
    v11_seg = ('emdata4:8900', '20631f94c3f446d7864bc55bf515706e', 'segmentation')
    mito_mask = ('emdata4.int.janelia.org:8900', 'fbd7db9ebde6426e9f8474af10fe4386', 'mito_20190717.46637005.combined')
    measure_tbar_mito_distances(v11_seg, mito_mask, body, scale=3, npclient=c, tbars=tbars.iloc[:10])
