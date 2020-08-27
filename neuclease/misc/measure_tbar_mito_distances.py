import numpy as np
import pandas as pd

from skimage.graph import MCP_Geometric
import vigra
from vigra.analysis import labelMultiArrayWithBackground

from neuprint import SynapseCriteria as SC, fetch_synapses
from neuclease.util import tqdm_proxy
from neuclease.dvid.labelmap import fetch_labelmap_specificblocks, fetch_seg_around_point


def measure_tbar_mito_distances(seg_src, mito_src, body,
                                initial_radius_s0=512, max_radius_s0=1024, radius_step_s0=128,
                                scale=0, mito_min_size_s0=10_000, npclient=None, tbars=None):
    # Fetch tbars
    print('fetching tbars')
    if tbars is None:
        tbars = fetch_synapses(body, SC(rois='FB', type='pre', primary_only=True), client=npclient)

    tbars['mito-distance'] = np.inf
    tbars['done'] = False
    tbars['mito-x'] = 0
    tbars['mito-y'] = 0
    tbars['mito-z'] = 0

    tbars = tbars.iloc[:10]

    for row in tqdm_proxy(tbars.itertuples(), total=len(tbars)):
        p = (row.z, row.y, row.x)
        radius = initial_radius_s0
        print(f"p = {p}")
        mito_distance, mito_point = _measure_tbar_mito_distances( seg_src, mito_src, body, p, tbars, radius, scale, mito_min_size_s0, locate_mito=True)
        tbars.loc[row.Index, 'mito-distance'] = mito_distance
        tbars.loc[row.Index, 'done'] = True
        tbars.loc[row.Index, ['mito-z', 'mito-y', 'mito-x']] = mito_point

    return tbars


def _measure_tbar_mito_distances(seg_src, mito_src, body, main_tbar_point_s0, tbar_points_s0,
                                 radius_s0, scale, mito_min_size_s0, mito_scale_offset=1, locate_mito=False):
    p = np.asarray(main_tbar_point_s0) // (2**scale)
    mito_min_size = mito_min_size_s0 // ((2**scale)**3)
    radius = radius_s0 // (2**scale)

    seg, box, p_local, block_corners = fetch_seg_around_point(*seg_src, p, radius, scale, body)

    # Due to downsampling effects, it's possible that the
    # tbar fell off its body in the downsampled image.
    # Force it to have the correct label.
    seg[(*p_local,)] = body

    assert scale - mito_scale_offset >= 0, \
        "FIXME: need to upsample the mito seg if using scale 0.  Not implemented yet."

    mito_seg = fetch_labelmap_specificblocks(*mito_src, block_corners, scale-mito_scale_offset, threads=4)

    # Compute mito CC within body mask
    body_mask = (seg == body)
    mito_mask = np.array([0,1,1,1,0], np.uint8)[mito_seg]  # mito mask class 4 means "empty"

    body_mito_mask = np.where(body_mask, mito_mask, 0)
    body_mito_mask = vigra.taggedView(body_mito_mask, 'zyx')
    body_mito_cc = labelMultiArrayWithBackground(body_mito_mask)

    # Erase small mitos from body_mito_mask
    mito_sizes = np.bincount(body_mito_cc.reshape(-1))
    mito_sizes[0] = 0
    body_mito_mask = (mito_sizes > mito_min_size)[body_mito_cc]
    if body_mito_mask.sum() == 0:
        if locate_mito:
            return np.inf, (0,0,0)
        return np.inf

    # Find the minimum cost from any mito point to the p_local,
    # restricted to the body mask.
    #
    # TODO:
    #   Check if it would be faster to go in the other direction.
    #   If not, try finding paths to all tbars

    body_costs = np.where(body_mask, np.float32(1.0), np.float32(np.inf))
    mcp = MCP_Geometric(body_costs)
    distances, _ = mcp.find_costs(np.argwhere(body_mito_mask), [p_local])

    d = distances[(*p_local,)]
    d_s0 = (2**scale) * d

    if locate_mito:
        if d_s0 == np.inf:
            return np.inf, (0,0,0)

        # The necessary path traceback data is retained
        # in MCP internally after find_costs() above.
        p_mito = mcp.traceback(p_local)[0] + box[0]
        p_mito_s0 = (2**scale) * p_mito
        return d_s0, p_mito_s0
    else:
        return d_s0


if __name__ == "__main__":
    from neuprint import Client
    c = Client('neuprint.janelia.org', 'hemibrain:v1.1')
    body = 519046655
    tbars = fetch_synapses(body, SC(rois='FB', type='pre', primary_only=True))
    v11_seg = ('emdata4:8900', '20631f94c3f446d7864bc55bf515706e', 'segmentation')
    mito_mask = ('emdata4.int.janelia.org:8900', 'fbd7db9ebde6426e9f8474af10fe4386', 'mito_20190717.46637005.combined')
    measure_tbar_mito_distances(v11_seg, mito_mask, body, scale=3, npclient=c, tbars=tbars.iloc[6:])
