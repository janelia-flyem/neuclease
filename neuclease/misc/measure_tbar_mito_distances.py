import logging
from collections import namedtuple
from collections.abc import Mapping
from typing import Sequence

import numpy as np
import pandas as pd

from skimage.util import view_as_blocks
from skimage.graph import MCP_Geometric
import vigra
from vigra.analysis import labelMultiArrayWithBackground

from neuprint import SynapseCriteria as SC, fetch_synapses
from neuclease.util import tqdm_proxy, Timer, box_to_slicing, box_intersection, round_box
from neuclease.dvid.labelmap import fetch_labelmap_specificblocks, fetch_seg_around_point

try:
    from flyemflows.volumes import VolumeService
    _have_flyemflows = True
except ImportError:
    _have_flyemflows = False

logger = logging.getLogger(__name__)
EXPORT_DEBUG_VOLUMES = False
SearchConfig = namedtuple('SearchConfig', 'radius_s0 scale')

DEFAULT_SEARCH_CONFIGS = [
    SearchConfig(radius_s0=250, scale=3),  # 2 microns (empirically, this captures ~90% of FB tbars)
    SearchConfig(radius_s0=625, scale=3),  # 5 microns
    SearchConfig(radius_s0=1250, scale=3)  # 10 microns
]


def measure_tbar_mito_distances(seg_src,
                                mito_src,
                                body,
                                search_configs=DEFAULT_SEARCH_CONFIGS,
                                mito_min_size_s0=10_000,
                                mito_scale_offset=1,
                                npclient=None,
                                tbars=None):
    """
    Search for the closest mito to each tbar in a list of tbars
    (or any set of points, really).

    TODO:
        - Distinguish between "download scale" and "analysis scale".
          The data can be downloaded at a higher scale and then downscaled
          using continuity-preserving downsampling before analysis.

        - Try morphological closing (or simply dilating) the body before
          running the path search, to close small gaps between segments.

        - Right now, disconnected components result in an early stop,
          marking a tbar as 'done' without trying more aggressive
          search criteria. Should that be changed, or will the above two
          changes be sufficient to ensure that remaining gaps are really
          unclosable even at higher resolution?

    Args:
        seg_src:
            (server, uuid, instance) OR a flyemflows VolumeService
            Labelmap instance for the neuron segmentation.
        mito_src:
            (server, uuid, instance) OR a flyemflows VolumeService
            Labelmap instance for the mitochondria "mask"
            (actually a segmentation with a few classes.)
        body:
            The body ID of interest, on which the tbars reside.
        search_configs:
            A list of pairs ``[(radius_s0, scale), (radius_s0, scale), ...]``.
            For each tbar, this function tries to locate a mitochondria within
            he given radius, using data downloaded from the given scale.
            If the search fails and no mito can be found, the function tries
            again using the next search criteria in the list.
            The radius should always be specified in scale-0 units,
            regardless of the scale at which you want to perform the analysis.
            Notes:
                - Scale 4 is too low-res.  Stick with scale-3 or better.
                - Higher radius is more expensive, but some of that expense is
                  recouped because all points that fall within the radius are
                  analyzed at once.  See _measure_tbar_mito_distances()
                  implementation for details.
        mito_min_size_s0:
            Mito mask voxels that fall outside the body mask will be discarded,
            and then the mito mask is segmented via a connected components step.
            Components below this size threshold will be discarded before
            distances are computed.  Specify this threshold in units of scale 0
            voxels, regardless of the scale at which the analysis is being performed.
        mito_scale_offset:
            If the mito mask layer is stored at a lower resolution than the
            neuron segmentation, specify the difference between the two scales
            using this parameter. (It's assumed that the scales differ by a power of two.)
            For instance, if the segmentation is stored at 8nm resolution,
            but the mito masks are stored at 16nm resolution, use mito_scale_offset=1.
        npclient:
            ``neuprint.Client`` to use when fetching the list of tbars that belong
            to the given body, unless you provide your own tbar points in the next
            argument.
        tbars:
            A DataFrame of tbar coordinates at least with columns ``['x', 'y', 'z']``.
    Returns:
        DataFrame of tbar coordinates, mito distances, and mito coordinates.
        Points for which no nearby mito could be found (after trying all the given search_configs)
        will be marked with `done=False` in the results.
    """
    # Fetch tbars
    if tbars is None:
        tbars = fetch_synapses(body, SC(type='pre', primary_only=True), client=npclient)
    else:
        tbars = tbars.copy()

    tbars['body'] = body
    tbars['mito-distance'] = np.inf
    tbars['done'] = False
    tbars['mito-x'] = 0
    tbars['mito-y'] = 0
    tbars['mito-z'] = 0

    with tqdm_proxy(total=len(tbars)) as progress:
        for row in tbars.itertuples():
            if row.done:
                continue

            for radius_s0, scale in search_configs:
                num_done = _measure_tbar_mito_distances(seg_src, mito_src, body, tbars, row.Index, radius_s0, scale, mito_min_size_s0, mito_scale_offset)
                progress.update(num_done)
                done = (tbars['done'].loc[row.Index])
                if done:
                    break
                logger.info("Search failed for primary tbar. Trying next search config!")

            if not done:
                logger.warn(f"Failed to find a nearby mito for tbar at point {(row.x, row.y, row.z)}")
                progress.update(1)

    num_done = tbars['done'].sum()
    num_failed = (~tbars['done']).sum()
    logger.info(f"Found mitos for {num_done} tbars, failed for {num_failed} tbars")

    return tbars


def _measure_tbar_mito_distances(seg_src, mito_src, body, tbar_points_s0, primary_point_index,
                                 radius_s0, scale, mito_min_size_s0, mito_scale_offset):
    """
    Download the segmentation for a single body around one tbar point as a mask,
    and also the corresponding mitochondria mask for those voxels.
    Then compute the minimum distance from any mitochondria voxel to all tbar
    points in the region (not just the one tbar we chose as our focal point).

    Not all of the computed distances are used, however.
    Only the results for tbars which are closer to their nearest mitochondria
    than they are to the subvolume edge can be trusted.

    The results are written into the columns of tbar_points_s0.
    Points for which a mito was found are marked as 'done', and the
    mito-distance is recorded. Also, the closest point in the mito is stored
    in the mito-x/y/z columns.

    Args:
        seg_src:
            (server, uuid, instance) OR a flyemflows VolumeService
            Labelmap instance for the neuron segmentation.
        mito_src:
            (server, uuid, instance) OR a flyemflows VolumeService
            Labelmap instance for the mitochondria "mask"
            (actually a segmentation with a few classes.)
        body:
            The body ID on which the tbars reside.
        tbar_points_s0:
            DataFrame with ALL tbar coordinates you plan to analyze.
            The coordinates should be specified at scale 0,
            even if you are specifying a different scale to use for the analysis.
            We update the row of the "primary" point, but we also update any
            other rows we can, since the mask we download might happen to
            catch other tbars, too.
        primary_point_index:
            An index value, indicating which row of tbar_points_s0 should be
            the "primary" point around which the body/mito masks are downloaded.
        radius_s0:
            The radius of segmentation around the "primary" point to fetch and
            analyze for mito-tbar distances. Specified at scale 0, regardless of
            the scale you want to be used for performing the analysis.
        scale:
            To save time and RAM, it's faster to perform the analysis using a
            lower resolution.  Specify which scale to use.
        mito_min_size_s0:
            Mito mask voxels that fall outside the body mask will be discarded,
            and then the mito mask is segmented via a connected components step.
            Components below this size threshold will be discarded before
            distances are computed.  Specify this threshold in units of scale 0
            voxels, regardless of the scale at which the analysis is being performed.
        mito_scale_offset:
            If the mito mask layer is stored at a lower resolution than the
            neuron segmentation, specify the difference between the two scales
            using this parameter. (It's assumed that the scales differ by a power of two.)
            For instance, if the segmentation is stored at 8nm resolution,
            but the mito masks are stored at 16nm resolution, use mito_scale_offset=1.

    Returns:
        The number of tbars for which a nearby mitochondria was found in this batch.
        (Where "batch" is the set of not-yet-done tbars that overlap with the body mask
        near the "primary" tbar point.)
    """
    assert not tbar_points_s0['done'].loc[primary_point_index]
    primary_point_s0 = tbar_points_s0[[*'zyx']].loc[primary_point_index].values
    batch_tbars = tbar_points_s0.copy()

    # Adjust for scale
    primary_point = np.asarray(primary_point_s0) // (2**scale)
    mito_min_size = mito_min_size_s0 // ((2**scale)**3)
    radius = radius_s0 // (2**scale)
    batch_tbars[[*'zyx']] //= (2**scale)

    body_mask, mask_box, body_block_corners = _fetch_body_mask(seg_src, primary_point, radius, scale, body, batch_tbars[[*'zyx']].values)
    mito_mask = _fetch_mito_mask(mito_src, body_mask, mask_box, body_block_corners, scale, mito_min_size, mito_scale_offset)

    if EXPORT_DEBUG_VOLUMES:
        print(f"Primary point in the local volume is: {(primary_point - mask_box[0])[::-1]}")
        np.save('/tmp/body_mask.npy', 1*body_mask.astype(np.uint64))
        np.save('/tmp/mito_mask.npy', 2*mito_mask.astype(np.uint64))

    if (body_mask & mito_mask).sum() == 0:
        # The body mask contains no mitochondria at all.
        if ( body_mask[0, :, :].any() or body_mask[-1, :, :].any() or
             body_mask[:, 0, :].any() or body_mask[:, -1, :].any() or
             body_mask[:, :, 0].any() or body_mask[:, :, -1].any() ):
            # The body mask touches the edge of the volume,
            # so we should expand our radius and keep trying.
            return 0
        else:
            # Doesn't touch volume edges.
            # We're done with it, even though we can't find a mito.
            tbar_points_s0.loc[primary_point_index, 'done'] = True
            return 1

    # Find the set of all points that fall within the mask.
    # That's that batch of tbars we'll find mito distances for.
    batch_tbars = batch_tbars.query('not done')

    in_box = (batch_tbars[[*'zyx']] >= mask_box[0]).all(axis=1) & (batch_tbars[[*'zyx']] < mask_box[1]).all(axis=1)
    batch_tbars = batch_tbars.loc[in_box]

    tbars_local = batch_tbars[[*'zyx']] - mask_box[0]
    in_mask = body_mask[tuple(tbars_local.values.transpose())]
    batch_tbars = batch_tbars.iloc[in_mask]
    assert len(batch_tbars) >= 1

    with Timer(f"Calculating distances for batch of {len(batch_tbars)} points", logger):
        tbars_local = batch_tbars[[*'zyx']] - mask_box[0]
        distances, mito_points_local = _calc_distances(body_mask, mito_mask, tbars_local.values)

    mito_points = mito_points_local + mask_box[0]
    batch_tbars['mito-distance'] = distances
    batch_tbars.loc[:, ['mito-z', 'mito-y', 'mito-x']] = mito_points

    batch_cube = [primary_point - radius, primary_point + radius + 1]

    valid_rows = []
    for i in batch_tbars.index:
        # If we found a mito for this tbar, we can only keep it if
        # the tbar is closer to the mito than it is to the edge of
        # the mask volume. Otherwise, we can't guarantee that this
        # mito is the globally closest mito to the tbar.  (There
        # could be one just outside the mask subvolume that is
        # closer.)

        # Define a box (cube) around the point,
        # whose radius is the mito distance.
        p = batch_tbars[[*'zyx']].loc[i].values
        d = batch_tbars['mito-distance'].loc[i]
        p_cube = [p - d, p + d + 1]

        # If the cube around our point doesn't exceed the box that was
        # searched for this batch, we can believe this mito distance.
        if (p_cube == box_intersection(p_cube, batch_cube)).all():
            valid_rows.append(i)

    logger.info(f"Kept {len(valid_rows)}/{len(batch_tbars)} mito distances (R={radius_s0})")
    batch_tbars = batch_tbars.loc[valid_rows]

    # Update the input DataFrame (and rescale)
    tbar_points_s0.loc[batch_tbars.index, 'mito-distance'] = (2**scale)*batch_tbars['mito-distance']
    tbar_points_s0.loc[batch_tbars.index, ['mito-z', 'mito-y', 'mito-x']] = (2**scale)*batch_tbars[['mito-z', 'mito-y', 'mito-x']]
    tbar_points_s0.loc[batch_tbars.index, 'done'] = True

    return len(batch_tbars)


def _fetch_body_mask(seg_src, p, radius, scale, body, tbar_points):
    """
    Fetch a mask for the given body around the given point, with the given radius.
    Only the connected component that covers the given point will be returned.
    If the component doesn't extend out to the given radius in all dimensions,
    then the returned subvolume may be smaller than the requested radius would seem to imply.
    """
    with Timer("Fetching body segmentation", logger):
        if _have_flyemflows and isinstance(seg_src, VolumeService):
            p = np.asarray(p)
            R = radius
            seg_box = [p-R, p+R+1]

            # Align to 64-px for consistency with the dvid case,
            # and compatibility with the code below.
            seg_box = round_box(seg_box, 64, 'out')
            p_local = (p - seg_box[0])
            seg = seg_src.get_subvolume(seg_box, scale)
        else:
            assert len(seg_src) == 3 and all(isinstance(s, str) for s in seg_src)
            seg, seg_box, p_local, _ = fetch_seg_around_point(*seg_src, p, radius, scale, body,
                                                              sparse_component_only=True, map_on_client=True, threads=16)

    # Due to downsampling effects, it's possible that the
    # main tbar fell off its body in the downsampled image.
    # Force it to have the correct label.
    seg[(*p_local,)] = body

    # The same is true for all of the other tbars that fall within the mask box.
    # Fix them all.
    in_box = (tbar_points >= seg_box[0]).all(axis=1) & (tbar_points < seg_box[1]).all(axis=1)
    tbar_points = tbar_points[in_box]
    local_tbar_points = tbar_points - seg_box[0]
    seg[(*local_tbar_points.transpose(),)] = body

    # Compute mito CC within body mask
    body_mask = (seg == body).view(np.uint8)
    del seg

    # Find the connected component that contains our point of interest
    # amd limit our analysis to those voxels.
    body_mask = vigra.taggedView(body_mask, 'zyx')
    body_cc = labelMultiArrayWithBackground(body_mask)

    # Update mask. Limit to extents of the main cc, but align box to nearest 64px
    body_mask = (body_cc == body_cc[(*p_local,)])
    body_block_mask = view_as_blocks(body_mask, (64,64,64)).any(axis=(3,4,5))
    body_block_corners = seg_box[0] + (64 * np.argwhere(body_block_mask))
    mask_box = (body_block_corners.min(axis=0),
                body_block_corners.max(axis=0) + 64)

    local_box = mask_box - seg_box[0]
    body_mask = body_mask[box_to_slicing(*local_box)]

    assert body_mask[(*p - mask_box[0],)]
    return body_mask, mask_box, body_block_corners


def _fetch_mito_mask(mito_src, body_mask, mask_box, body_block_corners, scale, mito_min_size, mito_scale_offset):
    assert scale - mito_scale_offset >= 0, \
        "FIXME: need to upsample the mito seg if using scale 0.  Not implemented yet."

    with Timer("Fetching mito mask", logger):
        if _have_flyemflows and isinstance(mito_src, VolumeService):
            mito_seg = mito_src.get_subvolume(mask_box, scale)
        else:
            assert len(mito_src) == 3 and all(isinstance(s, str) for s in mito_src)
            mito_seg = fetch_labelmap_specificblocks(*mito_src, body_block_corners, scale - mito_scale_offset, supervoxels=True, threads=4)

    # mito classes 1,2,3 are valid;
    # mito mask class 4 means "empty", as does 0.
    mito_mask = np.array([0,1,1,1,0], np.uint8)[mito_seg]

    body_mito_mask = np.where(body_mask, mito_mask, 0)
    body_mito_mask = vigra.taggedView(body_mito_mask, 'zyx')
    body_mito_cc = labelMultiArrayWithBackground(body_mito_mask)

    # Erase small mitos from body_mito_mask
    mito_sizes = np.bincount(body_mito_cc.reshape(-1))
    mito_sizes[0] = 0
    body_mito_mask = (mito_sizes > mito_min_size)[body_mito_cc]
    return body_mito_mask


def _calc_distances(body_mask, mito_mask, local_points_zyx):
    """
    Calculate the distances from a set of mito points to a set of tbar points,
    restricting paths to the given body mask.

    Uses skimage.graph.MCP_Geometric to perform the graph search.
    """
    if mito_mask.sum() == 0:
        return np.inf, (0,0,0)

    # MCP uses float64, so we may as well use that now and avoid copies
    body_costs = np.where(body_mask, 1.0, np.inf)

    # Insanely, MCP source code copies to fortran order internally,
    # so let's pass in fortran order to start with.
    # That shaves ~10-20% from the initialization time.
    body_costs = body_costs.transpose()
    mito_mask = mito_mask.transpose()
    local_points_xyz = np.asarray(local_points_zyx)[:, ::-1]

    mcp = MCP_Geometric(body_costs)
    distance_vol, _ = mcp.find_costs(np.argwhere(mito_mask), local_points_xyz)
    point_distances = distance_vol[(*local_points_xyz.transpose(),)]

    # The necessary path traceback data is retained
    # in MCP internally after find_costs() above.
    p_mito_xyz = np.zeros((len(local_points_xyz), 3), np.int16)

    for i, (p_xyz, d) in enumerate(zip(local_points_xyz, point_distances)):
        if d != np.inf:
            p_mito_xyz[i] = mcp.traceback(p_xyz)[0]

    # Transpose back to C-order
    p_mito_zyx = p_mito_xyz[:, ::-1]
    return point_distances, p_mito_zyx


if __name__ == "__main__":
    from neuprint import Client
    from neuclease import configure_default_logging
    configure_default_logging()

    c = Client('neuprint.janelia.org', 'hemibrain:v1.1')
    body = 519046655
    tbars = fetch_synapses(body, SC(rois='FB', type='pre', primary_only=True))

    #v11_seg = ('emdata4:8900', '20631f94c3f446d7864bc55bf515706e', 'segmentation')
    #mito_mask = ('emdata4.int.janelia.org:8900', 'fbd7db9ebde6426e9f8474af10fe4386', 'mito_20190717.46637005.combined')
    #mito_scale_offset = 1

    seg_cfg = {
        "zarr": {
            "path": "/Users/bergs/data/hemibrain-v1.1.zarr",
            "dataset": "s3",
            "store-type": "NestedDirectoryStore",
            "out-of-bounds-access": "permit-empty"
        },
        "adapters": {
            "rescale-level": -3
        }
    }

    seg_svc = VolumeService.create_from_config(seg_cfg)

    mito_cfg = {
        "zarr": {
            "path": "/Users/bergs/data/hemibrain-mito-mask.zarr",
            "dataset": "s3",
            "store-type": "NestedDirectoryStore",
            "out-of-bounds-access": "permit-empty"
        },
        "adapters": {
            "rescale-level": -3
        }
    }
    mito_svc = VolumeService.create_from_config(mito_cfg)
    mito_scale_offset = 0

    processed_tbars = measure_tbar_mito_distances(seg_svc, mito_svc, body, npclient=c, tbars=tbars.iloc[:10],
                                                  mito_scale_offset=mito_scale_offset)
    print(processed_tbars)
