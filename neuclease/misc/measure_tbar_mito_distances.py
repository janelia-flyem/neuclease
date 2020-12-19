import os
import logging
from collections import namedtuple

import numpy as np
import pandas as pd

from skimage.util import view_as_blocks
from skimage.graph import MCP_Geometric

import vigra
from vigra.analysis import labelMultiArrayWithBackground

from neuprint import SynapseCriteria as SC, fetch_synapses
from neuclease.util import tqdm_proxy, Timer, box_to_slicing, round_box, apply_mask_for_labels, downsample_mask
from neuclease.logging_setup import PrefixedLogger

try:
    from flyemflows.volumes import VolumeService
    _have_flyemflows = True
except ImportError:
    _have_flyemflows = False

logger = logging.getLogger(__name__)
EXPORT_DEBUG_VOLUMES = False
DEBUG_BODY_MASK = False
SearchConfig = namedtuple('SearchConfig', 'radius_s0 download_scale analysis_scale')

DEFAULT_SEARCH_CONFIGS = [
    SearchConfig(radius_s0=250,  download_scale=2, analysis_scale=3),  #  2 microns (empirically, this captures ~90% of FB tbars)
    SearchConfig(radius_s0=625,  download_scale=2, analysis_scale=3),  #  5 microns
    SearchConfig(radius_s0=1250, download_scale=2, analysis_scale=4)   # 10 microns
]

# Fake mito ID to mark the places where a body exits the analysis volume.
# Must not conflict with any real mito IDs
FACE_MARKER = np.uint64(1e15-1)


def measure_tbar_mito_distances(seg_src,
                                mito_src,
                                body,
                                search_configs=DEFAULT_SEARCH_CONFIGS,
                                dilation_radius_s0=16,
                                npclient=None,
                                tbars=None):
    """
    Search for the closest mito to each tbar in a list of tbars
    (or any set of points, really).

    Args:
        seg_src:
            (server, uuid, instance) OR a flyemflows VolumeService
            Labelmap instance for the neuron segmentation.
        mito_src:
            (server, uuid, instance) OR a flyemflows VolumeService
            Labelmap instance for the mitochondria "supervoxel"
            segmentation -- not just the "masks".
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
        dilation_radius_s0:
            The neuron will be dilated before distances are analyzed.
            This will close small gaps in the segmentation, but it will have
            a slight effect on the distances returned.
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
    tbars['mito-id'] = np.uint64(0)

    with tqdm_proxy(total=len(tbars)) as progress:
        for row in tbars.itertuples():
            if row.done:
                continue

            for radius_s0, download_scale, analysis_scale in search_configs:
                loop_logger = PrefixedLogger(logger, f"({row.x}, {row.y}, {row.z}) [ds={download_scale} as={analysis_scale} r={radius_s0:4}] ")

                num_done = _measure_tbar_mito_distances(
                    seg_src, mito_src, body, tbars, row.Index,
                    radius_s0, download_scale, analysis_scale, dilation_radius_s0,
                    loop_logger)
                progress.update(num_done)
                done = (tbars['done'].loc[row.Index])
                if done:
                    break
                loop_logger.info("Search failed for primary tbar. Trying next search config!")

            if not done:
                loop_logger.warn(f"Failed to find a nearby mito for tbar at point {(row.x, row.y, row.z)}")
                progress.update(1)

    num_done = tbars['done'].sum()
    num_failed = (~tbars['done']).sum()
    logger.info(f"Found mitos for {num_done} tbars, failed for {num_failed} tbars")

    return tbars


def _measure_tbar_mito_distances(seg_src, mito_src, body, tbar_points_s0, primary_point_index,
                                 radius_s0, download_scale, analysis_scale, dilation_radius_s0,
                                 logger):
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
            Labelmap instance for the mitochondria "supervoxel"
            segmentation -- not just the "masks".
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
        download_scale:
            Body segmentation will be downloaded as this scale,
            but downsampled according to analysis scale.
        analysis_scale:
            Body and mito segmentation will be analyzed at this scale.
            The body segmentation will be downsampled (if necessary) to match this scale.

    Returns:
        The number of tbars for which a nearby mitochondria was found in this batch.
        (Where "batch" is the set of not-yet-done tbars that overlap with the body mask
        near the "primary" tbar point.)
    """
    assert not tbar_points_s0['done'].loc[primary_point_index]
    primary_point_s0 = tbar_points_s0[[*'zyx']].loc[primary_point_index].values
    batch_tbars = tbar_points_s0.copy()

    body_mask, mask_box, body_block_corners = _fetch_body_mask(
        seg_src, primary_point_s0, radius_s0, download_scale, analysis_scale, body, dilation_radius_s0, batch_tbars[[*'zyx']].values, logger)

    mito_seg = _fetch_body_mito_seg(
        mito_src, body_mask, mask_box, body_block_corners, analysis_scale, logger)

    primary_point = primary_point_s0 // (2**analysis_scale)
    _mark_mito_seg_faces(body_mask, mito_seg, mask_box, primary_point, radius_s0 // (2**analysis_scale))

    if EXPORT_DEBUG_VOLUMES:
        print(f"Primary point in the local volume is: {(primary_point - mask_box[0])[::-1]}")
        p = ' '.join(str(x) for x in primary_point_s0[::-1])
        d = f'/tmp/{p}'
        os.makedirs(d, exist_ok=True)
        np.save(f'{d}/body_mask.npy', body_mask.astype(np.uint64))
        np.save(f'{d}/mito_seg.npy', mito_seg)

    # Body mask should be binary for the rest of this function.
    body_mask = body_mask.astype(bool)

    if not mito_seg.any():
        # The body mask contains no mitochondria at all.
        # Does the body mask come near enough to the volume edge that it
        # could, conceivably, be joined to another part of the body mask
        # outside this block after dilation is performed?

        cr = max(1, dilation_radius_s0 // (2**analysis_scale))
        if ( body_mask[0:cr+1, :, :].any() or body_mask[-cr-1:, :, :].any() or
             body_mask[:, 0:cr+1, :].any() or body_mask[:, -cr-1:, :].any() or
             body_mask[:, :, 0:cr+1].any() or body_mask[:, :, -cr-1:].any() ):
            # The body mask approaches the edge of the volume,
            # so we should expand our radius and keep trying.
            return 0
        else:
            # The mask doesn't even come close to the volume edges.
            # We'll give up, even though we can't find a mito.
            tbar_points_s0.loc[primary_point_index, 'done'] = True
            return 1

    # Find the set of all points that fall within the body mask.
    # That's that batch of tbars we'll find mito distances for.
    batch_tbars = batch_tbars.query('not done')

    batch_tbars[[*'zyx']] //= (2**analysis_scale)
    in_box = (batch_tbars[[*'zyx']] >= mask_box[0]).all(axis=1) & (batch_tbars[[*'zyx']] < mask_box[1]).all(axis=1)
    batch_tbars = batch_tbars.loc[in_box]

    tbars_local = batch_tbars[[*'zyx']] - mask_box[0]
    in_mask = body_mask[tuple(tbars_local.values.transpose())]
    batch_tbars = batch_tbars.iloc[in_mask]
    assert len(batch_tbars) >= 1

    with Timer(f"Calculating distances for batch of {len(batch_tbars)} points", logger):
        tbars_local = batch_tbars[[*'zyx']] - mask_box[0]
        mito_ids, distances, mito_points_local = _calc_distances(body_mask, mito_seg, tbars_local.values, logger)

    mito_points = mito_points_local + mask_box[0]
    batch_tbars['mito-id'] = mito_ids
    batch_tbars['mito-distance'] = distances
    batch_tbars.loc[:, ['mito-z', 'mito-y', 'mito-x']] = mito_points

    # If the closest "mito" to the tbar was actually a "fake mito", inserted
    # onto the face of the volume (indicated by FACE_MARKER id value),
    # then the tbar is closer to the edge of the volume than it is to
    # the closest mito within the volume. We need to try again for those tbars.
    valid = (batch_tbars['mito-id'] != FACE_MARKER)
    logger.info(f"Kept {valid.sum()}/{len(batch_tbars)} mito distances (R={radius_s0})")
    batch_tbars = batch_tbars.loc[valid]

    # Update the input DataFrame (and rescale)
    tbar_points_s0.loc[batch_tbars.index, 'mito-id'] = batch_tbars['mito-id']
    tbar_points_s0.loc[batch_tbars.index, 'mito-distance'] = (2**analysis_scale)*batch_tbars['mito-distance']
    tbar_points_s0.loc[batch_tbars.index, ['mito-z', 'mito-y', 'mito-x']] = (2**analysis_scale)*batch_tbars[['mito-z', 'mito-y', 'mito-x']]
    tbar_points_s0.loc[batch_tbars.index, 'done'] = True

    return len(batch_tbars)


def _fetch_body_mask(seg_src, primary_point_s0, radius_s0, download_scale, analysis_scale, body, dilation_radius_s0, tbar_points_s0, logger):
    """
    Fetch a mask for the given body around the given point, with the given radius.

    The mask will be downloaded using download_scale and then rescaled to
    the analysis_scale using continuity-preserving downsampling.

    The mask will be post-processed in two ways:
        - Dilation is performed to close gaps
        - Only the connected component that covers the given point will be returned.
          If the component doesn't extend out to the given radius in all dimensions,
          then the returned subvolume may be smaller than the requested radius would
          otherwise have required.

    The returned mask is NOT a binary (boolean) volume. Instead, a uint8 volume is returned,
    with labels 1 and 2, indicating which portion of the mask belongs to the body (2) and
    which portion was added due to dilation (1).

    Later, when this mask is used to filter the mito segmentation, only label 1 will be used.
    When it's used to calculate path distances, both labels will be used.
    """
    scale_diff = analysis_scale - download_scale
    with Timer("Fetching body segmentation", logger):
        assert _have_flyemflows and isinstance(seg_src, VolumeService)
        p = np.asarray(primary_point_s0) // (2**download_scale)
        R = radius_s0 // (2**download_scale)
        seg_box = [p-R, p+R+1]

        # Align to 64-px for consistency with the dvid case,
        # and compatibility with the code below.
        seg_box = round_box(seg_box, 64 * (2**scale_diff), 'out')
        p_local = (p - seg_box[0])
        seg = seg_src.get_subvolume(seg_box, download_scale)

    # Extract mask
    raw_mask = (seg == body).view(np.uint8)
    del seg

    # Downsample mask conservatively, i.e. keeping 'on' pixels no matter what
    seg_box //= (2**scale_diff)
    p //= (2**scale_diff)
    p_local //= (2**scale_diff)
    raw_mask = downsample_mask(raw_mask, 2**scale_diff, 'or')
    pstr = ' '.join(str(x) for x in primary_point_s0[::-1])
    os.makedirs(f'/tmp/{pstr}', exist_ok=True)
    np.save(f'/tmp/{pstr}/raw_mask.npy', raw_mask)

    # Due to downsampling effects in the original data, it's possible
    # that the main tbar fell off its body in the downsampled image.
    # Make sure it's part of the mask
    raw_mask[(*p_local,)] = True

    # The same is true for all of the other tbars that fall within the mask box.
    # Fix them all.
    tbar_points = tbar_points_s0 // (2**analysis_scale)
    in_box = (tbar_points >= seg_box[0]).all(axis=1) & (tbar_points < seg_box[1]).all(axis=1)
    tbar_points = tbar_points[in_box]
    local_tbar_points = tbar_points - seg_box[0]
    raw_mask[(*local_tbar_points.transpose(),)] = True

    assert raw_mask.dtype == bool
    raw_mask = vigra.taggedView(raw_mask.view(np.uint8), 'zyx')

    # Perform light dilation on the mask to fix gaps in the
    # segmentation due to hot knife seams, downsampling, etc.
    if dilation_radius_s0 == 0:
        dilated_mask = raw_mask
    else:
        dilation_radius = max(1, dilation_radius_s0 // (2**analysis_scale))
        dilated_mask = vigra.filters.multiBinaryDilation(raw_mask, dilation_radius)

    if DEBUG_BODY_MASK:
        cc_mask = dilated_mask
    else:
        # Find the connected component that contains our point of interest
        # amd limit our analysis to those voxels.
        body_cc = labelMultiArrayWithBackground(dilated_mask)

        # Keep only the main CC.
        cc_mask = (body_cc == body_cc[(*p_local,)]).view(np.uint8)

    # Label the voxels:
    # 1: dilated mask (main CC only)
    # 2: dilated mask (main CC only) AND raw
    body_mask = np.where(cc_mask, raw_mask + cc_mask, 0)

    # Shrink the volume size to fit the data, but align box to nearest 64px
    body_block_mask = view_as_blocks(body_mask, (64,64,64)).any(axis=(3,4,5))
    body_block_corners = seg_box[0] + (64 * np.argwhere(body_block_mask))
    mask_box = (body_block_corners.min(axis=0),
                body_block_corners.max(axis=0) + 64)

    local_box = mask_box - seg_box[0]
    body_mask = body_mask[box_to_slicing(*local_box)]

    assert body_mask[(*p - mask_box[0],)]
    return body_mask, mask_box, body_block_corners


def _fetch_body_mito_seg(mito_src, body_mask, mask_box, body_block_corners, scale, logger):
    """
    Return the mito segmentation for only those mitos which
    overlap with the given body mask (not elsewhere).

    Args:
        mito_src:
            VolumeService to obtain mito segmentation
        body_mask:
            Volume with labels 1+2 as described in _fetch_body_mask()
    """
    with Timer("Fetching mito segmentation", logger):
        assert _have_flyemflows and isinstance(mito_src, VolumeService)
        mito_seg = mito_src.get_subvolume(mask_box, scale)

    core_body_mask = (body_mask == 2)
    body_mito_seg = np.where(core_body_mask, mito_seg, 0)

    # Due to downsampling discrepancies between the mito seg and neuron seg,
    # mito from neighboring neurons may slightly overlap this neuron.
    # Keep only mitos which have more of their voxels in the body mask than not.
    body_mito_sizes = pd.Series(body_mito_seg.ravel()).value_counts()
    del body_mito_seg
    mito_sizes = pd.Series(mito_seg.ravel()).value_counts()
    mito_sizes, body_mito_sizes = mito_sizes.align(body_mito_sizes, fill_value=0)
    core_mitos = {*mito_sizes[(body_mito_sizes > mito_sizes / 2)].index} - {0}
    core_mito_seg = apply_mask_for_labels(mito_seg, core_mitos, inplace=True)
    return core_mito_seg


def _mark_mito_seg_faces(body_mask, mito_seg, mask_box, primary_point, radius):
    """
    Create "fake mitos" on the edge of the mito volume,
    where the neuron exits the analysis volume.

    The analysis volume is defined by the center point and radius.
    Our actual mask volumes are often smaller, if the body doesn't fill the analysis box.
    But in cases where the mask volume is smaller than the analysis box,
    we don't want to mark edge voxels.  The mask box was truncated intentionally because
    we determined that the body doesn't actually extend beyond it.

    Note:
        We take pains here to avoid marking the edge of neurons that just barely
        touch the edge of the mask, but do not extend outside of it.
        That's why we care so much about the difference between the "analysis box"
        vs. the "mask box".
    """
    assert tuple(mask_box[1] - mask_box[0]) == body_mask.shape == mito_seg.shape
    p = primary_point
    r = radius
    mask_box = np.asarray(mask_box)
    radius_box = np.array([p - r, p + r + 1])  # analysis box
    mito_mask = mito_seg.astype(bool)

    for axis in [0,1,2]:
        if mask_box[0, axis] <= radius_box[0, axis]:
            sl = list(np.s_[:, :, :])
            sl[axis] = 0
            sl = tuple(sl)
            body_face = body_mask[sl].astype(bool)
            mito_face = mito_mask[sl]
            mito_seg[sl] = np.where(body_face & ~mito_face, FACE_MARKER, mito_seg[sl])

        if mask_box[1, axis] >= radius_box[1, axis]:
            sl = list(np.s_[:, :, :])
            sl[axis] = -1
            sl = tuple(sl)
            body_face = body_mask[sl].astype(bool)
            mito_face = mito_mask[sl]
            mito_seg[sl] = np.where(body_face & ~mito_face, FACE_MARKER, mito_seg[sl])


def _calc_distances(body_mask, mito_seg, local_points_zyx, logger):
    """
    Calculate the distances from a set of mito segments to a set of
    input points (tbars), restricting paths to the given body mask.

    Uses skimage.graph.MCP_Geometric to perform the graph search.

    Returns:
        (mito_ids, distances, mito_point_zyx)
        Three arrays, each ordered corresponding to the input points.

        Where:
            - mito_ids indicates the closest mito to each input point,
            - distances indicates the path distance from each point to
              its closest mito segments
            - mito_point_zyx indicates the "starting point" on the mito segment
              that yielded the closest distance to the target tbar point.
    """
    if mito_seg.sum() == 0:
        return 0, np.inf, (0,0,0)

    # MCP uses float64, so we may as well use that now and avoid copies
    body_costs = np.where(body_mask, 1.0, np.inf)

    # Insanely, MCP source code copies to fortran order internally,
    # so let's pass in fortran order to start with.
    # That shaves ~10-20% from the initialization time.
    body_costs = body_costs.transpose()
    mito_seg = mito_seg.transpose()
    local_points_xyz = np.asarray(local_points_zyx)[:, ::-1]

    mcp = MCP_Geometric(body_costs)
    distance_vol, _ = mcp.find_costs(np.argwhere(mito_seg), local_points_xyz)
    point_distances = distance_vol[(*local_points_xyz.transpose(),)]

    # The necessary path traceback data is retained
    # in MCP internally after find_costs() above.
    p_mito_xyz = np.zeros((len(local_points_xyz), 3), np.int16)

    for i, (p_xyz, d) in enumerate(zip(local_points_xyz, point_distances)):
        if d != np.inf:
            p_mito_xyz[i] = mcp.traceback(p_xyz)[0]

    mito_ids = mito_seg[tuple(p_mito_xyz.transpose())]
    mito_ids[np.isinf(point_distances)] = 0

    # Transpose back to C-order
    p_mito_zyx = p_mito_xyz[:, ::-1]
    return mito_ids, point_distances, p_mito_zyx


if __name__ == "__main__":
    from neuprint import Client
    from neuclease import configure_default_logging
    configure_default_logging()

    c = Client('neuprint.janelia.org', 'hemibrain:v1.1')

    body = 519046655
    tbars = fetch_synapses(body, SC(rois='FB', type='pre', primary_only=True))
    tbars = tbars.iloc[:10]

    # EXPORT_DEBUG_VOLUMES = True
    # body = 295474876
    # tbars = fetch_synapses(body, SC(type='pre', primary_only=True))
    # selections = (tbars[[*'xyz']] == (24721, 21717, 22518)).all(axis=1)
    # print(selections.sum())
    # tbars = tbars.loc[selections]

    # EXPORT_DEBUG_VOLUMES = True
    # body = 1002848124
    # tbars = fetch_synapses(body, SC(type='pre', primary_only=True))
    # selections = (tbars[[*'xyz']] == (18212,12349,16592)).all(axis=1)
    # tbars = tbars.loc[selections]

    # DEBUG_BODY_MASK = True
    # EXPORT_DEBUG_VOLUMES = True
    # body = 203253072
    # tbars = fetch_synapses(body, SC(type='pre', primary_only=True))
    # selections = (tbars[[*'xyz']] == (21362,23522,15106)).all(axis=1)
    # tbars = tbars.loc[selections]

    DEBUG_BODY_MASK = True
    EXPORT_DEBUG_VOLUMES = True
    body = 1005308608
    tbars = fetch_synapses(body, SC(type='pre', primary_only=True))
    selections = (tbars[[*'xyz']] == (25435,26339,21900)).all(axis=1)
    tbars = tbars.loc[selections]

    seg_cfg = {
        "zarr": {
            "path": "/Users/bergs/data/hemibrain-v1.2.zarr",
            "dataset": "s2",
            "store-type": "NestedDirectoryStore",
            "out-of-bounds-access": "permit-empty"
        },
        "adapters": {
            "rescale-level": -2
        }
    }
    mito_cfg = {
        "zarr": {
            "path": "/Users/bergs/data/hemibrain-v1.2-filtered-mito-cc.zarr",
            "dataset": "s3",
            "store-type": "NestedDirectoryStore",
            "out-of-bounds-access": "permit-empty"
        },
        "adapters": {
            "rescale-level": -3
        }
    }
    seg_svc = VolumeService.create_from_config(seg_cfg)
    mito_svc = VolumeService.create_from_config(mito_cfg)
    processed_tbars = measure_tbar_mito_distances(seg_svc, mito_svc, body, npclient=c, tbars=tbars, dilation_radius_s0=32)
    print(processed_tbars)
