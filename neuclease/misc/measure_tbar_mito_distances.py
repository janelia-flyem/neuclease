import os
import logging
from collections import namedtuple

import numpy as np
import pandas as pd

from skimage.graph import MCP_Geometric

import vigra
from vigra.analysis import labelMultiArrayWithBackground

from dvidutils import LabelMapper
from neuprint import SynapseCriteria as SC, fetch_synapses
from neuclease.util import tqdm_proxy, Timer, box_to_slicing, round_box, apply_mask_for_labels, downsample_mask, compute_nonzero_box, mask_for_labels, box_intersection
from neuclease.logging_setup import PrefixedLogger

try:
    from flyemflows.volumes import VolumeService
    _have_flyemflows = True
except ImportError:
    _have_flyemflows = False

logger = logging.getLogger(__name__)
EXPORT_DEBUG_VOLUMES = False
SearchConfig = namedtuple('SearchConfig', 'radius_s0 download_scale analysis_scale dilation_radius_s0 dilation_exclusion_buffer_s0')

DEFAULT_SEARCH_CONFIGS = [
    SearchConfig(radius_s0=250,  download_scale=1, analysis_scale=3, dilation_radius_s0=0, dilation_exclusion_buffer_s0=0),    #  2 microns (empirically, this captures ~90% of FB tbars)
    SearchConfig(radius_s0=625,  download_scale=1, analysis_scale=3, dilation_radius_s0=0, dilation_exclusion_buffer_s0=0),    #  5 microns
    SearchConfig(radius_s0=1250, download_scale=2, analysis_scale=3, dilation_radius_s0=16, dilation_exclusion_buffer_s0=625)  # 10 microns
]

# Fake mito ID to mark the places where a body exits the analysis volume.
# Must not conflict with any real mito IDs
FACE_MARKER = np.uint64(1e15-1)


def measure_tbar_mito_distances(seg_src,
                                mito_src,
                                body,
                                *,
                                search_configs=DEFAULT_SEARCH_CONFIGS,
                                npclient=None,
                                tbars=None,
                                valid_mitos=None):
    """
    Search for the closest mito to each tbar in a list of tbars
    (or any set of points, really).

    FIXME: Rename this function.  It works for more than just tbars.

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
            A list ``SearchConfig`` tuples.
            For each tbar, this function tries to locate a mitochondria within
            he given search radius, using data downloaded from the given scale.
            If the search fails and no mito can be found, the function tries
            again using the next search criteria in the list.
            The radius should always be specified in scale-0 units,
            regardless of the scale at which you want to perform the analysis.
            Additionally, the data will be downloaded at the specified scale,
            then downsampled (with continuity preserving downsampling) to a lower scale for analysis.
            Notes:
                - Scale 4 is too low-res.  Stick with scale-3 or better.
                - Higher radius is more expensive, but some of that expense is
                  recouped because all points that fall within the radius are
                  analyzed at once.  See _measure_tbar_mito_distances()
                  implementation for details.
            dilation_radius_s0:
                If dilation_radius_s0 is non-zero, the segmentation will be "repaired" to close
                gaps, using a procedure involving a dilation of the given radius.
            dilation_exclusion_buffer_s0:
                We want to close small gaps in the segmentation, but only if we think
                they're really a gap in the actual segmentation, not if they are merely
                fingers of the same branch that are actually connected outside of our
                analysis volume. The dilation procedure tends to form such spurious
                connections near the volume border, so this parameter can be used to
                exclude a buffer (inner halo) near the border from dilation repairs.
        npclient:
            ``neuprint.Client`` to use when fetching the list of tbars that belong
            to the given body, unless you provide your own tbar points in the next
            argument.
        tbars:
            A DataFrame of tbar coordinates at least with columns ``['x', 'y', 'z']``.
        valid_mitos:
            If provided, only the listed mito IDs will be considered valid as search targets.
    Returns:
        DataFrame of tbar coordinates, mito distances, and mito coordinates.
        Points for which no nearby mito could be found (after trying all the given search_configs)
        will be marked with `done=False` in the results.
    """
    # Fetch tbars
    if tbars is None:
        tbars = fetch_synapses(body, SC(type='pre', primary_only=True), client=npclient)

    tbars = initialize_results(body, tbars)

    if valid_mitos is None or len(valid_mitos) == 0:
        valid_mito_mapper = None
    else:
        valid_mitos = np.asarray(valid_mitos, dtype=np.uint64)
        valid_mito_mapper = LabelMapper(valid_mitos, valid_mitos)

    with tqdm_proxy(total=len(tbars)) as progress:
        for row in tbars.itertuples():
            # can't use row.done -- itertuples might be out-of-sync
            done = (tbars['done'].loc[row.Index])
            if done:
                continue

            loop_logger = None
            for cfg in search_configs:
                (radius_s0, download_scale, analysis_scale,
                    dilation_radius_s0, dilation_exclusion_buffer_s0) = cfg

                prefix = f"({row.x}, {row.y}, {row.z}) [ds={download_scale} as={analysis_scale} r={radius_s0:4} dil={dilation_radius_s0:2}] "
                loop_logger = PrefixedLogger(logger, prefix)

                num_done = _measure_tbar_mito_distances(
                    seg_src, mito_src, body, tbars, row.Index,
                    radius_s0, download_scale, analysis_scale,
                    dilation_radius_s0, dilation_exclusion_buffer_s0,
                    valid_mito_mapper, loop_logger)

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


def initialize_results(body, tbars):
    tbars = tbars.copy()
    tbars['body'] = body
    tbars['mito-distance'] = np.inf
    tbars['done'] = False
    tbars['mito-x'] = 0
    tbars['mito-y'] = 0
    tbars['mito-z'] = 0
    tbars['mito-id'] = np.uint64(0)
    tbars['search-radius'] = np.int32(0)
    tbars['download-scale'] = np.int8(0)
    tbars['analysis-scale'] = np.int8(0)
    tbars['crossed-gap'] = False
    tbars['focal-x'] = 0
    tbars['focal-y'] = 0
    tbars['focal-z'] = 0
    return tbars


def _measure_tbar_mito_distances(seg_src, mito_src, body, tbar_points_s0, primary_point_index,
                                 radius_s0, download_scale, analysis_scale,
                                 dilation_radius_s0, dilation_exclusion_buffer_s0,
                                 valid_mito_mapper, logger):
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
    batch_tbars = tbar_points_s0.query('not done').copy()

    body_mask, mask_box = _fetch_body_mask(
        seg_src, primary_point_s0, radius_s0, download_scale, analysis_scale, body,
        dilation_radius_s0, dilation_exclusion_buffer_s0, batch_tbars[[*'zyx']].values, logger)

    mito_seg = _fetch_body_mito_seg(
        mito_src, body_mask, mask_box, analysis_scale, valid_mito_mapper, logger)

    primary_point = primary_point_s0 // (2**analysis_scale)

    p = d = orig_mask_box = None
    if EXPORT_DEBUG_VOLUMES:
        print(f"Primary point in the local volume is: {(primary_point - mask_box[0])[::-1]}")
        p = '-'.join(str(x) for x in primary_point_s0[::-1])
        d = f'/tmp/{p}'
        os.makedirs(d, exist_ok=True)
        np.save(f'{d}/body_mask_unfiltered.npy', body_mask.astype(np.uint64))
        np.save(f'{d}/mito_seg_unfiltered.npy', mito_seg)
        orig_mask_box = mask_box

    if not mito_seg.any():
        # The body mask contains no mitochondria at all.
        # Does the body mask come near enough to the volume edge that it
        # could, conceivably, be joined to another part of the body mask
        # outside this block after dilation is performed?

        cr = max(1, dilation_radius_s0 // (2**analysis_scale))
        if ( body_mask[0:cr+1, :, :].any() or body_mask[-cr-1:, :, :].any() or  # noqa
             body_mask[:, 0:cr+1, :].any() or body_mask[:, -cr-1:, :].any() or  # noqa
             body_mask[:, :, 0:cr+1].any() or body_mask[:, :, -cr-1:].any() ):
            # The body mask approaches the edge of the volume,
            # so we should expand our radius and keep trying.
            return 0
        else:
            # The mask doesn't even come close to the volume edges.
            # We'll give up, even though we can't find a mito.
            tbar_points_s0.loc[primary_point_index, 'done'] = True
            return 1

    body_mask, mito_seg, mask_box = _crop_body_mask_and_mito_seg(
        body_mask, mito_seg, mask_box, analysis_scale, batch_tbars[[*'zyx']].values, logger)

    if body_mask is None:
        # Nothing left after cropping.
        # Give up on this search config, but try again.
        return 0

    _mark_mito_seg_faces(
        body_mask, mito_seg, mask_box, primary_point, radius_s0 // (2**analysis_scale))

    if EXPORT_DEBUG_VOLUMES:
        export_shape = orig_mask_box[1] - orig_mask_box[0]
        export_box = mask_box - orig_mask_box[0]
        body_mask_filtered = np.zeros(export_shape, np.uint64)
        body_mask_filtered[box_to_slicing(*export_box)] = body_mask

        mito_seg_filtered = np.zeros(export_shape, np.uint64)
        mito_seg_filtered[box_to_slicing(*export_box)] = mito_seg

        np.save(f'{d}/body_mask_filtered.npy', body_mask_filtered)
        np.save(f'{d}/mito_seg_filtered.npy', mito_seg_filtered)

    # Body mask should be binary for the rest of this function.
    body_mask = body_mask.astype(bool)

    # Find the set of all points that fall within the body mask.
    # That's that batch of tbars we'll find mito distances for.
    batch_tbars[[*'zyx']] //= (2**analysis_scale)
    in_box = (batch_tbars[[*'zyx']] >= mask_box[0]).all(axis=1) & (batch_tbars[[*'zyx']] < mask_box[1]).all(axis=1)
    batch_tbars = batch_tbars.loc[in_box]

    tbars_local = batch_tbars[[*'zyx']] - mask_box[0]
    in_mask = body_mask[tuple(tbars_local.values.transpose())]
    batch_tbars = batch_tbars.iloc[in_mask]
    assert len(batch_tbars) >= 1, f"Lost all tbars around {primary_point_s0[::-1]}"

    with Timer(f"Calculating distances for batch of {len(batch_tbars)} points", logger):
        tbars_local = batch_tbars[[*'zyx']] - mask_box[0]
        mito_ids, distances, mito_points_local, crossed_gaps = _calc_distances(body_mask, mito_seg, tbars_local.values, logger)

    mito_points = mito_points_local + mask_box[0]
    batch_tbars['mito-id'] = mito_ids
    batch_tbars['mito-distance'] = distances
    batch_tbars.loc[:, ['mito-z', 'mito-y', 'mito-x']] = mito_points
    batch_tbars['crossed-gap'] = crossed_gaps

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
    tbar_points_s0.loc[batch_tbars.index, 'search-radius'] = radius_s0
    tbar_points_s0.loc[batch_tbars.index, 'download-scale'] = download_scale
    tbar_points_s0.loc[batch_tbars.index, 'analysis-scale'] = analysis_scale
    tbar_points_s0.loc[batch_tbars.index, 'crossed-gap'] = batch_tbars['crossed-gap']
    tbar_points_s0.loc[batch_tbars.index, ['focal-z', 'focal-y', 'focal-x']] = primary_point_s0[None, :]
    tbar_points_s0.loc[batch_tbars.index, 'done'] = True

    return len(batch_tbars)


def _fetch_body_mask(seg_src, primary_point_s0, radius_s0, download_scale, analysis_scale, body,
                     dilation_radius_s0, dilation_exclusion_buffer_s0, tbar_points_s0, logger):
    """
    Fetch a mask for the given body around the given point, with the given radius.

    The mask will be downloaded using download_scale and then rescaled to
    the analysis_scale using continuity-preserving downsampling.

    The returned mask is NOT a binary (boolean) volume. Instead, a uint8 volume is returned,
    with labels 1 and 2, indicating which portion of the mask belongs to the body (2) and
    which portion was added due to dilation (1).

    Providing a non-binary result is convenient for debugging.  It is also used to
    restrict the voxels that are preserved when filtering mitos, if no valid_mitos
    are provided to that function.
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

    dilation_buffer = dilation_exclusion_buffer_s0 // (2**analysis_scale)
    dilation_box = np.array((seg_box[0] + dilation_buffer, seg_box[1] - dilation_buffer))

    # Shrink to fit the data.
    local_box = compute_nonzero_box(raw_mask)
    raw_mask = raw_mask[box_to_slicing(*local_box)]
    mask_box = local_box + seg_box[0]
    p_local -= local_box[0]

    # Fill gaps
    repaired_mask = _fill_gaps(raw_mask, mask_box, analysis_scale, dilation_radius_s0, dilation_box)

    # Label the voxels:
    # 1: filled gaps
    # 2: filled gaps AND raw
    assert raw_mask.dtype == repaired_mask.dtype == np.uint8
    body_mask = np.where(repaired_mask, raw_mask + repaired_mask, 0)

    assert body_mask[(*p - mask_box[0],)]
    return body_mask, mask_box


def _fill_gaps(mask, mask_box, analysis_scale, dilation_radius_s0, dilation_box):
    """
    Fill gaps between segments in the mask by dilating each segment
    and keeping the voxels that were covered by more than one dilation.
    """
    # Perform light dilation on the mask to fix gaps in the
    # segmentation due to hot knife seams, downsampling, etc.
    if dilation_radius_s0 == 0:
        return mask

    # We limit the dilation repair to a central box, to avoid joining
    # dendrites that just barely enter the volume in multiple places.
    # We only want to make repairs that aren't near the volume edge.
    dilation_box = box_intersection(mask_box, dilation_box)
    if (dilation_box[1] - dilation_box[0] <= 0).any():
        return mask

    # Perform dilation on each connected component independently,
    # and mark the areas where two dilated components overlap.
    # We'll add those overlapping voxels to the mask, to span
    # small gap defects in the segmentation.
    cc = labelMultiArrayWithBackground((mask != 0).view(np.uint8))
    cc_max = cc.max()
    if cc_max <= 1:
        return mask

    central_box = dilation_box - mask_box[0]
    cc_central = cc[box_to_slicing(*central_box)]

    dilation_radius = dilation_radius_s0 // (2**analysis_scale)
    dilated_intersections = np.zeros(cc_central.shape, bool)
    dilated_all = vigra.filters.multiBinaryDilation((cc_central == 1), dilation_radius)
    for i in range(2, cc_max+1):
        cc_dilated = vigra.filters.multiBinaryDilation((cc_central == i), dilation_radius)
        dilated_intersections[:] |= (dilated_all & cc_dilated)
        dilated_all[:] |= cc_dilated

    # Return a new array; don't modify the original in-place.
    mask = mask.astype(bool, copy=True)
    mask[box_to_slicing(*central_box)] |= dilated_intersections
    return mask.view(np.uint8)


def _fetch_body_mito_seg(mito_src, body_mask, mask_box, scale, valid_mito_mapper, logger):
    """
    Return the mito segmentation for only those mitos which
    overlap with the given body mask (not elsewhere).

    Args:
        mito_src:
            VolumeService to obtain mito segmentation
        body_mask:
            Volume with labels 1+2 as described in _fetch_body_mask()
        valid_mito_mapper:
            LabelMapper that keeps only valid mitos when its apply_with_default() method is called.
    """
    with Timer("Fetching mito segmentation", logger):
        assert _have_flyemflows and isinstance(mito_src, VolumeService)
        mito_seg = mito_src.get_subvolume(mask_box, scale)

    if valid_mito_mapper:
        return valid_mito_mapper.apply_with_default(mito_seg)

    core_body_mask = (body_mask == 2)
    body_mito_seg = np.where(core_body_mask, mito_seg, 0)

    # Due to downsampling discrepancies between the mito seg and neuron seg,
    # mito from neighboring neurons may slightly overlap this neuron.
    # Keep only mitos which have more of their voxels in the body mask than not.
    #
    # FIXME:
    #   This heuristic fails at the volume edge, where we might see just
    #   part of the mito.
    #   Need to overwrite small mitos on the volume edge with FACE_MARKER
    #   to indicate that they can't be trusted, and if such a mito is
    #   the "winning" mito, then we need to try a different search config.
    body_mito_sizes = pd.Series(body_mito_seg.ravel()).value_counts()
    del body_mito_seg
    mito_sizes = pd.Series(mito_seg.ravel()).value_counts()
    mito_sizes, body_mito_sizes = mito_sizes.align(body_mito_sizes, fill_value=0)
    core_mitos = {*mito_sizes[(body_mito_sizes > mito_sizes / 2)].index} - {0}
    core_mito_seg = apply_mask_for_labels(mito_seg, core_mitos, inplace=True)
    return core_mito_seg


def _crop_body_mask_and_mito_seg(body_mask, mito_seg, mask_box, analysis_scale, tbar_points_s0, logger):
    """
    To reduce the size of the analysis volumes during distance computation
    (the most expensive step), we pre-filter out components of the body mask
    don't actually contain both points of interest and mito.
    """
    body_cc = labelMultiArrayWithBackground((body_mask != 0).view(np.uint8))

    # Keep only components which contain both mito and points
    tbar_points = tbar_points_s0 // (2**analysis_scale)
    in_box = (tbar_points >= mask_box[0]).all(axis=1) & (tbar_points < mask_box[1]).all(axis=1)
    tbar_points = tbar_points[in_box]
    pts_local = tbar_points - mask_box[0]
    point_ccs = pd.unique(body_cc[tuple(np.transpose(pts_local))])
    mito_ccs = pd.unique(body_cc[mito_seg != 0])
    keep_ccs = set(point_ccs) & set(mito_ccs)
    keep_mask = mask_for_labels(body_cc, keep_ccs)

    body_mask = np.where(keep_mask, body_mask, 0)
    mito_seg = np.where(keep_mask, mito_seg, 0)
    logger.info(f"Dropped {body_cc.max() - len(keep_ccs)} components, kept {len(keep_ccs)}")

    # Shrink the volume bounding box to encompass only the
    # non-zero portion of the filtered body mask.
    nz_box = compute_nonzero_box(keep_mask)
    if not nz_box.any():
        return None, None, nz_box

    body_mask = body_mask[box_to_slicing(*nz_box)]
    mito_seg = mito_seg[box_to_slicing(*nz_box)]
    mask_box = mask_box[0] + nz_box
    return body_mask, mito_seg, mask_box


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
    body_mask = body_mask.transpose()
    body_costs = body_costs.transpose()
    mito_seg = mito_seg.transpose()
    local_points_xyz = np.asarray(local_points_zyx)[:, ::-1]

    mcp = MCP_Geometric(body_costs)
    distance_vol, _ = mcp.find_costs(np.argwhere(mito_seg), local_points_xyz)
    point_distances = distance_vol[(*local_points_xyz.transpose(),)]

    # The necessary path traceback data is retained
    # in MCP internally after find_costs() above.
    p_mito_xyz = np.zeros((len(local_points_xyz), 3), np.int16)

    crossed_gaps = np.zeros(len(local_points_xyz), bool)
    for i, (p_xyz, d) in enumerate(zip(local_points_xyz, point_distances)):
        if d != np.inf:
            tb = np.asarray(mcp.traceback(p_xyz))
            p_mito_xyz[i] = tb[0]

            # If the traceback had to use a gap-crossing, note it.
            # See the meaning of body_mask label 1 vs. label 2, above.
            # (label 1 means it's a filled gap voxel)
            crossed_gaps[i] = (body_mask[tuple(tb.transpose())] == 1).any()

    mito_ids = mito_seg[tuple(p_mito_xyz.transpose())]
    mito_ids[np.isinf(point_distances)] = 0

    # Transpose back to C-order
    p_mito_zyx = p_mito_xyz[:, ::-1]
    return mito_ids, point_distances, p_mito_zyx, crossed_gaps


if __name__ == "__main__":
    from neuprint import Client
    from neuclease.dvid import fetch_label, fetch_supervoxels
    from neuclease import configure_default_logging
    configure_default_logging()

    c = Client('neuprint.janelia.org', 'hemibrain:v1.2')

    # body = 519046655
    # tbars = fetch_synapses(body, SC(rois='FB', type='pre', primary_only=True))
    # tbars = tbars.iloc[:10]

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

    # EXPORT_DEBUG_VOLUMES = True
    # body = 203253072
    # tbars = fetch_synapses(body, SC(type='pre', primary_only=True))
    # selections = (tbars[[*'xyz']] == (21362,23522,15106)).all(axis=1)
    # tbars = tbars.loc[selections]

    EXPORT_DEBUG_VOLUMES = True
    body = 5813105172  # DPM neuron -- very dense
    tbars = fetch_synapses(body, SC(type='pre', primary_only=True))
    tbars = tbars.iloc[:10]

    # EXPORT_DEBUG_VOLUMES = True
    # #body = 1005308608
    # body = 2178626284
    # #tbars = fetch_synapses(body, SC(type='pre', primary_only=True))
    # tbars = fetch_label('emdata4:8900', '3159', 'synapses', body, format='pandas')[[*'xyz', 'conf', 'kind']]
    # #selections = (tbars[[*'xyz']] == (25435,26339,21900)).all(axis=1)
    # #tbars = tbars.loc[selections]

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

    valid_mitos = fetch_supervoxels('emdata4:8900', '3159', 'mito-objects', body)

    processed_tbars = measure_tbar_mito_distances(seg_svc, mito_svc, body, tbars=tbars, valid_mitos=valid_mitos)
    cols = ['bodyId', 'type', *'xyz', 'mito-distance', 'done', 'mito-id', 'mito-x', 'mito-y', 'mito-z', 'search-radius', 'download-scale', 'analysis-scale']
    print(processed_tbars[cols])
    processed_tbars.to_csv('/tmp/tbar-test-results.csv')
