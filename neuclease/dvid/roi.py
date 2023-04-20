import logging
from functools import partial
from collections.abc import Mapping

import ujson
import numpy as np
import pandas as pd

from ..util import Timer, tqdm_proxy, extract_labels_from_volume, box_shape, extract_subvol, box_intersection, compute_parallel
from . import dvid_api_wrapper, fetch_generic_json
from .rle import runlength_decode_from_ranges, runlength_decode_from_ranges_to_mask, runlength_encode_mask_to_ranges

logger = logging.getLogger(__name__)

@dvid_api_wrapper
def fetch_roi(server, uuid, instance, format='ranges', *, mask_box=None, session=None): # @ReservedAssignment
    """
    Fetch an ROI from dvid.
    
    Note: This function returns coordinates (or masks, etc.) at SCALE 5,
          since that the resolution at which DVID stores ROIs.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid ROI instance name, e.g. 'antenna-lobe'
        
        format:
            Determines the format of the return value, as described below.
            Either 'ranges', 'coords' or 'mask'.

        mask_box:
            Only valid when format='mask'.
            If provided, specifies the box within which the ROI mask should
            be returned, in scale-5 coordinates.
            Voxels outside the box will be omitted from the returned mask.

        Returns:
            If 'ranges':
                np.ndarray, [[Z,Y,X0,X1], [Z,Y,X0,X1], ...]
                Return the RLE block ranges as received from DVID.
                Note: By DVID conventions, the interval [X0,X1] is inclusive,
                      i.e. X1 is IN the range -- not one beyond the range,
                      which would normally be the Python convention.

            If 'coords':
                np.ndarray, [[Z,Y,X], [Z,Y,X], ...]
                Expand the ranges into a list of ROI-block coordinates (scale 5).

            If 'mask':
                (mask, mask_box)
                Return a binary mask of the ROI, where each voxel represents one ROI block (scale 5).
                The mask will be cropped to the bounding box of the ROI,
                and the bounding box is also returned.

            If 'raw':
                Just return the raw response from the server.
                convenient for copying ROIs from one server to another.
    """
    assert format in ('coords', 'ranges', 'mask', 'raw')
    if mask_box is not None:
        mask_box =  np.asarray(mask_box)

    endpoint = f'{server}/api/node/{uuid}/{instance}/roi'
    if format == 'raw':
        r = session.get(endpoint)
        r.raise_for_status()
        return r.content

    rle_ranges = fetch_generic_json(endpoint, session=session)
    rle_ranges = np.asarray(rle_ranges, np.int32, order='C')

    # Special cases for empty ROI
    if len(rle_ranges) == 0:
        if format == 'ranges':
            return np.ndarray( (0,4), np.int32 )

        if format == 'coords':
            return np.ndarray( (0,3), np.int32 )

        if format == 'mask':
            if mask_box is None:
                mask_box = np.array([[0,0,0], [0,0,0]], np.int32)
            mask = np.ndarray( box_shape(mask_box), np.int32 )
            return mask, mask_box

        assert False, "Shouldn't get here"

    assert rle_ranges.shape[1] == 4
    if format == 'ranges':
        return rle_ranges

    if format == 'coords':
        return runlength_decode_from_ranges(rle_ranges)

    if format == 'mask':
        mask, mask_box = runlength_decode_from_ranges_to_mask(rle_ranges, mask_box)
        return mask, mask_box

    assert False, "Shouldn't get here."

# Synonym, to avoid conflicts with annotation.fetch_roi()
fetch_roi_roi = fetch_roi


@dvid_api_wrapper
def post_roi(server, uuid, instance, roi_ranges, *, session=None):
    """
    Post a set of RLE ranges to DVID as an ROI.
    The ranges must be provided in SCALE-5 coordinates.

    For generating RLE ranges from a list of coordinates, see:
        neuclease.dvid.rle.runlength_encode_to_ranges()

    Note:
        When you post to an ROI instance, the old ROI contents are completely erased.
        There is no way to partially write an ROI and then expand it later.
        You're always overwriting the whole thing.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid ROI instance name, e.g. 'antenna-lobe'

        ranges:
            list or ndarray of ranges, specified in SCALE-5 coordinates:
            [[Z,Y,X0,X1], [Z,Y,X0,X1], ...]

            Alternatively, you can pass the ranges as pre-encoded JSON bytes,
            which is convenient when copying ROIs from one server to another.
            See fetch_roi(..., format='raw').
    """
    if isinstance(roi_ranges, bytes):
        # The caller is providing pre-encoded bytes
        encoded_ranges = roi_ranges
    else:
        if isinstance(roi_ranges, np.ndarray):
            roi_ranges = roi_ranges.tolist()
        encoded_ranges = ujson.dumps(roi_ranges)

    r = session.post(f'{server}/api/node/{uuid}/{instance}/roi', data=encoded_ranges)
    r.raise_for_status()


@dvid_api_wrapper
def post_roi_from_mask(server, uuid, instance, mask, mask_box=None, *, session):
    """
    Same as ``post_roi()``, but takes a binary mask
    volume as input, rather than pre-formatted ranges.
    """
    ranges = runlength_encode_mask_to_ranges(mask, mask_box)
    post_roi(server, uuid, instance, ranges, session=session)


@dvid_api_wrapper
def fetch_roi_size(server, uuid, rois, *, processes=1, session=None):
    """
    Return the size (at scale-5) of the given roi or rois.
    If a single name is passed as a string, the ROI size is returned.
    If a list of strings is passed, all of the sizes are returned as a pd.Series
    """
    if isinstance(rois, str):
        return _fetch_roi_size(server, uuid, rois, session=session)[1]

    _fn = partial(_fetch_roi_size, server, uuid)
    roi_sizes = compute_parallel(_fn, rois, ordered=False, processes=processes)
    roi_sizes = pd.DataFrame(roi_sizes, columns=['roi', 'size'])
    roi_sizes = roi_sizes.set_index('roi')['size']
    return roi_sizes.loc[rois]


def _fetch_roi_size(server, uuid, roi, session=None):
    ranges = fetch_roi(server, uuid, roi, 'ranges', session=session)
    z, y, x0, x1 = ranges.transpose()
    return roi, (x1 + 1 - x0).sum()


@dvid_api_wrapper
def fetch_combined_roi_volume(server, uuid, rois, as_bool=False, box_zyx=None, *, session=None, processes=0):
    """
    Fetch several ROIs from DVID and combine them into a single label volume or mask.
    The label values in the returned volume correspond to the order in which the ROI
    names were passed in, starting at label 1.

    If the ROIs overlap, the location of the overlapping voxels is not preserved in the result,
    but a list of detected overlapping ROI pairs is included in the results.

    Note: All results are returned at SCALE 5, i.e. resolution = 2**5.

    Two progress bars are shown: one for downloading the ROI RLEs, and another for
    constructing the output volume.

    Caveat for pathological cases:
        Note that if more than 2 ROIs overlap at a common location,
        then some pathological cases can omit pairs of overlaps.
        For example, if ROIs 1,2,3 all overlap at a single pixel,
        then overlapping_pairs = [(1,2),(2,3)], and (1,3) is not mentioned.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        rois:
            Either:
            - a mapping of `{ roi_name : label }`, indicating each ROI's
              label ID in the output image, or
            - a list of dvid ROI instance names, e.g. `['PB', 'FB', 'EB', 'NO']`,
              in which case the ROIs will be enumerated in order (starting at 1)

        as_bool:
            If True, return a boolean mask instead of a label volume.
            Note: When using is_bool, the `overlapping_pairs` result is undefined.

        box_zyx:
            Optional. Specifies the box `[start, stop] == [(z0,y0,x0), (z1,y1,x1)]`
            of the returned result.
            If this is smaller than the ROIs' combined bounding box, then ROIs
            will of course be truncated at the box boundaries.
            If this is larger than the ROIs' combined bounding box, then the
            result will be padded with zeros.
            As a convenience, you may specify None for either start or stop,
            in which case it will be replaced with the corresponding bound
            from the ROIs' combined bounding box.
            For example, `box_zyx=[(0,0,0), None]` can be used to produce an output
            volume whose coordinates are aligned to the underlying data (at scale 5)
            with no offset.

    Returns:
        (combined_vol, combined_box, overlap_stats)
        where combined_vol is an image volume (ndarray) (resolution: scale 5),
        combined_box indicates the location of combined_vol (scale 5),
        and overlap_stats indicates which ROIs overlap, and are thus not
        completely represented in the output volume (see caveat above).

        Unless as_bool is used, combined_vol is a label volume, whose dtype
        will be wide enough to allow a unique value for each ROI in the list.

    Example:

        from neuclease.dvid import fetch_repo_instances, fetch_combined_roi_volume

        # Select ROIs of interest
        rois = fetch_repo_instances(*master, 'roi').keys()
        rois = filter(lambda roi: not roi.startswith('(L)'), rois)
        rois = filter(lambda roi: not roi.endswith('-lm'), rois)

        # Combine into volume
        roi_vol, box, overlaps = fetch_combined_roi_volume('emdata3:8900', '7f0c', rois, box_zyx=[(0,0,0), None])
    """
    with Timer(f"Fetching {len(rois)} rois", logger):
        roi_ranges, roi_boxes = fetch_roi_ranges_and_boxes(
            server, uuid, rois, session=session, processes=processes)

    with Timer(f"Unpacking {len(rois)} rois", logger):
        combined_vol, box_zyx, overlap_stats = unpack_roi_ranges_to_combined_volume(
            rois, roi_ranges, roi_boxes, box_zyx, as_bool)

    return combined_vol, box_zyx, overlap_stats


@dvid_api_wrapper
def fetch_roi_ranges_and_boxes(server, uuid, rois, *, session=None, processes=0):
    """
    Helper for fetch_combined_roi_volume().
    Fetches the RLEs in 'range' format for a list of ROIs,
    and also computes the bounding box of each ROI.
    """
    if isinstance(rois, str):
        rois = [rois]
    rois = list(rois)

    if processes > 0:
        session = None

    _fetch = partial(fetch_roi, server, uuid, format='ranges', session=session)
    ranges = compute_parallel(_fetch, rois, processes=processes)
    all_rle_ranges = dict(zip(rois, ranges))

    roi_boxes = {}
    for roi, rle_ranges in all_rle_ranges.items():
        # If roi is completely empty, don't process it at all
        if len(rle_ranges) == 0:
            del rois[roi]
        else:
            roi_boxes[roi] = np.array([  rle_ranges[:, (0,1,2)].min(axis=0),
                                       1+rle_ranges[:, (0,1,3)].max(axis=0)])  # noqa
    return all_rle_ranges, roi_boxes


def unpack_roi_ranges_to_combined_volume(roi_labels, roi_ranges, roi_boxes, box_zyx=None, as_bool=False):
    """
    Helper for fetch_combined_roi_volume().

    Unpacks a list of RLE range-encoded ROIs into a single combined label volume,
    optionally restricted to a specified bounding box.
    For details, see the docs for fetch_combined_roi_volume()
    """
    if isinstance(roi_labels, str):
        roi_labels = [roi_labels]

    # roi_labels is a dict {name : label}
    if not isinstance(roi_labels, Mapping):
        roi_labels = {roi: i for i, roi in enumerate(roi_labels, start=1)}

    # Create a reverse-lookup {label : name} for reporting overlaps below.
    reverse_rois = {}
    for roi, label in roi_labels.items():
        if label in reverse_rois:
            # Caller is permitted to map more than one ROI to the same label,
            # so include both names in the overlap report.
            reverse_rois[label] = reverse_rois[label] + '+' + roi
        else:
            reverse_rois[label] = roi

    combined_vol, box_zyx = _initialize_combined_roi_volume(roi_boxes, box_zyx, roi_labels, as_bool)

    # Overlay ROIs one-by-one
    overlap_stats = []
    for roi, label in tqdm_proxy(roi_labels.items(), leave=False):
        roi_box = box_intersection(roi_boxes[roi], box_zyx)
        assert (roi_box[1] - roi_box[0] > 0).all(), "ROI box does not intersect the full box."
        roi_mask, _roi_box = runlength_decode_from_ranges_to_mask(roi_ranges[roi], roi_box)
        assert (_roi_box == roi_box).all()

        # If we're overwriting some areas of a ROI we previously wrote,
        # keep track of the overlapping pairs.
        combined_view = extract_subvol(combined_vol, roi_box - box_zyx[0])
        assert combined_view.base is combined_vol

        # Keep track of the overlapping sizes
        prev_labels_overlap_sizes = pd.Series(combined_view[roi_mask]).value_counts()
        for p, size in prev_labels_overlap_sizes.items():
            if p == 0:
                continue

            if p == label:
                # Note: In the case of combined ROIs, where the user is
                # mapping more than one ROI to the same label, we
                # overlaps among ROIs in the same group are still worth noting.
                # Hence, no special handling in this case.
                pass
            overlap_stats.append((reverse_rois[p], roi, size))

        # Overwrite view
        if as_bool:
            combined_view[roi_mask] = True
        else:
            combined_view[roi_mask] = label

    overlap_stats = pd.DataFrame(overlap_stats, columns=['roi_a', 'roi_b', 'overlap'])
    return combined_vol, box_zyx, overlap_stats


def _initialize_combined_roi_volume(roi_boxes, box_zyx, roi_labels, as_bool):
    """
    Helper for unpack_roi_ranges_to_combined_volume()
    """
    if box_zyx is None:
        box_zyx = [None, None]

    box_zyx = list(box_zyx)
    assert len(box_zyx) == 2

    roi_box_array = np.array([*roi_boxes.values()])
    if box_zyx[0] is None:
        box_zyx[0] = roi_box_array[:,0,:].min(axis=0)
    if box_zyx[1] is None:
        box_zyx[1] = roi_box_array[:,1,:].max(axis=0)

    box_zyx = np.asarray(box_zyx)
    combined_shape = (box_zyx[1] - box_zyx[0])

    if as_bool:
        dtype = np.bool
    else:
        # Choose smallest dtype that can hold enough unique values
        for d in [np.uint8, np.uint16, np.uint32]:
            if max(roi_labels.values()) <= np.iinfo(d).max:
                dtype = d
                break

    combined_vol = np.zeros(combined_shape, dtype)
    return combined_vol, box_zyx


@dvid_api_wrapper
def determine_point_rois(server, uuid, rois, points_df, combined_vol=None, combined_box=None, *, processes=0, session=None):
    """
    Convenience function that combines fetch_combined_roi_volume() and extract_labels_from_volume().
    Labels points with their corresponding ROI (if any).
    Points that are not contained in the given ROIs are not labeled.
    
    Given a list of ROI names and a DataFrame with (at least) columns ['x', 'y', 'z'],
    append columns 'roi_label' and 'roi', indicating which ROI each point falls in.
    A roi_label of 0 indicates an unspecified ROI, and a label of 1 indicates the
    first roi in the given list, etc.

    That is, for each row:
        
        roi = rois[roi_label-1]
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        rois:
            list of dvid ROI instance names, e.g. ['PB', 'FB', 'EB', 'NO']
        
        points_df:
            DataFrame with at least columns ['x', 'y', 'z'].
            The points in this DataFrame should be provided at SCALE-0,
            despite the fact that ROI analysis will be performed at scale 5.
            This function appends two additional columns to the DataFrame, IN-PLACE.
        
        combined_vol:
            Optional.  If you have already fetched the ROI volume for the given rois,
            you can provide it here as an optimization.
            See fetch_combined_roi_volume()
        
        combined_box:
            Optionally crop the ROIs according to the given box before using them.
            Must be provided if combined_vol is provided.

        processes:
            If given, fetch rois in parallel
    
    Returns:
        Nothing.  points_df is modified in-place.4
        Note:
            The 'roi' column will have a pandas.Categorical dtype.
            This is convenient for some uses and inconvenient for others.
            To convert it back to an ordinary dtype, use df['roi'] = df['roi].astype(str)
    """
    if isinstance(rois, str):
        rois = [rois]

    assert set(points_df.columns).issuperset(['x', 'y', 'z'])

    # This is a requirement of extract_labels_from_volume
    assert points_df.index.duplicated().sum() == 0, \
        "This function doesn't work if the input DataFrame's index has duplicate values."

    if combined_vol is None:
        combined_vol, combined_box, overlaps = fetch_combined_roi_volume(server, uuid, rois, False, combined_box, processes=processes, session=session)
        if len(overlaps):
            logger.warning(f"Some ROIs overlap!")
            logger.warning(f"Overlapping pairs:\n{overlaps}")

    assert combined_box is not None

    extract_labels_from_volume(points_df, combined_vol, combined_box, 5, rois, 'roi')

