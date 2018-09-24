import json
import logging
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from ..util import NumpyConvertingEncoder, tqdm_proxy
from . import dvid_api_wrapper, fetch_generic_json
from .rle import runlength_decode_from_ranges

logger = logging.getLogger(__name__)

@dvid_api_wrapper
def fetch_roi(server, uuid, instance, format='ranges', *, session=None): # @ReservedAssignment
    """
    Fetch an ROI from dvid.
    Note: This function returns coordinates (or masks, etc.) at SCALE 5.
    
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

        Returns:
            If 'ranges':
                np.ndarray, [[Z,Y,X0,X1], [Z,Y,X0,X1], ...]
                Return the RLE block ranges as received from DVID.

            If 'coords':
                np.ndarray, [[Z,Y,X], [Z,Y,X], ...]
                Expand the ranges into a list of ROI-block coordinates.

            If 'mask':
                (mask, mask_box)
                Return a binary mask of the ROI, where each voxel represents one ROI block (scale 5).
                The mask will be cropped to the bounding box of the ROI,
                and the bounding box is also returned.
    """
    assert format in ('coords', 'ranges', 'mask')
    rle_ranges = fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/roi', session=session)
    rle_ranges = np.asarray(rle_ranges, np.int32, order='C')

    # Special cases for empty ROI
    if len(rle_ranges) == 0:
        if format == 'ranges':
            return np.ndarray( (0,4), np.int32 )
    
        if format == 'coords':
            return np.ndarray( (0,3), np.int32 )
        
        if format == 'mask':
            mask_box = np.array([[0,0,0], [0,0,0]], np.int32)
            mask = np.ndarray( (0,0,0), np.int32 )
            return mask, mask_box
        assert False, "Shouldn't get here"
            
    assert rle_ranges.shape[1] == 4    
    if format == 'ranges':
        return rle_ranges

    coords = runlength_decode_from_ranges(rle_ranges)
    if format == 'coords':
        return coords

    if format == 'mask':
        mask_box = np.array([coords.min(axis=0), 1+coords.max(axis=0)])
        mask_shape = mask_box[1] - mask_box[0]

        coords -= mask_box[0]
        mask = np.zeros(mask_shape, bool)
        mask[tuple(coords.transpose())] = True
        return mask, mask_box

    assert False, "Shouldn't get here."


@dvid_api_wrapper
def post_roi(server, uuid, instance, roi_ranges, *, session=None):
    """
    Post a set of RLE ranges to DVID as an ROI.
    The ranges must be provided in SCALE-5 coordinates.
    
    For generating RLE ranges from a list of coordinates, see:
        neuclease.dvid.rle.runlength_encode_to_ranges()

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
    """
    encoded_ranges = json.dumps(roi_ranges, cls=NumpyConvertingEncoder)
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/roi', data=encoded_ranges)
    r.raise_for_status()


@dvid_api_wrapper
def fetch_combined_roi_volume(server, uuid, instances, as_bool=False, *, session=None):
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
        
        instances:
            list of dvid ROI instance names, e.g. ['PB', 'FB', 'EB', 'NO']
    
        as_bool:
            If True, return a boolean mask.
            Otherwise, return a label volume, whose dtype will be chosen such
            as to allow a unique value for each ROI in the list.
    
    Returns:
        (combined_vol, combined_box, overlapping_pairs)
        where combined_vol is an image volume (ndarray) (resolution: scale 5),
        combined_box indicates the location of combined_vol (scale 5),
        and overlapping_pairs
    
    Example:
    
        from neuclease.dvid import fetch_repo_instances, fetch_combined_roi_volume
        
        # Select ROIs of interest
        rois = fetch_repo_instances(*master, 'roi').keys()
        rois = filter(lambda roi: not roi.startswith('(L)'), rois)
        rois = filter(lambda roi: not roi.endswith('-lm'), rois))

        # Combine into volume
        roi_vol, box, overlaps = fetch_combined_roi_volume('emdata3:8900', '7f0c', rois)
    """
    all_ranges = [fetch_roi(server, uuid, roi, format='ranges', session=session)
                  for roi in tqdm_proxy(instances, leave=False)]
    
    roi_boxes = []
    for ranges in all_ranges:
        roi_boxes.append( [  ranges[:, (0,1,2)].min(axis=0),
                           1+ranges[:, (0,1,3)].max(axis=0)] )
    
    roi_boxes = np.array(roi_boxes)
    combined_box = [roi_boxes[:,0,:].min(axis=0),
                    roi_boxes[:,1,:].max(axis=0)]

    combined_shape = combined_box[1] - combined_box[0]
    
    if as_bool:
        dtype = np.bool
    else:
        # Choose smallest dtype that can hold enough unique values
        for d in [np.uint8, np.uint16, np.uint32]:
            if len(instances) <= np.iinfo(d).max:
                dtype = d
                break

    overlapping_pairs = []
    combined_vol = np.zeros(combined_shape, dtype)

    # Overlay ROIs one-by-one
    for i, rle_ranges in enumerate(tqdm_proxy(all_ranges, leave=False), start=1):
        coords = runlength_decode_from_ranges(rle_ranges)
        coords -= combined_box[0]

        # If we're overwriting some areas of a ROI we previously wrote,
        # keep track of the overlapping pairs.
        prev_rois = set(pd.unique(combined_vol[tuple(coords.transpose())]))
        prev_rois -= set([0])
        if prev_rois:
            overlapping_pairs += ((p,i) for p in prev_rois)

        if as_bool:
            combined_vol[tuple(coords.transpose())] = True
        else:
            combined_vol[tuple(coords.transpose())] = i

    return combined_vol, combined_box, overlapping_pairs


@dvid_api_wrapper
def determine_point_rois(server, uuid, rois, points_df, combined_vol=None, combined_box=None, *, session=None):
    """
    Given a list of ROI names and a DataFrame with (at least) columns ['x', 'y', 'z'],
    append columns 'roi_index' and 'roi', indicating which ROI each point falls in.
    A roi_index of 0 indicates an unspecified ROI, and a roi_index of 1 indicates the
    first roi in the given list, etc.

    That is, for each row:
        
        roi = rois[roi_index-1]
    
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
        
        combined_box: Must be provided if combined_vol is provided.
    """
    assert set(points_df.columns).issuperset(['x', 'y', 'z'])
    
    if combined_vol is None:
        combined_vol, combined_box, overlapping_pairs = fetch_combined_roi_volume(server, uuid, rois, session=session)
        if overlapping_pairs:
            logger.warning(f"Some ROIs overlap!")

    assert combined_box is not None

    # Rescale points to scale 5 (ROIs are given at scale 5)
    logger.info("Scaling points")
    downsampled_coords_zyx = points_df[['z', 'y', 'x']] // (2**5)

    # Drop everything outside the combined_box
    logger.info("Excluding OOB points")
    min_z, min_y, min_x = combined_box[0] #@UnusedVariable
    max_z, max_y, max_x = combined_box[1] #@UnusedVariable
    q = 'z >= @min_z and y >= @min_y and x >= @min_x and z < @max_z and y < @max_y and x < @max_x'
    downsampled_coords_zyx.query(q, inplace=True)

    logging.info(f"Extracting {len(downsampled_coords_zyx)} ROI index values")
    points_df['roi_index'] = 0
    downsampled_coords_zyx -= combined_box[0]
    points_df.loc[downsampled_coords_zyx.index, 'roi_index'] = combined_vol[tuple(downsampled_coords_zyx.values.transpose())]

    roi_categories = ['<unspecified>'] + list(rois)
    roi_index_mapping = { roi : category for roi, category in zip(range(len(roi_categories)), roi_categories) }
    points_df['roi'] = pd.Categorical(points_df['roi_index'].map(roi_index_mapping), categories=roi_categories)

