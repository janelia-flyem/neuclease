import numpy as np

from . import dvid_api_wrapper, fetch_generic_json
from .rle import runlength_decode_from_ranges

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


