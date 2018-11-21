import numpy as np
from . import dvid_api_wrapper

@dvid_api_wrapper
def fetch_raw(server, uuid, instance, box_zyx, throttle=False, *, dtype=np.uint8, session=None):
    """
    Fetch raw array data from an instance that contains voxels.
    
    Note:
        Most voxels data instances do not support a 'scale' parameter, so it is not included here.
        Instead, by convention, we typically create multiple data instances with a suffix indicating the scale.
        For instance, 'grayscale', 'grayscale_1', 'grayscale_2', etc.
        (For labelarray and labelmap instances, see fetch_labelarray_voxels(), which does support scale.)
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'grayscale'

        box_zyx:
            The bounds of the volume to fetch in the coordinate system for the requested scale.
            Given as a pair of coordinates (start, stop), e.g. [(0,0,0), (10,20,30)], in Z,Y,X order.
            The box need not be block-aligned.
        
        throttle:
            If True, passed via the query string to DVID, in which case DVID might return a '503' error
            if the server is too busy to service the request.
            It is your responsibility to catch DVIDExceptions in that case.

        dtype:
            The datatype of the underlying data instance.
            Must match the data instance dtype, e.g. np.uint8 for instances of type uint8blk.
    
    Returns:
        np.ndarray
    """
    box_zyx = np.asarray(box_zyx)
    assert box_zyx.shape == (2,3)

    params = {}
    if throttle:
        params['throttle'] = 'true'        
    
    shape_zyx = (box_zyx[1] - box_zyx[0])
    shape_str = '_'.join(map(str, shape_zyx[::-1]))
    offset_str = '_'.join(map(str, box_zyx[0, ::-1]))

    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/raw/0_1_2/{shape_str}/{offset_str}', params=params)
    r.raise_for_status()

    if len(r.content) != np.prod(shape_zyx) * np.dtype(dtype).itemsize:
        from neuclease.dvid import fetch_instance_info
        info = fetch_instance_info(server, uuid, instance)
        typename = info["Base"]["TypeName"]
        msg = ("Buffer from DVID is the wrong length for the requested array.\n"
               "Did you pass the correct dtype for this instance?\n"
               f"Instance '{instance}' has type '{typename}', and you passed dtype={np.dtype(dtype).name}")
        raise RuntimeError(msg)

    a = np.frombuffer(r.content, dtype=dtype)
    return a.reshape(shape_zyx)


@dvid_api_wrapper
def post_raw(server, uuid, instance, offset_zyx, volume, throttle=False, mutate=True, *, session=None):
    assert len(offset_zyx) == 3

    params = {}
    if throttle:
        params['throttle'] = 'true'        
    if mutate:
        params['mutate'] = 'true'

    shape_str = '_'.join(map(str, volume.shape[::-1]))
    offset_str = '_'.join(map(str, offset_zyx[::-1]))

    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/raw/0_1_2/{shape_str}/{offset_str}',
                    params=params, data=bytes(volume))
    r.raise_for_status()

