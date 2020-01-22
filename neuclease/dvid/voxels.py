import logging
import numpy as np

from .node import fetch_instance_info
from . import dvid_api_wrapper

logger = logging.getLogger(__name__)

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
    assert np.issubdtype(box_zyx.dtype, np.integer), \
        f"Box has the wrong dtype.  Use an integer type, not {box_zyx.dtype}"
    assert box_zyx.shape == (2,3)

    params = {}
    if throttle:
        params['throttle'] = 'true'        
    
    shape_zyx = (box_zyx[1] - box_zyx[0])
    shape_str = '_'.join(map(str, shape_zyx[::-1]))
    offset_str = '_'.join(map(str, box_zyx[0, ::-1]))

    r = session.get(f'{server}/api/node/{uuid}/{instance}/raw/0_1_2/{shape_str}/{offset_str}', params=params)
    r.raise_for_status()

    if len(r.content) != np.prod(shape_zyx) * np.dtype(dtype).itemsize:
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
    offset_zyx = np.asarray(offset_zyx)
    assert offset_zyx.shape == (3,)
    assert np.issubdtype(offset_zyx.dtype, np.integer), \
        f"Offset has the wrong dtype.  Use an integer type, not {offset_zyx.dtype}"

    params = {}
    if throttle:
        params['throttle'] = 'true'        
    if mutate:
        params['mutate'] = 'true'

    shape_str = '_'.join(map(str, volume.shape[::-1]))
    offset_str = '_'.join(map(str, offset_zyx[::-1]))

    r = session.post(f'{server}/api/node/{uuid}/{instance}/raw/0_1_2/{shape_str}/{offset_str}',
                    params=params, data=bytes(volume))
    r.raise_for_status()


@dvid_api_wrapper
def fetch_volume_box(server, uuid, instance, *, session=None):
    """
    Return the volume extents for the given instance as a box.
    
    Returns:
        np.ndarray [(z0,y0,x0), (z1,y1,x1)]
    
    Notes:
        - Returns *box*, shape=(box[1] - box[0])
        - Returns ZYX order
    """
    info = fetch_instance_info(server, uuid, instance, session=session)
    box_xyz = np.array((info["Extended"]["MinPoint"], info["Extended"]["MaxPoint"]))
    
    if box_xyz[0] is None or box_xyz[1] is None:
        # If the instance has been created, but not written to,
        # DVID will return null extents.
        # We return zeros, since that's nicer to work with.
        return np.array([[0,0,0], [0,0,0]])
    
    box_xyz[1] += 1
    
    box_zyx = box_xyz[:,::-1]
    return box_zyx


@dvid_api_wrapper
def post_resolution(server, uuid, instance, resolution, *, session=None):
    """
    Sets the resolution for the image volume.
    
    Args:
        server, uuid, instance:
            Refer to a voxels-like instance, e.g. uint8blk, labelmap, etc.
        
        resolution:
            For example: [8.0,  8.0, 8.0]
            
            Note:
                Following the python conventions used everywhere in this library,
                the resolution should be passed in ZYX order!
    """
    resolution = np.asarray(resolution).tolist()
    assert len(resolution) == 3
    r = session.post(f'{server}/api/node/{uuid}/{instance}/resolution', json=resolution[::-1])
    r.raise_for_status()


@dvid_api_wrapper
def post_extents(server, uuid, instance, box_zyx, *, session=None):
    """
    Post new volume extents for the given instance.
    
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'grayscale'.
            Must be a volume type, e.g. 'uint8blk' or 'labelmap', etc.

        box_zyx:
            The new extents: [[z0,y0,x0], [z1,y1,x1]].
    """
    box_zyx = np.asarray(box_zyx)
    assert box_zyx.shape == (2,3)
    
    min_point_xyz = box_zyx[0, ::-1]
    max_point_xyz = box_zyx[1, ::-1] - 1

    extents_json = { "MinPoint": min_point_xyz.tolist(),
                     "MaxPoint": max_point_xyz.tolist() }
    
    url = f'{server}/api/node/{uuid}/{instance}/extents'
    r = session.post(url, json=extents_json)
    r.raise_for_status()


def update_extents(server, uuid, instance, minimal_extents_zyx, *, session=None):
    """
    Convenience function. (Not a direct endpoint wrapper.)
    
    Ensure that the given data instance has at least the given extents.
    Update the instance extents metadata along axes that are smaller
    than the given extents box.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'grayscale'

        minimal_box_zyx:
            3D bounding box [min_zyx, max_zyx] = [(z0,y0,x0), (z1,y1,x1)].
            If provided, data extents will be at least this large (possibly larger).
            (The max extent should use python conventions, i.e. the MaxPoint + 1)
    """
    minimal_extents_zyx = np.array(minimal_extents_zyx, dtype=int)
    assert minimal_extents_zyx.shape == (2,3), \
        "Minimal extents must be provided as a 3D bounding box: [(z0,y0,x0), (z1,y1,x1)]"
    logger.info(f"Updating extents for {uuid}/{instance}")
    
    # Fetch original extents.
    info = fetch_instance_info(server, uuid, instance, session=session)
    
    orig_extents_xyz = np.array( [(1e9, 1e9, 1e9), (-1e9, -1e9, -1e9)], dtype=int )
    if info["Extended"]["MinPoint"] is not None:
        orig_extents_xyz[0] = info["Extended"]["MinPoint"]

    if info["Extended"]["MaxPoint"] is not None:
        orig_extents_xyz[1] = info["Extended"]["MaxPoint"]
        orig_extents_xyz[1] += 1

    minimal_extents_xyz = minimal_extents_zyx[:, ::-1].copy()
    minimal_extents_xyz[0] = np.minimum(minimal_extents_xyz[0], orig_extents_xyz[0])
    minimal_extents_xyz[1] = np.maximum(minimal_extents_xyz[1], orig_extents_xyz[1])

    if (minimal_extents_xyz != orig_extents_xyz).any():
        post_extents(server, uuid, instance, minimal_extents_xyz[:, ::-1])
