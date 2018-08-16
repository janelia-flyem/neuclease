import logging
from itertools import chain

import numpy as np
import pandas as pd

from . import dvid_api_wrapper, fetch_generic_json
from .labelmap import fetch_volume_box
from ..util import Grid, clipped_boxes_from_grid, round_box, tqdm_proxy

logger = logging.getLogger(__name__)


@dvid_api_wrapper
def post_sync(server, uuid, instance, sync_instances, replace=False, *, session=None):
    """
    Appends to list of data instances with which the annotations are synced.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid annotations instance name, e.g. 'synapses'
            
        sync_instances:
            list of dvid instances to which the annotations instance should be synchronized,
            e.g. ['segmentation']
        
        replace:
            If True, replace existing sync instances with the given sync_instances.
            Otherwise append the sync_instances.
    """
    body = { "sync": ",".join(sync_instances) }
    if replace:
        params = { 'replace': str(bool(replace)).lower() }

    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/sync', json=body, params=params)
    r.raise_for_status()


@dvid_api_wrapper
def fetch_label(server, uuid, instance, label, relationships=False, *, session=None):
    """
    Returns all point annotations within the given label as an array of elements.
    This endpoint is only available if the annotation data instance is synced with
    voxel label data instances (labelblk, labelarray, labelmap).
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid annotations instance name, e.g. 'synapses'
        
        label:
            Body ID
            
        relationships:
            Set to true to return all relationships for each annotation.

    Returns:
        JSON list
    """
    params = { 'relationships': str(bool(relationships)).lower() }
    
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/label/{label}', params=params)
    r.raise_for_status()
    return r.json()


@dvid_api_wrapper
def fetch_tag(server, uuid, instance, tag, relationships=False, *, session=None):
    """
    Returns all point annotations with the given tag as an array of elements.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid annotations instance name, e.g. 'synapses'
        
        tag:
            The tag to search for
            
        relationships:
            Set to true to return all relationships for each annotation.

    Returns:
        JSON list
    """
    params = { 'relationships': str(bool(relationships)).lower() }
    
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/tag/{tag}', params=params)
    r.raise_for_status()
    return r.json()


@dvid_api_wrapper
def fetch_roi(server, uuid, instance, roi, roi_uuid=None, *, session=None):
    """
    Returns all point annotations within the ROI.  Currently, this 
    request will only work for ROIs that have same block size as
    the annotation data instance.  Therefore, most ROIs (32px blocks) are not
    not compatible with most labelmap instances (64px blocks).
    
    Warning:
        The name 'fetch_roi()' clashes with a function in dvid.roi, so you
        may need to explicitly import dvid.annotations to access this function:
        
        from dvid.annotations import fetch_roi
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid annotations instance name, e.g. 'synapses'
        
        roi:
            The name of a roi instance, e.g. 'AL-lm'
        
        roi_uuid:
            If provided, the ROI will be fetched at this version.
            Otherwise, the ROI will be fetched at the same version
            as the requested annotation instance.

    Returns:
        JSON list
    """
    if roi_uuid:
        roi = roi + ',' + roi_uuid
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/roi/{roi}')
    r.raise_for_status()
    return r.json()


@dvid_api_wrapper
def fetch_elements(server, uuid, instance, box_zyx, *, session=None):
    """
    Returns all point annotations within the given box.
    Note: Automatically includes relationships.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid annotations instance name, e.g. 'synapses'
        
        box_zyx:
            The bounds of the subvolume from which to fetch annotation elements.
            Given as a pair of coordinates (start, stop), e.g. [(0,0,0), (10,20,30)],
            in Z,Y,X order.  It need not be block-aligned.
        
    Returns:
        JSON list
    """
    box_zyx = np.asarray(box_zyx)
    shape = box_zyx[1] - box_zyx[0]
    
    shape_str = '_'.join(map(str, shape[::-1]))
    offset_str = '_'.join(map(str, box_zyx[0, ::-1]))

    url = f'http://{server}/api/node/{uuid}/{instance}/elements/{shape_str}/{offset_str}'
    return fetch_generic_json(url, session=session)


@dvid_api_wrapper
def fetch_blocks(server, uuid, instance, box_zyx, *, session=None):
    """
    Returns all point annotations within all blocks that intersect the given box.
    
    This differs from fetch_elements() in the following ways:
        - All annotations in the intersecting blocks are returned,
          even annotations that lie outside of the specified box.
        - The return value is a dict instead of a list.

    Note: Automatically includes relationships.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid annotations instance name, e.g. 'synapses'
        
        box_zyx:
            The bounds of the subvolume from which to fetch annotation elements.
            Given as a pair of coordinates (start, stop), e.g. [(0,0,0), (10,20,30)],
            in Z,Y,X order.  It need not be block-aligned.
        
    Returns:
        JSON dict { block_id : element-list }
    """
    box_zyx = np.asarray(box_zyx)
    shape = box_zyx[1] - box_zyx[0]
    
    shape_str = '_'.join(map(str, shape[::-1]))
    offset_str = '_'.join(map(str, box_zyx[0, ::-1]))

    url = f'http://{server}/api/node/{uuid}/{instance}/blocks/{shape_str}/{offset_str}'
    return fetch_generic_json(url, session=session)


def load_synapses_as_dataframe(elements):
    """
    Load the given JSON elements as synapses a DataFrame.
    Note: Discards relationships.
    
    Args:
        elements:
            JSON list of synapse annotation elements as returned by
            fetch_elements(), etc.

    Returns:
        DataFrame with columns ['x', 'y', 'z', 'kind', 'conf', 'user']
    """
    if not elements:
        return pd.DataFrame([], columns=['x', 'y', 'z', 'kind', 'conf', 'user'])

    # Accumulating separate lists for each column ought to be
    # faster than building a list-of-tuples, I think.
    
    # Primary columns
    xs = []
    ys = []
    zs = []
    kinds = []
    confs = []
    users = []
    
    for e in elements:
        x,y,z = e['Pos']
        
        xs.append(x)
        ys.append(y)
        zs.append(z)
        
        kinds.append( e['Kind'] )
        confs.append( float(e.get('Prop', {}).get('conf', 0.0)) )
        users.append( e.get('Prop', {}).get('user', '') )
        
    df = pd.DataFrame( {'x': xs, 'y': ys, 'z': zs}, dtype=np.int32 )
    df['kind'] = pd.Series(kinds, dtype='category')
    df['conf'] = pd.Series(confs, dtype=np.float32)
    df['user'] = pd.Series(users, dtype='category')
    return df


@dvid_api_wrapper
def fetch_synapses_in_batches(server, uuid, synapses_instance, bounding_box_zyx, batch_shape_zyx=(64,64,64000), format='json', *, session=None): #@ReservedAssignment
    """
    Fetch all synapse annotations for the given labelmap volume (or subvolume) and synapse instance.
    Box-shaped regions are queried in batches according to the given batch shape.
    Returns either the raw JSON or a pandas DataFrame.
    
    Note: The DataFrame format discards relationship information.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        synapses_instance:
            dvid annotations instance name, e.g. 'synapses'
        
        bounding_box_zyx:
            The bounds of the subvolume from which to fetch synapse annotations.
            Given as a pair of coordinates (start, stop), e.g. [(0,0,0), (256,1024,1024)],
            in Z,Y,X order.  It must be block-aligned.

            Alternatively, a labelmap instance name can be passed here (e.g. 'segmentation'),
            in which case the labelmap's bounding box will be fetched and used.
            
        batch_shape_zyx:
            What box shape to use for each /elements request.
            Must be block-aligned (i.e. multiple of 64px in all dimensions).
        
        format:
            Either 'json' or 'pandas'. If 'pandas, return a DataFrame.
            Note: The DataFrame format discards relationship information.
        
        show_progress:
            If True, display a progress bar on the console.
    
    Returns:
        If format == 'json', a list of JSON elements.
        If format == 'pandas', a DataFrame with columns ['x', 'y', 'z', 'kind', 'conf', 'user'].
        Note: The pandas format discards relationship information.
    """
    assert format in ('pandas', 'json')
    if isinstance(bounding_box_zyx, str):
        bounding_box_zyx = fetch_volume_box(server, uuid, bounding_box_zyx, session=session)
    else:
        bounding_box_zyx = np.asarray(bounding_box_zyx)
        assert (bounding_box_zyx % 64 == 0).all(), "box must be block-aligned"
    
    batch_shape_zyx = np.asarray(batch_shape_zyx)
    assert (batch_shape_zyx % 64 == 0).all(), "batch shape must be block-aligned"

    results = []
    aligned_bounding_box = round_box(bounding_box_zyx, batch_shape_zyx, 'out')
    num_batches = np.prod((aligned_bounding_box[1] - aligned_bounding_box[0]) // batch_shape_zyx)
    
    boxes = clipped_boxes_from_grid(bounding_box_zyx, Grid(batch_shape_zyx))
    for box in tqdm_proxy(boxes, logger=logger, total=num_batches):
        elements = fetch_blocks(server, uuid, synapses_instance, box, session=session)
        if len(elements) == 0:
            continue

        elements = chain(*elements.values())

        if format == 'json':
            results.extend(elements)
        elif format == 'pandas':
            df = load_synapses_as_dataframe(elements)
            results.append(df)

    if format == 'json':
        return results
    elif format == 'pandas':
        return pd.concat(results)

    raise AssertionError("Shouldn't get here")


def load_synapses_from_csv(csv_path):
    """
    Convenience function for reading saved synapse table as CSV with the proper dtypes.
    """
    dtype = { 'x': np.int32,
              'y': np.int32,
              'z': np.int32,
              'kind': 'category',
              'conf': np.float32,
              'user': 'category',
              'label': np.uint64,
              'body': np.uint64,
              'sv': np.uint64 }

    return pd.read_csv(csv_path, header=0, dtype=dtype)
