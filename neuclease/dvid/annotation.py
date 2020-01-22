import os
import sys
import logging
import warnings
from itertools import chain
from functools import partial
from collections import namedtuple

import ujson
import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from . import dvid_api_wrapper, fetch_generic_json
from .node import fetch_instance_info
from .voxels import fetch_volume_box
from ..util import Timer, Grid, boxes_from_grid, round_box, tqdm_proxy, compute_parallel, gen_json_objects, encode_coords_to_uint64, decode_coords_from_uint64

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

    params = {}
    if replace:
        params['replace'] = str(bool(replace)).lower()

    r = session.post(f'{server}/api/node/{uuid}/{instance}/sync', json=body, params=params)
    r.raise_for_status()

# Synonym
post_annotation_sync = post_sync


@dvid_api_wrapper
def post_reload(server, uuid, instance, *, check=False, inmemory=True, session=None): # Note: See wrapper_proxies.post_reload()
    """
    Forces asynchronous recreation of its tag and label indexed denormalizations.
    Can be used to initialize a newly added instance.
    
    Notes:
        - This call merely triggers the reload and returns immediately.
          For sufficiently large volumes, the reloading process on DVID will take hours.
          The only way to determine that the reloading process has completed is to
          monitor the dvid log file for a message that includes the
          words ``Finished denormalization``.

        - The instance will return errors for any POST request
          while denormalization is ongoing.

    Args:
        server:
            dvid server, e.g. 'emdata4:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid annotations instance name, e.g. 'synapses'
        
        check:
            If True, check denormalizations, writing to log when issues
            are detected, and only replacing denormalization when it is incorrect.
        
        inmemory:
            If True, use in-memory reload, which assumes the server
            has enough memory to hold all annotations in memory.
    """
    params = {}
    if check:
        params['check'] = "true"
    if not inmemory:
        params['inmemory'] = "false"
    
    r = session.post(f'{server}/api/node/{uuid}/{instance}/reload', params=params)
    r.raise_for_status()

# Synonym
post_annotation_reload = post_reload


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
    
    r = session.get(f'{server}/api/node/{uuid}/{instance}/label/{label}', params=params)
    r.raise_for_status()
    return r.json()

# Synonym.  See wrapper_proxies.py
fetch_annotation_label = fetch_label

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
    
    r = session.get(f'{server}/api/node/{uuid}/{instance}/tag/{tag}', params=params)
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
    r = session.get(f'{server}/api/node/{uuid}/{instance}/roi/{roi}')
    r.raise_for_status()
    return r.json()

# Synonym to avoid conflicts with roi.fetch_roi()
fetch_annotation_roi = fetch_roi


@dvid_api_wrapper
def fetch_elements(server, uuid, instance, box_zyx, *, format='json', session=None):  #@ReservedAssignment
    """
    Returns all point annotations within the given box.
    
    Note:
        Automatically includes relationships if format=True,
        and automatically discards relationships if format=False.
    
    Note:
        This function is best for fetching relatively
        sparse annotations, to-do annotations.
        For synapse annotations, see ``fetch_synapses_in_batches()``.
    
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
        
        format:
            Either 'json' or 'pandas'
            If 'pandas', convert the elements into a dataframe
            with separate columns for X,Y,Z and each property.
            In the pandas case, relationships are discarded.
        
    Returns:
        JSON list
    """
    assert format in ('json', 'pandas')
    box_zyx = np.asarray(box_zyx)
    shape = box_zyx[1] - box_zyx[0]
    
    shape_str = '_'.join(map(str, shape[::-1]))
    offset_str = '_'.join(map(str, box_zyx[0, ::-1]))

    url = f'{server}/api/node/{uuid}/{instance}/elements/{shape_str}/{offset_str}'
    data = fetch_generic_json(url, session=session)
    
    # The endooint returns 'null' instead of an empty list, on old servers at least.
    # But we always return a list.
    data = data or []
    
    if format == 'pandas':
        return load_elements_as_dataframe(data)
    else:
        return data


def load_elements_as_dataframe(elements):
    """
    Convert the given elements from JSON to a pandas DataFrame.
    
    Note:
        For synapse annotations in particular,
        see ``load_synapses_as_dataframes()``
    """
    pos = np.zeros((len(elements), 3), dtype=np.int32)
    kinds = []
    tags = []
    
    prop_arrays = {}

    for i, e in enumerate(elements):
        pos[i] = e['Pos']
        kinds.append(e['Kind'])
        tags.append(e['Tags'])

        if 'Prop' not in e or not e['Prop']:
            continue

        for k, v in e['Prop'].items():
            pa = prop_arrays.get(k)
            if pa is None:
                pa = prop_arrays[k] = np.empty(len(elements), dtype=object)
            pa[i] = v
    
    return pd.DataFrame({'z': pos[:, 2], 'y': pos[:,1], 'x': pos[:,0],
                         'kind': kinds, 'tags': tags, **prop_arrays})


@dvid_api_wrapper
def post_elements(server, uuid, instance, elements, kafkalog=True, *, session=None):
    """
    Adds or modifies point annotations.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid annotations instance name, e.g. 'synapses'
        
        elements:
            Elements as JSON data (a python list-of-dicts).
            This is the same format as returned by fetch_elements().
            It is NOT the format returned by fetch_blocks().
            If your data came from fetch_blocks(), you must extract and concatenate the values of that dict.
        
        kafkalog:
            If True, log kafka events for each posted element.
        
        Example:
        
            from itertools import chain
            blocks = fetch_blocks(server, uuid, instance_1, box)
            elements = list(chain(*blocks.values()))
            post_elements(server, uuid, instance_2, elements)
        
    """
    params = {}
    if not kafkalog or kafkalog == 'off':
        params['kafkalog'] = 'off'
    
    r = session.post(f'{server}/api/node/{uuid}/{instance}/elements', json=elements, params=params)
    r.raise_for_status()


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

    url = f'{server}/api/node/{uuid}/{instance}/blocks/{shape_str}/{offset_str}'
    return fetch_generic_json(url, session=session)


@dvid_api_wrapper
def post_blocks(server, uuid, instance, blocks_json, kafkalog=False, *, session=None):
    """
    Unlike the POST /elements endpoint, the /blocks endpoint is the fastest way to store
    all point annotations and assumes the caller has (1) properly partitioned the elements
    int the appropriate block for the block size (default 64) and (2) will do a POST /reload
    to create the denormalized Label and Tag versions of the annotations after all
    ingestion is completed.

    This low-level ingestion also does not transmit subscriber events to associated
    synced data (e.g., labelsz).

    The POSTed JSON should be similar to the GET version with the block coordinate as 
    the key:

    {
        "10,381,28": [ array of point annotation elements ],
        "11,381,28": [ array of point annotation elements ],
        ...
    }
    """
    params = {}
    if not kafkalog:
        params['kafkalog'] = 'off'
    
    url = f'{server}/api/node/{uuid}/{instance}/blocks'
    data = ujson.dumps(blocks_json).encode('utf-8')
    r = session.post(url, data=data, params=params)
    r.raise_for_status()


@dvid_api_wrapper
def delete_element(server, uuid, instance, coord_zyx, kafkalog=True, *, session=None):
    """
    Deletes a point annotation given its location.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid annotations instance name, e.g. 'synapses'
        
        coord_zyx:
            coordinate (Z,Y,X)
        
        kafkalog:
            If True, log this deletion in kafka.  Otherwise, don't.
    """
    assert len(coord_zyx) == 3
    coord_str = '_'.join(map(str, coord_zyx[::-1]))
    
    params = {}
    if not kafkalog:
        params['kafkalog'] = 'off'
    
    r = session.delete(f'{server}/api/node/{uuid}/{instance}/element/{coord_str}', params=params)
    r.raise_for_status()


class SynapseWarning(UserWarning):
    pass

def load_synapses_as_dataframes(elements, return_both_partner_tables=False):
    """
    Load the given JSON elements as synapses a DataFrame.
    
    Args:
        elements:
            JSON list of synapse annotation elements as returned by
            fetch_elements(), etc.
        
        return_both_partner_tables:
            Debugging feature.
            Helps detect DVID data inconsistencies, if used correctly.
            If True, return two separate partner tables, computed
            from the PreSyn and PostSyn relationship data, respectively.
            
            That is, pre_partner_df contains the pre->post pairs found
            in the 'PreSynTo' relationships, and post_partner_df contains
            the pre->post found in the 'PostSynTo' relationships.
            
            Note that the two tables will likely NOT be identical,
            unless the given elements include every synapse in your volume.
            By default, combine (and de-duplicate) the two tables.

    Returns:
    
        point_df:
            One row for every t-bar and psd in the file, indicating its
            location, confidence, and synapse type (PostSyn or PreSyn)
            Columns: ['z', 'y', 'x', 'conf', 'kind', 'user']
            Index: np.uint64, an encoded version of [z,y,x]
        
        [post_]partner_df:
            Indicates which T-bar each PSD is associated with.
            One row for every psd in the file.
            Columns: ['pre_id', 'post_id']
            where the values correspond to the index of point_df.
            Note:
                It can generally be assumed that for the synapses we
                load into dvid, every PSD (PostSyn) is
                associated with exactly one T-bar (PreSyn).
        
        [pre_partner_df]:
            Only returned if return_both_partner_tables=True
    """
    #with warnings.catch_warnings():
    #    warnings.simplefilter("once", category=SynapseWarning)
    
    return _load_synapses_as_dataframes(elements, return_both_partner_tables)

def _load_synapses_as_dataframes(elements, return_both_partner_tables):
    if not elements:
        point_df = pd.DataFrame([], columns=['x', 'y', 'z', 'kind', 'conf', 'user'])
        partner_df = pd.DataFrame([], columns=['post_id', 'pre_id'], dtype=np.uint64)
        if return_both_partner_tables:
            return point_df, partner_df, partner_df
        else:
            return point_df, partner_df

    # Accumulating separate lists for each column ought to be
    # faster than building a list-of-tuples, I think.
    
    # Primary columns
    xs = []
    ys = []
    zs = []
    kinds = []
    confs = []
    users = []
    
    # Relationship coordinates
    # [(pre_z, pre_y, pre_x, post_z, post_y, post_x), ...]
    pre_rel_points = []
    post_rel_points = []
    
    need_fake_point = False
    for e in elements:
        x,y,z = e['Pos']
        
        xs.append(x)
        ys.append(y)
        zs.append(z)
        
        kinds.append( e['Kind'] )
        confs.append( float(e.get('Prop', {}).get('conf', 0.0)) )
        users.append( e.get('Prop', {}).get('user', '') )
        
        if 'Rels' not in e or len(e['Rels']) == 0:
            # In general, there should never be
            # a tbar or psd with no relationships at all.
            # That indicates an inconsistency in the database.
            # To keep track of such cases, we add a special connection to point (0,0,0).
            #warnings.warn("At least one synapse had no relationships! "
            #              "Adding artificial partner(s) to (0,0,0).",
            #              SynapseWarning)
            need_fake_point = True
            if e['Kind'] == 'PreSyn':
                pre_rel_points.append( (z,y,x, 0,0,0) )
            else:
                post_rel_points.append( (0,0,0, z,y,x) )
        else:
            for rel in e['Rels']:
                rx, ry, rz = rel['To']
                
                if rx == ry == rz == 0:
                    # We usually assume (0,0,0) is not a real synapse, so it can be used in the case of "orphan" synapses.
                    # But in this case, apparently a real synapse was found at (0,0,0), obfuscating the warning above.
                    warnings.warn("Huh? The fetched synapse data actually contains a relationship to point (0,0,0)!")
                
                if e['Kind'] == 'PreSyn':
                    pre_rel_points.append( (z,y,x, rz,ry,rx) )
                else:
                    post_rel_points.append( (rz,ry,rx, z,y,x) )

    # See warning above.
    if need_fake_point:
        xs.append(0)
        ys.append(0)
        zs.append(0)
        kinds.append('Fake')
        confs.append(0.0)
        users.append('neuclease.dvid.annotation.load_synapses_as_dataframes')

    point_df = pd.DataFrame( {'z': zs, 'y': ys, 'x': xs}, dtype=np.int32 )

    kind_dtype = pd.CategoricalDtype(categories=["PreSyn", "PostSyn", "Fake"], ordered=False)
    point_df['kind'] = pd.Series(kinds, dtype=kind_dtype)
    point_df['conf'] = pd.Series(confs, dtype=np.float32)
    point_df['user'] = pd.Series(users, dtype='category')
    point_df.index = encode_coords_to_uint64(point_df[['z', 'y', 'x']].values)
    point_df.index.name = 'point_id'

    def construct_partner_df(rel_points):
        if rel_points:
            rel_points = np.array(rel_points, np.int32)
            pre_partner_ids  = encode_coords_to_uint64(rel_points[:, :3])
            post_partner_ids = encode_coords_to_uint64(rel_points[:, 3:])
        else:
            pre_partner_ids = np.zeros((0,), dtype=np.uint64)
            post_partner_ids = np.zeros((0,), dtype=np.uint64)

        partner_df = pd.DataFrame({'pre_id': pre_partner_ids, 'post_id': post_partner_ids})
        return partner_df

    pre_partner_df = construct_partner_df(pre_rel_points)
    post_partner_df = construct_partner_df(post_rel_points)
    
    if return_both_partner_tables:
        return point_df, pre_partner_df, post_partner_df

    # For synapses near block borders, maybe only the PreSyn or
    # only the PostSyn happens to be in the given elements.
    # But in most cases, both PreSyn and PostSyn are present,
    # and therefore the relationship is probably listed twice.
    # Drop duplicates.
    partner_df = pd.concat((pre_partner_df, post_partner_df), ignore_index=True)
    partner_df.drop_duplicates(inplace=True)
    
    return point_df, partner_df


def fetch_bodies_for_synapses(server, uuid, seg_instance, point_df=None, partner_df=None, batch_size=10_000, threads=0, processes=0):
    from .labelmap import fetch_labels_batched
    
    if point_df is not None:
        bodies = fetch_labels_batched(server, uuid, seg_instance, point_df[['z', 'y', 'x']].values,
                                      batch_size=batch_size, threads=threads, processes=processes)
        point_df['body'] = bodies

    if partner_df is not None:
        pre_coords = decode_coords_from_uint64(partner_df['pre_id'].values)
        post_coords = decode_coords_from_uint64(partner_df['post_id'].values)
        partner_df['pre_body'] = fetch_labels_batched(server, uuid, seg_instance, pre_coords,
                                                      batch_size=batch_size, threads=threads, processes=processes)
        partner_df['post_body'] = fetch_labels_batched(server, uuid, seg_instance, post_coords,
                                                       batch_size=batch_size, threads=threads, processes=processes)


def fetch_synapses_in_batches(server, uuid, synapses_instance, bounding_box_zyx=None, batch_shape_zyx=(256,256,64000),
                              format='pandas', endpoint='blocks', processes=8, #@ReservedAssignment
                              check_consistency=False, return_both_partner_tables=False): 
    """
    Fetch all synapse annotations for the given labelmap volume (or subvolume) and synapse instance.
    Box-shaped regions are queried in batches according to the given batch shape.
    Returns either the raw JSON or a pandas DataFrame.
    
    Note:
        Every synapse should have at least one partner (relationship).
        If a synapse is found without a partner, that indicates a problem with the database.
        In that case, a warning is emitted and the synapse is given an artificial partner to point (0,0,0).
    
    Note:
        On the hemibrain dataset (~70 million points),
        this function takes ~4 minutes if you use 32 processes.
    
    Warning:
        For large volumes with many synapses, the 'json' format requires a lot of RAM,
        and is not particularly convenient to save/load.
    
    See also:
        ``save_synapses_npy()``, ``load_synapses_npy()``
    
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
            
            If not provided, the entire bounding box of the sync'd
            labelmap instance (e.g. 'segmentation') is used.
            
        batch_shape_zyx:
            What box shape to use for each /elements request.
            Must be block-aligned (i.e. multiple of 64px in all dimensions).
        
        format:
            Either 'json' or 'pandas'. If 'pandas, return a DataFrame.
        
        endpoint:
            Either 'blocks' (faster) or 'elements' (supported on older servers).
    
        check_consistency:
            DVID debug feature. Checks for consistency in the response to the /blocks endpoint.
        
        return_both_partner_tables:
            Debugging feature.
            Helps detect DVID data inconsistencies, if used correctly.
            If True, return two separate partner tables, computed
            from the PreSyn and PostSyn relationship data, respectively.
            
            That is, pre_partner_df contains the pre->post pairs found
            in the 'PreSynTo' relationships, and post_partner_df contains
            the pre->post found in the 'PostSynTo' relationships.
            
            Note that the two tables will likely NOT be identical,
            unless the given elements include every synapse in your volume.
            By default, combine (and de-duplicate) the two tables.

    Returns:
        If format == 'json', a list of JSON elements.
        If format == 'pandas', returns two or three dataframes,
        depending on return_both_partner_tables:

        point_df:
            One row for every t-bar and psd in the file, indicating its
            location, confidence, and synapse type (PostSyn or PreSyn)
            Columns: ['z', 'y', 'x', 'conf', 'kind', 'user']
            Index: np.uint64, an encoded version of [z,y,x]
        
        [post_]partner_df:
            Indicates which T-bar each PSD is associated with.
            One row for every psd in the file.
            Columns: ['pre_id', 'post_id']
            where the values correspond to the index of point_df.
            Note:
                It can generally be assumed that for the synapses we
                load into dvid, every PSD (PostSyn) is
                associated with exactly one T-bar (PreSyn).
        
        [pre_partner_df]:
            Only returned if return_both_partner_tables=True            
    """
    assert format in ('pandas', 'json')
    assert endpoint in ('blocks', 'elements')
    assert not return_both_partner_tables or format == 'pandas', \
        "return_both_partner_tables does not apply unless you're asking for pandas format"

    if bounding_box_zyx is None or isinstance(bounding_box_zyx, str):
        # Determine name of the segmentation instance that's
        # associated with the given synapses instance.
        syn_info = fetch_instance_info(server, uuid, synapses_instance)
        seg_instance = syn_info["Base"]["Syncs"][0]
        if isinstance(bounding_box_zyx, str):
            assert bounding_box_zyx == seg_instance, \
                ("The segmentation instance name you provided doesn't match the name of the sync'd instance.\n"
                 "Please provide an explicit bounding-box.")
        bounding_box_zyx = fetch_volume_box(server, uuid, seg_instance)
    else:
        bounding_box_zyx = np.asarray(bounding_box_zyx)
        assert (bounding_box_zyx % 64 == 0).all(), "box must be block-aligned"
    
    batch_shape_zyx = np.asarray(batch_shape_zyx)
    assert (batch_shape_zyx % 64 == 0).all(), "batch shape must be block-aligned"

    boxes = [*boxes_from_grid(bounding_box_zyx, Grid(batch_shape_zyx))]
    fn = partial(_fetch_synapse_batch, server, uuid, synapses_instance, 
                 format=format, endpoint=endpoint, check_consistency=check_consistency,
                 return_both_partner_tables=return_both_partner_tables)

    initializer = None
    #initializer = lambda: warnings.simplefilter("once", category=SynapseWarning)
    results = compute_parallel(fn, boxes, processes=processes, ordered=False, leave_progress=True, initializer=initializer)

    if format == 'json':
        return list(chain(*results))
    elif format == 'pandas':
        if return_both_partner_tables:
            point_dfs, pre_partner_dfs, post_partner_dfs = zip(*results)
            pre_partner_dfs = [*filter(len, pre_partner_dfs)]
            post_partner_dfs = [*filter(len, post_partner_dfs)]
        else:
            point_dfs, partner_dfs = zip(*results)
            partner_dfs = [*filter(len, partner_dfs)]
        
        # Any zero-length dataframes might have the wrong dtypes,
        # which would screw up the concat step.  Remove them.
        point_dfs = [*filter(len, point_dfs)]

        if len(point_dfs) == 0:
            # Return empty dataframe
            return load_synapses_as_dataframes([], return_both_partner_tables)

        point_df = pd.concat(point_dfs)
        
        # Make sure user and kind are Categorical
        point_df['kind'] = point_df['kind'].astype("category")
        point_df['user'] = point_df['user'].astype("category")

        # If any 'fake' synapses were added due to inconsistent data,
        # Drop duplicates among them.
        if (point_df['kind'] == "Fake").any():
            # All fake rows are the same.  Drop all but the first.
            fake_df = point_df.query('kind == "Fake"').iloc[0:1]
            point_df = pd.concat((fake_df, point_df.query('kind != "Fake"')))

        # Sort, mostly to ensure that the Fake point (if any) is at the top.        
        point_df.sort_values(['z', 'y', 'x'], inplace=True)
        
        if return_both_partner_tables:
            pre_partner_df = pd.concat(pre_partner_dfs, ignore_index=True)
            post_partner_df = pd.concat(post_partner_dfs, ignore_index=True)
            return point_df, pre_partner_df, post_partner_df
        else:
            partner_df = pd.concat(partner_dfs, ignore_index=True)
            partner_df.drop_duplicates(inplace=True)
            return point_df, partner_df


def _fetch_synapse_batch(server, uuid, synapses_instance, batch_box, format, endpoint, # @ReservedAssignment
                         check_consistency, return_both_partner_tables):
    """
    Helper for fetch_synapses_in_batches(), above.
    As a special check, if format 'pandas' is used, we also check for dvid inconsistencies.
    """
    assert not check_consistency or endpoint == 'blocks', \
        "check_consistency can only be used with the blocks endpoint."

    if endpoint == 'blocks':
        blocks = fetch_blocks(server, uuid, synapses_instance, batch_box)
        elements = list(chain(*blocks.values()))
        
        if check_consistency:
            for key, els in blocks.items():
                if len(els) == 0:
                    continue
                block = [int(c) for c in key.split(',')]
                block_box = 64*np.array((block, block))
                block_box[1] += 64
                pos = np.array([e['Pos'] for e in els])
                if (pos < block_box[0]).any() or (pos >= block_box[1]).any():
                    msg = ("Detected a DVID inconsistency: Some elements fetched from block "
                           f"at {block_box[0, ::-1].tolist()} (XYZ) fall outside the block!")
                    raise RuntimeError(msg)

    elif endpoint == 'elements':
        elements = fetch_elements(server, uuid, synapses_instance, batch_box)
    else:
        raise AssertionError("Invalid endpoint choice")

    if format == 'json':
        return elements

    if return_both_partner_tables:
        point_df, pre_partner_df, post_partner_df = load_synapses_as_dataframes(elements, True)
        return point_df, pre_partner_df, post_partner_df
    else:
        point_df, partner_df = load_synapses_as_dataframes(elements, False)
        return point_df, partner_df


def save_synapses_npy(synapse_point_df, npy_path, save_index=None):
    """
    Save the given synapse point DataFrame to a .npy file,
    with careful handling of strings to avoid creating any
    pickled objects (which are annoying to load).
    """
    assert save_index in (True, False, None)
    if save_index is None:
        save_index = (synapse_point_df.index.name is not None)
    
    dtypes = {}

    # Avoid 'pickle' objects (harder to load) by converting
    # categories/strings to fixed-width strings
    max_kind = synapse_point_df['kind'].map(len).astype(int).max()
    dtypes['kind'] = f'U{max_kind}'
    
    if 'user' in synapse_point_df:
        max_user = synapse_point_df['user'].map(len).astype(int).max()
        dtypes['user'] = f'U{max_user}'
    
    np.save(npy_path, synapse_point_df.to_records(index=save_index, column_dtypes=dtypes))


def load_synapses_npy(npy_path):
    """
    Load the given .npy file as a synapse point DataFrame,
    with special handling of the string columns to use
    categorical dtypes (saves RAM). 
    """
    records = np.load(npy_path, allow_pickle=True)

    numeric_cols = ['z', 'y', 'x', 'conf', 'label', 'body', 'sv']
    numeric_cols = [*filter(lambda c: c in records.dtype.names, numeric_cols)]

    df = pd.DataFrame(records[numeric_cols])

    if 'point_id' in records.dtype.names:
        df.index = records['point_id']

    df['kind'] = pd.Series(records['kind'], dtype='category')
    if 'user' in records.dtype.names:
        df['user'] = pd.Series(records['user'], dtype='category')

    return df


def save_synapses_csv(synapse_point_df, csv_path, index=False):
    """
    Save the given synapse points table to CSV.
    
    Note:
        Usually it's more efficient to read/write .npy files.
        See ``save_synapses_npy()``.
    """
    synapse_point_df.to_csv(csv_path, header=True, index=index)


def load_synapses_csv(csv_path):
    """
    Convenience function for reading saved synapse
    table from CSV with the proper dtypes.

    Note:
        Usually it's more efficient to read/write .npy files.
        See ``load_synapses_npy()``.

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


def load_synapses(path):
    """
    Load synapse points from the given file path.
    """
    if isinstance(path, pd.DataFrame):
        return path

    assert isinstance(path, str)
    _, ext = os.path.splitext(path)
    assert ext in ('.csv', '.npy', '.json')

    if ext == '.csv':
        points_df = load_synapses_csv(path)
    elif ext == '.npy':
        points_df = load_synapses_npy(path)
    elif ext == '.json':
        points_df, _partner_df = load_synapses_from_json()
    
    return points_df


def load_synapses_from_json(json_path, batch_size=1000):
    """
    Load the synapses to a dataframe from a JSON file
    (which must have the same structure as the elements response from DVID).
    The JSON file is consumed in batches, avoiding the need
    to load the entire JSON document in RAM at once.
    """
    point_dfs = []
    partner_dfs = []
    try:
        with open(json_path, 'r') as f:
            for elements in tqdm_proxy( gen_json_objects(f, batch_size) ):
                point_df, partner_df = load_synapses_as_dataframes(elements)
                point_dfs.append(point_df)
                partner_dfs.append(partner_df)
                
    except KeyboardInterrupt:
        msg = f"Stopping early due to KeyboardInterrupt. ({len(point_dfs)} batches completed)\n"
        sys.stderr.write(msg)

    point_df = pd.concat(point_dfs)
    partner_df = pd.concat(partner_dfs)
    return point_df, partner_df


def load_relationships(elements, kind=None):
    """
    Given a list of JSON elements, load all relationships as a table.
    """
    from_x = []
    from_y = []
    from_z = []
    
    to_x = []
    to_y = []
    to_z = []
    
    rels = []
    
    for element in tqdm_proxy(elements):
        if kind and (kind != element['Kind']):
            continue
        
        fx, fy, fz = element['Pos']
        
        for obj in element['Rels']:
            tx, ty, tz = obj['To']

            from_x.append(fx)
            from_y.append(fy)
            from_z.append(fz)
        
            to_x.append(tx)
            to_y.append(ty)
            to_z.append(tz)

            rels.append(obj['Rel'])
        
    df = pd.DataFrame( {'from_x': from_x,
                        'from_y': from_y,
                        'from_z': from_z,
                        'to_x': to_x,
                        'to_y': to_y,
                        'to_z': to_z,
                        }, dtype=np.int32 )

    df['rel'] = pd.Series(rels, dtype='category')
    return df


def compute_weighted_edge_table(relationships_df, synapses_df):
    """
    Given a synapse 'relationship table' with columns [from_x, from_y, from_z, to_x, to_y, to_z],
    and a synapse table with columns [x, y, z, body],
    Perform the necessary merge operations to determine from_body and to_body for each relationship,
    and then aggregate those relationships to to yield a table of weights for each unique body pair.
    """
    from_bodies = relationships_df.merge(synapses_df[['z', 'y', 'x', 'body']], how='left',
                                         left_on=['from_z', 'from_y', 'from_x'],
                                         right_on=['z', 'y', 'x'])['body']
    
    to_bodies = relationships_df.merge(synapses_df[['z', 'y', 'x', 'body']], how='left',
                                       left_on=['to_z', 'to_y', 'to_x'],
                                       right_on=['z', 'y', 'x'])['body']

    edge_table = pd.DataFrame({'from_body': from_bodies,
                               'to_body': to_bodies})

    weighted_edge_table = edge_table.groupby(['from_body', 'to_body']).size()
    weighted_edge_table.sort_values(ascending=False, inplace=True)
    weighted_edge_table.name = 'weight'
    return weighted_edge_table.reset_index()


def load_gary_synapse_json(path, processes=8, batch_size=100_000):
    """
    Load a synapse json file from Gary's format into two tables.
    
    Args:
        path:
            A path to a .json file.
            See ``neuclease/tests/test_annotation.py`` for an example.
        
        processes:
            How many processes to use in parallel to load the data.

        batch_size:
            The size (number of t-bars) to processes per batch during multiprocessing.
    
    Returns:
        point_df:
            One row for every t-bar and psd in the file.
            Columns: ['z', 'y', 'x', 'confidence', 'kind']
            Index: np.uint64, an encoded version of [z,y,x]
        
        partner_df:
            Indicates which T-bar each PSD is associated with.
            One row for every psd in the file.
            Columns: ['post_id', 'pre_id']
            where the values correspond to the index of point_df.
            Note:
                Gary guarantees that every PSD (PostSyn) is
                associated with exactly 1 T-bar (PreSyn).
    """
    logger.info(f"Loading JSON data from {path}")
    with open(path, 'r') as f:
        data = ujson.load(f)["data"]
    
    if processes == 0:
        logger.info("Generating tables in the main process (not parallel).")
        return _load_gary_synapse_data(data)
    
    batches = []
    for batch_start in range(0, len(data), batch_size):
        batches.append(data[batch_start:batch_start+batch_size])

    logger.info(f"Converting via {len(batches)} batches (using {processes} processes).")    
    results = compute_parallel(_load_gary_synapse_data, batches, processes=processes)
    point_dfs, partner_dfs = zip(*results)

    logger.info("Combining results")
    point_df = pd.concat(point_dfs)
    partner_df = pd.concat(partner_dfs, ignore_index=True)
    return point_df, partner_df


def _load_gary_synapse_data(data):
    """
    Helper for load_gary_synapse_json()
    """
    point_table = []
    confidences = []
    kinds = []
    partner_table = []
    
    for syn in data:
        tx, ty, tz = syn["T-bar"]["location"]
        confidence = float(syn["T-bar"]["confidence"])
        point_table.append( (tz, ty, tx) )
        confidences.append( confidence )
        kinds.append('PreSyn')

        for partner in syn["partners"]:
            px, py, pz = partner["location"]
            confidence = float(partner["confidence"])

            point_table.append( (pz, py, px) )
            confidences.append(confidence)
            kinds.append('PostSyn')
            partner_table.append( (tz, ty, tx, pz, py, px) )

    points = np.array(point_table, np.int32)
    point_df = pd.DataFrame(points, columns=['z', 'y', 'x'], dtype=np.int32)
    point_df['conf'] = np.array(confidences, np.float32)
    point_df['kind'] = pd.Series(kinds, dtype='category')

    point_ids = encode_coords_to_uint64(points)
    point_df.index = point_ids
    point_df.index.name = 'point_id'

    partner_points = np.array(partner_table, dtype=np.int32)
    tbar_partner_ids = encode_coords_to_uint64(partner_points[:,:3])
    psd_partner_ids = encode_coords_to_uint64(partner_points[:,3:])
    partner_df = pd.DataFrame({'post_id': psd_partner_ids, 'pre_id': tbar_partner_ids})
    
    return point_df, partner_df


def body_synapse_counts(synapse_samples):
    """
    Given a DataFrame of sampled synapses (or a path to a CSV file),
    Tally synapse totals (by kind) for each body.
    
    Returns:
        DataFrame with columns: ['PreSyn', 'PostSyn'], indexed by 'body'.
        (The PreSyn/PostSyn columns are synapse counts.)
    """
    if isinstance(synapse_samples, str):
        synapse_samples = pd.read_csv(synapse_samples)
    
    assert 'body' in synapse_samples.columns, "Samples must have a 'body' col."
    assert 'kind' in synapse_samples.columns, "Samples must have a 'kind' col"
    
    synapse_samples = synapse_samples[['body', 'kind']]
    synapse_counts = synapse_samples.pivot_table(index='body', columns='kind', aggfunc='size')
    synapse_counts.fillna(0.0, inplace=True)

    if 0 in synapse_counts.index:
        msg = ("*** Synapse table includes body 0 and was therefore probably generated "
               "from out-of-date data OR some synapses in your data fall on voxels with "
               "no label (label 0). ***")
        logger.warning(msg)

    synapse_counts['PostSyn'] = synapse_counts['PostSyn'].astype(np.int32)
    synapse_counts['PreSyn'] = synapse_counts['PreSyn'].astype(np.int32)

    # Convert columns from categorical index to normal index,
    # so the caller can easily append their own columns if they want.
    synapse_counts.columns = synapse_counts.columns.tolist()
    return synapse_counts[['PreSyn', 'PostSyn']]


def fetch_roi_synapses(server, uuid, synapses_instance, rois, fetch_labels=False, return_partners=False, processes=16):
    """
    Fetch the coordinates and (optionally) body labels for 
    all synapses that fall within the given ROIs.
    
    Args:
    
        server:
            DVID server, e.g. 'emdata4:8900'
        
        uuid:
            DVID uuid, e.g. 'abc9'
        
        synapses_instance:
            DVID synapses instance name, e.g. 'synapses'
        
        rois:
            A single DVID ROI instance names or a list of them, e.g. 'EB' or ['EB', 'FB']
        
        fetch_labels:
            If True, also fetch the supervoxel and body label underneath each synapse,
            returned in columns 'sv' and 'body'.
            
        return_partners:
            If True, also return the partners table.

        processes:
            How many parallel processes to use when fetching synapses and supervoxel labels.
    
    Returns:
        pandas DataFrame with columns:
        ``['z', 'y', 'x', 'kind', 'conf']`` and ``['sv', 'body']`` (if ``fetch_labels=True``)
        If return_partners is True, also return the partners table.

    Example:
        df = fetch_roi_synapses('emdata4:8900', '3c281', 'synapses', ['PB(L5)', 'PB(L7)'], True, 8)
    """
    # Late imports to avoid circular imports in dvid/__init__
    from neuclease.dvid import fetch_combined_roi_volume, determine_point_rois, fetch_labels_batched, fetch_mapping, fetch_mappings

    assert rois, "No rois provided, result would be empty. Is that what you meant?"
    
    if isinstance(rois, str):
        rois = [rois]
    
    # Determine name of the segmentation instance that's
    # associated with the given synapses instance.
    syn_info = fetch_instance_info(server, uuid, synapses_instance)
    seg_instance = syn_info["Base"]["Syncs"][0]
    
    logger.info(f"Fetching mask for ROIs: {rois}")
    # Fetch the ROI as a low-res array (scale 5, i.e. 32-px resolution)
    roi_vol_s5, roi_box_s5, overlapping_pairs = fetch_combined_roi_volume(server, uuid, rois)
    
    if len(overlapping_pairs) > 0:
        logger.warning("Some ROIs overlapped and are thus not completely represented in the output:\n"
                       f"{overlapping_pairs}")
    
    # Convert to full-res box
    roi_box = (2**5) * roi_box_s5
    
    # fetch_synapses_in_batches() requires a box that is 64-px-aligned
    roi_box = round_box(roi_box, 64, 'out')
    
    logger.info("Fetching synapse points")
    # points_df is a DataFrame with columns for [z,y,x]
    points_df, partners_df = fetch_synapses_in_batches(server, uuid, synapses_instance, roi_box, processes=processes)
    
    # Append a 'roi_name' column to points_df
    logger.info("Labeling ROI for each point")
    determine_point_rois(server, uuid, rois, points_df, roi_vol_s5, roi_box_s5)

    logger.info("Discarding points that don't overlap with the roi")
    rois = {*rois}
    points_df = points_df.query('roi in @rois').copy()
    
    columns = ['z', 'y', 'x', 'kind', 'conf', 'roi_label', 'roi']

    if fetch_labels:
        logger.info("Fetching supervoxel under each point")
        svs = fetch_labels_batched(server, uuid, seg_instance,
                                   points_df[['z', 'y', 'x']].values,
                                   supervoxels=True,
                                   processes=processes)

        with Timer("Mapping supervoxels to bodies", logger):
            # Arbitrary heuristic for whether to do the
            # body-lookups on DVID or on the client.
            if len(svs) < 100_000:
                bodies = fetch_mapping(server, uuid, seg_instance, svs)
            else:
                mapping = fetch_mappings(server, uuid, seg_instance)
                mapper = LabelMapper(mapping.index.values, mapping.values)
                bodies = mapper.apply(svs, True)

        points_df['sv'] = svs
        points_df['body'] = bodies
        columns += ['body', 'sv']
    
    if return_partners:
        # Filter
        #partners_df = partners_df.query('post_id in @points_df.index and pre_id in @points_df.index').copy()
        
        # Faster filter (via merge)
        partners_df = partners_df.merge(points_df[[]], 'inner', left_on='pre_id', right_index=True)
        partners_df = partners_df.merge(points_df[[]], 'inner', left_on='post_id', right_index=True)
        return points_df[columns], partners_df
    else:
        return points_df[columns]


def determine_bodies_of_interest(server, uuid, synapses_instance, rois=None, min_tbars=2, min_psds=10, processes=16, *, synapse_table=None, seg_instance=None):
    """
    Determine which bodies fit the given criteria
    for minimum synapse counts WITHIN the given ROIs.
    
    Note that the min_tbars and min_psds criteria are OR'd together.
    A body need only match at least one of the criteria to be considered "of interest".

    This function is just a convenience wrapper around calling
    fetch_roi_synapses(), fetch_labels_batched(), and body_synapse_counts().

    Note:
        If your synapse table is already loaded and already has a 'body' column,
        and you aren't providing any rois to filter with, then this function is
        merely equivalent to calling body_synapse_counts() and filtering it
        for tbar/psd requirements.

    Args:
        server:
            dvid server
        
        uuid:
            dvid uuid
        
        synapses_instance:
            synapses annotation instance name, e.g. 'synapses'
            If you are providing a pre-loaded synapse_table and overriding seg_instance,
            you can set synapses_instance=None.
        
        rois:
            A list of ROI instance names.  If provided, ONLY synapses
            within these ROIs will be counted when determining bodies of interest.
            If not provided, all synapses in the volume will be counted.

        min_tbars:
            All bodies with at least this many t-bars (PreSyn annotations) will be "of interest".
    
        min_psds:
            All bodies with at least this many PSDs (PostSyn annotations) will be "of interest".
        
        processes:
            How many parallel processes to use when fetching synapses and body labels.
        
        synapse_table:
            If you have a pre-loaded synapse table (or a path to one stored as .npy or .csv),
            you may provide it here, in which case the synapse points won't be fetched from DVID.
            Furthermore, if the table already contains a 'body' column, then it is presumed to be
            accurate and body labels will not be fetched from DVID.
    
        seg_instance:
            If you want to override the segmentation instance name to use
            (rather than inspecting the syanapse instance syncs), provide it here.
    
    Returns:
        pandas DataFrame, as returned by body_synapse_counts().
        That is, DataFrame with columns: ['PreSyn', 'PostSyn'], indexed by 'body',
        where only bodies of interest are included in the table.
    """
    from neuclease.dvid import fetch_labels_batched, fetch_combined_roi_volume, determine_point_rois
    
    # Download synapses if necessary
    if synapse_table is None:
        with Timer("Fetching synapse points", logger):
            if rois is None:
                # Fetch all synapses in the volume
                points_df, _partners_df = fetch_synapses_in_batches(server, uuid, synapses_instance, processes=processes)
            else:
                # Fetch only the synapses within the given ROIs
                points_df = fetch_roi_synapses(server, uuid, synapses_instance, rois, False, processes=processes)
    else:
        # User provided a pre-loaded synapse table (or a path to one)
        if isinstance(synapse_table, str):
            with Timer(f"Loading synapse table {synapse_table}", logger):
                _, ext = os.path.splitext(synapse_table)
                assert ext in ('.csv', '.npy')
                if ext == '.csv':
                    synapse_table = load_synapses_csv(synapse_table)
                elif ext == '.npy':
                    synapse_table = load_synapses_npy(synapse_table)

        assert isinstance(synapse_table, pd.DataFrame)
        assert not ({'z', 'y', 'x', 'kind'} - {*synapse_table.columns}), \
            "Synapse table does not contain all expected columns"
        
        points_df = synapse_table
        if rois:
            roi_vol_s5, roi_box_s5, _ = fetch_combined_roi_volume(server, uuid, rois)
            determine_point_rois(server, uuid, rois, points_df, roi_vol_s5, roi_box_s5)
            points_df = points_df.query('roi_label != 0')

    if 'body' in points_df:
        logger.info("Using user-provided body labels")
    else:
        with Timer("Fetching synapse body labels", logger):
            if seg_instance is None:
                syn_info = fetch_instance_info(server, uuid, synapses_instance)
                seg_instance = syn_info["Base"]["Syncs"][0]
    
            points_df['body'] = fetch_labels_batched( server, uuid, seg_instance,
                                                      points_df[['z', 'y', 'x']].values,
                                                      processes=processes )

    with Timer("Aggregating body-wise synapse counts"):
        body_synapses_df = body_synapse_counts(points_df)

    body_synapses_df = body_synapses_df.query('PreSyn >= @min_tbars or PostSyn >= @min_psds')
    return body_synapses_df



ConsistencyResults = namedtuple("ConsistencyResults",
                                ["orphan_tbars", "orphan_psds",
                                "pre_dupes", "post_dupes",
                                "only_in_tbar", "only_in_psd",
                                "bad_tbar_refs", "bad_psd_refs",
                                "oversubscribed_post", "oversubscribed_pre"])

def check_synapse_consistency(syn_point_df, pre_partner_df, post_partner_df):
    """
    Given a synapse point table and TWO partners tables as returned when
    calling ``fetch_synapses_in_batches(..., return_both_partner_tables=True)``,
    Analyze the relationships to look for inconsistencies.
    
    Note:
        There are different types of results returned,
        and they are not mutually exclusive.
        For example, "orphan tbars" will also count toward
        "non-reciprocal relationships", and also contribute to the "oversubscribed"
        counts (since the orphans are artificially partnered to (0,0,0), which ends
        up counting as oversubscribed). 
    """
    # 'Orphan' points (a tbar or psd with no relationships at all)
    orphan_tbars = pre_partner_df.query('post_id == 0')
    orphan_psds = post_partner_df.query('pre_id == 0')
    logger.info(f"Found {len(orphan_tbars)} orphan TBars")
    logger.info(f"Found {len(orphan_psds)} orphan psds")

    # Duplicate connections (one tbar references the same PSD twice or more)
    pre_dupes = pre_partner_df.loc[pre_partner_df.duplicated()].drop_duplicates()
    post_dupes = post_partner_df.loc[post_partner_df.duplicated()].drop_duplicates()
    logger.info(f"Found {len(pre_dupes)} duplicated tbar->psd relationships.")
    logger.info(f"Found {len(post_dupes)} duplicated psd<-tbar relationships.")

    # Non-reciprocal (Tbar references PSD, but not the other way around, or vice-versa)
    pre_nodupes_df = pre_partner_df.drop_duplicates()
    merged = pre_nodupes_df.merge(post_partner_df.drop_duplicates(), 'outer', ['pre_id', 'post_id'], indicator='which')
    only_in_tbar = merged.query('which == "left_only"')
    only_in_psd = merged.query('which == "right_only"')
    logger.info(f"Found {len(only_in_tbar)} non-reciprocal relationships from TBars")
    logger.info(f"Found {len(only_in_psd)} non-reciprocal relationships from PSDs")

    # Refs to nowhere (Tbar or PSD has a relationship to a point that doesn't exist)
    point_ids = syn_point_df.index
    bad_tbar_refs = pre_partner_df.query('post_id not in @point_ids')
    bad_psd_refs = post_partner_df.query('pre_id not in @point_ids')
    logger.info(f"Found {len(bad_tbar_refs)} references to non-existent PSDs")
    logger.info(f"Found {len(bad_psd_refs)} references to non-existent TBars")

    # Too many refs from a single PSD
    oversubscribed_post = post_partner_df.loc[post_partner_df.duplicated('post_id')]
    oversubscribed_pre = pre_nodupes_df.loc[pre_nodupes_df.duplicated('post_id')]

    logger.info(f"Found {len(oversubscribed_post)} PSDs that contain more than one relationship")
    logger.info(f"Found {len(oversubscribed_pre)} PSDs that are referenced by more than one TBar")

    return ConsistencyResults( orphan_tbars, orphan_psds,
                               pre_dupes, post_dupes,
                               only_in_tbar, only_in_psd,
                               bad_tbar_refs, bad_psd_refs,
                               oversubscribed_post, oversubscribed_pre )


def post_tbar_jsons(server, uuid, instance, partner_df, merge_existing=True, processes=32, chunk_shape=(256, 256, 64000)):
   
    logger.info("Computing chunk/block IDs")
    partner_df['cz_pre'] = partner_df['z_pre'] // chunk_shape[0]
    partner_df['cy_pre'] = partner_df['y_pre'] // chunk_shape[1]
    partner_df['cx_pre'] = partner_df['x_pre'] // chunk_shape[2]
    partner_df['cid_pre'] = encode_coords_to_uint64(partner_df[['cz_pre', 'cy_pre', 'cx_pre']].values)

    partner_df['bz_pre'] = partner_df['z_pre'] // 64
    partner_df['by_pre'] = partner_df['y_pre'] // 64
    partner_df['bx_pre'] = partner_df['x_pre'] // 64
    partner_df['bid_pre'] = encode_coords_to_uint64(partner_df[['bz_pre', 'by_pre', 'bx_pre']].values)

    num_chunks = partner_df['cid_pre'].nunique()
    _post = partial(_post_tbar_chunk, server, uuid, instance, chunk_shape, merge_existing)
    compute_parallel(_post, partner_df.groupby(['cz_pre', 'cy_pre', 'cx_pre']),
                     total=num_chunks, processes=processes, ordered=False, starmap=True)


def _post_tbar_chunk(server, uuid, instance, chunk_shape, merge_existing, c_zyx, chunk_df):
    block_jsons = {}
    for (bz, by, bx), block_df in chunk_df.groupby(['bz_pre', 'by_pre', 'bx_pre']):
        block_jsons[f"{bx},{by},{bz}"] = compute_tbar_jsons(block_df)
    
    if merge_existing:
        chunk_start = np.asarray(c_zyx) * chunk_shape
        chunk_stop = chunk_start + chunk_shape
        existing = fetch_blocks(server, uuid, instance, [chunk_start, chunk_stop])
        for key in existing.keys():
            if key in block_jsons:
                block_jsons[key].extend(existing[key])
            elif existing[key]:
                block_jsons[key] = existing[key]
    
    post_blocks(server, uuid, instance, block_jsons)


def post_psd_jsons(server, uuid, instance, partner_df, merge_existing=True, processes=32, chunk_shape=(256, 256, 64000)):
    logger.info("Computing chunk/block IDs")
    partner_df['cz_post'] = partner_df['z_post'] // chunk_shape[0]
    partner_df['cy_post'] = partner_df['y_post'] // chunk_shape[1]
    partner_df['cx_post'] = partner_df['x_post'] // chunk_shape[2]
    partner_df['cid_post'] = encode_coords_to_uint64(partner_df[['cz_post', 'cy_post', 'cx_post']].values)

    partner_df['bz_post'] = partner_df['z_post'] // 64
    partner_df['by_post'] = partner_df['y_post'] // 64
    partner_df['bx_post'] = partner_df['x_post'] // 64
    partner_df['bid_post'] = encode_coords_to_uint64(partner_df[['bz_post', 'by_post', 'bx_post']].values)

    num_chunks = partner_df['cid_post'].nunique()
    _post = partial(_post_psd_chunk, server, uuid, instance, chunk_shape, merge_existing)
    compute_parallel(_post, partner_df.groupby(['cz_post', 'cy_post', 'cx_post']),
                     total=num_chunks, processes=processes, ordered=False, starmap=True)


def _post_psd_chunk(server, uuid, instance, chunk_shape, merge_existing, c_zyx, chunk_df):
    block_jsons = {}
    for (bz, by, bx), block_df in chunk_df.groupby(['bz_post', 'by_post', 'bx_post']):
        block_jsons[f"{bx},{by},{bz}"] = compute_psd_jsons(block_df)
    
    if merge_existing:
        chunk_start = np.asarray(c_zyx) * chunk_shape
        chunk_stop = chunk_start + chunk_shape
        existing = fetch_blocks(server, uuid, instance, [chunk_start, chunk_stop])
        for key in existing.keys():
            if key in block_jsons:
                block_jsons[key].extend(existing[key])
            elif existing[key]:
                block_jsons[key] = existing[key]
    
    post_blocks(server, uuid, instance, block_jsons)


def delete_all_synapses(server, uuid, instance, box=None, chunk_shape=(256,256,64000)):
    if box is None or isinstance(box, str):
        # Determine name of the segmentation instance that's
        # associated with the given synapses instance.
        syn_info = fetch_instance_info(server, uuid, instance)
        seg_instance = syn_info["Base"]["Syncs"][0]
        if isinstance(box, str):
            assert box == seg_instance, \
                ("The segmentation instance name you provided doesn't match the name of the sync'd instance.\n"
                 "Please provide an explicit bounding-box.")
        box = fetch_volume_box(server, uuid, seg_instance)

    box = np.asarray(box)
    assert (box % 64 == 0).all(), "box must be block-aligned"

    chunk_boxes = boxes_from_grid(box, chunk_shape, clipped=True)
    _erase = partial(_erase_chunk, server, uuid, instance)
    compute_parallel(_erase, chunk_boxes, processes=32)


def _erase_chunk(server, uuid, instance, chunk_box):
    """
    Helper for delete_all_synapses().
    Fetch all blocks in the chunk (to see which blocks have data)
    and erase the ones that aren't empty.
    """
    EMPTY = []
    chunk_data = fetch_blocks(server, uuid, instance, chunk_box)
    empty_data = {k:EMPTY for k,v in chunk_data.items() if v}
    post_blocks(server, uuid, instance, empty_data, kafkalog=False)


def compute_tbar_jsons(partner_df):
    """
    Compute the element JSON data that corresponds to the tbars in the given partner table
    """
    block_ids = partner_df[['z_pre', 'y_pre', 'z_pre']].values // 64
    assert np.equal.reduce(block_ids, axis=0).all()
    
    tbar_jsons = []
    for _pre_id, tbar_df in partner_df.groupby('pre_id'):
        tbar_xyz = tbar_df[['x_pre', 'y_pre', 'z_pre']].values[0].tolist()
        tbar_conf = tbar_df['conf_pre'].iloc[0]
        tbar_jsons.append({
            "Pos": tbar_xyz,
            "Kind": "PreSyn",
            "Tags": [],
            "Prop": {"conf": str(tbar_conf), "user": "$fpl"},
            "Rels": [{"Rel": "PreSynTo", "To":c} for c in tbar_df[['x_post', 'y_post', 'z_post']].values.tolist()]
        })
    return tbar_jsons


def compute_psd_jsons(partner_df):
    """
    Compute the element JSON data that corresponds to the PSDs in the given partner table
    """
    block_ids = partner_df[['z_post', 'y_post', 'z_post']].values // 64
    assert np.equal.reduce(block_ids, axis=0).all()
    
    psd_jsons = []
    for row in partner_df.itertuples():
        psd_jsons.append({
            "Pos": [int(row.x_post), int(row.y_post), int(row.z_post)],
            "Kind": "PostSyn",
            "Tags": [],
            "Prop": {"conf": str(row.conf_post), "user": "$fpl"},
            "Rels": [{"Rel": "PostSynTo", "To": [int(row.x_pre), int(row.y_pre), int(row.z_pre)]}]
        })
    return psd_jsons



def load_gary_psds(pkl_path):
    """
    Load a pickle file as given by Gary's code and return a 'partner table'.
    """
    import pickle
    data = pickle.load(open(pkl_path, 'rb'))
    _table = []
    for tbar_coord, tbar_conf, psd_coords, psd_confs in tqdm_proxy(zip(data['locs'], data['conf'], data['psds'], data['psds_conf']), total=len(data['locs'])):
        for psd_coord, psd_conf in zip(psd_coords, psd_confs):
            _table.append([*(tbar_coord[::-1]), tbar_conf, *(psd_coord[::-1]), psd_conf])

    df = pd.DataFrame(_table, columns=['z_pre', 'y_pre', 'x_pre', 'conf_pre', 'z_post', 'y_post', 'x_post', 'conf_post'])

    for col in ['z_pre', 'y_pre', 'x_pre', 'z_post', 'y_post', 'x_post']:
        df[col] = df[col].astype(np.int32)

    df['pre_id'] = encode_coords_to_uint64(df[['z_pre', 'y_pre', 'x_pre']].values)
    df['post_id'] = encode_coords_to_uint64(df[['z_post', 'y_post', 'x_post']].values)

    df['user_pre'] = df['user_post'] = '$fpl'
    df['kind_pre'] = 'PreSyn'
    df['kind_post'] = 'PostSyn'
    
    df = df[['pre_id', 'z_pre', 'y_pre', 'x_pre', 'kind_pre', 'conf_pre', 'user_pre',
             'post_id', 'z_post', 'y_post', 'x_post', 'kind_post', 'conf_post', 'user_post']]
    return df


def add_synapses(point_df, partner_df, new_psd_partners_df):
    """
    Add the PSDs from new_psd_partners_df, which may reference
    existing tbars, or may reference new tbars, in which
    case the tbars will be added, too.
    """
    POINT_COLS = ['z', 'y', 'x', 'kind', 'conf', 'user']
    PARTNER_COLS_PRE = ['pre_id', 'z_pre', 'y_pre', 'x_pre', 'kind_pre', 'conf_pre', 'user_pre']
    PARTNER_COLS_POST = ['post_id', 'z_post', 'y_post', 'x_post', 'kind_post', 'conf_post', 'user_post']
    PARTNER_COLS = [*PARTNER_COLS_PRE, *PARTNER_COLS_POST]
    
    partner_df = partner_df[PARTNER_COLS]
    new_psd_partners_df = new_psd_partners_df[PARTNER_COLS]

    # Check for possible conflicts before we begin
    conflicts = (pd.Index(new_psd_partners_df['pre_id'].values)
                     .intersection(new_psd_partners_df['post_id'].values))
    if len(conflicts) > 0:
        raise RuntimeError("tbars and psds in the new set overlap!")
    
    conflicts = (pd.Index(new_psd_partners_df['pre_id'].values)
                     .intersection(partner_df['post_id'].values))
    if len(conflicts) > 0:
        raise RuntimeError("tbars in the new set overlap with psds in the old set!")

    conflicts = (pd.Index(new_psd_partners_df['post_id'].values)
                     .intersection(partner_df['pre_id'].values))
    if len(conflicts) > 0:
        raise RuntimeError("psds in the new set overlap with tbars in the old set!")

    partner_df = pd.concat((partner_df, new_psd_partners_df), ignore_index=True, sort=True)
    partner_df.drop_duplicates(['pre_id', 'post_id'], keep='last', inplace=True)

    # Update points
    new_points_pre = (new_psd_partners_df
                          .rename(columns={'pre_id': 'point_id', **dict(zip(PARTNER_COLS_PRE[1:], POINT_COLS))})
                          .drop_duplicates('point_id', keep='last')
                          .set_index('point_id'))

    new_points_post = (new_psd_partners_df
                          .rename(columns={'post_id': 'point_id', **dict(zip(PARTNER_COLS_POST[1:], POINT_COLS))})
                          .drop_duplicates('point_id', keep='last')
                          .set_index('point_id'))

    point_df = pd.concat((point_df, new_points_pre, new_points_post), sort=True)
    
    # Drop duplicate point_ids, keep new
    point_df = point_df.loc[~point_df.index.duplicated(keep='last')]

    return point_df, partner_df


def delete_psds(point_df, partner_df, obsolete_partner_df):
    """
    Delete the PSDs listed in the given obsolete_partner_df.
    If any tbars are left with no partners, delete those tbars, too.
    """
    obsolete_partner_df = obsolete_partner_df[['pre_id', 'post_id']]
    obsolete_pre_ids = obsolete_partner_df['pre_id'].values
    obsolete_post_ids = obsolete_partner_df['post_id'].values

    # Drop obsolete PSDs
    point_df = point_df.query('kind == "PreSyn" or point_id not in @obsolete_post_ids')
    partner_df = partner_df.query('post_id not in @obsolete_post_ids')

    # Delete empty tbars
    remaining_tbar_ids = partner_df['pre_id'].unique()
    dropped_tbar_ids = obsolete_partner_df.query('pre_id not in @remaining_tbar_ids')['pre_id'].unique()
    point_df = point_df.query('kind == "PostSyn" or point_id not in @dropped_tbar_ids')
    
    return point_df.copy(), partner_df.copy()


def delete_tbars(point_df, partner_df, obsolete_tbar_ids):
    """
    Delete the given tbars and all of their associated PSDs.
    """
    _obsolete_psd_ids = partner_df.query('pre_id in @obsolete_tbar_ids')['post_id'].values
    partner_df = partner_df.query('pre_id not in @obsolete_tbar_ids')

    q = ('    (kind == "PreSyn"  and point_id not in @obsolete_tbar_ids)'
         ' or (kind == "PostSyn" and point_id not in @_obsolete_psd_ids)')
    point_df = point_df.query(q)

    return point_df.copy(), partner_df.copy()


def select_autapses(partner_df):
    """
    Select rows from the given 'partner table' that correspond to autapses.
    Must have columns body_pre and body_post.
    """ 
    return partner_df.query('body_pre == body_post')


def select_redundant_psds(partner_df):
    """
    Select rows of the given 'partner table' that correspond to redundant PSDs.
    If a tbar has more than one connection to the same body, then all but one
    of them are considered redundant.
    This function returns the less confident PSD entries as redundant.
    """
    if 'conf_post' in partner_df:
        partner_df = partner_df.sort_values('conf_post')
    else:
        logger.warning("DataFrame has no 'conf_post' column.  Discarding redundant PSDs in arbitrary order.")

    dupe_psd_rows = partner_df.duplicated(['pre_id', 'body_post'], keep='last')
    dupe_partner_df = partner_df.loc[dupe_psd_rows]
    return dupe_partner_df.copy()


