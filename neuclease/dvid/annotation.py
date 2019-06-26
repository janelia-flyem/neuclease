import sys
import logging
from itertools import chain
from functools import partial

import ujson
import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from . import dvid_api_wrapper, fetch_generic_json
from .node import fetch_instance_info
from .voxels import fetch_volume_box
from ..util import Timer, Grid, boxes_from_grid, round_box, tqdm_proxy, gen_json_objects, encode_coords_to_uint64, compute_parallel

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

    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/sync', json=body, params=params)
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
    
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/reload', params=params)
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
    data = fetch_generic_json(url, session=session)
    
    # The endooint returns 'null' instead of an empty list, on old servers at least.
    # But we always return a list.
    return data or []


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
            Elements as JSON data (a python list-of-dicts)
        
        kafkalog:
            If True, log kafka events for each posted element.
    """
    params = {}
    if not kafkalog or kafkalog == 'off':
        params['kafkalog'] = 'off'
    
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/elements', json=elements, params=params)
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

    url = f'http://{server}/api/node/{uuid}/{instance}/blocks/{shape_str}/{offset_str}'
    return fetch_generic_json(url, session=session)


def load_synapses_as_dataframes(elements):
    """
    Load the given JSON elements as synapses a DataFrame.
    
    Args:
        elements:
            JSON list of synapse annotation elements as returned by
            fetch_elements(), etc.

    Returns:
        point_df:
            One row for every t-bar and psd in the file, indicating its
            location, confidence, and synapse type (PostSyn or PreSyn)
            Columns: ['z', 'y', 'x', 'conf', 'kind', 'user']
            Index: np.uint64, an encoded version of [z,y,x]
        
        partner_df:
            Indicates which T-bar each PSD is associated with.
            One row for every psd in the file.
            Columns: ['post_id', 'pre_id']
            where the values correspond to the index of point_df.
            Note:
                It can generally be assumed that for the synapses we
                load into dvid, every PSD (PostSyn) is
                associated with exactly one T-bar (PreSyn).
    """
    if not elements:
        point_df = pd.DataFrame([], columns=['x', 'y', 'z', 'kind', 'conf', 'user'])
        partner_df = pd.DataFrame([], columns=['post_id', 'pre_id'], dtype=np.uint64)
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
    rel_points = []
    
    for e in elements:
        x,y,z = e['Pos']
        
        xs.append(x)
        ys.append(y)
        zs.append(z)
        
        kinds.append( e['Kind'] )
        confs.append( float(e.get('Prop', {}).get('conf', 0.0)) )
        users.append( e.get('Prop', {}).get('user', '') )
        
        if 'Rels' in e:
            for rel in e['Rels']:
                rx, ry, rz = rel['To']
                if e['Kind'] == 'PreSyn':
                    rel_points.append( (z,y,x, rz,ry,rx) )
                else:
                    rel_points.append( (rz,ry,rx, z,y,x) )

    point_df = pd.DataFrame( {'z': zs, 'y': ys, 'x': xs}, dtype=np.int32 )
    point_df['kind'] = pd.Series(kinds, dtype='category')
    point_df['conf'] = pd.Series(confs, dtype=np.float32)
    point_df['user'] = pd.Series(users, dtype='category')
    point_df.index = encode_coords_to_uint64(point_df[['z', 'y', 'x']].values)
    point_df.index.name = 'point_id'
    
    rel_points = np.array(rel_points, np.int32)
    pre_partner_ids = encode_coords_to_uint64(rel_points[:,:3])
    post_partner_ids = encode_coords_to_uint64(rel_points[:,3:])
    partner_df = pd.DataFrame({'post_id': post_partner_ids, 'pre_id': pre_partner_ids})
    
    # For synapses near block borders, maybe only the PreSyn or
    # only the PostSyn happens to be in the given elements.
    # But in most cases, both PreSyn and PostSyn are present,
    # and therefore the relationship is probably listed twice.
    # Drop duplicates.
    partner_df.drop_duplicates(inplace=True)
    
    return point_df, partner_df


def fetch_synapses_in_batches(server, uuid, synapses_instance, bounding_box_zyx=None, batch_shape_zyx=(256,256,64000),
                              format='pandas', endpoint='blocks', processes=8): #@ReservedAssignment
    """
    Fetch all synapse annotations for the given labelmap volume (or subvolume) and synapse instance.
    Box-shaped regions are queried in batches according to the given batch shape.
    Returns either the raw JSON or a pandas DataFrame.
    
    Note:
        On the hemibrain dataset (~70 million points),
        this function takes ~4 minutes if you use 32 processes.
    
    Warning:
        For large volumes with many synapses, the 'json' format requires a lot of RAM,
        and is not particularly convenient to save/load.
    
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
    
    Returns:
        If format == 'json', a list of JSON elements.
        If format == 'pandas', returns two dataframes.
        (See ``load_synapses_as_dataframes()`` for details.)
    """
    assert format in ('pandas', 'json')
    assert endpoint in ('blocks', 'elements')

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

    boxes = boxes_from_grid(bounding_box_zyx, Grid(batch_shape_zyx))
    fn = partial(_fetch_synapse_batch, server, uuid, synapses_instance, format=format, endpoint=endpoint)
    results = compute_parallel(fn, boxes, processes=processes, ordered=False, leave_progress=True)

    if format == 'json':
        return list(chain(*results))
    elif format == 'pandas':
        point_dfs, partner_dfs = zip(*results)
        
        # Any zero-length dataframes might have the wrong dtypes,
        # which would screw up the concat step.  Remove them.
        point_dfs = filter(len, point_dfs)
        partner_dfs = filter(len, partner_dfs)

        point_df = pd.concat(point_dfs)
        partner_df = pd.concat(partner_dfs, ignore_index=True)
        partner_df.drop_duplicates(inplace=True)
        return point_df, partner_df


def _fetch_synapse_batch(server, uuid, synapses_instance, batch_box, format='json', endpoint='blocks'): # @ReservedAssignment
    """
    Helper for fetch_synapses_in_batches(), above.
    """
    if endpoint == 'blocks':
        elements = fetch_blocks(server, uuid, synapses_instance, batch_box)
        elements = list(chain(*elements.values()))
    elif endpoint == 'elements':
        elements = fetch_elements(server, uuid, synapses_instance, batch_box)
    else:
        raise AssertionError("Invalid endpoint choice")

    if format == 'json':
        return elements
    elif format == 'pandas':
        point_df, partner_df = load_synapses_as_dataframes(elements)
        return (point_df, partner_df)


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
        logger.warning("*** Synapse table includes body 0 and was therefore probably generated from out-of-date data. ***")
    
    return synapse_counts


def fetch_roi_synapses(server, uuid, synapses_instance, rois, fetch_labels=False, processes=16):
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
            list of DVID ROI instance names, e.g. ['EB', 'FB']
        
        fetch_labels:
            If True, also fetch the supervoxel and body label underneath each synapse,
            returned in columns 'sv' and 'body'.
        
        processes:
            How many parallel processes to use when fetching synapses and supervoxel labels.
    
    Returns:
        pandas DataFrame with columns:
        ``['z', 'y', 'x', 'kind', 'conf']`` and ``['sv', 'body']`` (if ``fetch_labels=True``)
    
    Example:
        df = fetch_roi_synapses('emdata4:8900', '3c281', 'synapses', ['PB(L5)', 'PB(L7)'], True, 8)
    """
    # Late imports to avoid circular imports in dvid/__init__
    from neuclease.dvid import fetch_combined_roi_volume, determine_point_rois, fetch_labels_batched, fetch_mapping, fetch_mappings
    
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
    points_df, _partners_df = fetch_synapses_in_batches(server, uuid, synapses_instance, roi_box, processes=processes)
    
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
                bodies = fetch_mapping(server, uuid, seg_instance, svs).values
            else:
                mapping = fetch_mappings(server, uuid, seg_instance)
                mapper = LabelMapper(mapping.index.values, mapping.values)
                bodies = mapper.apply(svs, True)

        points_df['sv'] = svs
        points_df['body'] = bodies
        columns += ['body', 'sv']
    
    return points_df[columns]
