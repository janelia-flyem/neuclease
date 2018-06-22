import json
import logging
import getpass
import threading
import functools
from datetime import datetime
from io import BytesIO
from collections import namedtuple

import requests

import numpy as np
import pandas as pd
from numba import jit

from libdvid import DVIDNodeService

from ..util import Timer, uuids_match

DEFAULT_DVID_SESSIONS = {}
DEFAULT_APPNAME = "neuclease"

logger = logging.getLogger(__name__)

DvidInstanceInfo = namedtuple("DvidInstanceInfo", "server uuid instance")


def default_dvid_session(appname=None):
    """
    Return a default requests.Session() object that automatically appends the
    'u' and 'app' query string parameters to every request.
    """
    if appname is None:
        appname = DEFAULT_APPNAME
    # Technically, request sessions are not threadsafe,
    # so we keep one for each thread.
    thread_id = threading.current_thread().ident
    try:
        s = DEFAULT_DVID_SESSIONS[(appname, thread_id)]
    except KeyError:
        s = requests.Session()
        s.params = { 'u': getpass.getuser(),
                     'app': appname }
        DEFAULT_DVID_SESSIONS[(appname, thread_id)] = s

    return s


def sanitize_server(f):
    """
    Decorator for functions whose first arg is either a string or a DvidInstanceInfo (or similar tuple).
    If the server address begins with 'http://', that prefix is stripped from it.
    """
    @functools.wraps(f)
    def wrapper(instance_info, *args, **kwargs):
        if isinstance(instance_info, str):
            if instance_info.startswith('http://'):
                instance_info = instance_info[len('http://'):]
        else:
            server, uuid, instance = instance_info
            if server.startswith('http://'):
                server = server[len('http://'):]
            instance_info = DvidInstanceInfo(server, uuid, instance)

        return f(instance_info, *args, **kwargs)
    return wrapper


def fetch_generic_json(url, json=None):
    session = default_dvid_session()
    r = session.get(url, json=json)
    r.raise_for_status()
    return r.json()


@sanitize_server
def fetch_supervoxels_for_body(instance_info, body_id, user=None):
    server, uuid, instance = instance_info
    query_params = {}
    if user:
        query_params['u'] = user

    url = f'http://{server}/api/node/{uuid}/{instance}/supervoxels/{body_id}'
    r = default_dvid_session().get(url, params=query_params)
    r.raise_for_status()
    supervoxels = np.array(r.json(), np.uint64)
    supervoxels.sort()
    return supervoxels


@sanitize_server
def fetch_body_size(instance_info, body_id, supervoxels=False):
    server, uuid, instance = instance_info
    supervoxels = str(bool(supervoxels)).lower()
    url = f'http://{server}/api/node/{uuid}/{instance}/size/{body_id}?supervoxels={supervoxels}'
    response = fetch_generic_json(url)
    return response['voxels']


@sanitize_server
def fetch_body_sizes(instance_info, body_ids, supervoxels=False):
    server, uuid, instance = instance_info
    body_ids = list(map(int, body_ids))
    supervoxels = str(bool(supervoxels)).lower()

    url = f'http://{server}/api/node/{uuid}/{instance}/sizes?supervoxels={supervoxels}'
    r = default_dvid_session().get(url, json=body_ids)
    r.raise_for_status()
    return r.json()


@sanitize_server
def fetch_supervoxel_sizes_for_body(instance_info, body_id, user=None):
    """
    Return the sizes of all supervoxels in a body 
    """
    server, uuid, instance = instance_info
    supervoxels = fetch_supervoxels_for_body(instance_info, body_id, user)
    
    query_params = {}
    if user:
        query_params['u'] = user

    url = f'http://{server}/api/node/{uuid}/{instance}/sizes?supervoxels=true'
    r = default_dvid_session().get(url, params=query_params, json=supervoxels.tolist())
    r.raise_for_status()
    sizes = np.array(r.json(), np.uint32)
    
    series = pd.Series(data=sizes, index=supervoxels)
    series.index.name = 'sv'
    series.name = 'size'
    return series


@sanitize_server
def fetch_label_for_coordinate(instance_info, coordinate_zyx, supervoxels=False):
    server, uuid, instance = instance_info
    session = default_dvid_session()
    coord_xyz = np.array(coordinate_zyx)[::-1]
    coord_str = '_'.join(map(str, coord_xyz))
    supervoxels = str(bool(supervoxels)).lower()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/label/{coord_str}?supervoxels={supervoxels}')
    r.raise_for_status()
    return np.uint64(r.json()["Label"])


@sanitize_server
def fetch_sparsevol_rles(instance_info, label, supervoxels=False, scale=0):
    server, uuid, instance = instance_info
    session = default_dvid_session()
    supervoxels = str(bool(supervoxels)).lower() # to lowercase string
    url = f'http://{server}/api/node/{uuid}/{instance}/sparsevol/{label}?supervoxels={supervoxels}&scale={scale}'
    r = session.get(url)
    r.raise_for_status()
    return r.content


def extract_rle_size_and_first_coord(rle_payload_bytes):
    """
    Given a binary RLE payload as returned by the /sparsevol endpoint,
    extract the count of voxels in the RLE and the first coordinate in the RLE. 
    
    Args:
        rle_payload_bytes:
            Bytes. Must be in DVID's "Legacy RLEs" format.

    Useful for sampling label value under a given RLE geometry
    (assuming all of the points in the RLE share the same label).
    
    Returns:
        voxel_count, coord_zyx
    """
    assert (len(rle_payload_bytes) - 3*4) % (4*4) == 0, \
        "Payload does not appear to be an RLE payload as defined by DVID's 'Legacy RLE' format."
    rles = np.frombuffer(rle_payload_bytes, dtype=np.uint32)[3:]
    rles = rles.reshape(-1, 4)
    first_coord_xyz = rles[0, :3]
    first_coord_zyx = first_coord_xyz[::-1]

    voxel_count = rles[:, 3].sum()
    return voxel_count, first_coord_zyx


@sanitize_server
def split_supervoxel(instance_info, supervoxel, rle_payload_bytes):
    """
    Split the given supervoxel according to the provided RLE payload, as specified in DVID's split-supervoxel docs.
    
    Returns:
        The two new IDs resulting from the split: (split_sv_id, remaining_sv_id)
    """
    server, uuid, instance = instance_info
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/split-supervoxel/{supervoxel}', data=rle_payload_bytes)
    r.raise_for_status()
    
    results = r.json()
    return (results["SplitSupervoxel"], results["RemainSupervoxel"] )


@sanitize_server
def fetch_mapping(instance_info, supervoxel_ids):
    server, uuid, instance = instance_info
    supervoxel_ids = list(map(int, supervoxel_ids))
    body_ids = fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/mapping', json=supervoxel_ids)
    return body_ids


@sanitize_server
def fetch_mappings(instance_info, include_identities=True, retired_supervoxels=[]):
    """
    Fetch the complete sv-to-label mapping table from DVID and return it as a pandas Series (indexed by sv).
    
    Args:
        include_identities:
            If True, add rows for identity mappings (which are not included in DVID's response).
        
        retired_supervoxels:
            A set of supervoxels NOT to automatically add as identity mappings,
            e.g. due to the fact that they were split.
    
    Returns:
        pd.Series(index=sv, data=body)
    """
    server, uuid, instance = instance_info
    session = default_dvid_session()
    
    # This takes ~30 seconds so it's nice to log it.
    uri = f"http://{server}/api/node/{uuid}/{instance}/mappings"
    with Timer(f"Fetching {uri}", logger):
        r = session.get(uri)
        r.raise_for_status()

    with Timer(f"Parsing mapping", logger), BytesIO(r.content) as f:
        df = pd.read_csv(f, sep=' ', header=None, names=['sv', 'body'], engine='c', dtype=np.uint64)

    if include_identities:
        with Timer(f"Appending missing identity-mappings", logger), BytesIO(r.content) as f:
            missing_idents = set(df['body']) - set(df['sv']) - set(retired_supervoxels)
            missing_idents = np.fromiter(missing_idents, np.uint64)
            missing_idents.sort()
            
            idents_df = pd.DataFrame({'sv': missing_idents, 'body': missing_idents})
            df = pd.concat((df, idents_df), ignore_index=True)

    df.set_index('sv', inplace=True)

    return df['body']


@sanitize_server
def fetch_mutation_id(instance_info, body_id):
    server, uuid, instance = instance_info
    response = fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/lastmod/{body_id}')
    return response["mutation id"]


@sanitize_server
def fetch_sparsevol_coarse(instance_info, body_id, supervoxels=False):
    """
    Return the 'coarse sparsevol' representation of a given body.
    This is similar to the sparsevol representation at scale=6,
    EXCEPT that it is generated from the label index, so no blocks
    are lost from downsampling.

    Return an array of coordinates of the form:

        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]
    """
    server, uuid, instance = instance_info
    supervoxels = str(bool(supervoxels)).lower()
    session = default_dvid_session()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/sparsevol-coarse/{body_id}?supervoxels={supervoxels}')
    r.raise_for_status()
    
    return parse_rle_response( r.content )


def fetch_sparsevol(instance_info, label, supervoxels=False, scale=0):
    """
    Return coordinates of all voxels in the given body/supervoxel at the given scale.

    Note: At scale 0, this will be a LOT of data for any reasonably large body.
          Use with caution.
    """
    rles = fetch_sparsevol_rles(instance_info, label, supervoxels, scale)
    return parse_rle_response(rles)

def parse_rle_response(response_bytes):
    """
    Parse a (legacy) RLE response from DVID, used by various endpoints
    such as 'sparsevol' and 'sparsevol-coarse'.
    
    Return an array of coordinates of the form:

        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]
    """
    descriptor = response_bytes[0]
    ndim = response_bytes[1]
    run_dimension = response_bytes[2]

    assert descriptor == 0, f"Don't know how to handle this payload. (descriptor: {descriptor})"
    assert ndim == 3, "Expected XYZ run-lengths"
    assert run_dimension == 0, "FIXME, we assume the run dimension is X"

    content_as_int32 = np.frombuffer(response_bytes, np.int32)
    _voxel_count = content_as_int32[1]
    run_count = content_as_int32[2]
    rle_items = content_as_int32[3:].reshape(-1,4)

    assert len(rle_items) == run_count, \
        f"run_count ({run_count}) doesn't match data array length ({len(rle_items)})"

    rle_starts_xyz = rle_items[:,:3]
    rle_starts_zyx = rle_starts_xyz[:,::-1]
    rle_lengths = rle_items[:,3]

    # Sadly, the decode function requires contiguous arrays, so we must copy.
    rle_starts_zyx = rle_starts_zyx.copy('C')
    rle_lengths = rle_lengths.copy('C')

    # For now, DVID always returns a voxel_count of 0, so we can't make this assertion.
    #assert rle_lengths.sum() == _voxel_count,\
    #    f"Voxel count ({voxel_count}) doesn't match expected sum of run-lengths ({rle_lengths.sum()})"

    dense_coords = runlength_decode_from_lengths(rle_starts_zyx, rle_lengths)
    
    assert rle_lengths.sum() == len(dense_coords), "Got the wrong number of coordinates!"
    return dense_coords


@jit("i4[:,:](i4[:,::1],i4[::1])", nopython=True) # See note about signature, below.
def runlength_decode_from_lengths(rle_start_coords_zyx, rle_lengths):
    """
    Given a 2D array of coordinates and a 1D array of runlengths, i.e.:
        
        [[Z,Y,X], [Z,Y,X], [Z,Y,X],...]

        and 
        
        [Length, Length, Length,...]

    Return an array of coordinates of the form:

        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]
    
    In which every run-length has been expanded into a run
    of consecutive coordinates in the result.
    That is, result.shape == (rle_lengths.sum(), 3)
    
    Note: The "runs" are expanded along the X AXIS.
    
    Note about Signature:
    
        Due to an apparent numba bug, it is dangerous to pass non-contiguous arrays to this function.
        (It returns incorrect results.)
        
        Therefore, the signature is explicitly written above to require contiguous arrays (e.g. i4[::1]),
        If you attempt to pass a non-contiguous array, you'll see an error like this:
        
            TypeError: No matching definition for argument type(s) readonly array(int32, 2d, A), readonly array(int32, 1d, C)
    """
    # Numba doesn't allow us to use empty lists at all,
    # so we have to initialize this list with a dummy row,
    # which we'll omit in the return value
    coords = [0,0,0]
    for i in range(len(rle_start_coords_zyx)):
        (z, y, x0) = rle_start_coords_zyx[i]
        length = rle_lengths[i]
        for x in range(x0, x0+length):
            coords.extend([z,y,x])

    coords = np.array(coords, np.int32).reshape((-1,3))
    return coords[1:, :] # omit dummy row (see above)


def compute_changed_bodies(instance_info_a, instance_info_b):
    """
    Returns the list of all bodies whose supervoxels changed
    between uuid_a and uuid_b.
    This includes bodies that were changed, added, or removed completely.
    """
    mapping_a = fetch_mappings(instance_info_a)
    mapping_b = fetch_mappings(instance_info_b)
    
    assert mapping_a.name == 'body'
    assert mapping_b.name == 'body'
    
    mapping_a = pd.DataFrame(mapping_a)
    mapping_b = pd.DataFrame(mapping_b)
    
    logger.info("Aligning mappings")
    df = mapping_a.merge(mapping_b, 'outer', left_index=True, right_index=True, suffixes=['_a', '_b'], copy=False)

    changed_df = df.query('body_a != body_b')
    changed_df.fillna(0)
    changed_bodies = np.unique(changed_df.values.astype(np.uint64))
    if changed_bodies[0] == 0:
        changed_bodies = changed_bodies[1:]
    return changed_bodies


@sanitize_server
def fetch_server_info(server):
    return fetch_generic_json(f'http://{server}/api/server/info')


@sanitize_server
def fetch_full_instance_info(instance_info):
    """
    Returns the full JSON instance info from DVID
    """
    server, uuid, instance = instance_info
    return fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/info')

@sanitize_server
def generate_sample_coordinate(instance_info, body_id, supervoxels=False):
    """
    Return an arbitrary coordinate that lies within the given body.
    Usually faster than fetching all the RLEs.
    """
    server, uuid, instance = instance_info
    
    # FIXME: I'm using sparsevol instead of sparsevol-coarse due to an apparent bug in DVID at the moment
    SCALE = 2
    coarse_block_coords = fetch_sparsevol(instance_info, body_id, supervoxels, scale=SCALE)
    first_block_coord = (2**SCALE) * np.array(coarse_block_coords[0]) // 64 * 64
    
    ns = DVIDNodeService(server, uuid)
    first_block = ns.get_labelarray_blocks3D( instance, (64,64,64), first_block_coord, supervoxels=supervoxels )
    nonzero_coords = np.transpose((first_block == body_id).nonzero())
    if len(nonzero_coords) == 0:
        term = {False: 'body', True: 'supervoxel'}[supervoxels]
        raise RuntimeError(f"The sparsevol-coarse info for this {term} ({body_id}) "
                           "appears to be out-of-sync with the scale-0 segmentation.")

    return first_block_coord + nonzero_coords[0]

@sanitize_server
def read_kafka_messages(instance_info, group_id=None, consumer_timeout=2.0, dag_filter='leaf-only', action_filter=None, return_format='json-values'):
    """
    Read the stream of available Kafka messages for the given DVID instance,
    and optionally filter them by UUID or Action.
    
    Args:
        instance_info:
            (server, uuid, instance)
        
        group_id:
            Kafka group ID to use when reading.  If not given, a new one is created.
        
        consumer_timeout:
            Seconds to timeout (after which we assume we've read all messages).
        
        dag_filter:
            How to filter out messages based on the UUID.
            One of:
            - 'leaf-only' (only messages whose uuid matches the provided instance_info),
            - 'leaf-and-parents' (only messages matching the given instance_info uuid or its ancestors), or
            - 'all' (no filtering by UUID).

        action_filter:
            A list of actions to use as a filter for the returned messages.
            For example, if action_filter=['split', 'split-complete'],
            all messages with other actions will be filtered out.

        return_format:
            Either 'records' (return list of kafka ConsumerRecord objects),
            or 'json-values' (return list of parsed JSON structures from each record.value)
    """
    from kafka import KafkaConsumer
    server, uuid, instance = instance_info
    
    assert dag_filter in ('leaf-only', 'leaf-and-parents', 'all')
    assert return_format in ('records', 'json-values')

    if group_id is None:
        # Choose a unique 'group_id' to use
        group_id = getpass.getuser() + '-' + datetime.now().isoformat()
    
    server_info = fetch_server_info(instance_info[0])
    kafka_server = server_info["Kafka Servers"]

    full_instance_info = fetch_full_instance_info(instance_info)
    data_uuid = full_instance_info["Base"]["DataUUID"]
    repo_uuid = full_instance_info["Base"]["RepoUUID"]

    consumer = KafkaConsumer( bootstrap_servers=[kafka_server],
                              group_id=group_id,
                              enable_auto_commit=False,
                              auto_offset_reset='earliest',
                              consumer_timeout_ms=int(consumer_timeout * 1000))

    consumer.subscribe([f'dvidrepo-{repo_uuid}-data-{data_uuid}'])

    logger.info(f"Reading kafka messages from {kafka_server} for {server} / {uuid} / {instance}")
    with Timer() as timer:
        # Read all messages (until consumer timeout)
        records = list(consumer)
    logger.info(f"Reading {len(records)} kafka messages took {timer.seconds} seconds")

    values = [json.loads(msg.value) for msg in records]
    records_and_values = zip(records, values)

    if dag_filter == 'leaf-only':
        records_and_values = filter(lambda r_v: uuids_match(r_v[1]["UUID"], uuid), records_and_values)
    elif dag_filter == 'leaf-and-parents':
        raise NotImplementedError("FIXME")
    elif dag_filter == 'all':
        pass
    else:
        assert False

    if action_filter is not None:
        if isinstance(action_filter, str):
            action_filter = [action_filter]
        action_filter = set(action_filter)
        records_and_values = filter(lambda r_v: r_v[1]["Action"] in action_filter, records_and_values)
        
    records, values = zip(*records_and_values)

    if return_format == 'records':
        return records
    elif return_format == 'json-values':
        return values
    else:
        assert False


