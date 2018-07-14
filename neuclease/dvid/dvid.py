import os
import json
import logging
import getpass
import threading
import functools

from datetime import datetime
from io import BytesIO
from itertools import starmap
from collections import namedtuple, Iterable

import requests
import networkx as nx

import numpy as np
import pandas as pd
from numba import jit

from libdvid import DVIDNodeService

from ..util import Timer, uuids_match, round_box, extract_subvol

# The labelops_pb2 file was generated with the following commands:
# $ cd neuclease/dvid
# $ protoc --python_out=. labelops.proto
# $ sed -i '' s/labelops_pb2/neuclease.dvid.labelops_pb2/g labelops_pb2.py
from .labelops_pb2 import LabelIndex


DEFAULT_DVID_SESSIONS = {}
DEFAULT_DVID_NODE_SERVICES = {}
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
    pid = os.getpid()
    try:
        s = DEFAULT_DVID_SESSIONS[(appname, thread_id, pid)]
    except KeyError:
        s = requests.Session()
        s.params = { 'u': getpass.getuser(),
                     'app': appname }
        DEFAULT_DVID_SESSIONS[(appname, thread_id, pid)] = s

    return s

def default_node_service(server, uuid, appname=None):
    if appname is None:
        appname = DEFAULT_APPNAME

    # One per thread/process
    thread_id = threading.current_thread().ident
    pid = os.getpid()

    try:
        ns = DEFAULT_DVID_NODE_SERVICES[(appname, thread_id, pid, server, uuid)]
    except KeyError:
        ns = DVIDNodeService(server, str(uuid), getpass.getuser(), appname)

    return ns

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

        try:
            return f(instance_info, *args, **kwargs)
        except requests.HTTPError as ex:
            # If the error response had content (and it's not super-long),
            # show that in the traceback, too.  DVID error messages are often helpful.
            if ( not hasattr(ex, 'response_content_appended')
                 and ex.response is not None
                 and ex.response.content
                 and len(ex.response.content) <= 200 ):
                
                msg = str(ex.args[0]) + "\n" + ex.response.content.decode('utf-8')
                new_ex = requests.HTTPError(msg, *args[1:])
                new_ex.response_content_appended = True
                raise new_ex from ex
            else:
                raise
    return wrapper


def fetch_generic_json(url, json=None):
    session = default_dvid_session()
    r = session.get(url, json=json)
    r.raise_for_status()
    return r.json()


INSTANCE_TYPENAMES = """\
annotation
float32blk
googlevoxels
imagetile
keyvalue
labelarray
labelblk
labelgraph
labelmap
labelsz
labelvol
multichan16
rgba8blk
roi
tarsupervoxels
uint16blk
uint32blk
uint64blk
uint8blk
""".split()


@sanitize_server
def create_instance(instance_info, typename, versioned=True, compression=None, tags=[], type_specific_settings={}):
    """
    Create a data instance of the given type.
    
    typename:
        Valid instance names are listed in INSTANCE_TYPENAMES
    
    versioned:
        Whether or not the instance should be versioned.

    compression:
        Which compression DVID should use when storing the data in the instance.
        Different instance types support different compression options.
        Typical choices are: ['none', 'snappy', 'lz4', 'gzip'].
        
        Note: Here, the string 'none' means "use no compression",
              whereas a Python None value means "Let DVID choose a default compression type".

    tags:
        Optional 'tags' to initialize the instance with, e.g. "type=meshes".
    
    Returns:
        True if the instance was newly created, False if it already existed.

    """
    server, uuid, instance = instance_info
    assert typename in INSTANCE_TYPENAMES, f"Unknown typename: {typename}"

    settings = {}
    settings["dataname"] = instance
    settings["typename"] = typename

    if not versioned:
        settings["versioned"] = 'false'
    
    if typename == 'tarsupervoxels':
        # Will DVID return an error for us in these cases?
        # If so, we can remove these asserts...
        assert not versioned, "Instances of tarsupervoxels must be unversioned"
        assert compression in (None, 'none'), "Compression not supported for tarsupervoxels"

    if compression is not None:
        assert compression in ('none', 'snappy', 'lz4', 'gzip') # jpeg is also supported, but then we need to parse e.g. jpeg:80
        settings["Compression"] = compression
    
    if tags:
        settings["Tags"] = ','.join(tags)
    
    settings.update(type_specific_settings)
    
    r = requests.post(f"http://{server}/api/repo/{uuid}/instance", json=settings)
    r.raise_for_status()


def create_voxel_instance(instance_info, typename, versioned=True, compression=None, tags=[],
                          block_size=64, voxel_size=8.0, voxel_units='nanometers', background=None,
                          type_specific_settings={}):
    """
    Create an instance of a voxel datatype.
    """
    assert typename in ("uint8blk", "uint16blk", "uint32blk", "uint64blk", "float32blk", "labelblk", "labelarray", "labelmap")

    if not isinstance(Iterable, block_size):
        block_size = 3*(block_size,)

    if not isinstance(Iterable, voxel_size):
        voxel_size = 3*(voxel_size,)

    block_size_str = ','.join(map(str, block_size))
    voxel_size_str = ','.join(map(str, voxel_size))

    type_specific_settings = dict(type_specific_settings)
    type_specific_settings["BlockSize"] = block_size_str
    type_specific_settings["VoxelSize"] = voxel_size_str
    type_specific_settings["VoxelUnits"] = voxel_units
    
    if background is not None:
        assert typename in ("uint8blk", "uint16blk", "uint32blk", "uint64blk", "float32blk"), \
            "Background value is only valid for block-based instance types."
        type_specific_settings["Background"] = background
    
    create_instance(instance_info, typename, versioned, compression, tags, type_specific_settings)


def create_labelmap_instance(instance_info, tags=[], block_size=64, voxel_size=8.0,
                             voxel_units='nanometers', enable_index=True, max_scale=0):
    """
    Create a labelmap instance.

    Args:
        enable_index:
            Whether or not to support indexing on this label instance
            Should usually be True, except for benchmarking purposes.
        
        max_scale:
            The maximum downres level of this labelmap instance.
        
        Other args passed directly to create_voxel_instance().
    """
    type_specific_settings = {
        "IndexedLabels": str(enable_index).lower(),
        "CountLabels": str(enable_index).lower(),
        "MaxDownresLevel": str(max_scale)
    }
    create_voxel_instance( instance_info, tags=tags, block_size=block_size, voxel_size=voxel_size,
                           voxel_units=voxel_units, type_specific_settings=type_specific_settings )
    

def create_tarsupervoxel_instance(instance_info, sync_instance, extension, tags=[]):
    """
    Create a tarsupervoxel instance and sync it to a labelmap instance.
    """
    if extension[0] == '.':
        extension = extension[1:]

    create_instance(instance_info, "tarsupervoxels", versioned=False, tags=tags,
                    type_specific_settings={"Extension": extension})
    
    post_tarsupervoxel_sync(instance_info, sync_instance)


@sanitize_server
def post_tarsupervoxel_sync(instance_info, sync_instance, replace=False):
    server, uuid, instance = instance_info
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/sync',
                     params={ "replace": str(bool(replace)).lower() },
                     json={ "sync": sync_instance } )
    r.raise_for_status()


@sanitize_server
def create_branch(server, uuid, branch_name, note=None, custom_uuid=None):
    """
    Create a branch from the given UUID with the given new branch name.
    Branch name must be unique (not used previously in the repo).
    
    If a custom UUID is provided, it will be used by DVID instead of auto-generating one.
    Note: Providing a custom UUID is unusual and should probably be avoided.

    Returns:
        The uuid of the new branch.
    """
    session = default_dvid_session()
    body = {"branch": branch_name}
    if note:
        body["note"] = note

    if custom_uuid:
        body["uuid"] = custom_uuid
        
    r = session.post(f'http://{server}/api/node/{uuid}/branch', json=body)
    r.raise_for_status()
    return r.json()["child"]


@sanitize_server
def fetch_keys(instance_info):
    """
    Fetches the complete list of keys in the instance (not their values).
    
    WARNING: In the current version of DVID, which uses the basholeveldb backend,
             this will be VERY SLOW for instances with a lot of data.
             (The speed depends on the total size of the values, not the number of keys.)
    """
    server, uuid, instance = instance_info
    return fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/keys')


@sanitize_server
def fetch_keyrange(instance_info, key1, key2):
    """
    Returns all keys between 'key1' and 'key2' for
    the given data instance (not their values).
    
    WARNING: This can be slow for large ranges.
    """
    server, uuid, instance = instance_info
    url = f'http://{server}/api/node/{uuid}/{instance}/keyrange/{key1}/{key2}'
    return fetch_generic_json(url)
    

@sanitize_server
def fetch_key(instance_info, key, as_json=False):
    server, uuid, instance = instance_info
    session = default_dvid_session()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/key/{key}')
    r.raise_for_status()
    if as_json:
        return r.json()
    return r.content


@sanitize_server
def post_key(instance_info, key, data):
    server, uuid, instance = instance_info
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/key/{key}', data=data)
    r.raise_for_status()
    

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
def fetch_size(instance_info, label_id, supervoxels=False):
    server, uuid, instance = instance_info
    supervoxels = str(bool(supervoxels)).lower()
    url = f'http://{server}/api/node/{uuid}/{instance}/size/{label_id}?supervoxels={supervoxels}'
    response = fetch_generic_json(url)
    return response['voxels']

# Deprecated name
fetch_body_size = fetch_size


@sanitize_server
def fetch_sizes(instance_info, label_ids, supervoxels=False):
    server, uuid, instance = instance_info
    label_ids = list(map(int, label_ids))
    supervoxels = str(bool(supervoxels)).lower()

    url = f'http://{server}/api/node/{uuid}/{instance}/sizes?supervoxels={supervoxels}'
    return fetch_generic_json(url, label_ids)

# Deprecated name
fetch_body_sizes = fetch_sizes


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
def fetch_mappings(instance_info, as_array=False):
    """
    Fetch the complete sv-to-label in-memory mapping table
    from DVID and return it as a numpy array or a pandas Series (indexed by sv).
    (This takes 30-60 seconds for a hemibrain-sized volume.)
    
    NOTE: This returns the 'raw' mapping from DVID, which is usually not useful on its own.
          DVID does not store entries for 'identity' mappings, and it sometimes includes
          entries for supervoxels that have already been 'retired' due to splits.

          See fetch_complete_mappings(), which compensates for these issues.
    
    Args:
        as_array:
            If True, return the mapping as an array with shape (N,2),
            where supervoxel is the first column and body is the second.
            Otherwise, return a  pd.Series
    
    Returns:
        pd.Series(index=sv, data=body), unless as_array is True
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

    if as_array:
        return df.values

    df.set_index('sv', inplace=True)
    
    assert df.index.dtype == np.uint64
    assert df['body'].dtype == np.uint64
    return df['body']


@sanitize_server
def fetch_complete_mappings(instance_info, split_source='kafka', include_retired=False):
    """
    Fetch the complete mapping from DVID for all agglomerated bodies,
    including 'identity' mappings (for agglomerated bodies only)
    and taking split supervoxels into account (discard them, or map them to 0).
    
    This is similar to fetch_mappings() above, but compensates for the incomplete
    mapping from DVID due to identity rows, and filters out retired supervoxels.
    
    (This function takes ~2 minutes to run on the hemibrain volume.)
    
    Note: Single-supervoxel bodies are not necessarily included in this mapping.
          Any supervoxel IDs missing from the results of this function should be
          considered as implicitly mapped to themselves.
    
    Args:
        instance_info:
            server, uuid, instance
        
        split_source:
            Either 'dvid', 'kafka'.
            Required for properly filtering out 'retired' supervoxel IDs.
        
        include_retired:
            If True, include rows for 'retired' supervoxels, which all map to 0.

    Returns:
        pd.Series(index=sv, data=body)
    """
    split_events = fetch_supervoxel_splits(instance_info, split_source)
    split_tables = list(map(lambda t: np.asarray(t, np.uint64), split_events.values()))
    if split_tables:
        split_table = np.concatenate(split_tables)
        retired_svs = split_table[:, SplitEvent._fields.index('old')]
        retired_svs = set(retired_svs)
    else:
        retired_svs = set()

    # Fetch base mapping
    base_mapping = fetch_mappings(instance_info, as_array=True)
    base_svs = base_mapping[:,0]
    base_bodies = base_mapping[:,1]

    # Augment with identity rows, which aren't included in the base.
    with Timer(f"Constructing missing identity-mappings", logger):
        missing_idents = set(base_bodies) - set(base_svs) - retired_svs
        missing_idents = np.fromiter(missing_idents, np.uint64)
        missing_idents_mapping = np.array((missing_idents, missing_idents)).transpose()

    parts = [base_mapping, missing_idents_mapping]

    # Optionally include 'retired' supervoxels -- mapped to 0
    if include_retired:
        retired_svs_array = np.fromiter(retired_svs, np.uint64)
        retired_mapping = np.zeros((len(retired_svs_array), 2), np.uint64)
        retired_mapping[:, 0] = retired_svs_array
        parts.append(retired_mapping)

    # Combine into a single table
    if len(parts) > 1:
        full_mapping = np.concatenate(parts)
    else:
        full_mapping = parts[0]

    full_mapping = np.asarray(full_mapping, order='C')
    
    # View as 1D buffer of structured dtype to sort in-place.
    # (Sorted index is more efficient with speed and RAM in pandas)
    mapping_view = memoryview(full_mapping.reshape(-1))
    np.frombuffer(mapping_view, dtype=[('sv', np.uint64), ('body', np.uint64)]).sort()

    # Construct pd.Series for fast querying
    s = pd.Series(index=full_mapping[:,0], data=full_mapping[:,1])
    s.index.name = 'sv'
    s.name = 'body'
    
    # Drop all rows with retired supervoxels.
    # Someday, this will be unnecessary, but our current production server includes such rows.
    s.drop(retired_svs, inplace=True, errors='ignore')

    assert s.index.dtype == np.uint64
    assert s.dtype == np.uint64
    return s


@sanitize_server
def fetch_mutation_id(instance_info, body_id):
    server, uuid, instance = instance_info
    response = fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/lastmod/{body_id}')
    return response["mutation id"]


@sanitize_server
def fetch_sparsevol_coarse(instance_info, label_id, supervoxels=False):
    """
    Return the 'coarse sparsevol' representation of a given body/supervoxel.
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
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/sparsevol-coarse/{label_id}?supervoxels={supervoxels}')
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
    changed_df.fillna(0, inplace=True)
    changed_bodies = np.unique(changed_df.values.astype(np.uint64))
    if changed_bodies[0] == 0:
        changed_bodies = changed_bodies[1:]
    return changed_bodies


@sanitize_server
def fetch_server_info(server):
    return fetch_generic_json(f'http://{server}/api/server/info')


@sanitize_server
def fetch_repo_info(server, uuid):
    return fetch_generic_json(f'http://{server}/api/repo/{uuid}/info')
    

@sanitize_server
def fetch_full_instance_info(instance_info):
    """
    Returns the full JSON instance info from DVID
    """
    server, uuid, instance = instance_info
    return fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/info')


@sanitize_server
def generate_sample_coordinate(instance_info, label_id, supervoxels=False):
    """
    Return an arbitrary coordinate that lies within the given body.
    Usually faster than fetching all the RLEs.
    """
    server, uuid, instance = instance_info
    
    # FIXME: I'm using sparsevol instead of sparsevol-coarse due to an apparent bug in DVID at the moment
    SCALE = 2
    coarse_block_coords = fetch_sparsevol(instance_info, label_id, supervoxels, scale=SCALE)
    first_block_coord = (2**SCALE) * np.array(coarse_block_coords[0]) // 64 * 64
    
    ns = default_node_service(server, uuid)
    first_block = ns.get_labelarray_blocks3D( instance, (64,64,64), first_block_coord, supervoxels=supervoxels )
    nonzero_coords = np.transpose((first_block == label_id).nonzero())
    if len(nonzero_coords) == 0:
        label_type = {False: 'body', True: 'supervoxel'}[supervoxels]
        raise RuntimeError(f"The sparsevol-coarse info for this {label_type} ({label_id}) "
                           "appears to be out-of-sync with the scale-0 segmentation.")

    return first_block_coord + nonzero_coords[0]


@sanitize_server
def fetch_labelarray_voxels(instance_info, box, scale=0, throttle=False, supervoxels=False):
    """
    Fetch a volume of voxels from the given instance.
    
    Args:
        instance_info:
            server, uuid, instance
        
        box:
            The bounds of the volume to fetch in the coordinate system for the requested scale.
            Given as a pair of coordinates (start, stop), e.g. [(0,0,0), (10,20,30)].
            The box need not be block-aligned, but the request to DVID will be block aligned
            to 64px boundaries, and the retrieved volume will be truncated as needed before
            it is returned.
        
        scale:
            Which downsampling scale to fetch from
        
        throttle:
            If True, passed via the query string to DVID, in which case DVID might return a '503' error
            if the server is too busy to service the request.
            It is your responsibility to catch DVIDExceptions in that case.
        
        supervoxels:
            If True, request supervoxel data from the given labelmap instance.
    
    Returns:
        ndarray, with shape == (box[1] - box[0])
    """
    server, uuid, instance = instance_info
    ns = default_node_service(server, uuid)

    # Labelarray data can be fetched very efficiently if the request is block-aligned
    # So, block-align the request no matter what.
    aligned_box = round_box(box, 64, 'out')
    aligned_shape = aligned_box[1] - aligned_box[0]
    aligned_volume = ns.get_labelarray_blocks3D( instance, aligned_shape, aligned_box[0], throttle, scale, supervoxels )
    
    requested_box_within_aligned = box - aligned_box[0]
    return extract_subvol(aligned_volume, requested_box_within_aligned )


class KafkaReadError(RuntimeError):
    pass

@sanitize_server
def read_kafka_messages(instance_info, action_filter=None, dag_filter='leaf-and-parents', return_format='json-values', group_id=None, consumer_timeout=2.0):
    """
    Read the stream of available Kafka messages for the given DVID instance,
    and optionally filter them by UUID or Action.

    Args:
        instance_info:
            (server, uuid, instance)
        
        action_filter:
            A list of actions to use as a filter for the returned messages.
            For example, if action_filter=['split', 'split-complete'],
            all messages with other actions will be filtered out.

        dag_filter:
            How to filter out messages based on the UUID.
            One of:
            - 'leaf-only' (only messages whose uuid matches the provided instance_info),
            - 'leaf-and-parents' (only messages matching the given instance_info uuid or its ancestors), or
            - None (no filtering by UUID).

        return_format:
            Either 'records' (return list of kafka ConsumerRecord objects),
            or 'json-values' (return list of parsed JSON structures from each record.value)

        group_id:
            Kafka group ID to use when reading.  If not given, a new one is created.
            (FIXME: Frequently creating new group IDs like this is probably not best-practice, but it works for now.)
        
        consumer_timeout:
            Seconds to timeout (after which we assume we've read all messages).
        
    """
    from kafka import KafkaConsumer
    server, uuid, instance = instance_info
    
    assert dag_filter in ('leaf-only', 'leaf-and-parents', None)
    assert return_format in ('records', 'json-values')

    if group_id is None:
        # Choose a unique 'group_id' to use
        # FIXME: Frequently creating new group IDs like this is probably not best-practice, but it works for now.
        group_id = getpass.getuser() + '-' + datetime.now().isoformat()
    
    server_info = fetch_server_info(server)

    if "Kafka Servers" not in server_info or not server_info["Kafka Servers"]:
        raise KafkaReadError(f"DVID server ({server}) does not list a kafka server")

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

    values = [json.loads(rec.value) for rec in records]
    records_and_values = zip(records, values)

    if dag_filter == 'leaf-only':
        records_and_values = filter(lambda r_v: uuids_match(r_v[1]["UUID"], uuid), records_and_values)

    elif dag_filter == 'leaf-and-parents':
        # Load DAG structure as nx.DiGraph
        dag = fetch_and_parse_dag(server, repo_uuid)
        
        # Determine full name of leaf uuid, for proper set membership
        matching_uuids = list(filter(lambda u: uuids_match(u, uuid), dag.nodes()))
        assert matching_uuids != 0, f"DAG does not contain uuid: {uuid}"
        assert len(matching_uuids) == 1, f"More than one UUID in the server DAG matches the leaf uuid: {uuid}"
        full_uuid = matching_uuids[0]
        
        # Filter based on set of leaf-and-parents
        leaf_and_parents = {full_uuid} | nx.ancestors(dag, full_uuid)
        records_and_values = filter(lambda r_v: r_v[1]["UUID"] in leaf_and_parents, records_and_values)
        
    elif dag_filter is None:
        pass
    else:
        assert False

    if action_filter is not None:
        if isinstance(action_filter, str):
            action_filter = [action_filter]
        action_filter = set(action_filter)
        records_and_values = filter(lambda r_v: r_v[1]["Action"] in action_filter, records_and_values)
    
    # Evaluate
    records_and_values = list(records_and_values)
    
    # Unzip
    if records_and_values:
        records, values = zip(*records_and_values)
    else:
        records = values = []

    if return_format == 'records':
        return records
    elif return_format == 'json-values':
        return values
    else:
        assert False


@sanitize_server
def fetch_and_parse_dag(server, repo_uuid):
    """
    Read the /repo/info for the given repo UUID
    and extract the DAG structure from it.

    Return the DAG as a nx.DiGraph, whose nodes' attribute
    dicts contain the fields from the DAG json data.
    """
    repo_info = fetch_repo_info(server, repo_uuid)

    # The JSON response is a little weird.
    # The DAG nodes are given as a dict with uuids as keys,
    # but to define the DAG structure, parents and children are
    # referred to by their integer 'VersionID' (not their UUID).

    # Let's start by creating an easy lookup from VersionID -> node info
    node_infos = {}
    for node_info in repo_info["DAG"]["Nodes"].values():
        version_id = node_info["VersionID"]
        node_infos[version_id] = node_info
        
    g = nx.DiGraph()
    
    # Add graph nodes (with node info from the json as the nx node attributes)
    for version_id, node_info in node_infos.items():
        g.add_node(node_info["UUID"], **node_info)
        
    # Add edges from each parent to all children
    for version_id, node_info in node_infos.items():
        parent_uuid = node_info["UUID"]
        for child_version_id in node_info["Children"]:
            child_uuid = node_infos[child_version_id]["UUID"]
            g.add_edge(parent_uuid, child_uuid)

    return g


@sanitize_server
def perform_cleave(instance_info, body_id, supervoxel_ids):
    server, uuid, instance = instance_info
    supervoxel_ids = list(map(int, supervoxel_ids))

    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/cleave/{body_id}', json=supervoxel_ids)
    r.raise_for_status()
    cleaved_body = r.json()["CleavedLabel"]
    return cleaved_body


SplitEvent = namedtuple("SplitEvent", "mutid old remain split")

@sanitize_server
def post_merge(instance_info, main_label, other_labels):
    """
    Merges multiple bodies together.
    
    Args:
        instance_info:
            server, uuid, instance
        main_label:
            The label whose ID will be kept by the merged body
        other_labels:
            List of labels to merge into the main_label
    """
    server, uuid, instance = instance_info
    main_label = int(main_label)
    other_labels = list(map(int, other_labels))
    
    content = [main_label] + other_labels
    
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/merge', json=content)
    r.raise_for_status()
    

@sanitize_server
def fetch_supervoxel_splits(instance_info, source='kafka'):
    """
    Fetch supervoxel split events from dvid or kafka.
    (See fetch_supervoxel_splits_from_dvid() for details.)
    
    Note: If source='kafka', but no kafka server is found, 'dvid' is used as a fallback
    """
    assert source in ('dvid', 'kafka')

    if source == 'kafka':
        try:
            return fetch_supervoxel_splits_from_kafka(instance_info)
        except KafkaReadError:
            # Fallback to reading DVID
            source = 'dvid'

    if source == 'dvid':
        return fetch_supervoxel_splits_from_dvid(instance_info)

    assert False


@sanitize_server
def fetch_supervoxel_splits_from_dvid(instance_info):
    """
    Fetch the /supervoxel-splits info for the given instance.
    
    Args:
        instance_info:
            server, uuid, instance

    Returns:
        Dict of { uuid: event_list }, where event_list is a list of SplitEvent tuples.
        The UUIDs in the dict appear in the same order that DVID provides them in the response.
        According to the docs, they appear in reverse-chronological order, starting with the
        requested UUID and moving toward the repo root UUID.

    WARNING:
        The /supervoxel-splits endpoint was implemented relatively recently.  It is incapable
        of returning split events for supervoxels that were split before that endpoint was
        implemented and released.  On older server, the results returned by this function may
        be incomplete, since older UUIDs predate the /supervoxel-splits endpoint support.
        However the equivalent information can be extracted from the Kafka log, albeit
        somewhat more slowly.  See fetch_supervoxel_splits_from_kafka()

    """
    # From the DVID docs:
    #
    # GET <api URL>/node/<UUID>/<data name>/supervoxel-splits
    #
    #     Returns JSON for all supervoxel splits that have occured up to this version of the
    #     labelmap instance.  The returned JSON is of format:
    # 
    #         [
    #             "abc123",
    #             [[<mutid>, <old>, <remain>, <split>],
    #             [<mutid>, <old>, <remain>, <split>],
    #             [<mutid>, <old>, <remain>, <split>]],
    #             "bcd234",
    #             [[<mutid>, <old>, <remain>, <split>],
    #             [<mutid>, <old>, <remain>, <split>],
    #             [<mutid>, <old>, <remain>, <split>]]
    #         ]
    #     
    #     The UUID examples above, "abc123" and "bcd234", would be full UUID strings and are in order
    #     of proximity to the given UUID.  So the first UUID would be the version of interest, then
    #     its parent, and so on.

    server, uuid, instance = instance_info
    session = default_dvid_session()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/supervoxel-splits')
    r.raise_for_status()

    events = {}
    
    # Iterate in chunks of 2
    for uuid, event_list in zip(*2*[iter(r.json())]):
        assert isinstance(uuid, str)
        assert isinstance(event_list, list)
        events[uuid] = list(starmap(SplitEvent, event_list))
 
    return events

@sanitize_server
def fetch_supervoxel_splits_from_kafka(instance_info, actions=['split', 'split-supervoxel']):
    """
    Read the kafka log for the given instance and return a log of
    all supervoxel split events, partitioned by UUID.
    
    This produces the same output as fetch_supervoxel_splits_from_dvid(),
    but uses the kafka log instead of the DVID /supervoxel-splits endpoint.
    See the warning in that function's docstring for an explanation of why
    this function may give better results (although it is slower).

    To return all supervoxel splits, this function parses both 'split'
    and 'split-supervoxel' kafka messages.
    
    As a debugging feature, you can opt to select splits of only one
    type by specifying which actions to filter with.
    """
    assert not (set(actions) - set(['split', 'split-supervoxel'])), \
        f"Invalid actions: {actions}"
    
    msgs = read_kafka_messages(instance_info, action_filter=actions, dag_filter='leaf-and-parents')
    
    # Supervoxels can be split via either /split or /split-supervoxel.
    # We need to parse them both.
    body_split_msgs = list(filter(lambda msg: msg['Action'] == 'split', msgs))
    sv_split_msgs = list(filter(lambda msg: msg['Action'] == 'split-supervoxel', msgs))

    events = {}
    for msg in body_split_msgs:
        if msg["SVSplits"] is None:
            logger.error(f"SVSplits is null for body {msg['Target']}")
            continue
        for old_sv_str, split_info in msg["SVSplits"].items():
            event = SplitEvent(msg["MutationID"], int(old_sv_str), split_info["Split"], split_info["Remain"])
            events.setdefault(msg["UUID"], []).append( event )

    for msg in sv_split_msgs:
        event = SplitEvent( msg["MutationID"], msg["Supervoxel"], msg["SplitSupervoxel"], msg["RemainSupervoxel"] )
        events.setdefault(msg["UUID"], []).append( event )
    
    return events


def split_events_to_graph(events):
    """
    Load the split events into an annotated networkx.DiGraph, where each node is a supervoxel ID.
    The node annotations are: 'uuid' and 'mutid', indicating the uuid and mutation id at
    the time the supervoxel was CREATED.
    
    Args:
        events: dict as returned by fetch_supervoxel_splits()
    
    Returns:
        nx.DiGraph, which will be a tree
    """
    g = nx.DiGraph()

    for uuid, event_list in events.items():
        for (mutid, old, remain, split) in event_list:
            g.add_edge(old, remain)
            g.add_edge(old, split)
            if 'uuid' not in g.node[old]:
                # If the old ID is not a product of a split event, we don't know when it was created.
                # (Presumably, it originates from the root UUID, but for old servers the /split-supervoxels
                # endpoint is not comprehensive all the way to the root node.)
                g.node[old]['uuid'] = '<unknown>'
                g.node[old][mutid] = -1

            g.node[remain]['uuid'] = uuid
            g.node[split]['uuid'] = uuid
            g.node[remain]['mutid'] = mutid
            g.node[split]['mutid'] = mutid
    
    return g


def find_root(g, start):
    """
    Find the root node in a tree, given as a nx.DiGraph,
    tracing up the tree starting with the given start node.
    """
    parents = [start]
    while parents:
        root = parents[0]
        parents = list(g.predecessors(parents[0]))
    return root


def extract_split_tree(events, sv_id):
    """
    Construct a nx.DiGraph from the given list of SplitEvents,
    exract the particular tree (a subgraph) that contains the given supervoxel ID.
    
    Args:
        events:
            Either a nx.DiGraph containing all split event trees of interest (so, a forest),
            or a dict of SplitEvent tuples from which a DiGraph forest can be constructed.
            
        sv_id:
            The tree containing this supervoxel ID will be extracted.
            The ID does not necessarily need to be the roof node of it's split tree.
    
    Returns:
        nx.DiGraph -- A subgraph of the forest passed in.
    """
    if isinstance(events, dict):
        g = split_events_to_graph(events)
    elif isinstance(events, nx.DiGraph):
        g = events
    else:
        raise RuntimeError(f"Unexpected input type: {type(events)}")
    
    if sv_id not in g.nodes():
        raise RuntimeError(f"Supervoxel {sv_id} is not referenced in the given split events.")
    
    root = find_root(g, sv_id)
    tree_nodes = {root} | nx.descendants(g, root)
    tree = g.subgraph(tree_nodes)
    return tree


def tree_to_dict(tree, root, display_fn=str, _d=None):
    """
    Convert the given tree (nx.DiGraph) into a dict,
    suitable for display via the asciitree module.
    
    Args:
        tree:
            nx.DiGraph
        root:
            Where to start in the tree (ancestors of this node will be ignored)
        display_fn:
            Callback used to convert node values into strings, which are used as the dict keys.
        _d:
            Internal use only.
    """
    if _d is None:
        _d = {}
    d_desc = _d[display_fn(root)] = {}
    for n in sorted(tree.successors(root)):
        tree_to_dict(tree, n, display_fn, _d=d_desc)
    return _d


def render_split_tree(tree, root=None, uuid_len=4):
    """
    Render the given split tree as ascii text.
    
    Requires the 'asciitree' module (conda install -c conda-forge asciitree)
    
    Args:
        tree:
            nx.DiGraph with 'uuid' annotations (as returned by extract_split_tree())
        root:
            Node to use as the root of the tree for display purposes.
            If None, root will be searched for.
        uuid_len:
            How many characters of the uuid to print for each node in the tree.
            Set to 0 to disable UUID display.
    
    Returns:
        str
        Example:
            >>> events = fetch_supervoxel_splits(('emdata3:8900', '52f9', 'segmentation'))
            >>> tree = extract_split_tree(events, 677527463)
            >>> print(render_split_tree(tree))
            677527463 (<unknown>)
             +-- 5813042205 (25dc)
             +-- 5813042206 (25dc)
                 +-- 5813042207 (25dc)
                 +-- 5813042208 (25dc)
    """
    root = root or find_root(tree, next(iter(tree.nodes())))

    def display_fn(n):
        uuid = tree.node[n]['uuid']
        if uuid != '<unknown>':
            uuid = uuid[:uuid_len]
        return f"{n} ({uuid})"

    if uuid_len == 0:
        display_fn = str

    d = tree_to_dict(tree, root, display_fn)

    from asciitree import LeftAligned
    return LeftAligned()(d)


def fetch_and_render_split_tree(instance_info, sv_id, split_source='kafka'):
    """
    Fetch all split supervoxel provenance data from DVID and then
    extract the provenance tree containing the given supervoxel.
    Then render it as a string to be displayed on the console.
    """
    events = fetch_supervoxel_splits(instance_info, split_source)
    tree = extract_split_tree(events, sv_id)
    return render_split_tree(tree)


def fetch_and_render_split_trees(instance_info, sv_ids, split_source='kafka'):
    """
    For each of the given supervoxels, produces an ascii-renderered split
    tree showing all of its ancestors,descendents, siblings, etc.

    Supervoxels that have never been involved in splits will be skipped,
    and no corresponding tree is returned.
    
    Note: If two supervoxels in the list have a common parent,
    the tree will be returned twice.
    
    TODO: It would be nice if this were smarter about eliminating
          duplicates in such cases.
    
    Returns:
        dict of { sv_id : str }
    """
    sv_ids = set(sv_ids)
    events = fetch_supervoxel_splits(instance_info, split_source)
    event_forest = split_events_to_graph(events)
    all_split_ids = set(event_forest.nodes())
    
    rendered_trees = {}
    for sv_id in sv_ids:
        if sv_id in all_split_ids:
            tree = extract_split_tree(event_forest, sv_id)
            rendered_trees[sv_id] = render_split_tree(tree)
    return rendered_trees


def fetch_split_supervoxel_sizes(instance_info, include_retired=False, split_source='kafka'):
    """
    Fetch the list of all current split supervoxel fragments from DVID or Kafka,
    then fetch the sizes of each of those supervoxels.
    
    Args:
        instance_info:
            server, uuid, instance
        
        include_retired:
            If True, include 'retired' supervoxel IDs in the result.
            They will have a size of 0.
        
        split_source:
            Where to pull split events from.
            Either 'kafka' (slower) or 'dvid' (some servers return incomplete histories).
    
    Returns:
        pd.Series, indexed by SV ID
    """
    leaf_fragment_svs, retired_svs = fetch_supervoxel_fragments(instance_info, split_source)

    with Timer(f"Fetching sizes for {len(leaf_fragment_svs)} split supervoxels", logger):
        sizes = fetch_sizes(instance_info, leaf_fragment_svs, supervoxels=True)
        sizes = np.array(sizes, np.uint32)

    sv_sizes = pd.Series(data=sizes, index=leaf_fragment_svs)
    sv_sizes.name = 'size'
    sv_sizes.index.name = 'sv'
    sv_sizes.sort_index(inplace=True)

    if include_retired:
        retired_sv_sizes = pd.Series(data=np.uint32(0), index=retired_svs)
        retired_sv_sizes.name = 'size'
        retired_sv_sizes.index.name = 'sv'
        retired_sv_sizes.sort_index(inplace=True)
        sv_sizes = pd.concat((sv_sizes, retired_sv_sizes))
        
    assert sv_sizes.index.dtype == np.uint64
    assert sv_sizes.dtype == np.uint32
    return sv_sizes


def fetch_supervoxel_fragments(instance_info, split_source='kafka'):
    """
    Fetch the list of all supervoxels that have been split and their resulting fragments.
    
    Args:
        instance_info:
            server, uuid, instance
        
        split_source:
            Where to pull split events from.
            Either 'kafka' (slower) or 'dvid' (some servers return incomplete histories).
    
    Returns:
        (leaf_fragment_svs, retired_svs)
        where leaf_fragment_svs is the list of all supervoxel fragments that still exist in the instance,
        and retired_svs is the list of all supervoxels that have ever been split in the instance.
        
    """
    split_events = fetch_supervoxel_splits(instance_info, split_source)
    if len(split_events) == 0:
        # No splits on this node
        return (np.array([], np.uint64), np.array([], np.uint64))
    
    split_tables = list(map(lambda t: np.asarray(t, np.uint64), split_events.values()))
    split_table = np.concatenate(split_tables)

    retired_svs = split_table[:, SplitEvent._fields.index('old')]
    remain_fragment_svs = split_table[:, SplitEvent._fields.index('remain')]
    split_fragment_svs = split_table[:, SplitEvent._fields.index('split')]

    leaf_fragment_svs = (set(remain_fragment_svs) | set(split_fragment_svs)) - set(retired_svs)
    leaf_fragment_svs = np.fromiter(leaf_fragment_svs, np.uint64)
    
    return (leaf_fragment_svs, retired_svs)


@sanitize_server
def fetch_tarfile(instance_info, body_id, output=None):
    """
    Fetch a .tar file from a tarsupervoxels instance for the given body,
    and save it to bytes, a file object, or a file path.
    
    Args:
        instance_info:
            server, uuid, instance
        
        body_id:
            The body whose supervoxel files will be fetched in the tar.
        
        output:
            If None, tarfile is returned in-memory, as bytes.
            If str, it is interpreted as a path to which the .tar file will be written.
            Otherwise, must be a file object to write the bytes to (e.g. a BytesIO object).
    
    Returns:
        None, unless no output file object/path is provided,
        in which case the tarfile bytes are returned.
    """
    server, uuid, instance = instance_info
    url = f'http://{server}/api/node/{uuid}/{instance}/tarfile/{body_id}'
    return fetch_file(url, output)


def fetch_file(url, output=None, chunksize=2**10):
    """
    Fetch a file from the given endpoint,
    and save it to bytes, a file object, or a file path.

    Args:
        url:
            Complete url to fetch from.
        
        output:
            If None, file is returned in-memory, as bytes.
            If str, it is interpreted as a path to which the file will be written.
            Otherwise, must be a file object to write the bytes to (e.g. a BytesIO object).
        
        chunksize:
            Data will be streamed in chunks, with the given chunk size.

    Returns:
        None, unless no output file object/path is provided,
        in which case the fetched bytes are returned.
    """
    session = default_dvid_session()

    with session.get(url, stream=True) as r:
        r.raise_for_status()

        if output is None:
            return r.content

        if isinstance(output, str):
            # Create a file on disk and write to it.
            with open(output, 'wb') as f:
                for chunk in r.iter_content(chunksize):
                    f.write(chunk)
        else:
            # output is a file object
            for chunk in r.iter_content(chunksize):
                output.write(chunk)


@sanitize_server
def fetch_labelindex(instance_info, label, format='protobuf'): # @ReservedAssignment
    """
    Fetch the LabelIndex for the given label ID from DVID,
    and return it as the native protobuf structure, or as a more-convenient
    structure that encodes all block counts into a single big DataFrame.
    (See convert_labelindex_to_pandas())
    
    Note that selecting the 'pandas' format takes ~10x longer.
    """
    server, uuid, instance = instance_info
    assert format in ('protobuf', 'pandas')

    session = default_dvid_session()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/index/{label}')
    r.raise_for_status()
    labelindex = LabelIndex()
    labelindex.ParseFromString(r.content)

    if format == 'protobuf':
        return labelindex
    elif format == 'pandas':
        return convert_labelindex_to_pandas(labelindex)


PandasLabelIndex = namedtuple("PandasLabelIndex", "blocks label last_mutid last_mod_time last_mod_user")
def convert_labelindex_to_pandas(labelindex):
    """
    Convert a protobuf LabelIndex object into a PandasLabelIndex tuple,
    which returns supervoxel counts for all blocks in one big pd.DataFrame.
    
    Args:
        labelindex:
            Instance of neuclease.dvid.labelops_pb2.LabelIndex, as returned by fetch_labelindex()
    
    Returns:
        PandasLabelIndex (a namedtuple), in which the `blocks` member is a pd.DataFrame
        with the following columns: ['z', 'y', 'x', 'sv', 'count'].
        Note that the block coordinates are given in VOXEL units.
        That is, all coordinates in the table are multiples ofo 64.
    """
    encoded_block_coords = np.fromiter(labelindex.blocks.keys(), np.uint64, len(labelindex.blocks))
    coords_zyx = decode_labelindex_blocks(encoded_block_coords)

    block_svs = []
    block_counts = []
    block_coords = []
    
    # Convert each block's data into arrays
    for coord_zyx, sv_counts in zip(coords_zyx, labelindex.blocks.values()):
        svs = np.fromiter(sv_counts.counts.keys(), np.uint64, count=len(sv_counts.counts))
        counts = np.fromiter(sv_counts.counts.values(), np.uint32, count=len(sv_counts.counts))
        
        block_svs.append(svs)
        block_counts.append(counts)
        block_coords.append( np.repeat(coord_zyx[None], len(svs), axis=0) )

    # Concatenate all block data and load into one big DataFrame
    all_coords = np.concatenate(block_coords)
    all_svs = np.concatenate(block_svs)
    all_counts = np.concatenate(block_counts)
    
    blocks_df = pd.DataFrame( all_coords, columns=['z', 'y', 'x'] )
    blocks_df['sv'] = all_svs
    blocks_df['count'] = all_counts
    
    return PandasLabelIndex( blocks_df,
                             labelindex.label,
                             labelindex.last_mutid,
                             labelindex.last_mod_time,
                             labelindex.last_mod_user )


@jit(nopython=True)
def decode_labelindex_blocks(encoded_blocks):
    """
    Calls decode_labelindex_block() on a 1-D array of encoded coordinates.
    """
    decoded_blocks = np.zeros((len(encoded_blocks), 3), dtype=np.int32)
    for i in range(len(encoded_blocks)):
        encoded = encoded_blocks[i]
        decoded_blocks[i,:] = decode_labelindex_block(encoded)
    return decoded_blocks


@jit(nopython=True)
def decode_labelindex_block(encoded_block):
    """
    Helper function.
    Decodes a block coordinate from a LabelIndex entry.
    
    DVID encodes the block coordinates into a single uint64,
    as three signed 21-bit integers, in zyx order
    (leaving the top bit of the uint64 set to 0).
    
    So, the encoded layout is as follows (S == sign bit):
    0SzzzzzzzzzzzzzzzzzzzzSyyyyyyyyyyyyyyyyyyyySxxxxxxxxxxxxxxxxxxxx
    
    
    NOTE: The encoded coordinates from DVID are in 'block coordinate space',
          not 'voxel coordinate space', but we nonetheless return
          VOXEL coordinates, not block coordinates.
          (That is, we multiply the block coordinates by 64.)
    """
    z = np.int32((encoded_block >> 2*21) & 0x1F_FFFF) # 21 bits
    y = np.int32((encoded_block >>   21) & 0x1F_FFFF) # 21 bits
    x = np.int32((encoded_block >>    0) & 0x1F_FFFF) # 21 bits
    
    # Check sign bits and extend if necessary
    if encoded_block & (1 << (3*21-1)):
        z |= np.int32(0xFFFF_FFFF << 21)

    if encoded_block & (1 << (21*2-1)):
        y |= np.int32(0xFFFF_FFFF << 21)

    if encoded_block & (1 << (21*1-1)):
        x |= np.int32(0xFFFF_FFFF << 21)
    
    return np.array((64*z, 64*y, 64*x), dtype=np.int32)

