import logging
from io import BytesIO

import numpy as np
import pandas as pd

from ...util import Timer, round_box, extract_subvol

from .. import dvid_api_wrapper, default_dvid_session, default_node_service, fetch_generic_json
from ..repo import create_voxel_instance
from ..node import fetch_full_instance_info
from ..kafka import read_kafka_messages
from ..rle import parse_rle_response

from ._split import SplitEvent, fetch_supervoxel_splits_from_kafka

logger = logging.getLogger(__name__)

def create_labelmap_instance(server, uuid, instance, tags=[], block_size=64, voxel_size=8.0,
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
    type_specific_settings = { "IndexedLabels": str(enable_index).lower(), "CountLabels": str(enable_index).lower(), "MaxDownresLevel": str(max_scale) }
    create_voxel_instance( server, uuid, instance, 'labelmap', tags=tags, block_size=block_size, voxel_size=voxel_size,
                       voxel_units=voxel_units, type_specific_settings=type_specific_settings )



@dvid_api_wrapper
def fetch_supervoxels_for_body(server, uuid, instance, body_id, user=None):
    query_params = {}
    if user:
        query_params['u'] = user

    url = f'http://{server}/api/node/{uuid}/{instance}/supervoxels/{body_id}'
    r = default_dvid_session().get(url, params=query_params)
    r.raise_for_status()
    supervoxels = np.array(r.json(), np.uint64)
    supervoxels.sort()
    return supervoxels


@dvid_api_wrapper
def fetch_size(server, uuid, instance, label_id, supervoxels=False):
    supervoxels = str(bool(supervoxels)).lower()
    url = f'http://{server}/api/node/{uuid}/{instance}/size/{label_id}?supervoxels={supervoxels}'
    response = fetch_generic_json(url)
    return response['voxels']

# Deprecated name
fetch_body_size = fetch_size


@dvid_api_wrapper
def fetch_sizes(server, uuid, instance, label_ids, supervoxels=False):
    label_ids = list(map(int, label_ids))
    supervoxels = str(bool(supervoxels)).lower()

    url = f'http://{server}/api/node/{uuid}/{instance}/sizes?supervoxels={supervoxels}'
    return fetch_generic_json(url, label_ids)

# Deprecated name
fetch_body_sizes = fetch_sizes


@dvid_api_wrapper
def fetch_supervoxel_sizes_for_body(server, uuid, instance, body_id, user=None):
    """
    Return the sizes of all supervoxels in a body 
    """
    supervoxels = fetch_supervoxels_for_body(server, uuid, instance, body_id, user)
    
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


@dvid_api_wrapper
def fetch_label_for_coordinate(server, uuid, instance, coordinate_zyx, supervoxels=False):
    session = default_dvid_session()
    coord_xyz = np.array(coordinate_zyx)[::-1]
    coord_str = '_'.join(map(str, coord_xyz))
    supervoxels = str(bool(supervoxels)).lower()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/label/{coord_str}?supervoxels={supervoxels}')
    r.raise_for_status()
    return np.uint64(r.json()["Label"])


@dvid_api_wrapper
def fetch_sparsevol_rles(server, uuid, instance, label, supervoxels=False, scale=0):
    session = default_dvid_session()
    supervoxels = str(bool(supervoxels)).lower() # to lowercase string
    url = f'http://{server}/api/node/{uuid}/{instance}/sparsevol/{label}?supervoxels={supervoxels}&scale={scale}'
    r = session.get(url)
    r.raise_for_status()
    return r.content


@dvid_api_wrapper
def split_supervoxel(server, uuid, instance, supervoxel, rle_payload_bytes):
    """
    Split the given supervoxel according to the provided RLE payload, as specified in DVID's split-supervoxel docs.
    
    Returns:
        The two new IDs resulting from the split: (split_sv_id, remaining_sv_id)
    """
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/split-supervoxel/{supervoxel}', data=rle_payload_bytes)
    r.raise_for_status()
    
    results = r.json()
    return (results["SplitSupervoxel"], results["RemainSupervoxel"] )


@dvid_api_wrapper
def fetch_mapping(server, uuid, instance, supervoxel_ids):
    supervoxel_ids = list(map(int, supervoxel_ids))
    body_ids = fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/mapping', json=supervoxel_ids)
    return body_ids


@dvid_api_wrapper
def fetch_mappings(server, uuid, instance, as_array=False):
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


@dvid_api_wrapper
def fetch_complete_mappings(server, uuid, instance, include_retired=False, kafka_msgs=None):
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
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        include_retired:
            If True, include rows for 'retired' supervoxels, which all map to 0.

    Returns:
        pd.Series(index=sv, data=body)
    """
    # Read complete kafka log; we need both split and cleave info
    if kafka_msgs is None:
        kafka_msgs = read_kafka_messages(server, uuid, instance)
    split_events = fetch_supervoxel_splits_from_kafka(server, uuid, instance, kafka_msgs=kafka_msgs)
    split_tables = list(map(lambda t: np.asarray(t, np.uint64), split_events.values()))
    if split_tables:
        split_table = np.concatenate(split_tables)
        retired_svs = split_table[:, SplitEvent._fields.index('old')] #@UndefinedVariable
        retired_svs = set(retired_svs)
    else:
        retired_svs = set()

    def extract_cleave_fragments():
        for msg in kafka_msgs:
            if msg["Action"] == "cleave":
                yield msg["CleavedLabel"]

    # Cleave fragment IDs (i.e. bodies that were created via a cleave)
    # should not be included in the set of 'identity' rows.
    # (These IDs are guaranteed to be disjoint from supervoxel IDs.)
    cleave_fragments = set(extract_cleave_fragments())

    # Fetch base mapping
    base_mapping = fetch_mappings(server, uuid, instance, as_array=True)
    base_svs = base_mapping[:,0]
    base_bodies = base_mapping[:,1]

    # Augment with identity rows, which aren't included in the base.
    with Timer(f"Constructing missing identity-mappings", logger):
        missing_idents = set(base_bodies) - set(base_svs) - retired_svs - cleave_fragments
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
    
    # Drop all rows with retired supervoxels.
    s.drop(retired_svs, inplace=True, errors='ignore')
    
    # Reload index to ensure most RAM-efficient implementation.
    # (This seems to make a big difference in RAM usage!)
    s.index = s.index.values

    s.index.name = 'sv'
    s.name = 'body'

    assert s.index.dtype == np.uint64
    assert s.dtype == np.uint64
    return s


@dvid_api_wrapper
def fetch_mutation_id(server, uuid, instance, body_id):
    response = fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/lastmod/{body_id}')
    return response["mutation id"]


@dvid_api_wrapper
def fetch_sparsevol_coarse(server, uuid, instance, label_id, supervoxels=False):
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
    supervoxels = str(bool(supervoxels)).lower()
    session = default_dvid_session()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/sparsevol-coarse/{label_id}?supervoxels={supervoxels}')
    r.raise_for_status()
    
    return parse_rle_response( r.content )


def fetch_sparsevol(server, uuid, instance, label, supervoxels=False, scale=0):
    """
    Return coordinates of all voxels in the given body/supervoxel at the given scale.

    Note: At scale 0, this will be a LOT of data for any reasonably large body.
          Use with caution.
    """
    rles = fetch_sparsevol_rles(server, uuid, instance, label, supervoxels, scale)
    return parse_rle_response(rles)


def compute_changed_bodies(instance_info_a, instance_info_b):
    """
    Returns the list of all bodies whose supervoxels changed
    between uuid_a and uuid_b.
    This includes bodies that were changed, added, or removed completely.
    
    Args:
        instance_info_a:
            (server, uuid, instance)

        instance_info_b:
            (server, uuid, instance)
    """
    mapping_a = fetch_mappings(*instance_info_a)
    mapping_b = fetch_mappings(*instance_info_b)
    
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


def fetch_volume_box(server, uuid, instance):
    """
    Return the volume extents for the given instance as a box.
    
    Returns:
        np.ndarray [(z0,y0,x0), (z1,y1,x1)]
    
    Notes:
        - Returns *box*, shape=(box[1] - box[0])
        - Returns ZYX order
    """
    info = fetch_full_instance_info(server, uuid, instance)
    box_xyz = np.array((info["Extended"]["MinPoint"], info["Extended"]["MaxPoint"]))
    box_xyz[1] += 1
    
    box_zyx = box_xyz[:,::-1]
    return box_zyx


@dvid_api_wrapper
def generate_sample_coordinate(server, uuid, instance, label_id, supervoxels=False):
    """
    Return an arbitrary coordinate that lies within the given body.
    Usually faster than fetching all the RLEs.
    """
   
    # FIXME: I'm using sparsevol instead of sparsevol-coarse due to an apparent bug in DVID at the moment
    SCALE = 2
    coarse_block_coords = fetch_sparsevol(server, uuid, instance, label_id, supervoxels, scale=SCALE)
    first_block_coord = (2**SCALE) * np.array(coarse_block_coords[0]) // 64 * 64
    
    ns = default_node_service(server, uuid)
    first_block = ns.get_labelarray_blocks3D( instance, (64,64,64), first_block_coord, supervoxels=supervoxels )
    nonzero_coords = np.transpose((first_block == label_id).nonzero())
    if len(nonzero_coords) == 0:
        label_type = {False: 'body', True: 'supervoxel'}[supervoxels]
        raise RuntimeError(f"The sparsevol-coarse info for this {label_type} ({label_id}) "
                           "appears to be out-of-sync with the scale-0 segmentation.")

    return first_block_coord + nonzero_coords[0]


@dvid_api_wrapper
def fetch_labelarray_voxels(server, uuid, instance, box, scale=0, throttle=False, supervoxels=False):
    """
    Fetch a volume of voxels from the given instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'

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
    ns = default_node_service(server, uuid)

    # Labelarray data can be fetched very efficiently if the request is block-aligned
    # So, block-align the request no matter what.
    aligned_box = round_box(box, 64, 'out')
    aligned_shape = aligned_box[1] - aligned_box[0]
    aligned_volume = ns.get_labelarray_blocks3D( instance, aligned_shape, aligned_box[0], throttle, scale, supervoxels )
    
    requested_box_within_aligned = box - aligned_box[0]
    return extract_subvol(aligned_volume, requested_box_within_aligned )


@dvid_api_wrapper
def perform_cleave(server, uuid, instance, body_id, supervoxel_ids):
    supervoxel_ids = list(map(int, supervoxel_ids))

    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/cleave/{body_id}', json=supervoxel_ids)
    r.raise_for_status()
    cleaved_body = r.json()["CleavedLabel"]
    return cleaved_body


@dvid_api_wrapper
def post_merge(server, uuid, instance, main_label, other_labels):
    """
    Merges multiple bodies together.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'

        main_label:
            The label whose ID will be kept by the merged body

        other_labels:
            List of labels to merge into the main_label
    """
    main_label = int(main_label)
    other_labels = list(map(int, other_labels))
    
    content = [main_label] + other_labels
    
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/merge', json=content)
    r.raise_for_status()
    

