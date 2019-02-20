import gzip
import logging
from io import BytesIO
from multiprocessing.pool import ThreadPool
from collections import defaultdict

import numpy as np
import pandas as pd
from requests import HTTPError

from libdvid import DVIDNodeService, encode_label_block

from ...util import Timer, round_box, extract_subvol, DEFAULT_TIMESTAMP, tqdm_proxy, ndrange, box_to_slicing

from .. import dvid_api_wrapper, fetch_generic_json
from ..repo import create_voxel_instance, fetch_repo_dag
from ..kafka import read_kafka_messages, kafka_msgs_to_df
from ..rle import parse_rle_response

from ._split import SplitEvent, fetch_supervoxel_splits_from_kafka

logger = logging.getLogger(__name__)

@dvid_api_wrapper
def create_labelmap_instance(server, uuid, instance, versioned=True, tags=[], block_size=64, voxel_size=8.0,
                             voxel_units='nanometers', enable_index=True, max_scale=0, *, session=None):
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
    create_voxel_instance( server, uuid, instance, 'labelmap', versioned, tags=tags, block_size=block_size, voxel_size=voxel_size,
                       voxel_units=voxel_units, type_specific_settings=type_specific_settings, session=session )


@dvid_api_wrapper
def fetch_maxlabel(server, uuid, instance, *, session=None, dag=None):
    """
    Read the MaxLabel for the given segmentation instance at the given node.
    
    This implementation includes a workaround for issue 284:
    
      - https://github.com/janelia-flyem/dvid/issues/284
    
    If the ``/maxlabel`` endpoint returns an error stating "No maximum label",
    we recursively check the parent node(s) for a valid maxlabel until we find one.
    """
    url = f'http://{server}/api/node/{uuid}/{instance}/maxlabel'
    
    try:
        return fetch_generic_json(url, session=session)["maxlabel"]
    except HTTPError as ex:
        if ex.response is None or 'No maximum label' not in ex.response.content.decode('utf-8'):
            raise

        # Oops, Issue 284
        # Search upwards in the DAG for a uuid with a valid max label
        if dag is None:
            dag = fetch_repo_dag(server, uuid)

        # Check the parent nodes
        parents = list(dag.predecessors(uuid))
        if not parents:
            # No parents to check
            raise

        # Find the maxlabel of all parents
        # (In theory, there can be more than one.)
        parent_maxes = []
        for parent_uuid in parents:
            try:
                parent_maxes.append( fetch_maxlabel(server, parent_uuid, instance, dag=dag) )
            except HTTPError as ex:
                parent_maxes.append(0)
        
        parent_max = max(parent_maxes)
        if parent_max == 0:
            # Could not find a max label for any parent.
            raise

        return parent_max

@dvid_api_wrapper
def post_maxlabel(server, uuid, instance, maxlabel, *, session=None):
    """
    Update the MaxLabel for the given segmentation instance at the given node.
    Usually DVID auto-updates the MaxLabel for an instance upon cleave/split/split-supervoxel,
    so there is rarely any need to use this function.
    But it is useful when you are writing raw supervoxel data "under DVID's feet".
    
    Note:
        It is an error to post a smaller maxlabel value than the
        current MaxLabel value for the instance.
        See ``fetch_maxlabel()``
    """
    url = f'http://{server}/api/node/{uuid}/{instance}/maxlabel/{maxlabel}'
    r = session.post(url)
    r.raise_for_status()


@dvid_api_wrapper
def fetch_nextlabel(server, uuid, instance, *, session=None):
    url = f'http://{server}/api/node/{uuid}/{instance}/nextlabel'
    r_json = fetch_generic_json(url, session=session)
    return r_json['nextlabel']


@dvid_api_wrapper
def post_nextlabel(server, uuid, instance, num_labels, *, session=None):
    """
    Ask DVID to reserve a number of new label IDs that are not yet
    present in the given labelmap instance.
    For typical merge/cleave/split operations, this is not necessary.
    DVID handles the generation of label IDs as needed.
    
    But in the (very rare) case where you want to overwrite segmentation supervoxels directly,
    you should use this function to reserve a number of unique IDs that you can
    use without conflicting with existing objects in the volume.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        num_labels:
            How many labels to reserve
    
    Returns:
        (start, end), where start and end represent the first and last
        labels in a contiguous block of label IDs that have been reserved.
        
        Note:
            The returned range is INCLUSIVE, meaning 'end' is reserved
            (unlike python's range() function).
    """
    url = f'http://{server}/api/node/{uuid}/{instance}/nextlabel/{num_labels}'
    r = session.post(url)
    r.raise_for_status()
    d = r.json()
    start, end = d["start"], d["end"]
    
    num_reserved = end-start+1
    assert num_reserved == num_labels, \
        "Unexpected response from DVID. "\
        f"Requested {num_labels}, but received {num_reserved} labels ({start}, {end})"
    
    return (np.uint64(start), np.uint64(end))


@dvid_api_wrapper
def fetch_supervoxels(server, uuid, instance, body_id, user=None, *, session=None):
    # FIXME: Rename to 'fetch_supervoxels()'
    # FIXME: Remove 'user' in favor of session arg
    query_params = {}
    if user:
        query_params['u'] = user

    url = f'http://{server}/api/node/{uuid}/{instance}/supervoxels/{body_id}'
    r = session.get(url, params=query_params)
    r.raise_for_status()
    supervoxels = np.array(r.json(), np.uint64)
    supervoxels.sort()
    return supervoxels

# Deprecated name
fetch_supervoxels_for_body = fetch_supervoxels 

@dvid_api_wrapper
def fetch_size(server, uuid, instance, label_id, supervoxels=False, *, session=None):
    supervoxels = str(bool(supervoxels)).lower()
    url = f'http://{server}/api/node/{uuid}/{instance}/size/{label_id}?supervoxels={supervoxels}'
    response = fetch_generic_json(url, session=session)
    return response['voxels']

# FIXME: Deprecated name
fetch_body_size = fetch_size


@dvid_api_wrapper
def fetch_sizes(server, uuid, instance, label_ids, supervoxels=False, *, session=None):
    label_ids = np.asarray(label_ids, np.uint64)
    sv_param = str(bool(supervoxels)).lower()

    url = f'http://{server}/api/node/{uuid}/{instance}/sizes?supervoxels={sv_param}'
    sizes = fetch_generic_json(url, label_ids.tolist(), session=session)
    
    sizes = pd.Series(sizes, index=label_ids, name='size')
    if supervoxels:
        sizes.index.name = 'sv'
    else:
        sizes.index.name = 'body'
    
    return sizes

# FIXME: Deprecated name
fetch_body_sizes = fetch_sizes


@dvid_api_wrapper
def fetch_supervoxel_sizes_for_body(server, uuid, instance, body_id, user=None, *, session=None):
    """
    Return the sizes of all supervoxels in a body.
    Convenience function to call fetch_supervoxels() followed by fetch_sizes()
    
    Returns: 
        pd.Series, indexed by supervoxel
    """
    
    # FIXME: Remove 'user' param in favor of 'session' param.
    supervoxels = fetch_supervoxels_for_body(server, uuid, instance, body_id, user, session=session)
    
    query_params = {}
    if user:
        query_params['u'] = user

    # FIXME: Call fetch_sizes() with a custom session instead of rolling our own request here.
    url = f'http://{server}/api/node/{uuid}/{instance}/sizes?supervoxels=true'
    r = session.get(url, params=query_params, json=supervoxels.tolist())
    r.raise_for_status()
    sizes = np.array(r.json(), np.uint32)
    
    series = pd.Series(data=sizes, index=supervoxels)
    series.index.name = 'sv'
    series.name = 'size'
    return series


@dvid_api_wrapper
def fetch_label(server, uuid, instance, coordinate_zyx, supervoxels=False, scale=0, *, session=None):
    """
    Fetch the label at a single coordinate.
    
    See also: ``fetch_labels()``
    """
    coord_xyz = np.array(coordinate_zyx)[::-1]
    coord_str = '_'.join(map(str, coord_xyz))
    
    params = {}
    if supervoxels:
        params['supervoxels'] = str(bool(supervoxels)).lower()
    if scale != 0:
        params['scale'] = str(scale)

    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/label/{coord_str}', params=params)
    r.raise_for_status()
    return np.uint64(r.json()["Label"])

# Old name (FIXME: remove)
fetch_label_for_coordinate = fetch_label


@dvid_api_wrapper
def fetch_labels(server, uuid, instance, coordinates_zyx, supervoxels=False, scale=0, *, session=None):
    """
    Fetch the labels at a list of coordinates.
    """
    coordinates_zyx = np.asarray(coordinates_zyx, np.int32)
    assert coordinates_zyx.ndim == 2 and coordinates_zyx.shape[1] == 3

    params = {}
    if supervoxels:
        params['supervoxels'] = str(bool(supervoxels)).lower()
    if scale != 0:
        params['scale'] = str(scale)

    coords_xyz = np.array(coordinates_zyx)[:, ::-1].tolist()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/labels', json=coords_xyz, params=params)
    r.raise_for_status()
    
    labels = np.array(r.json(), np.uint64)
    return labels


def fetch_labels_batched(server, uuid, instance, coordinates_zyx, supervoxels=False, scale=0, batch_size=10_000, threads=0):
    """
    Like fetch_labels, but fetches in batches, optionally multithreaded.
    """
    coords_df = pd.DataFrame(coordinates_zyx, columns=['z', 'y', 'x'], dtype=np.int32)
    coords_df['label'] = np.uint64(0)
    
    def fetch_batch(batch_start):
        batch_stop = min(batch_start+batch_size, len(coords_df))
        batch_df = coords_df.iloc[batch_start:batch_stop].copy()
        batch_coords = batch_df[['z', 'y', 'x']].values
        # don't pass session: We want a unique session per thread
        batch_df.loc[:, 'label'] = fetch_labels(server, uuid, instance, batch_coords, supervoxels, scale)
        return batch_df

    batch_starts = list(range(0, len(coords_df), batch_size))
    if threads <= 1:
        batch_dfs = map(fetch_batch, batch_starts)
        batch_dfs = tqdm_proxy(batch_dfs, total=len(batch_starts), leave=False)
        batch_dfs = list(batch_dfs)
    else:
        with ThreadPool(threads) as pool:
            batch_dfs = pool.imap_unordered(fetch_batch, batch_starts)
            batch_dfs = tqdm_proxy(batch_dfs, total=len(batch_starts), leave=False)
            batch_dfs = list(batch_dfs)

    return pd.concat(batch_dfs).sort_index()['label'].values

    
@dvid_api_wrapper
def fetch_sparsevol_rles(server, uuid, instance, label, supervoxels=False, scale=0, *, session=None):
    """
    Fetch the sparsevol RLE representation for a given label.
    
    See also: neuclease.dvid.rle.parse_rle_response()
    """
    supervoxels = str(bool(supervoxels)).lower() # to lowercase string
    url = f'http://{server}/api/node/{uuid}/{instance}/sparsevol/{label}?supervoxels={supervoxels}&scale={scale}'
    r = session.get(url)
    r.raise_for_status()
    return r.content


@dvid_api_wrapper
def post_split_supervoxel(server, uuid, instance, supervoxel, rle_payload_bytes, downres=True, *, split_id=None, remain_id=None, session=None):
    """
    Split the given supervoxel according to the provided RLE payload,
    as specified in DVID's split-supervoxel docs.
    
    Args:
    
        server, uuid, intance:
            Segmentation instance
        
        supervoxel:
            ID of the supervoxel to split
        
        rle_payload_bytes:
            RLE binary payload, in the format specified by the DVID docs.
        
        split_id, remain_id:
            DANGEROUS.  Instead of letting DVID choose the ID of the new 'split' and
            'remain' supervoxels, these parameters allow you to specify them yourself.
    
    Returns:
        The two new IDs resulting from the split: (split_sv_id, remaining_sv_id)
    """
    url = f'http://{server}/api/node/{uuid}/{instance}/split-supervoxel/{supervoxel}'


    if bool(split_id) ^ bool(remain_id):
        msg = ("I'm not sure if DVID allows you to specify the split_id "
               "without specifying remain_id (or vice-versa).  "
               "Please specify both (or neither).")
        raise RuntimeError(msg)
    
    params = {}
    if not downres:
        # true by default; not supported in older versions of dvid.
        params['downres'] = 'false'
    
    if split_id is not None:
        params['split'] = str(split_id)
    if remain_id is not None:
        params['remain'] = str(remain_id)
    
    r = session.post(url, data=rle_payload_bytes, params=params)
    r.raise_for_status()
    
    results = r.json()
    return (results["SplitSupervoxel"], results["RemainSupervoxel"] )


# Legacy name
split_supervoxel = post_split_supervoxel


@dvid_api_wrapper
def fetch_mapping(server, uuid, instance, supervoxel_ids, *, session=None):
    """
    For each of the given supervoxels, ask DVID what body they belong to.
    If the supervoxel no longer exists, it will map to label 0.
    
    Returns:
        pd.Series, with index named 'sv' and values named 'body'
    """
    supervoxel_ids = list(map(int, supervoxel_ids))
    body_ids = fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/mapping', json=supervoxel_ids, session=session)
    mapping = pd.Series(body_ids, index=np.asarray(supervoxel_ids, np.uint64), dtype=np.uint64, name='body')
    mapping.index.name = 'sv'
    return mapping


@dvid_api_wrapper
def fetch_mappings(server, uuid, instance, as_array=False, *, session=None):
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
def fetch_complete_mappings(server, uuid, instance, include_retired=True, kafka_msgs=None, sort=None, *, session=None):
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
    assert sort in (None, 'sv', 'body')
    
    # Read complete kafka log; we need both split and cleave info
    if kafka_msgs is None:
        kafka_msgs = read_kafka_messages(server, uuid, instance)
    split_events = fetch_supervoxel_splits_from_kafka(server, uuid, instance, kafka_msgs=kafka_msgs, session=session)
    split_tables = list(map(lambda t: np.asarray([row[:-1] for row in t], np.uint64), split_events.values()))
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
    base_mapping = fetch_mappings(server, uuid, instance, as_array=True, session=session)
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
    full_mapping = np.concatenate(parts)
    full_mapping = np.asarray(full_mapping, order='C')

    # Drop duplicates that may have been introduced via retired svs
    # (if DVID didn't filter them out)
    dupes = pd.Series(full_mapping[:,0]).duplicated(keep='last')
    full_mapping = full_mapping[(~dupes).values]
    
    # View as 1D buffer of structured dtype to sort in-place.
    # (Sorted index is more efficient with speed and RAM in pandas)
    mapping_view = memoryview(full_mapping.reshape(-1))
    np.frombuffer(mapping_view, dtype=[('sv', np.uint64), ('body', np.uint64)]).sort()

    # Construct pd.Series for fast querying
    s = pd.Series(index=full_mapping[:,0], data=full_mapping[:,1])
    
    if not include_retired:
        # Drop all rows with retired supervoxels, including:
        # identities we may have added that are now retired
        # any retired SVs erroneously included by DVID itself in the fetched mapping
        s.drop(retired_svs, inplace=True, errors='ignore')
    
    # Reload index to ensure most RAM-efficient implementation.
    # (This seems to make a big difference in RAM usage!)
    s.index = s.index.values

    s.index.name = 'sv'
    s.name = 'body'

    if sort == 'sv':
        s.sort_index(inplace=True)
    elif sort == 'body':
        s.sort_values(inplace=True)

    assert s.index.dtype == np.uint64
    assert s.dtype == np.uint64

    return s


@dvid_api_wrapper
def fetch_mutation_id(server, uuid, instance, body_id, *, session=None):
    response = fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/lastmod/{body_id}', session=session)
    return response["mutation id"]


@dvid_api_wrapper
def fetch_sparsevol_coarse(server, uuid, instance, label_id, supervoxels=False, *, session=None):
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
    
    See also: ``fetch_sparsevol_coarse_via_labelindex()``
    
    Note: The returned coordinates are not necessarily sorted.
    """
    supervoxels = str(bool(supervoxels)).lower()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/sparsevol-coarse/{label_id}?supervoxels={supervoxels}')
    r.raise_for_status()
    
    return parse_rle_response( r.content )


def fetch_sparsevol_coarse_threaded(server, uuid, instance, labels, supervoxels=False, num_threads=2):
    """
    Call fetch_sparsevol_coarse() for a list of threads using a ThreadPool.
    
    Returns:
        dict of { label: coords }
        If any of the sparsevols can't be found due to error 404,
        'coords' for that label will be None.
    """
    def fetch_coords(label_id):
        try:
            coords = fetch_sparsevol_coarse(server, uuid, instance, label_id, supervoxels)
            return (label_id, coords)
        except HTTPError as ex:
            if (ex.response is not None and ex.response.status_code == 404):
                return (label_id, None)
            raise
    
    with ThreadPool(num_threads) as pool:
        labels_coords = pool.imap_unordered(fetch_coords, labels)
        labels_coords = list(tqdm_proxy(labels_coords, total=len(labels)))
    
    return dict(labels_coords)


@dvid_api_wrapper
def fetch_sparsevol(server, uuid, instance, label, supervoxels=False, scale=0, dtype=np.int32, *, session=None):
    """
    Return coordinates of all voxels in the given body/supervoxel at the given scale.

    For dtype arg, see parse_rle_response()

    Note: At scale 0, this will be a LOT of data for any reasonably large body.
          Use with caution.
    """
    rles = fetch_sparsevol_rles(server, uuid, instance, label, supervoxels, scale, session=session)
    return parse_rle_response(rles, dtype)


def compute_changed_bodies(instance_info_a, instance_info_b, *, session=None):
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
    mapping_a = fetch_mappings(*instance_info_a, session=session)
    mapping_b = fetch_mappings(*instance_info_b, session=session)
    
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


@dvid_api_wrapper
def generate_sample_coordinate(server, uuid, instance, label_id, supervoxels=False, *, session=None):
    """
    Return an arbitrary coordinate that lies within the given body.
    Usually faster than fetching all the RLEs.
    """
    SCALE = 6 # sparsevol-coarse is always scale 6
    coarse_block_coords = fetch_sparsevol_coarse(server, uuid, instance, label_id, supervoxels, session=session)
    num_blocks = len(coarse_block_coords)
    middle_block_coord = (2**SCALE) * np.array(coarse_block_coords[num_blocks//2]) // 64 * 64
    middle_block_box = (middle_block_coord, middle_block_coord + 64)
    
    block = fetch_labelarray_voxels(server, uuid, instance, middle_block_box, supervoxels=supervoxels, session=session)
    nonzero_coords = np.transpose((block == label_id).nonzero())
    if len(nonzero_coords) == 0:
        label_type = {False: 'body', True: 'supervoxel'}[supervoxels]
        raise RuntimeError(f"The sparsevol-coarse info for this {label_type} ({label_id}) "
                           "appears to be out-of-sync with the scale-0 segmentation.")

    return middle_block_coord + nonzero_coords[0]


@dvid_api_wrapper
def fetch_labelmap_voxels(server, uuid, instance, box_zyx, scale=0, throttle=False, supervoxels=False, *, inflate=True, session=None):
    """
    Fetch a volume of voxels from the given instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'

        box_zyx:
            The bounds of the volume to fetch in the coordinate system for the requested scale.
            Given as a pair of coordinates (start, stop), e.g. [(0,0,0), (10,20,30)], in Z,Y,X order.
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
        
        inflate:
            If True, inflate the compressed voxels from DVID and return an ordinary ndarray
            If False, return a callable proxy that stores the compressed data internally,
            and that will inflate the data when called.
    
    Returns:
        ndarray, with shape == (box[1] - box[0])
    """
    box_zyx = np.asarray(box_zyx)
    assert np.issubdtype(box_zyx.dtype, np.integer), \
        f"Box has the wrong dtype.  Use an integer type, not {box_zyx.dtype}"
    assert box_zyx.shape == (2,3)

    # Labelarray data can be fetched very efficiently if the request is block-aligned
    # So, block-align the request no matter what.
    aligned_box = round_box(box_zyx, 64, 'out')
    aligned_shape = aligned_box[1] - aligned_box[0]

    shape_str = '_'.join(map(str, aligned_shape[::-1]))
    offset_str = '_'.join(map(str, aligned_box[0, ::-1]))
    
    params = {}
    params['compression'] = 'blocks'

    # We don't bother adding these to the query string if we
    # don't have to, just to avoid cluttering the http logs.
    if scale:
        params['scale'] = scale
    if throttle:
        params['throttle'] = str(bool(throttle)).lower()
    if supervoxels:
        params['supervoxels'] = str(bool(supervoxels)).lower()

    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/blocks/{shape_str}/{offset_str}', params=params)
    r.raise_for_status()

    def inflate_labelarray_blocks():
        aligned_volume = DVIDNodeService.inflate_labelarray_blocks3D_from_raw(r.content, aligned_shape, aligned_box[0])
        requested_box_within_aligned = box_zyx - aligned_box[0]
        return extract_subvol(aligned_volume, requested_box_within_aligned )
        
    inflate_labelarray_blocks.content = r.content
    
    if inflate:
        return inflate_labelarray_blocks()
    else:
        return inflate_labelarray_blocks


# Deprecated name
fetch_labelarray_voxels = fetch_labelmap_voxels


def post_labelmap_voxels(server, uuid, instance, offset_zyx, volume, scale=0, downres=False, noindexing=False, throttle=False, *, session=None):
    """
    Post a supervoxel segmentation subvolume to a labelmap instance.
    Internally, breaks the volume into blocks and uses the ``/blocks``
    endpoint to post the data, so the volume must be block-aligned.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'

        offset_zyx:
            The upper-left coordinate of the volume to be written.
            Must be block-aligned, in ``(z,y,x)`` order.
        
        volume:
            Data to post, ``uint64``.  Shape must be divisible by DVID's blockshape for labelmap instance.
        
        scale:
            Which pyramid scale to post this block to.

        downres:
            Specifies whether the given write should trigger regeneration
            of donwres pyramids for this block on the DVID server.
            Only permitted for scale 0 posts.
        
        noindexing:
            If True, will not compute label indices from the received voxel data.
            Normally only used during initial ingestion, when an external tool is
            used to overwrite the label indexes en masse.
            (See ``neuclease/bin/ingest_label_indexes.py``)
        
        throttle:
            If True, passed via the query string to DVID, in which case DVID might return a '503' error
            if the server is too busy to service the request.
            It is your responsibility to catch DVIDExceptions in that case.
        
        supervoxels:
            If True, request supervoxel data from the given labelmap instance.
    """
    offset_zyx = np.asarray(offset_zyx)
    shape = np.array(volume.shape)

    assert (offset_zyx % 64 == 0).all(), "Data must be block-aligned"
    assert (shape % 64 == 0).all(), "Data must be block-aligned"
    
    corners = []
    blocks = []
    for corner in ndrange(offset_zyx, offset_zyx + shape, (64,64,64)):
        corners.append(corner)
        vol_corner = corner - offset_zyx
        block = volume[box_to_slicing(vol_corner, vol_corner+64)]
        blocks.append( block )

    post_labelarray_blocks(server, uuid, instance, corners, blocks, scale, downres, noindexing, throttle, session=session)

# Deprecated name
post_labelarray_voxels = post_labelmap_voxels


@dvid_api_wrapper
def post_labelmap_blocks(server, uuid, instance, corners_zyx, blocks, scale=0, downres=False, noindexing=False, throttle=False, *, session=None):
    """
    Post supervoxel data to a labelmap instance, from a list of blocks.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'

        corners_zyx:
            The starting coordinates of each block in the list (in full-res voxel coordinates)
        
        blocks:
            An iterable of uint64 blocks, each with shape (64,64,64)
        
        scale:
            Which pyramid scale to post this block to.

        downres:
            Specifies whether the given blocks should trigger regeneration
            of donwres pyramids for this block on the DVID server.
            Only permitted for scale 0 posts.
        
        noindexing:
            If True, will not compute label indices from the received voxel data.
            Normally only used during initial ingestion, when an external tool is
            used to overwrite the label indexes en masse.
            (See ``neuclease/bin/ingest_label_indexes.py``)

        throttle:
            If True, passed via the query string to DVID, in which case DVID might return a '503' error
            if the server is too busy to service the request.
            It is your responsibility to catch DVIDExceptions in that case.
    """
    assert not downres or scale == 0, "downres option is only valid for scale 0"

    body_data = encode_labelarray_blocks(corners_zyx, blocks)
    if not body_data:
        return # No blocks

    # These options are already false by default, so we'll only include them if we have to.
    opts = { 'downres': downres, 'noindexing': noindexing, 'throttle': throttle }

    params = { 'scale': str(scale) }
    for opt, value in opts.items():
        if value:
            params[opt] = str(bool(value)).lower()

    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/blocks', params=params, data=body_data)
    r.raise_for_status()

# Deprecated name
post_labelarray_blocks = post_labelmap_blocks


def encode_labelarray_blocks(corners_zyx, blocks):
    """
    Encode a sequence of labelmap blocks to bytes, in the
    format expected by dvid's ``/blocks`` endpoint.

    The format for each block is:
    - 12 bytes for the block (X,Y,Z) location (its upper corner)
    - 4 bytes for the length of the encoded block data (N)
    - N bytes for the encoded block data, which consists of labelmap-compressed
      data, which is then further compressed using gzip after that.
    
    Args:
        corners_zyx:
            List or array of corners at which each block starts.
            Corners must be block-aligned.
        
        blocks:
            Iterable of blocks to encode.
            Each block must be 64px wide, uint64.
    
    Returns:
        bytes
    """
    if not hasattr(corners_zyx, '__len__'):
        corners_zyx = list(corners_zyx)

    if len(corners_zyx) == 0:
        return b''
    
    corners_zyx = np.asarray(corners_zyx)
    assert np.issubdtype(corners_zyx.dtype, np.integer), \
        f"corners array has the wrong dtype.  Use an integer type, not {corners_zyx.dtype}"
    
    corners_zyx = np.asarray(corners_zyx, np.int32)
    assert corners_zyx.ndim == 2
    assert corners_zyx.shape[1] == 3
    if hasattr(blocks, '__len__'):
        assert len(blocks) == len(corners_zyx)

    corners_xyz = corners_zyx[:, ::-1].copy('C')

    # dvid wants block coordinates, not voxel coordinates
    encoded_corners = map(bytes, corners_xyz // 64)

    def _encode_label_block(block):
        # We wrap the C++ call in this little pure-python function
        # solely for the sake of nice profiler output.
        return encode_label_block(block)

    encoded_blocks = []
    for block in blocks:
        assert block.shape == (64,64,64)
        block = np.asarray(block, np.uint64, 'C')
        encoded_blocks.append( gzip.compress(_encode_label_block(block)) )
        del block
    assert len(encoded_blocks) == len(corners_xyz)

    encoded_lengths = np.fromiter(map(len, encoded_blocks), np.int32)
    
    stream = BytesIO()
    for corner_buf, len_buf, block_buf in zip(encoded_corners, encoded_lengths, encoded_blocks):
        stream.write(corner_buf)
        stream.write(len_buf)
        stream.write(block_buf)

    body_data = stream.getbuffer()
    return body_data


def encode_labelarray_volume(offset_zyx, volume):
    """
    Encode a uint64 volume as labelarray data, located at the given offset coordinate.
    The coordinate and volume shape must be 64-px aligned.
    
    See ``decode_labelarray_volume()`` for the corresponding decode function.
    """
    offset_zyx = np.asarray(offset_zyx)
    shape = np.array(volume.shape)

    assert (offset_zyx % 64 == 0).all(), "Data must be block-aligned"
    assert (shape % 64 == 0).all(), "Data must be block-aligned"
    
    corners_zyx = list(ndrange(offset_zyx, offset_zyx + volume.shape, (64,64,64)))

    def gen_blocks():
        for corner in corners_zyx:
            vol_corner = corner - offset_zyx
            block = volume[box_to_slicing(vol_corner, vol_corner+64)]
            yield block
            del block

    return encode_labelarray_blocks(corners_zyx, gen_blocks())


def encode_nonaligned_labelarray_volume(offset_zyx, volume):
    """
    Encode a uint64 volume as labelarray data, located at the given offset coordinate.
    The volume need not be aligned to 64-px blocks, but the encoded result will be
    padded with zeros to ensure alignment.
    
    Returns:
        aligned_box, encoded_data
        where aligned_box indicates the extent of the encoded result.
    """
    offset_zyx = np.asarray(offset_zyx)
    shape = np.array(volume.shape)
    box_zyx = np.array([offset_zyx, offset_zyx + shape])
    
    from ...util import Grid, boxes_from_grid
    
    full_boxes = boxes_from_grid(box_zyx, Grid((64,64,64)), clipped=False)
    clipped_boxes = boxes_from_grid(box_zyx, Grid((64,64,64)), clipped=True)

    full_boxes = np.array(list(full_boxes))
    clipped_boxes = np.array(list(clipped_boxes))

    def gen_blocks():
        for full_box, clipped_box in zip(full_boxes, clipped_boxes):
            if (full_box == clipped_box).all():
                vol_corner = full_box[0] - offset_zyx
                block = volume[box_to_slicing(vol_corner, vol_corner+64)]
                yield block
                del block
            else:
                # Must extract clipped and copy it into a full box.
                vol_box = clipped_box - offset_zyx
                clipped_block = volume[box_to_slicing(*vol_box)]
                full_block = np.zeros((64,64,64), np.uint64)
                full_block[box_to_slicing(*(clipped_box - full_box[0]))] = clipped_block
                yield full_block
                del full_block
                del clipped_block

    aligned_box = np.array( (full_boxes[:,0,:].min(axis=0), 
                             full_boxes[:,1,:].max(axis=0)) )
    
    return aligned_box, encode_labelarray_blocks(full_boxes[:,0,:], gen_blocks())
    

def decode_labelarray_volume(box_zyx, encoded_data):
    """
    Decode the payload from the labelmap instance ``GET /blocks`` endpoint,
    or from the output of ``encode_labelarray_volume()``.
    """
    box_zyx = np.asarray(box_zyx)
    shape = box_zyx[1] - box_zyx[0]
    return DVIDNodeService.inflate_labelarray_blocks3D_from_raw(encoded_data, shape, box_zyx[0])


@dvid_api_wrapper
def post_cleave(server, uuid, instance, body_id, supervoxel_ids, *, session=None):
    """
    Execute a cleave operation on the given body.
    This "cleaves away" the given list of supervoxel ids into a new body,
    whose ID will be chosen by DVID.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'

        body_id:
            The body ID from which to cleave the supervoxels
        
        supervoxel_ids:
            The list of supervoxels to cleave out of the given body.
            (All of the given supervoxel IDs must be mapped to the given body)
    
    Returns:
        The label ID of the new body created by the cleave operation.
    """
    supervoxel_ids = list(map(int, supervoxel_ids))

    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/cleave/{body_id}', json=supervoxel_ids)
    r.raise_for_status()
    cleaved_body = r.json()["CleavedLabel"]
    return cleaved_body


@dvid_api_wrapper
def post_merge(server, uuid, instance, main_label, other_labels, *, session=None):
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
    
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/merge', json=content)
    r.raise_for_status()
    

def labelmap_kafka_msgs_to_df(kafka_msgs, default_timestamp=DEFAULT_TIMESTAMP):
    """
    Convert the kafka messages for a labelmap instance into a DataFrame.
    """
    df = kafka_msgs_to_df(kafka_msgs, drop_duplicates=False, default_timestamp=default_timestamp)

    # Append action and 'body'
    df['action'] = [msg['Action'] for msg in df['msg']]
    
    mutation_bodies = defaultdict(lambda: 0)
    mutation_svs = defaultdict(lambda: 0)
    
    target_bodies = []
    target_svs = []
    
    # This logic is somewhat more complex than you might think is necesary,
    # but that's because the kafka logs (sadly) contain duplicate mutation IDs,
    # i.e. the mutation ID was not unique in our earlier logs.
    for msg in df['msg'].values:
        action = msg['Action']
        mutid = msg['MutationID']

        if not action.endswith('complete'):
            target_body = 0
            target_sv = 0

            if action == 'cleave':
                target_body = msg['OrigLabel']
            elif action in ('merge', 'split'):
                target_body = msg['Target']
            elif action == 'split-supervoxel':
                target_sv = msg['Supervoxel']

            target_bodies.append(target_body)
            target_svs.append(target_sv)

            mutation_bodies[mutid] = target_body
            mutation_svs[mutid] = target_sv

        else:
            # The ...-complete messages contain nothing but the action, uuid, and mutation ID,
            # but as a convenience we will match them with the target_body or target_sv,
            # based on the most recent message with a matching mutation ID.
            target_bodies.append( mutation_bodies[mutid] )
            target_svs.append( mutation_svs[mutid] )

    df['target_body'] = target_bodies
    df['target_sv'] = target_svs

    return df[['timestamp', 'uuid', 'mutid', 'action', 'target_body', 'target_sv', 'msg']]


def compute_affected_bodies(kafka_msgs):
    """
    Given a list of json messages from a labelmap instance,
    Compute the set of all bodies that are mentioned in the log as either new, changed, or removed.
    Also return the set of new supervoxels from 'supervoxel-split' actions.
    
    Note: Supervoxels from 'split' actions are not included in new_supervoxels.
          If you're interested in all supervoxel splits, see fetch_supervoxel_splits().
    
    Note:
        These results do not consider any '-complete' messsages in the list.
        If an operation failed, it may still be included in these results.   

    See also:
        neuclease.dvid.kafka.filter_kafka_msgs_by_timerange()

    Args:
        Kafka log for a labelmap instance, obtained via read_kafka_messages().
    
    Returns:
        new_bodies, changed_bodies, removed_bodies, new_supervoxels
    
    Example:
    
        >>> # Compute the list of bodies whose meshes are possibly outdated.
    
        >>> kafka_msgs = read_kafka_messages(server, uuid, seg_instance)
        >>> filtered_kafka_msgs = filter_kafka_msgs_by_timerange(kafka_msgs, min_timestamp="2018-11-22")
        
        >>> new_bodies, changed_bodies, _removed_bodies, new_supervoxels = compute_affected_bodies(filtered_kafka_msgs)
        >>> sv_split_bodies = set(fetch_mapping(server, uuid, seg_instance, new_supervoxels)) - set([0])

        >>> possibly_outdated_bodies = (new_bodies | changed_bodies | sv_split_bodies)
    
    """
    new_bodies = set()
    changed_bodies = set()
    removed_bodies = set()
    new_supervoxels = set()
    
    for msg in kafka_msgs:
        if msg['Action'].endswith('complete'):
            continue
        
        if msg['Action'] == 'cleave':
            changed_bodies.add( msg['OrigLabel'] )
            new_bodies.add( msg['CleavedLabel'] )
    
        if msg['Action'] == 'merge':
            changed_bodies.add( msg['Target'] )
            labels = set( msg['Labels'] )
            removed_bodies |= labels
            changed_bodies -= labels
            new_bodies -= labels
        
        if msg['Action'] == 'split':
            changed_bodies.add( msg['Target'] )
            new_bodies.add( msg['NewLabel'] )
            
        if msg['Action'] == 'split-supervoxel':
            new_supervoxels.add(msg['SplitSupervoxel'])
            new_supervoxels.add(msg['RemainSupervoxel'])

    return new_bodies, changed_bodies, removed_bodies, new_supervoxels
