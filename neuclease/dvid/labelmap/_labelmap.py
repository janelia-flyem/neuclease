import re
import gzip
import logging
from io import BytesIO
from functools import partial
from multiprocessing.pool import ThreadPool
from collections import defaultdict

import numpy as np
import pandas as pd
from requests import HTTPError

from libdvid import DVIDNodeService, encode_label_block

from ...util import Timer, round_box, extract_subvol, DEFAULT_TIMESTAMP, tqdm_proxy, ndrange, box_to_slicing, compute_parallel

from .. import dvid_api_wrapper, fetch_generic_json, fetch_repo_info
from ..repo import create_voxel_instance, fetch_repo_dag
from ..kafka import read_kafka_messages, kafka_msgs_to_df
from ..rle import parse_rle_response

from ._split import SplitEvent, fetch_supervoxel_splits_from_kafka
from .labelops_pb2 import MappingOps, MappingOp
from neuclease.dvid.server import fetch_server_info

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
    """
    Fetch the list of supervoxel IDs that are associated with the given body.
    """
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
    """
    Wrapper for DVID's /size endpoint.
    Returns the size (voxel count) of a single body (or supervoxel)
    which DVID obtains by reading the body's label indexes.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        label_id:
            A single label ID to fetch size of.
            Should be a body ID, unless supervoxels=True,
            in which case it should be a supervoxel ID.
        
        supervoxels:
            If True, interpret label_id as supervoxel ID,
            and return a supervoxel size, not a body size.
    Returns:
        The voxel count of the given body/supervoxel, as an integer.
    """
    supervoxels = str(bool(supervoxels)).lower()
    url = f'http://{server}/api/node/{uuid}/{instance}/size/{label_id}?supervoxels={supervoxels}'
    response = fetch_generic_json(url, session=session)
    return response['voxels']

# FIXME: Deprecated name
fetch_body_size = fetch_size


@dvid_api_wrapper
def fetch_sizes(server, uuid, instance, label_ids, supervoxels=False, *, session=None):
    """
    Wrapper for DVID's /sizes endpoint.
    Returns the size (voxel count) of the given bodies (or supervoxels),
    which DVID obtains by reading the bodies' label indexes.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        label_ids:
            List of label IDs to fetch sizes for.
            Should be a list of body IDs, unless supervoxels=True,
            in which case it should be a list of supervoxel IDs.
        
        supervoxels:
            If True, interpret label_ids as a list of supervoxel IDs,
            and return supervoxel sizes, not body sizes.
    Returns:
        pd.Series, of the size results, in the same order as the labels passed in.
        Indexed by label ID.
    """
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

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        coordinate_zyx:
            A single coordinate ``[z,y,x]``
        
        supervoxels:
            If True, read supervoxel IDs from DVID, not body IDs.
        
        scale:
            Which scale of the data to read from.
            (Your coordinates must be correspondingly scaled.)

    Returns:
        ``np.uint64``
    
    See also:
        ``fetch_labels()``, ``fectch_labels_batched()``
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

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        coordinates_zyx:
            array of shape (N,3) with coordinates to sample.
            Rows must be ``[[z,y,x], [z,y,x], ...``
        
        supervoxels:
            If True, read supervoxel IDs from DVID, not body IDs.
        
        scale:
            Which scale of the data to read from.
            (Your coordinates must be correspondingly scaled.)

    Returns:
        ndarray of N labels

    See also:
        ``fetch_label()``, ``fectch_labels_batched()``
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


def fetch_labels_batched(server, uuid, instance, coordinates_zyx, supervoxels=False, scale=0, batch_size=10_000, threads=0, processes=0, presort=True):
    """
    Like fetch_labels, but fetches in batches, optionally multithreaded or multiprocessed.

    See also: ``fetch_label()``, ``fectch_labels()``
    """
    assert not threads or not processes, "Choose either threads or processes (not both)"
    coords_df = pd.DataFrame(coordinates_zyx, columns=['z', 'y', 'x'], dtype=np.int32)
    coords_df['label'] = np.uint64(0)
    
    if presort:
        with Timer(f"Pre-sorting {len(coords_df)} coordinates by block index", logger):
            # Sort coordinates by their block index,
            # so DVID will be able to service the requests faster.
            coords_df['bz'] = coords_df['z'] // 64
            coords_df['by'] = coords_df['y'] // 64
            coords_df['bx'] = coords_df['x'] // 64
            coords_df.sort_values(['bz', 'by', 'bx'], inplace=True)
            del coords_df['bz']
            del coords_df['by']
            del coords_df['bx']

    fetch_batch = partial(_fetch_labels_batch, server, uuid, instance, supervoxels, scale)

    batch_dfs = []
    for batch_start in range(0, len(coords_df), batch_size):
        batch_stop = min(batch_start+batch_size, len(coords_df))
        batch_df = coords_df.iloc[batch_start:batch_stop].copy()
        batch_dfs.append(batch_df)

    with Timer("Fetching labels from DVID", logger):
        batch_starts = list(range(0, len(coords_df), batch_size))
        if threads <= 1 and processes <= 1:
            batch_result_dfs = map(fetch_batch, batch_dfs)
            batch_result_dfs = tqdm_proxy(batch_result_dfs, total=len(batch_starts), leave=False, logger=logger)
            batch_result_dfs = list(batch_result_dfs)
        else:
            batch_result_dfs = compute_parallel(fetch_batch, batch_dfs, 1, threads, processes, ordered=False, leave_progress=False)

    return pd.concat(batch_result_dfs).sort_index()['label'].values


def _fetch_labels_batch(server, uuid, instance, supervoxels, scale, batch_df):
    """
    Helper for fetch_labels_batched(), defined at top-level so it can be pickled.
    """
    batch_coords = batch_df[['z', 'y', 'x']].values
    # don't pass session: We want a unique session per thread
    batch_df.loc[:, 'label'] = fetch_labels(server, uuid, instance, batch_coords, supervoxels, scale)
    return batch_df
    

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
def fetch_mappings(server, uuid, instance, as_array=False, *, format=None, session=None): # @ReservedAssignment
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
        
        format:
            The format in which dvid should return the data, either 'csv' or 'binary'.
            By default, the DVID server version is checked to see if 'binary' is supported,
            and 'binary' is used if possible.  (The 'binary' format saves some time,
            since there is no need to parse CSV.)

    Returns:
        pd.Series(index=sv, data=body), unless as_array is True
    """
    if format is None:
        # The first DVID version to support the 'binary' format is v0.8.24-15-g40b8b77
        dvid_version = fetch_server_info(server)["DVID Version"]
        assert dvid_version.startswith('v')
        parts = re.split(r'[.-]', dvid_version[1:])
        parts = parts[:4]
        parts = tuple(int(p) for p in parts)
        if parts >= (0, 8, 24, 15):
            format = 'binary' # @ReservedAssignment
        else:
            format = 'csv' # @ReservedAssignment

    if format == 'binary':
        # This takes ~30 seconds so it's nice to log it.
        uri = f"http://{server}/api/node/{uuid}/{instance}/mappings?format=binary"
        with Timer(f"Fetching {uri}", logger):
            r = session.get(uri)
            r.raise_for_status()
        
        a = np.frombuffer(r.content, np.uint64).reshape(-1,2)
        if as_array:
            return a

        df = pd.DataFrame(a, columns=['sv', 'body'])

    else:
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
        
        kafka_msgs:
            Optionally provide the complete labelmap kafka log if you've got it,
            in which case this function doesn't need to re-fetch it.
        
        sort:
            Optional.
            If 'sv', sort by supervoxel column.
            If 'body', sort by body. Otherwise, don't sort.

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
def post_mappings(server, uuid, instance, mappings, mutid, *, batch_size=None, session=None):
    """
    Post a list of SV-to-body mappings to DVID, provided as a ``pd.Series``.
    Will be converted to protobuf structures for upload.
    
    Note:
        This is not intended for general use. It is used to initialize
        the mapping after an initial ingestion of supervoxels.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        mappings:
            ``pd.Series``, indexed by sv, bodies as value
        
        mutid:
            The mutation ID to use when posting the mappings
        
        batch_size:
            If provided, the mappings will be sent in batches, whose sizes will
            roughly correspond to the given size, in terms of the number of
            supervoxels in the batch.
    """
    assert isinstance(mappings, pd.Series)
    df = pd.DataFrame(mappings.rename('body'))
    df.index.name = 'sv'
    df = df.reset_index()

    def _post_mapping_ops(ops_list):
        ops = MappingOps()
        ops.mappings.extend(ops_list)
        payload = ops.SerializeToString()
        r = session.post(f'http://{server}/api/node/{uuid}/{instance}/mappings', data=payload)
        r.raise_for_status()

    progress_bar = tqdm_proxy(total=len(df), disable=(batch_size is None), logger=logger)
    progress_bar.update(0)

    batch_ops_so_far = 0
    ops_list = []
    for body_id, body_df in df.groupby('body'):
        op = MappingOp()
        op.mutid = mutid
        op.mapped = body_id
        op.original.extend( body_df['sv'] )

        # Add to this chunk of ops
        ops_list.append(op)
        batch_ops_so_far += len(op.original)

        # Send if chunk is full
        if (batch_size is not None) and (batch_ops_so_far >= batch_size):
            _post_mapping_ops(ops_list)
            progress_bar.update(batch_ops_so_far)
            ops_list = [] # reset
            batch_ops_so_far = 0

    # send last chunk, if there are leftovers
    if ops_list:
        _post_mapping_ops(ops_list)
        progress_bar.update(batch_ops_so_far)


def copy_mappings(src_info, dest_info, batch_size=None, *, session=None):
    """
    Copy the complete in-memory mapping from one server to another,
    performed in batches and with a progress display.

    Args:
        src_triple:
            tuple (server, uuid, instance) to copy from
        
        dest_triple:
            tuple (server, uuid, instance) to copy to

        batch_size:
            If provided, the mappings will be sent in batches, whose sizes will
            roughly correspond to the given size, in terms of the number of
            supervoxels in the batch.
    """
    # Pick the higher mutation id between the source and destination
    src_mutid = fetch_repo_info(*src_info[:2])["MutationID"]
    dest_mutid = fetch_repo_info(*dest_info[:2])["MutationID"]
    mutid = max(src_mutid, dest_mutid)
    
    mappings = fetch_mappings(*src_info)
    post_mappings(*dest_info, mappings, mutid, batch_size=batch_size, session=session)
    

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
    Call fetch_sparsevol_coarse() for a list of labels using a ThreadPool.
    
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
        labels_coords = list(tqdm_proxy(labels_coords, total=len(labels)), logger=logger)
    
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
    
    This function fetches the sparsevol-coarse to select a block
    in which the body of interest can be found, then it fetches the segmentation
    for that block and picks a point within it that lies within the body of interest.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        label_id:
            A body or supervoxel ID (if supervoxels=True)
        
        supervoxels:
            If True, treat ``label_id`` as a supervoxel ID.
    
    Returns:
        [Z,Y,X] -- An arbitrary point within the body of interest.
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
def fetch_labelmap_voxels(server, uuid, instance, box_zyx, scale=0, throttle=False, supervoxels=False, *, format='array', session=None):
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
        
        format:
            If 'array', inflate the compressed voxels from DVID and return an ordinary ndarray
            If 'lazy-array', return a callable proxy that stores the compressed data internally,
            and that will inflate the data when called.
            If 'raw-response', return DVID's raw /blocks response buffer without inflating it.
    
    Returns:
        ndarray, with shape == (box[1] - box[0])
    """
    assert format in ('array', 'lazy-array', 'raw-response')
    box_zyx = np.asarray(box_zyx)
    assert np.issubdtype(box_zyx.dtype, np.integer), \
        f"Box has the wrong dtype.  Use an integer type, not {box_zyx.dtype}"
    assert box_zyx.shape == (2,3)

    if format == 'raw-response':
        assert (box_zyx % 64 == 0).all(), \
            f"If requesting raw-response, the requested box must be block-aligned, not {box_zyx.tolist()}."

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
    
    if format == 'array':
        return inflate_labelarray_blocks()
    elif format == 'lazy-array':
        return inflate_labelarray_blocks
    elif format == 'raw-response':
        return r.content
    else:
        raise AssertionError(f"Unknown format: {format}")


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

    post_labelmap_blocks(server, uuid, instance, corners, blocks, scale, downres, noindexing, throttle, session=session)

# Deprecated name
post_labelarray_voxels = post_labelmap_voxels


@dvid_api_wrapper
def post_labelmap_blocks(server, uuid, instance, corners_zyx, blocks, scale=0, downres=False, noindexing=False, throttle=False, *, is_raw=False, gzip_level=6, session=None):
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
        
        is_raw:
            If you have already encoded the blocks in DVID's compressed labelmap format
            (or fetched them directly from another node), you may pass them as a raw buffer,
            and set ``is_raw`` to True, and set corners_zyx to ``None``.

        gzip_level:
            The level of gzip compression to use, from 0 (no compression) to 9.
    """
    assert not downres or scale == 0, "downres option is only valid for scale 0"

    if is_raw:
        assert isinstance(blocks, (bytes, memoryview))
        body_data = blocks
    else:
        body_data = encode_labelarray_blocks(corners_zyx, blocks, gzip_level)
    
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


def encode_labelarray_blocks(corners_zyx, blocks, gzip_level=6):
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
        
        gzip_level:
            The level of gzip compression to use, from 0 (no compression) to 9.
    
    Returns:
        memoryview
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
        encoded_blocks.append( gzip.compress(_encode_label_block(block), gzip_level) )
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


def encode_labelarray_volume(offset_zyx, volume, gzip_level=6):
    """
    Encode a uint64 volume as labelarray data, located at the given offset coordinate.
    The coordinate and volume shape must be 64-px aligned.

    Args:
        offset_zyx:
            The upper-left coordinate of the volume to be written.
            Must be block-aligned, in ``(z,y,x)`` order.

        volume:
            ndarray, uint64.  Must be block-aligned.
        
        gzip_level:
            The level of gzip compression to use, from 0 (no compression) to 9.
    
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

    return encode_labelarray_blocks(corners_zyx, gen_blocks(), gzip_level)


def encode_nonaligned_labelarray_volume(offset_zyx, volume, gzip_level=6):
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
    
    return aligned_box, encode_labelarray_blocks(full_boxes[:,0,:], gen_blocks(), gzip_level)
    

def decode_labelarray_volume(box_zyx, encoded_data):
    """
    Decode the payload from the labelmap instance ``GET /blocks`` endpoint,
    or from the output of ``encode_labelarray_volume()``.
    """
    box_zyx = np.asarray(box_zyx)
    shape = box_zyx[1] - box_zyx[0]
    return DVIDNodeService.inflate_labelarray_blocks3D_from_raw(encoded_data, shape, box_zyx[0])


def parse_labelarray_data(encoded_data, extract_labels=True):
    """
    For a buffer of encoded labelarray/labelmap data,
    extract the block IDs and label list for each block,
    and return them, along with the spans of the buffer
    in which each block resides.
    
    Args:
        encoded_data:
            Raw gzip-labelarray-compressed block data as returned from dvid via
            ``GET .../blocks?compression=blocks``,
            e.g. via ``fetch_labelmap_voxels(..., format='raw-response')``
        
        extract_labels:
            If True, extract the list of labels contained within the block.
            This is somewhat expensive because it requires decompressing the block's
            gzip-compressed portion.
    
    Returns:
        spans, or (spans, labels) depending on whether or not extract_labels is True,
        where spans and labels are both dicts using block_ids as keys:
            spans: { block_id: (start, stop) }
            labels: { block_id: label_ids }
    """
    # The format for each block is:
    # - 12 bytes for the block (X,Y,Z) location (its upper corner)
    # - 4 bytes for the length of the encoded block data (N)
    # - N bytes for the encoded block data, which consists of labelmap-compressed
    #   data, which is then further compressed using gzip after that.
    # 
    # The N bytes of the encoded block data is gzip-compressed.
    # After unzipping, it starts with:
    # 
    #   3 * uint32      values of gx, gy, and gz
    #   uint32          # of labels (L), cannot exceed uint32.
    #   L * uint64      packed labels in little-endian format.  Label 0 can be used to represent
    #                       deleted labels, e.g., after a merge operation to avoid changing all
    #                       sub-block indices.
    # 
    # See dvid's ``POST .../blocks`` documentation for more details.

    assert isinstance(encoded_data, (bytes, memoryview))

    spans = {}
    labels = {}

    pos = 0
    while pos < len(encoded_data):
        start_pos = pos
        x, y, z, num_bytes = np.frombuffer(encoded_data[pos:pos+16], np.int32)
        pos += 16
        end_pos = pos + num_bytes
        spans[(z, y, x)] = (start_pos, end_pos)

        if extract_labels:
            block_data = gzip.decompress( encoded_data[pos:end_pos] )
            gx, gy, gz, num_labels = np.frombuffer(block_data[:16], np.uint32)
            assert gx == gy == gz == 8, "Invalid block data"
            block_labels = np.frombuffer(block_data[16:16+8*num_labels], np.uint64)
            labels[(z, y, x)] = block_labels
        
        pos = end_pos

    if extract_labels:
        return (spans, labels)
    else:
        return spans


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
def post_hierarchical_cleaves(server, uuid, instance, body_id, group_mapping, *, session=None):
    """
    When you want to perform a lot of cleaves on a single
    body (e.g. a "frankenbody") whose labelindex is really big,
    it is more efficient to perform some coarse cleaves at first,
    which will iteratively divide the labelindex into big chunks,
    and then re-cleave the coarsely cleaved objects.
    That will run faster than cleaving off little bits one at a time,
    thanks to the reduced labelindex sizes at each step.
    
    Given a set of N supervoxel groups that belong to a single body,
    this function will issue N cleaves to leave each group as its own body,
    but using a strategy that should be more efficient than naively cleaving
    each group off independently.
    
    Note:
        If the given groups include all supervoxels in the body, then
        one of the groups (the last one) will be left with the original
        body ID. Otherwise, every group will be given a new body ID.
    
    Performance:
        One infamous "frankenbody" in the hemibrain dataset,
        which had 100k supervoxels touching 400k blocks, required 75k cleaves.
        With this function, those 75k cleaves can be completed in 8 minutes.
        (A naive series of cleaves would have taken 20-40 hours.)
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'

        body_id:
            The body ID from which to cleave the supervoxels.
            All supervoxels in the ``group_mapping`` index must belong to this body.
        
        group_mapping:
            A ``pd.Series`` whose index is supervoxel IDs of the cleave,
            and whose values are arbitrary component IDs indicating the final
            supervoxel grouping for the cleaves that will be performed.
            (The actual value of the group IDs are not used in the cleave operation.)

    Returns:
        A DataFrame indexed by the SVs in your group_mapping (though not necessarily in the same order),
        with columns for the group you provided and the new body ID it now maps to after cleaving.
    
    Example:

        .. code-block:: python
        
            body_id = 20
            svs    = [1,2,3,4,5,6,7,8,9,10]
            groups = [1,1,2,2,3,3,3,3,3,4] # Arbitrary IDs (not body IDs)
            group_mapping = pd.Series(index=svs, data=groups)
            final_mapping_df = post_hierarchical_cleaves(server, uuid, instance, body_id, group_mapping)
    """
    from . import fetch_labelindex # late import to avoid recursive import

    assert isinstance(group_mapping, pd.Series)
    assert group_mapping.index.dtype == np.uint64
    group_mapping = group_mapping.rename('group', copy=False)
    group_mapping = group_mapping.rename_axis('sv', copy=False)
    
    dvid_mapping = fetch_mapping(server, uuid, instance, group_mapping.index.values, session=session)
    assert (dvid_mapping == body_id).all(), \
        "All supervoxels in the group_mapping index must map (in DVID) to the given body_id"

    group_df = pd.DataFrame(group_mapping)
    assert group_df.index.duplicated().sum() == 0, \
        "Your group_mapping includes duplicate values for some supervoxels."
    
    def _cleave_groups(body, li_df, bodies):
        """
        Recursively split the given dataframe (a view of li_df, defined below) in half,
        and post a cleave to split the SVs in the top half from the bottom half in DVID.

        Args:
            li_df:
                A slice view of li_df (defined below).
                It has the same columns as a labelindex dataframe,
                but it's sorted by group, and has an extra column to
                mark the group boundaries.
            
            bodies: 
                A slice view of the above 'bodies' array (defined below),
                and will be overwritten to with the body IDs that DVID returns.
        """
        assert (bodies == body).all()
        if li_df['mark'].sum() == 1:
            # Only one group present; no cleaves necessary
            assert (li_df['group'] == li_df['group'].iloc[0]).all()
            return
        
        # Choose a marked row that approximately divides the DF in half
        N = len(li_df)
        (h1_marked_rows,) = li_df['mark'].iloc[:N//2].values.nonzero()
        (h2_marked_rows,) = li_df['mark'].iloc[N//2:].values.nonzero()
        h2_marked_rows += N//2
        
        if len(h1_marked_rows) == 0:
            # Top half has no marks, choose first mark in bottom half
            split_row = h2_marked_rows[0]
        elif len(h2_marked_rows) == 0:
            # Bottom half has no marks, choose last mark of top half
            split_row = h1_marked_rows[-1]
        else:
            # Both halves have marks, choose either the last-of-top or first-of-bottom,
            # depending on which one yields a closer split.
            if h1_marked_rows[-1] < (N-h2_marked_rows[0]):
                split_row = h2_marked_rows[0]
            else:
                split_row = h1_marked_rows[-1]

        top_df = li_df.iloc[:split_row]
        bottom_df = li_df.iloc[split_row:]
        
        top_bodies = bodies[:split_row]
        bottom_bodies = bodies[split_row:]
        
        # Cleave the top away
        top_svs = np.sort(pd.unique(top_df['sv'].values))
        top_body = post_cleave(server, uuid, instance, body, top_svs, session=session)
        
        progress_bar.update(1)
        
        # Update the shared body column (a view)
        top_bodies[:] = top_body
        
        # Recurse
        _cleave_groups(top_body, top_df, top_bodies)
        _cleave_groups(body, bottom_df, bottom_bodies)


    li_df = fetch_labelindex(server, uuid, instance, body_id, format='pandas', session=session).blocks
    orig_len = len(li_df)
    li_df = li_df.query('sv in @group_df.index')
    
    # If we the groups don't include every supervoxel in the body,
    # then start by cleaving the entire set out from its body, for two reasons:
    # 1. _cleave_groups() below terminates once all groups have unique body IDs,
    #   so the last group will get skipped, (unless we follow-up with a final cleave),
    #   leaving it with the main body ID (not what we wanted).
    # 2. If the non-cleaved supervoxels in the body are large, then they will
    #   add a lot of overhead in the subsequent cleaves if we don't cleave them away first.
    need_initial_cleave = (len(li_df) != orig_len)
    
    li_df = li_df.merge(group_df, 'left', 'sv')
    li_df = li_df.sort_values(['group', 'sv']).reset_index(drop=True).copy()

    # Mark the first row of each group
    groups = li_df['group'].values
    li_df['mark'] = True
    li_df.loc[1:, 'mark'] = (groups[1:] != groups[:-1])

    bodies = np.full(len(li_df), body_id, np.uint64)

    num_cleaves = int(li_df['mark'].sum())
    if not need_initial_cleave:
        num_cleaves -= 1

    with tqdm_proxy(total=num_cleaves, logger=logger) as progress_bar:
        progress_bar.update(0)
    
        if need_initial_cleave:
            body_id = post_cleave(server, uuid, instance, body_id, pd.unique(li_df['sv'].values), session=session)
            progress_bar.update(1)
            bodies[:] = body_id

        # Perform the hierarchical (recursive) cleaves
        _cleave_groups(body_id, li_df, bodies)
    
        li_df['body'] = bodies

    return li_df[['sv', 'group', 'body']].drop_duplicates('sv').set_index('sv')


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
    assert main_label not in other_labels, \
        (f"Can't merge {main_label} with itself.  "
        "DVID does not behave correctly if you attempt to merge a body into itself!")
    main_label = int(main_label)
    other_labels = list(map(int, other_labels))
    
    content = [main_label] + other_labels
    
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/merge', json=content)
    r.raise_for_status()
    

def read_labelmap_kafka_df(server, uuid, instance='segmentation', default_timestamp=DEFAULT_TIMESTAMP, drop_completes=True):
    """
    Convenience function for reading the kafka log for
    a labelmap instance and loading it into a DataFrame.
    Basically a combination of 
    
    Args:
        server, uuid, instance:
            A labelmap instance for which a kafka log exists.
        
        default_timestamp:
            See labelmap_kafka_msgs_to_df()
        
        drop_completes:
            See labelmap_kafka_msgs_to_df()

    Returns:
        DataFrame with columns:
            ['timestamp', 'uuid', 'mutid', 'action', 'target_body', 'target_sv', 'msg']
    """
    msgs = read_kafka_messages(server, uuid, instance)
    msgs_df = labelmap_kafka_msgs_to_df(msgs, default_timestamp, drop_completes)
    return msgs_df


def labelmap_kafka_msgs_to_df(kafka_msgs, default_timestamp=DEFAULT_TIMESTAMP, drop_completes=True):
    """
    Convert the kafka messages for a labelmap instance into a DataFrame.

    Args:
        kafka_msgs:
            A list of JSON messages from the kafka log emitted by a labelmap instance.
        
        default_timestamp:
            Old versions of DVID did not emit a timestamp with each message.
            For such messages, we'll assign a default timestamp, specified by this argument.

        drop_completes:
            If True, don't return the completion confirmation messages.
            That is, drop 'merge-complete', 'cleave-complete', 'split-complete',
            and 'split-supervoxel-complete'.
            Note: No attempt is made to ensure that all returned messages actually
            had a corresponding 'complete' message.  If some operation in the log
            failed (and thus has no corresponding 'complete' message), it will
            still be included in the output.
    
    """
    df = kafka_msgs_to_df(kafka_msgs, drop_duplicates=False, default_timestamp=default_timestamp)

    # Append action and 'body'
    df['action'] = [msg['Action'] for msg in df['msg']]

    if drop_completes:
        completes = df['action'].map(lambda s: s.endswith('-complete'))
        df = df[~completes].copy()
    
    mutation_bodies = defaultdict(lambda: 0)
    mutation_svs = defaultdict(lambda: 0)
    
    target_bodies = []
    target_svs = []
    
    # This logic is somewhat more complex than you might think is necessary,
    # but that's because the kafka logs (sadly) contain duplicate mutation IDs,
    # i.e. the mutation ID was not unique in our earlier logs.
    for msg in df['msg'].values:
        action = msg['Action']
        try:
            mutid = msg['MutationID']
        except KeyError:
            mutid = 0

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
    
    new_bodies = np.fromiter(new_bodies, np.uint64)
    changed_bodies = np.fromiter(changed_bodies, np.uint64)
    removed_bodies = np.fromiter(removed_bodies, np.uint64)
    new_supervoxels = np.fromiter(new_supervoxels, np.uint64)

    return new_bodies, changed_bodies, removed_bodies, new_supervoxels
