import re
import gzip
import logging
from io import BytesIO
from functools import partial, lru_cache, wraps
from itertools import starmap
from multiprocessing.pool import ThreadPool
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from requests import HTTPError

from libdvid import DVIDNodeService, encode_label_block
from dvidutils import LabelMapper
from vigra.analysis import labelMultiArrayWithBackground

from ...util import (Timer, round_box, extract_subvol, DEFAULT_TIMESTAMP, tqdm_proxy,
                     ndrange, ndrange_array, box_to_slicing, compute_parallel, boxes_from_grid, box_shape,
                     overwrite_subvol, iter_batches, extract_labels_from_volume, box_intersection, downsample_mask)

from .. import dvid_api_wrapper, fetch_generic_json, fetch_repo_info
from ..repo import create_voxel_instance, fetch_repo_dag, resolve_ref, expand_uuid
from ..kafka import read_kafka_messages, kafka_msgs_to_df
from ..rle import parse_rle_response, runlength_decode_from_ranges_to_mask

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
    url = f'{server}/api/node/{uuid}/{instance}/maxlabel'

    try:
        return fetch_generic_json(url, session=session)["maxlabel"]
    except HTTPError as ex:
        if ex.response is None or 'No maximum label' not in ex.response.content.decode('utf-8'):
            raise

        uuid = resolve_ref(server, uuid)
        uuid = expand_uuid(server, uuid)

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
    url = f'{server}/api/node/{uuid}/{instance}/maxlabel/{maxlabel}'
    r = session.post(url)
    r.raise_for_status()


@dvid_api_wrapper
def fetch_nextlabel(server, uuid, instance, *, session=None):
    url = f'{server}/api/node/{uuid}/{instance}/nextlabel'
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
    url = f'{server}/api/node/{uuid}/{instance}/nextlabel/{num_labels}'
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

    url = f'{server}/api/node/{uuid}/{instance}/supervoxels/{body_id}'
    r = session.get(url, params=query_params)
    r.raise_for_status()
    supervoxels = np.array(r.json(), np.uint64)
    supervoxels.sort()
    return supervoxels


def fetch_supervoxels_for_bodies(server, uuid, instance, bodies, *, threads=0, processes=0):
    """
    Fetch the supervoxels for all of the bodies in the given list,
    and return them as a DataFrame with columns ['sv','body']

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid instance name, e.g. 'segmentation'

        bodies:
            list of body IDs

        threads:
            If non-zero, fetch the results in parallel using a threadpool

        processes:
            If non-zero, fetch the results in parallel using a process pool

    Returns:
        DataFrame with columns ['sv', 'body']


    Notes:
        - The order of the bodies in the result will not
          necessarily be the same as the order of the input list.
        - If an error is encountered when fetching the supervoxels for a body
          (e.g. if the body doesn't exist at the given node), it will be
          omitted from the results and an error will be logged.
        - The input list is de-duplicated before fetching the results.
    """
    bodies = pd.unique(bodies).copy() # Apparently this copy is needed or else we get a segfault

    _fetch = partial(_fetch_supervoxels_for_body, server, uuid, instance)

    if threads == 0 and processes == 0:
        bodies_and_svs = [*map(_fetch, tqdm_proxy(bodies, leave=False))]
    else:
        bodies_and_svs = compute_parallel(_fetch, bodies, threads=threads, processes=processes, ordered=False)

    bodies = []
    all_svs = []
    bad_bodies = []
    for body, svs in bodies_and_svs:
        if svs is None:
            bad_bodies.append(body)
            continue
        bodies.append(np.array([body]*len(svs)))
        all_svs.append(svs)

    if len(bodies) > 0:
        bodies = np.concatenate(bodies)
        all_svs = np.concatenate(all_svs)

    if bad_bodies:
        if len(bad_bodies) < 100:
            msg = f"Could not obtain supervoxel list for {len(bad_bodies)} bodies: {bad_bodies}"
        else:
            msg = f"Could not obtain supervoxel list for {len(bad_bodies)} labels."
        logger.error(msg)

    return pd.DataFrame({"sv": all_svs, "body": bodies}, dtype=np.uint64)


def _fetch_supervoxels_for_body(server, uuid, instance, body):
    """
    Helper for fetch_supervoxels_for_bodies()
    """
    try:
        svs = fetch_supervoxels(server, uuid, instance, body)
        return (body, svs)
    except HTTPError as ex:
        if (ex.response is not None and ex.response.status_code == 404):
            return (body, None)
        raise


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
    url = f'{server}/api/node/{uuid}/{instance}/size/{label_id}?supervoxels={supervoxels}'
    response = fetch_generic_json(url, session=session)
    return response['voxels']

# FIXME: Deprecated name
fetch_body_size = fetch_size


@dvid_api_wrapper
def fetch_sizes(server, uuid, instance, label_ids, supervoxels=False, *, session=None, batch_size=1000, threads=0, processes=0):
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
    orig_label_ids = np.asarray(label_ids, np.uint64)
    if batch_size is None and (threads == 0 and processes == 0):
        return _fetch_sizes(server, uuid, instance, orig_label_ids, supervoxels, session)

    if batch_size is None:
        batch_size = 1000

    label_ids = pd.unique(orig_label_ids)
    batches = iter_batches(label_ids, batch_size)

    if len(label_ids) == 0:
        unordered_sizes = pd.Series([], name='size')
    elif (threads == 0 and processes == 0):
        unordered_sizes = []
        for batch in tqdm_proxy(batches, disable=(len(batches) == 1)):
            s = _fetch_sizes(server, uuid, instance, batch, supervoxels, session)
            unordered_sizes.append(s)
        unordered_sizes = pd.concat(unordered_sizes)
    else:
        fn = partial(_fetch_sizes, server, uuid, instance, supervoxels=supervoxels)
        unordered_sizes = compute_parallel(fn, batches, threads=threads, processes=processes, ordered=False)
        unordered_sizes = pd.concat(unordered_sizes)

    sizes = unordered_sizes.reindex(orig_label_ids)
    if supervoxels:
        sizes.index.name = 'sv'
    else:
        sizes.index.name = 'body'

    assert len(sizes) == len(orig_label_ids)
    return sizes


def _fetch_sizes(server, uuid, instance, label_ids, supervoxels, session=None):
    if len(label_ids) == 0:
        sizes = pd.Series([], name='size')
    else:
        sv_param = str(bool(supervoxels)).lower()
        url = f'{server}/api/node/{uuid}/{instance}/sizes?supervoxels={sv_param}'
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
def fetch_supervoxel_sizes_for_body(server, uuid, instance, body_id, *, session=None):
    """
    Return the sizes of all supervoxels in a body.

    Equivalent to calling fetch_supervoxels() followed by fetch_sizes(),
    except that this implementation fetching the labelindex for the whole body
    and computes the sizes locally.  For a large body, this is much faster than
    reading the supervoxel sizes via fetch_sizes(..., supervoxels=True), because
    the labelindex will only be read once. (DVID would read it N times for
    N supervoxels.)

    Returns:
        pd.Series, indexed by supervoxel
    """
    from . import fetch_labelindex # late import to avoid recursive import
    li = fetch_labelindex(server, uuid, instance, body_id, format='pandas')
    return li.blocks.groupby('sv')['count'].sum().rename('size')


@dvid_api_wrapper
def fetch_listlabels(server, uuid, instance, start=None, number=1_000_000, sizes=False, *, session=None):
    """
    Fetch the set of all labels in the entire labelmap instance.
    (DVID obtains this list by scanning the label index keys.)

    Since list is too long for a single request, you must paginate.
    Use the start and count arguments to request the list in parts.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid labelmap instance name, e.g. 'segmentation'

        start:
            The start of the range query.
            Not required to be a label that actually exists in the database.
            By default, DVID uses 0.

        number:
            How many labels to fetch, starting with the given ``start``
            value (or the first valid label above that value).

        sizes:
            If True, also fetch the sizes of the labels.
            (In that case a Series is returned.)

    Returns:
        If sizes=True, returns pd.Series, indexed by body ID.
        Otherwise, returns np.ndarray of body IDs.
    """
    params = {}
    if start is not None:
        params['start'] = int(start)
    if number:
        params['number'] = int(number)
    if sizes:
        params['sizes'] = 'true'

    r = session.get(f"{server}/api/node/{uuid}/{instance}/listlabels", params=params)
    r.raise_for_status()

    if sizes:
        labels_sizes = np.frombuffer(r.content, np.uint64).reshape(-1, 2)
        sizes = pd.Series(labels_sizes[:, 1], index=labels_sizes[:, 0], name='size')
        sizes.index.name = 'body'
        return sizes
    else:
        labels = np.frombuffer(r.content, np.uint64)
        return labels


@dvid_api_wrapper
def fetch_listlabels_all(server, uuid, instance, sizes=False, *, batch_size=100_000, session=None):
    """
    Convenience function for calling fetch_listlabels() repeatedly
    to obtain the complete list of all labels in the instance.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid labelmap instance name, e.g. 'segmentation'

        sizes:
            If True, also fetch the sizes of the labels.
            (In that case a Series is returned.)

        batch_size:
            The labels will be fetched via multiple requests.
            This specifies the size of each request.

    Returns:
        If sizes=True, returns pd.Series, indexed by body ID.
        Otherwise, returns np.ndarray of body IDs.
    """
    all_bodies = []
    start = 0

    progress = tqdm_proxy()
    progress.update(0)

    while True:
        b = fetch_listlabels(server, uuid, instance, start, batch_size, sizes, session=session)
        if len(b) == 0:
            break
        all_bodies.append(b)
        progress.update(len(b))
        start = b[-1] + 1

    if sizes:
        return pd.concat(all_bodies)
    else:
        return np.concatenate(all_bodies)


@dvid_api_wrapper
def compute_roi_distributions(server, uuid, labelmap_instance, label_ids, rois, *, session=None, batch_size=None, processes=1):
    """
    For a list of bodies and a list of ROIs, determine the voxel
    distribution of each body within all of the ROIs.
    The distribution is calculated using the label indexes,
    rather than the segmentation.

    Note:
        Although ROIs are specified at scale-5 resolution (32px),
        Label index coordinates are specified at scale-6 (64px).
        Therefore, this function is not 100% precise in cases where
        a neuron crosses from one ROI into another.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        labelmap_instance:
            dvid instance name, e.g. 'segmentation'

        label_ids:
            List of body IDs to process.

        rois:
            List of ROIs to include in the results.
            If the ROIs overlap, the results are undefined.

        batch_size:
            Labels will be processed in batches, rather than all at once.
            This specifies the batch size.

        processes:
            Label indexes will be fetched in parallel, in a process pool.
            This is the size of the process pool.

    Returns:
        pd.DataFrame
    """
    from ..roi import fetch_combined_roi_volume
    from . import fetch_labelindices  # late import to avoid recursive import

    assert processes, "Must use at least one process."

    label_ids = np.asarray(label_ids)
    rois = sorted(rois)

    logger.info(f"Fetching ROI segmentation for {len(rois)} ROIs")
    combined_vol, combined_box, overlaps = fetch_combined_roi_volume(server, uuid, rois, session=session)

    batch_size = batch_size or len(label_ids)
    batches = iter_batches(label_ids, batch_size)

    _fetch_indices = partial(fetch_labelindices, server, uuid, labelmap_instance, format='single-dataframe')

    results = []
    logger.info(f"Processing {len(label_ids)} labels in {len(batches)} batches")
    for batch in tqdm_proxy(batches):
        #minibatch_size = max(1, len(batch) // processes // 2)
        minibatch_size = 2
        minibatches = iter_batches(batch, minibatch_size)
        dfs = compute_parallel(_fetch_indices, minibatches, processes=processes, ordered=False)
        index_df = pd.concat(dfs, ignore_index=True)

        # Offset by 32 to move coordinate to the center of the 64px block
        index_df[[*'zyx']] += (2**5)

        # extract_labels_from_volume() uses the name 'label' for its own
        # purposes, so rename it 'body' before calling.
        index_df.rename(columns={'label': 'body'}, inplace=True)

        extract_labels_from_volume(index_df, combined_vol, combined_box, 5, rois)
        index_df.rename(inplace=True, columns={'label': 'roi_label', 'label_name': 'roi'})

        roi_dist_df = (index_df[['body', 'roi_label', 'roi', 'count']]
                        .groupby(['body', 'roi_label'], sort=False)
                        .agg({'roi': 'first', 'count': 'sum'})
                        .reset_index())

        results.append(roi_dist_df)

    full_dist_df = pd.concat(results, ignore_index=True)
    full_dist_df = full_dist_df.rename(columns={'count': 'voxels'})
    full_dist_df = full_dist_df.sort_values(['body', 'voxels'], ascending=[True, False])
    full_dist_df['roi'] = full_dist_df['roi'].cat.rename_categories({'<unspecified>': '<none>'})
    return full_dist_df[['body', 'roi', 'voxels']]


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

    r = session.get(f'{server}/api/node/{uuid}/{instance}/label/{coord_str}', params=params)
    r.raise_for_status()
    return np.uint64(r.json()["Label"])

# Synonym. See wrapper_proxies.py
fetch_labelmap_label = fetch_label

# Old name (FIXME: remove)
fetch_label_for_coordinate = fetch_label


@dvid_api_wrapper
def fetch_labels(server, uuid, instance, coordinates_zyx, scale=0, supervoxels=False, *, session=None):
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
    # I changed the order of these two args, so let's verify
    # that no old code is sending them in the wrong order.
    assert isinstance(supervoxels, bool)
    assert np.issubdtype(type(scale), np.integer)

    coordinates_zyx = np.asarray(coordinates_zyx, np.int32)
    assert coordinates_zyx.ndim == 2 and coordinates_zyx.shape[1] == 3

    params = {}
    if supervoxels:
        params['supervoxels'] = str(bool(supervoxels)).lower()
    if scale != 0:
        params['scale'] = str(scale)

    coords_xyz = np.array(coordinates_zyx)[:, ::-1].tolist()
    r = session.get(f'{server}/api/node/{uuid}/{instance}/labels', json=coords_xyz, params=params)
    r.raise_for_status()

    labels = np.array(r.json(), np.uint64)
    return labels


def fetch_labels_batched(server, uuid, instance, coordinates_zyx, supervoxels=False, scale=0, batch_size=10_000, threads=0, processes=0, presort=True):
    """
    Like fetch_labels, but fetches in batches, optionally multithreaded or multiprocessed.

    See also: ``fetch_label()``, ``fectch_labels()``
    """
    assert not threads or not processes, "Choose either threads or processes (not both)"
    coordinates_zyx = np.asarray(coordinates_zyx)
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

    fetch_batch = partial(_fetch_labels_batch, server, uuid, instance, scale, supervoxels)

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


def _fetch_labels_batch(server, uuid, instance, scale, supervoxels, batch_df):
    """
    Helper for fetch_labels_batched(), defined at top-level so it can be pickled.
    """
    batch_coords = batch_df[['z', 'y', 'x']].values
    # don't pass session: We want a unique session per thread
    batch_df.loc[:, 'label'] = fetch_labels(server, uuid, instance, batch_coords, scale, supervoxels)
    return batch_df


@dvid_api_wrapper
def fetch_sparsevol_rles(server, uuid, instance, label, supervoxels=False, scale=0, *, session=None):
    """
    Fetch the sparsevol RLE representation for a given label.

    See also: neuclease.dvid.rle.parse_rle_response()
    """
    supervoxels = str(bool(supervoxels)).lower() # to lowercase string
    url = f'{server}/api/node/{uuid}/{instance}/sparsevol/{label}?supervoxels={supervoxels}&scale={scale}'
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
    url = f'{server}/api/node/{uuid}/{instance}/split-supervoxel/{supervoxel}'


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
def fetch_mapping(server, uuid, instance, supervoxel_ids, *, session=None, as_series=False):
    """
    For each of the given supervoxels, ask DVID what body they belong to.
    If the supervoxel no longer exists, it will map to label 0.

    Returns:
        If as_series=True, return pd.Series, with index named 'sv' and values named 'body'.
        Otherwise, return the bodies as an array, in the same order in which the supervoxels were given.
    """
    supervoxel_ids = list(map(int, supervoxel_ids))
    body_ids = fetch_generic_json(f'{server}/api/node/{uuid}/{instance}/mapping', json=supervoxel_ids, session=session)
    mapping = pd.Series(body_ids, index=np.asarray(supervoxel_ids, np.uint64), dtype=np.uint64, name='body')
    mapping.index.name = 'sv'

    if as_series:
        return mapping
    else:
        return mapping.values


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
        uri = f"{server}/api/node/{uuid}/{instance}/mappings?format=binary"
        with Timer(f"Fetching {uri}", logger):
            r = session.get(uri)
            r.raise_for_status()

        a = np.frombuffer(r.content, np.uint64).reshape(-1,2)
        if as_array:
            return a

        df = pd.DataFrame(a, columns=['sv', 'body'])

    else:
        # This takes ~30 seconds so it's nice to log it.
        uri = f"{server}/api/node/{uuid}/{instance}/mappings"
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
        r = session.post(f'{server}/api/node/{uuid}/{instance}/mappings', data=payload)
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
    response = fetch_generic_json(f'{server}/api/node/{uuid}/{instance}/lastmod/{body_id}', session=session)
    return response["mutation id"]


@dvid_api_wrapper
def _fetch_sparsevol_coarse_impl(server, uuid, instance, label_id, supervoxels=False, *,
                                 format='coords', mask_box=None, session=None, cache=False):
    """
    Return the 'coarse sparsevol' representation of a given body/supervoxel.
    This is similar to the sparsevol representation at scale=6,
    EXCEPT that it is generated from the label index, so no blocks
    are lost from downsampling.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid labelmap instance name, e.g. 'segmentation'

        label_id:
            body ID in dvid, or supervoxel ID is supervoxels=True

        supervoxels:
            Whether or not to interpret label_id as a body or supervoxel

        format:
            Either 'coords', 'ranges', or 'mask'.

            If 'coords, return the coordinates (at scale 6) of the
            dvid blocks intersected by the body.

            If 'ranges', return the decoded ranges from DVID.
            See ``runlength_decode_from_ranges()`` for details on the ranges format.

            If 'mask', return a dense boolean volume (scale 6) indicating
            which blocks the body intersects, and return the volume's box.

        mask_box:
            Only valid when format='mask'.
            If provided, specifies the box within which the body mask should
            be returned, in scale-6 coordinates.
            Voxels outside the box will be omitted from the returned mask.

        cache:
            If True, the results will be stored in a ``lru_cache``.

    Returns:

        If format='coords', return an array of coordinates of the form:

            [[Z,Y,X],
             [Z,Y,X],
             [Z,Y,X],
             ...
            ]

        If 'mask':
            (mask, mask_box)
            Return a binary mask of the blocks intersected by the body,
            where each voxel represents one ROI block (scale 6).
            Unless you provide a custom ``mask_box``, the mask will be
            cropped to the bounding box of the body. The mask's bounding box
            is also returned. (If you passed in a custom ``mask_box``, it
            will be unchanged.)

    See also: ``fetch_sparsevol_coarse_via_labelindex()``

    Note: The returned coordinates are not necessarily sorted.
    """
    assert format in ('coords', 'ranges', 'mask')

    # The cache arg is only listed on this function for the sake of the docstring.
    assert not cache, \
        "Don't call this function directly.  Call fetch_sparsevol_coarse(), "\
        "which wraps this function and implements caching logic."

    supervoxels = str(bool(supervoxels)).lower()
    r = session.get(f'{server}/api/node/{uuid}/{instance}/sparsevol-coarse/{label_id}?supervoxels={supervoxels}')
    r.raise_for_status()

    if format == 'coords':
        return parse_rle_response( r.content, format='coords' )

    rle_ranges = parse_rle_response( r.content, format='ranges' )
    if format == 'ranges':
        return rle_ranges

    if format == 'mask':
        mask, mask_box = runlength_decode_from_ranges_to_mask(rle_ranges, mask_box)
        return mask, mask_box


@lru_cache(maxsize=1000)
def _fetch_sparsevol_coarse_cached(*args, **kwargs):
    result = _fetch_sparsevol_coarse_impl(*args, **kwargs)

    # Since we're caching the result,
    # make sure it can't be overwritten by the caller!
    if isinstance(result, np.ndarray):
        result.flags['WRITEABLE'] = False
    else:
        for item in result:
            item.flags['WRITEABLE'] = False

    return result


@wraps(_fetch_sparsevol_coarse_impl)
def fetch_sparsevol_coarse(*args, mask_box=None, cache=False, **kwargs):
    if not cache:
        return _fetch_sparsevol_coarse_impl(*args, mask_box=mask_box, **kwargs)

    # Convert to mask_box to tuple so it can be hashed for the cache
    if mask_box is not None:
        ((z0, y0, x0), (z1, y1, x1)) = mask_box
        mask_box = ((z0, y0, x0), (z1, y1, x1))

    return _fetch_sparsevol_coarse_cached(*args, mask_box=mask_box, **kwargs)


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
def fetch_sparsevol(server, uuid, instance, label, supervoxels=False, scale=0,
                    *, format='coords', dtype=np.int32, mask_box=None, session=None): #@ReservedAssignment
    """
    Return coordinates of all voxels in the given body/supervoxel at the given scale.

    For dtype arg, see parse_rle_response()

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid labelmap instance name, e.g. 'segmentation'

        label:
            body ID in dvid, or supervoxel ID is supervoxels=True

        supervoxels:
            Whether or not to interpret label_id as a body or supervoxel

        scale:
            The scale at which the sparsevol should be returned.
            Note that at higher scales, some bodies may appear discontiguous
            since the downsampled segmentation is used to construct the sparsevol.

        dtype:
            (Not used when format='mask'.)
            The dtype of the returned coords/ranges/rles.
            If you know the sparsevol coordinates will not exceed 65535,
            you can set this to np.int16 to save RAM.

        format:
            Either 'rle', 'ranges', 'coords', or 'mask'.
            See return value details below.

        mask_box:
            Only valid when format='mask'.
            If provided, specifies the box within which the body mask should
            be returned, in scale-N coordinates where N is the scale you're requesting.
            Voxels outside the box will be omitted from the returned mask.

    Return:
        If format == 'rle', returns the RLE start coordinates and RLE lengths as two arrays:

            (start_coords, lengths)

            where start_coords is in the form:

                [[Z,Y,X], [Z,Y,X], ...]

            and lengths is a 1-D array:

                [length, length, ...]

        If format == 'ranges':
            Return the RLEs as ranges, in the form:

                [[Z,Y,X0,X1], [Z,Y,X0,X1], ...]

            Note: By DVID conventions, the interval [X0,X1] is inclusive,
                  i.e. X1 is IN the range -- not one beyond the range,
                  which would normally be the Python convention.

        If format == 'coords', returns an array of coordinates of the form:

            [[Z,Y,X],
             [Z,Y,X],
             [Z,Y,X],
             ...
            ]

            Note: At scale 0, this will be a LOT of data for any reasonably large body.
                  Use with caution.

        If format == 'mask':
            (mask, mask_box)
            Return a binary mask (at the requested scale).
            Unless you provide a custom ``mask_box``, the mask will be
            cropped to the bounding box of the body. The mask's bounding box
            is also returned. (If you passed in a custom ``mask_box``, it
            will be unchanged.)
    """
    assert format in ('coords', 'rle', 'ranges', 'mask')

    rles = fetch_sparsevol_rles(server, uuid, instance, label, supervoxels, scale, session=session)

    if format in ('coords', 'rle', 'ranges'):
        return parse_rle_response(rles, dtype, format)

    if format == 'mask':
        rle_ranges = parse_rle_response( rles, format='ranges' )
        mask, mask_box = runlength_decode_from_ranges_to_mask(rle_ranges, mask_box)
        return mask, mask_box


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
    SCALE = 6  # sparsevol-coarse is always scale 6
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

    return middle_block_coord + nonzero_coords[len(nonzero_coords)//2]


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

    r = session.get(f'{server}/api/node/{uuid}/{instance}/blocks/{shape_str}/{offset_str}', params=params)
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


@dvid_api_wrapper
def fetch_labelmap_specificblocks(server, uuid, instance, corners_zyx, scale=0, supervoxels=False, format='array',
                                  *, map_on_client=False, threads=0, session=None):
    """
    Fetch a set of blocks from a labelmap instance.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid instance name, e.g. 'segmentation'

        corners_zyx:
            List of blocks to fetch, specified via their starting corners,
            in units corresponding to the given scale.

        scale:
            Which downsampling scale to fetch from

        supervoxels:
            If True, request supervoxel data from the given labelmap instance.

        format:
            One of the following:
                ('array', 'lazy-array', 'raw-response', 'blocks', 'lazy-blocks', 'raw-blocks')

            If 'array', inflate the compressed voxels from DVID into a single combined array, and return it.
            (Omitted or empty blocks will be filled with zeros in the result.)
            If 'lazy-array', return a callable proxy that stores the compressed data internally,
            and that will inflate the data into a single combined array when called.
            If 'raw-response', return DVID's raw /blocks response buffer without inflating it.
            If 'blocks', return a dict of {corner: block}
            If 'lazy-blocks': return a callable proxy that stores the compressed data internally,
            and that will inflate the data into a blocks dict when called.
            If 'raw-blocks', return a dict {corner: compressed_block}, in which each compressed
            block has not been inflated. Each block can be inflated with
            ``libdvid.DVIDNodeService.inflate_labelarray_blocks3D_from_raw(buf, (64,64,64), corner)``
            If 'callable-blocks`, return a dict {corner: callable}, where the callable returns the
            inflated block when called.  (This is the same as 'raw-blocks', but spares you some typing
            when you want to inflate the block.)

        map_on_client:
            If True, request supervoxels from dvid, then fetch the mapping for those supervoxels
            separately, and use that mapping to transform the volume into body IDs locally.
            This spares DVID the CPU burden of decompressing the labelblocks and applying the mapping
            mapping, and recompressing the blocks before sending them.
            Using this option with threads=0 will generally be slower than normal, for multiple reasons:
                - It incurs the extra latency of fetching the mapping from DVID
                - On the server, DVID uses multithreading to perform the remapping
            But if you set threads=8 or 16, you can acheive comparable or (barely) faster results than
            letting DVID do the remapping, despite the extra call to dvid.
            The main scenario in which this option is useful is when you are hitting DVID from several
            threads, and thus DVID's CPU is the bottleneck for your workload.

        threads:
            How many threads to use to inflate (and remap) blocks in parallel.
            This only has a modest effect on performance (e.g. ~20%), unless using
            map_on_client=True, in which case the improvement is non-trivial (~50%).

    Returns:
        See ``format`` argument.
    """
    assert format in ('array', 'lazy-array', 'raw-response', 'blocks', 'lazy-blocks', 'raw-blocks', 'callable-blocks')
    corners_zyx = np.asarray(corners_zyx)
    assert corners_zyx.ndim == 2
    assert corners_zyx.shape[1] == 3
    assert not map_on_client or not supervoxels, \
        "If you're fetching supervoxels, there's no mapping "\
        "necessary, on the client or otherwise."
    assert not map_on_client or (format not in ('raw-blocks', 'callable-blocks')), \
        "The map_on_client feature is not supported for the raw-blocks and callable-blocks formats."

    assert not (corners_zyx % 64).any(), "corners_zyx must be block-aligned!"

    block_ids = corners_zyx[:, ::-1] // 64
    params = {
        'blocks': ','.join(map(str, block_ids.reshape(-1)))
    }
    if scale:
        params['scale'] = scale
    if supervoxels or map_on_client:
        params['supervoxels'] = 'true'

    url = f"{server}/api/node/{uuid}/{instance}/specificblocks"
    r = session.get(url, params=params)
    r.raise_for_status()

    # From the DVID docs, regarding the block stream format:
    #   int32  Block 1 coordinate X (Note that this may not be starting block coordinate if it is unset.)
    #   int32  Block 1 coordinate Y
    #   int32  Block 1 coordinate Z
    #   int32  # bytes for first block (N1)
    #   byte0  Bytes of block data in compressed format.
    #   byte1
    #   ...
    #   byteN1

    if format == 'raw-response':
        return r.content

    min_corner = corners_zyx.min(axis=0)
    max_corner = corners_zyx.max(axis=0) + 64
    full_shape = max_corner - min_corner

    blocks = {}
    start = 0
    while start < len(r.content):
        header = np.frombuffer(r.content[start:start+16], dtype=np.int32)
        bx, by, bz, nbytes = header
        blocks[(64*bz, 64*by, 64*bx)] = r.content[start:start+16+nbytes]
        start += 16+nbytes

    def map_blocks_inplace(block_vols):
        svs = []
        for vol in block_vols:
            svs.append(pd.unique(vol.ravel()))
        svs = pd.unique(np.concatenate(svs))
        bodies = fetch_mapping(server, uuid, instance, svs)
        assert svs.dtype == bodies.dtype == np.uint64
        mapper = LabelMapper(svs, bodies)
        for vol in block_vols:
            mapper.apply_inplace(vol)

    def inflate_blocks(threads=threads):
        if threads == 0:
            inflated_blocks = [*starmap(_inflate_block, blocks.items())]
        else:
            inflated_blocks = compute_parallel(
                _inflate_block, blocks.items(), starmap=True, threads=threads, show_progress=False)

        if map_on_client:
            _corners, block_vols = zip(*inflated_blocks)
            if threads == 0:
                map_blocks_inplace(block_vols)
            else:
                block_batches = iter_batches(block_vols, (len(block_vols)+threads-1) // threads)
                compute_parallel(map_blocks_inplace, block_batches, ordered=False, threads=threads, show_progress=False)

        if format in ('blocks', 'lazy-blocks'):
            return dict(inflated_blocks)
        elif format in ('array', 'lazy-array'):
            vol = np.zeros(full_shape, np.uint64)
            for corner, block in inflated_blocks:
                z,y,x = corner - min_corner
                vol[z:z+64, y:y+64, x:x+64] = block
            return vol

    if format == 'raw-blocks':
        return blocks
    elif format == 'lazy-blocks':
        return inflate_blocks
    elif format == 'blocks':
        return inflate_blocks()
    elif format == 'array':
        return inflate_blocks()
    elif format == 'lazy-array':
        return inflate_blocks
    elif format == 'callable-blocks':
        # Wrap each buffer in a callable that will inflate it
        for coord, buf in [*blocks.items()]:
            f = DVIDNodeService.inflate_labelarray_blocks3D_from_raw
            blocks[coord] = partial(f, buf, (64,64,64), coord)
        return blocks


def _inflate_block(corner, buf):
    block = DVIDNodeService.inflate_labelarray_blocks3D_from_raw(buf, (64,64,64), corner)
    return (corner, block)


def fetch_labelmap_voxels_chunkwise(server, uuid, instance, box_zyx, scale=0, throttle=False, supervoxels=False,
                                    *, chunk_shape=(64,64,4096), threads=0, format='array', out=None):
    """
    Same as fetch_labelmap_voxels, but internally fetches the volume in
    pieces to avoid fetching too much from DVID in one call.

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

        chunk_shape:
            The size of each chunk

        threads:
            If non-zero, fetch chunks in parallel

        format:
            If 'array', inflate the compressed voxels from DVID and return an ordinary ndarray
            If 'lazy-array', return a callable proxy that stores the compressed data internally,
            and that will inflate the data when called.

        out:
            If given, write results into the given array.
            Must have the correct shape for the given ``box_zyx``,
            but need not have a dtype of ``np.uint64`` if you happen to know the
            label IDs will not exceed the max value for the output dtype.

    Returns:
        ndarray, with shape == (box[1] - box[0])
    """
    assert format in ('array', 'lazy-array')
    if out is None:
        full_vol = np.zeros(box_shape(box_zyx), np.uint64)
    else:
        assert (out.shape == box_shape(box_zyx)).all()
        full_vol = out

    chunk_boxes = boxes_from_grid(box_zyx, chunk_shape, clipped=True)

    _fetch = partial(_fetch_chunk, server, uuid, instance, scale, throttle, supervoxels)

    logger.info("Fetching compressed chunks")
    if threads == 0:
        boxes_and_chunks = [*tqdm_proxy(map(_fetch, chunk_boxes), total=len(chunk_boxes))]
    else:
        boxes_and_chunks = compute_parallel(_fetch, chunk_boxes, ordered=False, threads=threads)

    def inflate_labelarray_chunks():
        logger.info("Inflating chunks")
        for block_box, lazy_chunk in tqdm_proxy(boxes_and_chunks):
            overwrite_subvol(full_vol, block_box - box_zyx[0], lazy_chunk())
        return full_vol

    if format == 'array':
        return inflate_labelarray_chunks()
    elif format == 'lazy-array':
        return inflate_labelarray_chunks
    return full_vol


def _fetch_chunk(server, uuid, instance, scale, throttle, supervoxels, box):
    """
    Helper for fetch_labelmap_voxels_chunkwise()
    """
    return box, fetch_labelmap_voxels(server, uuid, instance, box, scale, throttle, supervoxels, format='lazy-array')


def fetch_seg_around_point(server, uuid, instance, point_zyx, radius, scale=0, sparse_body=None, sparse_component_only=False,
                           *, session=None, cache_svc=True, map_on_client=False, threads=0):
    """
    Fetch the segmentation around a given point, possibly
    limited to the blocks that contain a particular body.

    Args:
        server:
            dvid server
        uuid:
            dvid uuid
        instance:
            labelmap instance
        point_zyx:
            The point around which the segmentation will be fetched,
            specified in units that correspond to the given scale.
        radius:
            The radius of the volume to be fetched around the point,
            specified in units that correspond to the given scale
        scale:
            The scale at which to fetch the data
        sparse_body:
            If provided, only fetch blocks which contain this body ID.
            The returned volume will be mostly empty (zeros),
            except for the blocks which intersect this body's coarse sparsevol blocks.
            Note: This is not the same as the body's sparsevol representation,
            becuase the other bodies in those blocks are not masked out.
        sparse_component_only:
            If True, this indicates that you are only interested in the connected component
            of the segmentation that overlaps with the given point, so blocks of segmentation
            that are clearly disconnected from that component can be omitted from the results,
            at least as can be seen from the sparsevol-coarse representation (scale-6).
            This doesn't guarantee that all returned blocks contain the sparse component,
            but it will omit the blocks that can be easily omitted, since they aren't connected
            to the component of interest at scale-6 resolution.
            It is assumed that the given point lies on the given sparse_body;
            if not, an error will be raised if sparse_body doesn't intersect the same block as the point.
        cache_svc:
            If True, make use of the lru_cache offered by fetch_sparsevol_coarse().
            That way, you can call this function repeatedly for different points
            without hitting DVID every time, or parsing the sparsevol-coarse
            response every time.
        map_on_client:
            Whether or not to use map_on_client when calling fetch_labelmap_specificblocks().
            See that function for details.
        threads:
            How many threads to use when calling fetch_labelmap_specificblocks().
            See that function for details.

    Returns:
        seg:
            A volume of segmentation at large enough to contain the point
            and the given radius in all directions, and then expanded to
            align with dvid's 64-px grid.
        box:
            The box ([z0, y0, x0], [z1, y1, x1]) of the returned segmentation subvolume
        p:
            Where the requested point lies in the output subvolume.
        block_coords:
            The block coordinates that are included in the result,
            in units corresponding to the requested scale.
    """
    assert not sparse_component_only or sparse_body, \
        "Can't use sparse_component_only without providing a sparse_body"

    p = np.array(point_zyx)
    R = radius

    box = [p-R, p+R+1]
    aligned_box = round_box(box, 64, 'out')

    if not sparse_body:
        corners = ndrange_array(*aligned_box, 64)
        seg = fetch_labelmap_voxels_chunkwise(server, uuid, instance, aligned_box, scale=scale)
    else:
        svc_mask, svc_box = fetch_sparsevol_coarse(server, uuid, instance, sparse_body, format='mask', session=session, cache=cache_svc)
        aligned_box = box_intersection((2**(6-scale))*svc_box, aligned_box)
        svc_mask = svc_mask[box_to_slicing(*(aligned_box//(2**(6-scale)) - svc_box[0]))]
        svc_box = aligned_box // (2**(6-scale))

        if sparse_component_only:
            svc_cc = labelMultiArrayWithBackground(svc_mask.view(np.uint8))
            p_cc = svc_cc[tuple(p // (2**(6-scale)) - svc_box[0])]
            if p_cc == 0:
                raise RuntimeError(f"The given point {p.tolist()} does not lie on the given sparse_body {sparse_body}")
            svc_mask = (svc_cc == p_cc)

        corners = (np.argwhere(svc_mask) + svc_box[0]) * (2**(6-scale))
        corners = pd.DataFrame(corners // 64 * 64).drop_duplicates().values

        # Update aligned box, if the blocks can fit within a smaller bounding-box
        aligned_box = np.array([corners.min(axis=0), 64+corners.max(axis=0)])
        seg = fetch_labelmap_specificblocks(server, uuid, instance, corners, scale=scale, format='array',
                                            map_on_client=map_on_client, threads=threads, session=session)

    p_out = np.array([R,R,R]) + (box[0] - aligned_box[0])
    assert (seg.shape == (aligned_box[1] - aligned_box[0])).all()
    return seg, aligned_box, p_out, corners


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
def post_labelmap_blocks(server, uuid, instance, corners_zyx, blocks, scale=0, downres=False, noindexing=False, throttle=False, *, is_raw=False, gzip_level=6, session=None, progress=False):
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
        body_data = encode_labelarray_blocks(corners_zyx, blocks, gzip_level, progress)

    if not body_data:
        return # No blocks

    # These options are already false by default, so we'll only include them if we have to.
    opts = { 'downres': downres, 'noindexing': noindexing, 'throttle': throttle }

    params = { 'scale': str(scale) }
    for opt, value in opts.items():
        if value:
            params[opt] = str(bool(value)).lower()

    r = session.post(f'{server}/api/node/{uuid}/{instance}/blocks', params=params, data=body_data)
    r.raise_for_status()

# Deprecated name
post_labelarray_blocks = post_labelmap_blocks


def encode_labelarray_blocks(corners_zyx, blocks, gzip_level=6, progress=False):
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
            Each block must be 64px wide.
            Will be converted to uint64 if necessary.

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
    for block in tqdm_proxy(blocks, disable=not progress):
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


def encode_labelarray_volume(offset_zyx, volume, gzip_level=6, omit_empty_blocks=False):
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

        omit_empty_blocks:
            If True, don't encode blocks that are completely zero-filled.
            Omit them from the output.

    See ``decode_labelarray_volume()`` for the corresponding decode function.
    """
    offset_zyx = np.asarray(offset_zyx)
    shape = np.array(volume.shape)

    assert (offset_zyx % 64 == 0).all(), "Data must be block-aligned"
    assert (shape % 64 == 0).all(), "Data must be block-aligned"

    corners_zyx = ndrange_array(offset_zyx, offset_zyx + volume.shape, (64,64,64))

    if omit_empty_blocks:
        old_corners = corners_zyx.copy()
        corners_zyx = []
        for corner in old_corners:
            vol_corner = corner - offset_zyx
            block = volume[box_to_slicing(vol_corner, vol_corner+64)]
            if block.any():
                corners_zyx.append(corner)

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

    r = session.post(f'{server}/api/node/{uuid}/{instance}/cleave/{body_id}', json=supervoxel_ids)
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

    dvid_mapping = fetch_mapping(server, uuid, instance, group_mapping.index.values, as_series=True, session=session)
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

    r = session.post(f'{server}/api/node/{uuid}/{instance}/merge', json=content)
    r.raise_for_status()


@dvid_api_wrapper
def fetch_mutations(server, uuid, instance, userid=None, *, action_filter=None, dag_filter='leaf-only', format='pandas', session=None):
    """
    Fetch the log of successfully completed mutations.
    The log is returned in the same format as the kafka log.

    For consistency with :py:func:``read_kafka_msgs()``, this function adds
    the ``dag_filter`` and ``action_filter`` options, which are not part
    of the DVID REST API.

    Args:
        server, uuid, instance:
            A labelmap instance for which a kafka log exists.

        userid:
            If given, limit the query to only include mutations
            which were performed by the given user.
            Note: This need not be the same as the current user
            calling this function.

        action_filter:
            A list of actions to use as a filter for the returned messages.
            For example, if action_filter=['split', 'split-supervoxel'],
            all messages with other actions will be filtered out.
            (This is not part of the DVID API.  It's implemented in this
            python function a post-processing step.)

        dag_filter:
            Specifies which UUIDs for which to fetch mutations,
            relative to the specified ``uuid``.

            One of:
            - 'leaf-only' (only messages whose uuid matches the one provided),
            - 'leaf-and-parents' (only messages matching the given uuid or its ancestors), or
            - None (no filtering by UUID).
            (This is not part of the DVID API.  It's implemented in this
            python function by calling the /mutations endpoint for multiple UUIDs.)

        format:
            How to return the data. Either 'pandas' or 'json'.

    Returns:
        Either a DataFrame or list of parsed json values, depending
        on what you passed as 'format'.
    """
    assert dag_filter in ('leaf-only', 'leaf-and-parents', None)

    # json-values is a synonym, for compatibility with read_kafka_messages
    assert format in ('pandas', 'json', 'json-values')

    uuid = resolve_ref(server, uuid)

    if userid:
        params = {'userid': userid}
    else:
        params = {}

    if dag_filter == 'leaf-only':
        uuids = [uuid]
    else:
        dag = fetch_repo_dag(server, uuid, session=session)
        if dag_filter == 'leaf-and-parents':
            uuids = {uuid} | nx.ancestors(dag, uuid)
        else:
            assert dag_filter is None
            uuids = dag.nodes()
        uuids = nx.topological_sort(dag.subgraph(uuids))

    msgs = []
    for uuid in uuids:
        r = session.get(f'{server}/api/node/{uuid}/{instance}/mutations', params=params)
        r.raise_for_status()
        msgs.extend(r.json())

    if isinstance(action_filter, str):
        action_filter = [action_filter]

    if action_filter is not None:
        action_filter = {*action_filter}
        msgs = [*filter(lambda m: m['Action'] in action_filter, msgs)]

    if format == 'pandas':
        return labelmap_kafka_msgs_to_df(msgs)
    else:
        return msgs


def read_labelmap_kafka_df(server, uuid, instance='segmentation', action_filter=None, dag_filter='leaf-and-parents',
                           default_timestamp=DEFAULT_TIMESTAMP, drop_completes=True, group_id=None):
    """
    Convenience function for reading the kafka log for
    a labelmap instance and loading it into a DataFrame.
    A convenience function that combines ``read_kafka_messages()``
    and ``labelmap_kafka_msgs_to_df``.

    See also:
        :py:func:`fetch_mutations`

    Args:
        server, uuid, instance:
            A labelmap instance for which a kafka log exists.

        action_filter:
            A list of actions to use as a filter for the returned messages.
            For example, if action_filter=['split', 'split-complete'],
            all messages with other actions will be filtered out.

        dag_filter:
            How to filter out messages based on the UUID.
            One of:
            - 'leaf-only' (only messages whose uuid matches the one provided),
            - 'leaf-and-parents' (only messages matching the given uuid or its ancestors), or
            - None (no filtering by UUID).

        default_timestamp:
            See labelmap_kafka_msgs_to_df()

        drop_completes:
            See labelmap_kafka_msgs_to_df()

    Returns:
        DataFrame with columns:
            ['timestamp', 'uuid', 'mutid', 'action', 'target_body', 'target_sv', 'msg']
    """
    msgs = read_kafka_messages(server, uuid, instance, action_filter, dag_filter, group_id=group_id)
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
    FINAL_COLUMNS = ['timestamp', 'uuid', 'mutid', 'action', 'target_body', 'target_sv', 'user', 'msg']
    df = kafka_msgs_to_df(kafka_msgs, drop_duplicates=False, default_timestamp=default_timestamp)

    if len(df) == 0:
        return pd.DataFrame([], columns=FINAL_COLUMNS)

    # Append action
    df['action'] = [msg['Action'] for msg in df['msg']]
    df['user'] = [msg['User'] if 'User' in msg else '' for msg in df['msg']]

    if drop_completes:
        completes = df['action'].map(lambda s: s.endswith('-complete'))
        df = df[~completes].copy()

    if len(df) == 0:
        return pd.DataFrame([], columns=FINAL_COLUMNS)

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

    for col in FINAL_COLUMNS:
        if col not in df:
            df[col] = np.nan

    return df[FINAL_COLUMNS]


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
