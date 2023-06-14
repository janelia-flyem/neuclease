import re
import gzip
import time
import logging
from io import BytesIO
from functools import partial, lru_cache, wraps, reduce
from itertools import starmap, chain
from operator import or_
from multiprocessing.pool import ThreadPool
from collections import namedtuple
from collections.abc import Collection

import numpy as np
import pandas as pd
import networkx as nx
from requests import HTTPError

from libdvid import DVIDNodeService, encode_label_block
from dvidutils import LabelMapper
from vigra.filters import multiBinaryErosion, distanceTransform
from vigra.analysis import labelMultiArrayWithBackground

from ...util import (Timer, round_box, extract_subvol, DEFAULT_TIMESTAMP, tqdm_proxy, find_root,
                     ndrange, ndrange_array, box_to_slicing, compute_parallel, boxes_from_grid, box_shape,
                     overwrite_subvol, iter_batches, extract_labels_from_volume, box_intersection, lexsort_columns,
                     toposorted_ancestors, distance_transform, thickest_point_in_mask, encode_coords_to_uint64,
                     sort_blockmajor)

from .. import dvid_api_wrapper, fetch_generic_json, fetch_repo_info
from ..server import fetch_server_info
from ..repo import create_voxel_instance, fetch_repo_dag, resolve_ref, expand_uuid, find_repo_root
from ..kafka import read_kafka_messages, kafka_msgs_to_df
from ..rle import parse_rle_response, runlength_decode_from_ranges_to_mask, rle_ranges_box, construct_rle_payload_from_ranges, split_ranges_for_grid

from ._split import fetch_supervoxel_splits_from_dvid

# $ protoc --python_out=. neuclease/dvid/labelmap/labelops.proto
from .labelops_pb2 import MappingOps, MappingOp


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
    type_specific_settings = {
        "IndexedLabels": str(enable_index).lower(),
        "CountLabels": str(enable_index).lower(),
        "MaxDownresLevel": str(max_scale)
    }
    create_voxel_instance(server, uuid, instance, 'labelmap', versioned,
                          tags=tags, block_size=block_size, voxel_size=voxel_size,
                          voxel_units=voxel_units, type_specific_settings=type_specific_settings,
                          session=session)


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
            except HTTPError:
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
def fetch_lastmod(server, uuid, instance, body, *, session=None):
    """
    Returns last modification metadata for a label in JSON.

    Time is returned in RFC3339 string format. Returns a status code 404 (Not Found)
    if label does not exist.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid instance name, e.g. 'segmentation'

        body:
            Body ID

    Returns:
        dict

    Example response:

    .. code-block:: json

        {
            "mutation id": 2314,
            "last mod user": "johndoe",
            "last mod time": "2000-02-01 12:13:14 +0000 UTC", "last mod app": "Neu3"
        }

    """
    url = f'{server}/api/node/{uuid}/{instance}/lastmod/{body}'
    r = session.get(url)
    r.raise_for_status()
    return r.json()


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
    bodies = pd.unique(bodies).copy()  # Apparently this copy is needed or else we get a segfault

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


def fetch_supervoxel_counts(server, uuid, instance, bodies, *, processes=0):
    fn = partial(_fetch_svcount, server, uuid, instance)
    if not processes:
        counts = [*map(fn, tqdm_proxy(bodies))]
    else:
        counts = compute_parallel(fn, bodies, ordered=False, processes=32)
    counts = pd.DataFrame(counts, columns=['body', 'svcount'])
    counts = counts.set_index('body')['svcount']
    return counts.reindex(bodies)


def _fetch_svcount(server, uuid, instance, body):
    return (body, len(fetch_supervoxels(server, uuid, instance, body)))


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
    from . import fetch_labelindex  # late import to avoid recursive import
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
def fetch_listlabels_all(server, uuid, instance, sizes=False, *, start=0, stop=None, batch_size=100_000, session=None):
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
    start = start or 0

    progress = tqdm_proxy()
    progress.update(0)

    try:
        while True:
            b = fetch_listlabels(server, uuid, instance, start, batch_size, sizes, session=session)
            if len(b) == 0:
                break

            all_bodies.append(b)
            progress.update(len(b))

            if isinstance(b, pd.Series):
                start = b.index[-1] + 1
            else:
                start = b[-1] + 1

            if stop is not None and start > stop:
                break
    except KeyboardInterrupt:
        logger.warning("Interrupted.  Returning partial results.")

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

        roi_dist_df = (
            index_df[['body', 'roi_label', 'roi', 'count']]
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
        ``fetch_label()``, ``fetch_labels_batched()``
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


def fetch_labels_batched(server, uuid, instance, coordinates_zyx, supervoxels=False, scale=0, batch_size=10_000,
                         threads=0, processes=0, presort=True, progress=True):
    """
    Like fetch_labels(), but fetches in batches, optionally multithreaded or multiprocessed.

    See also: ``fetch_label()``, ``fectch_labels()``

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

        batch_size:
            How many points to query in each request.
            For best performance, keep this at 10k or lower.

        threads:
            If non-zero, use a thread pool to process batches in parallel.

        processes:
            If non-zero, use a process pool to process batces in parallel.

        presort:
            If True, pre-sort the coordinates in block-sorted order
            (i.e. binned into 64px blocks, and then sorted in ZYX order by block index).
            That way, the requests will be better aligned to the native block ordering in
            DVID's database, which helps throughput.
            Does not affect the output result order, which always corresponds to the input data order.
            DVID natively sorts within each request by block before accessing the data,
            but presorting is essential when spliting a large list of coordinates across many batches,
            especially when many coordinates may share blocks.

    Returns:
        ndarray of N labels (corresponding to the order you passed in)
    """
    assert not threads or not processes, "Choose either threads or processes (not both)"
    coordinates_zyx = np.asarray(coordinates_zyx)
    coords_df = pd.DataFrame(coordinates_zyx, columns=[*'zyx'], dtype=np.int32)
    coords_df['label'] = np.uint64(0)

    if presort:
        with Timer(f"Pre-sorting {len(coords_df)} coordinates by block index", logger):
            # Sort coordinates by their block index,
            # so each block usually won't appear in multiple batches,
            # which would incur repeated reads of the same block.
            # Here we preserve the index, which will be used to
            # restore the caller's original point order.
            sort_blockmajor(coords_df, inplace=True)

    with Timer("Fetching labels from DVID", logger):
        fetch_batch = partial(_fetch_labels_batch, server, uuid, instance, scale, supervoxels)
        batches = iter_batches(coords_df, batch_size)
        batch_result_dfs = compute_parallel(
            fetch_batch, batches, 1, threads, processes, ordered=False,
            leave_progress=False, show_progress=progress)

    # Restore the caller's original point order.
    return pd.concat(batch_result_dfs).sort_index()['label'].values


def _fetch_labels_batch(server, uuid, instance, scale, supervoxels, batch_df):
    """
    Helper for fetch_labels_batched(), defined at top-level so it can be pickled.
    """
    batch_df = batch_df.copy()
    batch_coords = batch_df[['z', 'y', 'x']].values
    # don't pass session: We want a unique session per thread
    batch_df.loc[:, 'label'] = fetch_labels(server, uuid, instance, batch_coords, scale, supervoxels)
    return batch_df


def fetch_bodies_for_many_points(server, uuid, seg_instance, point_df, mutations=None, mapping=None, batch_size=10_000, threads=0, processes=0):
    """
    Fetch the supervoxel and body for each point in a massive list of points.
    The results are written onto the input DataFrame, IN-PLACE, as columns for 'sv' and 'body'.

    If the input dataframe contains a 'sv' column, then this function saves time by only updating
    the rows for which the supervoxel *could* have changed since your input data was generated.

    Note:
        For short lists of points, it's faster to just use fetch_labels() or fetch_labels_batched().
        This function incurs significant overhead by fetching the complete in-memory mapping and
        also the complete mutation log.
        That takes a minute or two, but then constructing the 'body' column is comparatively fast.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        seg_instance:
            dvid labelmap instance name, e.g. 'segmentation'

        point_df:
            A DataFrame with at least columns for ['x', 'y', 'z']. See fetch_synapses_in_batches().
            Ideally, it also contains a pre-fetched 'sv' column, in which case most points
            can be processed quickly, without asking DVID to determine the underlying
            supervoxel again.

        mutations:
            Optional.  A cached copy of the complete mutation log for the instance,
            as fetched via fetch_mutations().
            If provided, it must correspond to the mutation log of the given UUID.

        mapping:
            Optional. A cached copy of the labelmap instance's in-memory map,
            as fetched via fetch_mappings().

        batch_size:
            Coordinates will be processed in batches.
            This specifies the number of points in each batch.

        threads:
            If non-zero, then multiple threads will be used to fetch labels in parallel.
            Can't be used with processes.  Pick one or the other.

        processes:
            If non-zero, then multiple processes will be used to fetch labels in parallel.
            Can't be used with threads. Pick one or the other.

    Returns:
        None.  Results are written IN-PLACE as columns 'sv' and 'body'.
        To speed up this function the next time you call it, cache the results
        somewhere and use them as the input in your next call.
    """
    assert not threads or not processes, "Choose either threads or processes (not both)"
    dvid_seg = (server, uuid, seg_instance)

    if 'sv' not in point_df.columns:
        # Fetch supervoxel under all coordinates
        coords = point_df[[*'zyx']].values
        point_df['sv'] = fetch_labels_batched(*dvid_seg, coords, supervoxels=True,
                                              batch_size=batch_size, threads=threads, processes=processes)
    else:
        # Only update the supervoxel IDs for coordinates which reside on a supervoxel which
        # was somehow involved in a split (as either parent or child).
        # For those points, there's no guarantee that the supervoxel in the given table
        # is in-sync with the given UUID, so we must check.
        if mutations is None:
            # We fetch the  mutations for the entire DAG (even other branches).
            # That way, we don't care at all which UUID was used to produce the
            # cached 'sv' column of the input table.
            mutations = fetch_mutations(*dvid_seg, dag_filter=None)

        new, changed, deleted, new_svs, deleted_svs = compute_affected_bodies(mutations)
        out_of_date = point_df.query('sv in @deleted_svs or sv in @new_svs').index
        if len(out_of_date) > 0:
            ood_coords = point_df.loc[out_of_date, [*'zyx']].values
            point_df.loc[out_of_date, 'sv'] = fetch_labels_batched(*dvid_seg, ood_coords, supervoxels=True,
                                                                   batch_size=5_000, threads=threads, processes=processes)
    #
    # Now map from sv -> body
    #
    if mapping is None:
        mapping = fetch_mappings(server, uuid, seg_instance, as_array=True)
    elif len(mapping) == 0:
        point_df['body'] = point_df['sv']
        return
    elif isinstance(mapping, pd.Series):
        # Convert to ndarray
        assert mapping.index.name == 'sv'
        assert mapping.name == 'body'
        mapping = mapping.reset_index().values

    assert mapping.shape[1] == 2
    assert mapping.dtype == np.uint64

    max_sv = max(point_df['sv'].max(), mapping[:, 0].max())
    if max_sv < 6*len(mapping):
        # Flat LUT is faster than a hash table,
        # as long as it doesn't take too much RAM.
        lut = np.arange(max_sv+np.uint64(1), dtype=np.uint64)
        lut[mapping[:, 0]] = mapping[:, 1]
        point_df['body'] = lut[point_df['sv'].values]
    else:
        mapper = LabelMapper(*mapping.T)
        point_df['body'] = mapper.apply(point_df['sv'].values, True)


@dvid_api_wrapper
def fetch_sparsevol_rles(server, uuid, instance, label, scale=0, supervoxels=False, *, box_zyx=None, session=None):
    """
    Fetch the sparsevol RLE representation for a given label.

    Args:
        server, uuid, instance:
            segmentation instance
        label:
            body ID or supervoxel ID
        scale:
            Which scale of segmentation blocks DVID should use when extracting the sparsvol mask
        supervoxels:
            If True, interpret 'label' as a supervoxel ID, not a body ID
        box_zyx:
            If provided, request only a portion of the given sparsevol,
            limited to the specified bounding box.
            This DVID feature was only implemented on 2023-04-09.
            If using a DVID that was built prior to that, using this
            parameter will result in invalid results.

    See also: neuclease.dvid.rle.parse_rle_response()
    """
    params = {
        'scale': str(scale)
    }

    if supervoxels:
        params['supervoxels'] = str(bool(supervoxels)).lower()

    if box_zyx is not None:
        box_zyx = box_zyx - [(0,0,0), (1,1,1)]
        box_keys = ['minz', 'miny', 'minx', 'maxz', 'maxy', 'maxx']
        params.update(dict(zip(box_keys, box_zyx.flat)))

    url = f'{server}/api/node/{uuid}/{instance}/sparsevol/{label}'
    r = session.get(url, params=params)
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

    Note:
        DVID will not return until the mapping and label index have been updated,
        but DVID may still be updating the voxel data (and multires pyramids).
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
def fetch_mapping(server, uuid, instance, supervoxel_ids, *, session=None, nolookup=False, batch_size=None, threads=0, processes=0, as_series=False):
    """
    For each of the given supervoxels, ask DVID what body they belong to.
    If the supervoxel no longer exists, it will map to label 0.

    Returns:
        If as_series=True, return pd.Series, with index named 'sv' and values named 'body'.
        Otherwise, return the bodies as an array, in the same order in which the supervoxels were given.
    """
    batch_size = batch_size or len(supervoxel_ids)
    if processes > 0:
        # Don't pass the session to child processes.
        session = None

    fn = partial(_fetch_mapping, server, uuid, instance, nolookup=nolookup, session=session)
    sv_batches = iter_batches(supervoxel_ids, batch_size)
    batch_results = compute_parallel(fn, sv_batches, threads=threads, processes=processes, show_progress=batch_size < len(supervoxel_ids))
    mapping = pd.concat(batch_results)

    if as_series:
        return mapping
    else:
        return mapping.values

@dvid_api_wrapper
def _fetch_mapping(server, uuid, instance, supervoxel_ids, *, nolookup=False, session=None):
    supervoxel_ids = list(map(int, supervoxel_ids))
    params = {}
    if nolookup:
        params['nolookup'] = 'true'

    url = f'{server}/api/node/{uuid}/{instance}/mapping'
    r = session.get(url, json=supervoxel_ids, params=params)
    r.raise_for_status()
    body_ids = r.json()

    mapping = pd.Series(body_ids, index=np.asarray(supervoxel_ids, np.uint64), dtype=np.uint64, name='body')
    mapping.index.name = 'sv'
    return mapping

@dvid_api_wrapper
def fetch_mappings(server, uuid, instance, as_array=False, *, format=None, consistent=False, session=None):
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

        consistent:
            If True, DVID will lock the labelmap instance to prevent
            mutations while the mappings request is being served.
            Otherwise, DVID will permit concurrent writes to the labelmap instance while this
            request is being served, meaning that you may possibly get an inconsistent view of
            the data if the requested UUID is not currently committed (locked).

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
            format = 'binary'
        else:
            format = 'csv'

    params = {}
    if format == 'binary':
        params['format'] = 'binary'
    if consistent:
        params['consistent'] = 'true'

    # This takes ~30 seconds so it's nice to log it.
    uri = f"{server}/api/node/{uuid}/{instance}/mappings"
    with Timer(f"Fetching {uri}", logger):
        r = session.get(uri, params=params)
        r.raise_for_status()

    if format == 'binary':
        a = np.frombuffer(r.content, np.uint64).reshape(-1,2)
        if as_array:
            return a
        df = pd.DataFrame(a, columns=['sv', 'body'])
    else:
        with Timer("Parsing mapping", logger), BytesIO(r.content) as f:
            df = pd.read_csv(f, sep=' ', header=None, names=['sv', 'body'], engine='c', dtype=np.uint64)
            if as_array:
                return df.values

    df.set_index('sv', inplace=True)

    assert df.index.dtype == np.uint64
    assert df['body'].dtype == np.uint64
    return df['body']


@dvid_api_wrapper
def fetch_complete_mappings(server, uuid, instance, mutations=None, sort=None, *, session=None):
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

        mutations:
            Optionally provide the complete labelmap mutation log if you've got it,
            in which case this function doesn't need to re-fetch it.
            Should be the messages obtained via fetch_mutations().

        sort:
            Optional.
            If 'sv', sort by supervoxel column.
            If 'body', sort by body. Otherwise, don't sort.

    Returns:
        pd.Series(index=sv, data=body)
    """
    assert sort in (None, 'sv', 'body')

    if mutations is None:
        mutations = fetch_mutations(server, uuid, instance)

    if not isinstance(mutations, pd.DataFrame):
        mutations = labelmap_kafka_msgs_to_df(mutations)
    assert 'msg' in mutations.columns

    # Fetch base mapping
    base_mapping = fetch_mappings(server, uuid, instance, session=session)

    cleave_labels = [m['CleavedLabel'] for m in mutations.query('action == "cleave-complete"')['msg']]
    cleave_labels = np.asarray(cleave_labels, np.uint64)
    split_df = fetch_supervoxel_splits_from_dvid(server, uuid, instance, format='pandas')
    renumber_targets = mutations.query('action == "renumber-complete"')['target_body'].values.astype(np.uint64)

    possible_retired = np.concatenate((split_df['old'].values.astype(np.uint64), cleave_labels, renumber_targets))
    possible_retired = pd.Series(0, index=possible_retired, dtype=np.uint64).rename('body').rename_axis('sv')

    mapped_bodies = base_mapping.unique()
    possible_identities = pd.Series(mapped_bodies, index=mapped_bodies).rename('body').rename_axis('sv')
    possible_identities = possible_identities[(possible_identities != 0)]

    # We only add the 'retired' IDs in cases where the supervoxel DIDN'T exist in the mapping already.
    # We only add 'identity' IDs in cases where the supervoxel DIDN'T exist in the mapping AND it wasn't retired.
    mapping = pd.concat((base_mapping, possible_retired, possible_identities))
    mapping = mapping.loc[~mapping.index.duplicated(keep='first')].copy()

    if sort == 'sv':
        mapping.sort_index(inplace=True)
    elif sort == 'body':
        mapping.sort_values(inplace=True)

    assert mapping.index.dtype == np.uint64
    assert mapping.dtype == np.uint64
    return mapping


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
            ops_list = []  # reset
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
def post_renumber(server, uuid, instance, renumbering, *, batch_size=None, session=None):
    """
    Change the body ID of a set of bodies.

    Args:
        server:
            DVID server
        uuid:
            DVID node
        instance:
            DVID labelmap instance
        renumbering:
            pd.Series named 'new_body', indexed by 'body'
        batch_size:
            Provide this to split the operation into smaller batches
    """
    assert isinstance(renumbering, pd.Series)
    assert renumbering.index.name == 'body'
    assert renumbering.name == 'new_body'

    batch_size = batch_size or len(renumbering)
    batches = iter_batches(renumbering, batch_size)
    for batch in tqdm_proxy(batches, disable=len(batches) <= 1):
        payload = batch.reset_index()[['new_body', 'body']].values.reshape(-1).tolist()
        r = session.post(f'{server}/api/node/{uuid}/{instance}/renumber', json=payload)
        r.raise_for_status()


@dvid_api_wrapper
def fetch_mutation_id(server, uuid, instance, body_id, *, session=None, handle_404=False):
    """
    Fetch the mutation ID for a particular body.
    The mutation ID is an integer that indicates when the body was most recently modified.
    This function obtains it via the /lastmod endpoint, but only returns the mutation ID, nothing more.

    If handle_404 is True, return -1 if the body doesn't exist.
    """
    try:
        response = fetch_generic_json(f'{server}/api/node/{uuid}/{instance}/lastmod/{body_id}', session=session)
    except HTTPError as ex:
        if handle_404 and ex.response is not None and ex.response.status_code == 404:
            return -1
        raise

    return response["mutation id"]


@dvid_api_wrapper
def fetch_mutation_ids(server, uuid, instance, bodies, *, session=None, processes=0):
    """
    Fetch the mutation IDs for several bodies, via parallel calls to fetch_mutation_id.

    If a body doesn't exist, its mutation ID will be returned as -1.
    (Note that legitimate bodies CAN have a mutation ID of 0.)

    Returns:
        pd.Series, indexed by body.
    """
    bodies = np.asarray(bodies, np.uint64)
    if processes == 0:
        mutids = []
        for body in bodies:
            m = fetch_mutation_id(server, uuid, instance, body, session=session, handle_404=True)
            mutids.append(m)
    else:
        fn = partial(fetch_mutation_id, server, uuid, instance, handle_404=True)
        mutids = compute_parallel(fn, bodies, processes=processes)

    return pd.Series(mutids, index=bodies, name='mutid').rename_axis('body')


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
        labels_coords = list(tqdm_proxy(labels_coords, total=len(labels), logger=logger))

    return dict(labels_coords)


def fetch_sparsevol_coarse_box(server, uuid, instance, label, supervoxels=False, *, missing='raise', session=None):
    """
    Convenience function for obtaining the approximate bounding box
    of a body via it's coarse sparsevol representation, which is
    derived from the label index.

    The results are returned at scale 6.
    """
    assert missing in ('raise', 'ignore')
    try:
        rle = fetch_sparsevol_coarse(server, uuid, instance, label, supervoxels, format='ranges', session=session)
        return rle_ranges_box(rle)
    except HTTPError as ex:
        status_code = (ex.response is not None) and ex.response.status_code
        if missing == 'ignore' and status_code == 404:
            return np.array([[0,0,0], [0,0,0]], dtype=np.int32)
        raise


@dvid_api_wrapper
def fetch_sparsevol(server, uuid, instance, label, scale=0, supervoxels=False,
                    *, format='coords', dtype=np.int32, mask_box=None, session=None):
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
            If provided, specifies the box within which the body mask (or RLEs/ranges/coords) should
            be returned, in scale-N coordinates where N is the scale you're requesting.
            Voxels outside the box will be omitted from the returned mask (or RLEs/ranges/coords).

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
    assert isinstance(supervoxels, bool)
    assert np.issubdtype(type(scale), np.integer)
    assert format in ('coords', 'rle', 'ranges', 'mask')

    rles = fetch_sparsevol_rles(server, uuid, instance, label, scale, supervoxels, box_zyx=mask_box, session=session)

    if format in ('coords', 'rle', 'ranges'):
        return parse_rle_response(rles, dtype, format)

    if format == 'mask':
        rle_ranges = parse_rle_response( rles, format='ranges' )
        mask, mask_box = runlength_decode_from_ranges_to_mask(rle_ranges, mask_box)
        return mask, mask_box


def fetch_sparsevol_box(server, uuid, instance, label, scale=0, supervoxels=False, *, missing='raise', session=None):
    """
    Convenience function for obtaining the bounding box of a body at a given scale, via it's sparsevol representation.
    """
    assert missing in ('raise', 'ignore')
    try:
        rle = fetch_sparsevol(server, uuid, instance, label, scale, supervoxels, format='ranges', session=session)
        return rle_ranges_box(rle)
    except HTTPError as ex:
        status_code = (ex.response is not None) and ex.response.status_code
        if missing == 'ignore' and status_code == 404:
            return np.array([[0,0,0], [0,0,0]], dtype=np.int32)
        raise


def fetch_sparsevol_boxes(server, uuid, instance, labels, scale=0, supervoxels=False, processes=4, missing='raise'):
    """
    Fetch the bounding box of several bodies (or supervoxels).
    Returns the boxes as a single array (N, 2, 3)
    """
    assert missing in ('raise', 'ignore')
    _fn = partial(fetch_sparsevol_box, server, uuid, instance, scale=scale, supervoxels=supervoxels, missing=missing)
    boxes = compute_parallel(_fn, labels, processes=processes)
    return np.array(boxes)


@dvid_api_wrapper
def fetch_sparsevol_head(server, uuid, instance, label, supervoxels=False, *, session=None):
    """
    Returns True if the given label exists at all on the DVID server,
    False otherwise.
    """
    supervoxels = str(bool(supervoxels)).lower()  # to lowercase string
    url = f'{server}/api/node/{uuid}/{instance}/sparsevol/{label}?supervoxels={supervoxels}'
    r = session.head(url)

    if r.status_code == 200:
        return True
    if r.status_code == 204:
        return False

    r.raise_for_status()


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
def generate_sample_coordinate(server, uuid, instance, label_id, supervoxels=False, *, interior=False, session=None):
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

        interior:
            By default (``interior=False``), this function might return points that
            fall close to the edge of the body surface, since it merely picks
            the "middle" voxel according to the scan order, without regard
            for any morphological attributes.
            When ``interior=True``, this function tries to avoid points which lie
            near the exterior of the body. It starts by eroding the low-res mask,
            thus eliminating very thin branches, and then selects the "most interior"
            voxel from the "middle" block of those blocks which remain, where "most interior"
            is determined via the distance transform.

    Returns:
        [Z,Y,X] -- An arbitrary point within the body of interest.
    """
    SCALE = 6  # sparsevol-coarse is always scale 6
    if not interior:
        coarse_block_coords = fetch_sparsevol_coarse(server, uuid, instance, label_id, supervoxels, session=session)
    else:
        # Fetch the SVC mask and erode by 1 (if possible)
        mask_s6, mask_box_s6 = fetch_sparsevol_coarse(server, uuid, instance, label_id, supervoxels, format='mask', session=session)
        eroded = multiBinaryErosion(mask_s6, 1)
        if eroded.sum() > 0:
            mask_s6 = eroded

        coarse_block_coords = np.transpose(mask_s6.nonzero()) + mask_box_s6[0]

    # Select the "middle" voxel in the SVC mask
    num_blocks = len(coarse_block_coords)
    middle_block_coord = (2**SCALE) * np.array(coarse_block_coords[num_blocks//2]) // 64 * 64
    middle_block_box = (middle_block_coord, middle_block_coord + 64)

    # Fetch the dense segmentation in the middle block.
    block = fetch_labelmap_voxels(server, uuid, instance, middle_block_box, supervoxels=supervoxels, session=session)
    block_mask = (block == label_id)
    if block_mask.sum() == 0:
        label_type = {False: 'body', True: 'supervoxel'}[supervoxels]
        raise RuntimeError(f"The sparsevol-coarse info for this {label_type} ({label_id}) "
                           "appears to be out-of-sync with the scale-0 segmentation.")

    if interior:
        # Find the "most interior" voxel in the block, according to the DT
        dt = distanceTransform(block_mask.astype(np.uint32), background=False)
        c = np.unravel_index(np.argmax(dt), dt.shape)
        c += middle_block_box[0]
        return c
    else:
        # Find the "middle" voxel, either in raster-scan order
        nonzero_coords = np.transpose(block_mask.nonzero())
        return middle_block_coord + nonzero_coords[len(nonzero_coords)//2]


def generate_sample_coordinates(server, uuid, instance, bodies, supervoxels=False, *, interior=False, processes=4, skip_out_of_sync=False):
    """
    Calls generate_sample_coordinate() in parallel for a batch of bodies.
    See that function for argument details.

    Args:
        skip_out_of_sync:
            If True, don't raise an exception for bodies whose
            label index is out-of-sync with the scale-0 segmentation.
            Instead, return [-1, -1, -1].
    """
    assert processes > 0
    gen_coord = partial(_generate_sample_coordinate_no404, skip_out_of_sync, server, uuid, instance, supervoxels=supervoxels, interior=interior)
    coords = compute_parallel(gen_coord, bodies, processes=processes)
    label_type = {False: 'body', True: 'supervoxel'}[supervoxels]
    return pd.DataFrame(coords, columns=[*'zyx'], index=bodies).rename_axis(label_type)


def _generate_sample_coordinate_no404(skip_out_of_sync, *args, **kwargs):
    try:
        return generate_sample_coordinate(*args, **kwargs)
    except HTTPError as ex:
        if ex.response.status_code == 404:
            return [0,0,0]
        raise
    except RuntimeError as ex:
        if 'out-of-sync' in str(ex) and skip_out_of_sync:
            return [-1, -1, -1]
        raise


# Alternative name
locate_bodies = generate_sample_coordinates


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
    assert (aligned_shape > 0).all(), \
        f"Requested box has zero or negative size (ZYX): {box_zyx.tolist()}"

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

    Note:
        Unlike the bare DVID /specificblocks endpoint, this function WILL produce an
        all-zero block in cases where there there are no labels in the block.
        Thus, the number of output blocks will exactly match the number of input blocks.

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

            Note:
                The keys of the results will not necessarily match the input order.
                The result is sorted by key, in scan-order.

        scale:
            Which downsampling scale to fetch from

        supervoxels:
            If True, request supervoxel data from the given labelmap instance.

        format:
            One of the following:
                ('array', 'lazy-array', 'raw-response', 'blocks', 'lazy-blocks', 'raw-blocks', 'callable-blocks')

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

    corners_zyx = lexsort_columns(corners_zyx)
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

    blocks = {c: None for c in map(tuple, corners_zyx.tolist())}
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
    if not buf:
        return corner, np.zeros((64,64,64), np.uint64)
    block = DVIDNodeService.inflate_labelarray_blocks3D_from_raw(buf, (64,64,64), corner)
    return (corner, block)


def fetch_labelmap_specificblocks_batched(server, uuid, instance, corners_zyx, scale=0, supervoxels=False, format='blocks',
                                          *, processes=0, batch_size=1000):
    """
    Same as fetch_labelmap_specificblocks(), but breaks the workload up into batches, optionally parallelized.

    For argument descriptions, see fetch_labelmap_specificblocks(),
    but note that this function doesn't support the 'array' output formats
    and doesn't support map_on_client.
    """
    assert format in ('blocks', 'raw-blocks', 'lazy-blocks', 'callable-blocks')
    batch_fmt = format
    if format in ('lazy-blocks', 'callable-blocks'):
        batch_fmt = 'raw-blocks'

    fn = partial(fetch_labelmap_specificblocks, server, uuid, instance, scale=scale, supervoxels=supervoxels, format=batch_fmt)
    corner_batches = iter_batches(corners_zyx, batch_size)
    block_batches = compute_parallel(fn, corner_batches, processes=processes)
    blocks = reduce(or_, block_batches)

    def inflate_blocks(processes=None, threads=None):
        # By default, this function is single-threaded but the user can
        # optionally specify parallelism by passing processes or threads
        # when calling it.
        inflated_blocks = compute_parallel(
            _inflate_block, blocks.items(), starmap=True, processes=processes, threads=threads, show_progress=False)
        return dict(inflated_blocks)

    if format in ('raw-blocks', 'blocks'):
        return blocks
    elif format == 'lazy-blocks':
        return inflate_blocks
    elif format == 'callable-blocks':
        # Wrap each buffer in a callable that will inflate it
        for coord, buf in [*blocks.items()]:
            f = DVIDNodeService.inflate_labelarray_blocks3D_from_raw
            blocks[coord] = partial(f, buf, (64,64,64), coord)
        return blocks


def fetch_labelmap_voxels_chunkwise(server, uuid, instance, box_zyx, scale=0, throttle=False, supervoxels=False,
                                    *, chunk_shape=(64,64,4096), threads=0, format='array', out=None, progress=True):
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

    if progress:
        logger.info("Fetching compressed chunks")
    if threads == 0:
        boxes_and_chunks = [*tqdm_proxy(map(_fetch, chunk_boxes), total=len(chunk_boxes), disable=not progress)]
    else:
        boxes_and_chunks = compute_parallel(_fetch, chunk_boxes, ordered=False, threads=threads)

    def inflate_labelarray_chunks():
        if progress:
            logger.info("Inflating chunks")
        for block_box, lazy_chunk in tqdm_proxy(boxes_and_chunks, disable=not progress):
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
def post_labelmap_blocks(server, uuid, instance, corners_zyx, blocks, scale=0, downres=False, noindexing=False, throttle=False,
                         *, is_raw=False, gzip_level=6, session=None, progress=False, ingestion_mode=False):
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

        ingestion_mode:
            If true, use ``POST .../ingest-supervoxels`` to send the data,
            which is similar to ``POST .../blocks`` except that it's faster because
            it doesn't use any mutexes to protect against multi-threaded access to
            the same blocks or label indexes.
            You can use this if you're ingesting data and you've disabled indexing and
            downres on the DVID side.
    """
    assert not downres or scale == 0, "downres option is only valid for scale 0"

    if is_raw:
        assert isinstance(blocks, (bytes, memoryview))
        body_data = blocks
    else:
        body_data = encode_labelarray_blocks(corners_zyx, blocks, gzip_level, progress)

    if not body_data:
        return  # No blocks

    # These options are already false by default, so we'll only include them if we have to.
    opts = { 'downres': downres, 'noindexing': noindexing, 'throttle': throttle }

    if ingestion_mode:
        endpoint = 'ingest-supervoxels'
        assert noindexing, "ingestion_mode=True requires noindexing=True"
        assert not downres, "ingestion_mode=True requires downres=False"
        del opts['noindexing']
        del opts['downres']
    else:
        endpoint = 'blocks'

    params = { 'scale': str(scale) }
    for opt, value in opts.items():
        if value:
            params[opt] = str(bool(value)).lower()

    r = session.post(f'{server}/api/node/{uuid}/{instance}/{endpoint}', params=params, data=body_data)
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
    assert len(supervoxel_ids) > 0, \
        "Can't cleave away an empty list of supervoxels"

    r = session.post(f'{server}/api/node/{uuid}/{instance}/cleave/{body_id}', json=supervoxel_ids)
    r.raise_for_status()
    cleaved_body = r.json()["CleavedLabel"]
    return cleaved_body


@dvid_api_wrapper
def post_hierarchical_cleaves(server, uuid, instance, body_id, group_mapping, leave_progress=True, *, session=None):
    """
    When you want to perform a lot of cleaves on a single
    body (e.g. a "frankenbody") whose labelindex is really big,
    it is more efficient to perform some coarse cleaves at first,
    which will iteratively divide the labelindex into big chunks,
    and then re-cleave the coarsely cleaved objects.
    That will run faster than cleaving off little bits one at a time,
    thanks to the reduced labelindex sizes at each step.
    (In other words, N*log(N) is faster than N^2.)

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
    from . import fetch_labelindex  # late import to avoid recursive import

    assert isinstance(group_mapping, pd.Series)
    assert group_mapping.index.dtype == np.uint64
    group_mapping = group_mapping.rename('group', copy=False)
    group_mapping = group_mapping.rename_axis('sv', copy=False)

    logger.info(f"Verifying mapping for {len(group_mapping)} supervoxels")
    dvid_mapping = fetch_mapping(server, uuid, instance, group_mapping.index.values, as_series=True,
                                 batch_size=10_000, threads=8, session=session)
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

    with Timer(f"Fetching label index for body {body_id}", logger):
        li_df = fetch_labelindex(server, uuid, instance, body_id, format='pandas', session=session).blocks
    orig_len = len(li_df)
    li_df = li_df.query('sv in @group_df.index')

    # If the groups don't include every supervoxel in the body,
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

    with Timer(f"Performing {num_cleaves} cleaves", logger), \
            tqdm_proxy(total=num_cleaves, leave=leave_progress, logger=logger) as progress_bar:
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
    main_label = int(main_label)
    other_labels = list(map(int, other_labels))
    assert main_label not in other_labels, \
        (f"Can't merge {main_label} with itself.  "
         "DVID does not behave correctly if you attempt to merge a body into itself!")

    content = [main_label] + other_labels

    r = session.post(f'{server}/api/node/{uuid}/{instance}/merge', json=content)
    r.raise_for_status()


@dvid_api_wrapper
def fetch_mutations(server, uuid, instance, userid=None, *, action_filter=None, dag_filter='leaf-and-parents', format='pandas', session=None):
    """
    Fetch the log of successfully completed mutations.
    The log is returned in the same format as the kafka log.

    For consistency with :py:func:``read_kafka_msgs()``, this function adds
    the ``dag_filter`` and ``action_filter`` options, which are not part
    of the DVID REST API. To emulate the bare-bones /mutations results,
    use dag_filter='leaf-only'.

    Note:
        By default, the dag_filter setting is 'leaf-and-parents'.
        So unlike the default behavior of the /mutations endpoint in the DVID REST API,
        this function returns all mutations for the given UUID and ALL of its ancestor UUIDs.
        (To achieve this, it calls the /mutations multiple times -- once per ancestor UUID.)

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
            (This is not part of the DVID API.  It's implemented in this
            python function by calling the /mutations endpoint for multiple UUIDs.)

            One of:
            - 'leaf-only' (only messages whose uuid matches the one provided),
            - 'leaf-and-parents' (only messages matching the given uuid or its ancestors), or
            - None (no filtering by UUID).

        format:
            How to return the data. Either 'pandas' or 'json'.

    Returns:
        Either a DataFrame or list of parsed json values, depending
        on what you passed as 'format'.
    """
    assert dag_filter in ('leaf-only', 'leaf-and-parents', None)

    # json-values is a synonym, for compatibility with read_kafka_messages
    assert format in ('pandas', 'json', 'json-values')

    uuid = resolve_ref(server, uuid, expand=True)

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
        # We don't need special handling of '*-complete' messages
        # because the mutation log only contains completed messages.
        # However in older mutation logs the message action isn't marked as a '*-complete'
        msg_df = labelmap_kafka_msgs_to_df(msgs, completes_only=False, fill_completes=False)

        # New convention is for DVID to emit '-complete' messages, but old servers didn't do that.
        # We just patch the log to make it look like that's what happened.
        replace = {k:k for k in msg_df['action'].unique()}
        replace.update({k: f'{k}-complete' for k in ('cleave', 'merge', 'split-supervoxel', 'split', 'renumber')})
        msg_df['action'] = msg_df['action'].map(replace)

        for msg in msg_df['msg']:
            msg['Action'] = replace[msg['Action']]
        return msg_df
    else:
        return msgs


@dvid_api_wrapper
def fetch_history(server, uuid, instance, body, from_uuid=None, to_uuid=None, format='pandas', *, session=None):
    """
    Returns JSON for the all mutations pertinent to the label in the given range of versions.

    Warning:
        At the time of this writing (2022-02-23), this feature in DVID doesn't work very well.
        Therefore, this function can't be used right now.

    See also:
        fetch_mutations(), compute_merge_hierarchies(), determine_owners()

    Args:
        server, uuid, instance:
            A labelmap instance.
        from_uuid:
            First UUID in the range of UUIDs to fetch mutations from.
            By default, this function determines the repo root UUID.
        to_uuid:
            The last UUID in the range of UUIDs (inclusive) to fetch mutations from.
            By default, this function uses whichever UUID you supplied as the
            second argument to this function, after 'server'.
    """
    assert format in ('pandas', 'json')
    to_uuid = to_uuid or uuid
    if from_uuid is None:
        from_uuid = find_repo_root(server, uuid)

    r = session.get(f'{server}/api/node/{uuid}/{instance}/history/{body}/{from_uuid}/{to_uuid}')
    r.raise_for_status()
    msgs = r.json()

    if format == 'pandas':
        return labelmap_kafka_msgs_to_df(msgs)
    else:
        return msgs


def read_labelmap_kafka_df(server, uuid, instance='segmentation', action_filter=None, dag_filter='leaf-and-parents',
                           default_timestamp=DEFAULT_TIMESTAMP, completes_only=True, group_id=None):
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

        completes_only:
            See labelmap_kafka_msgs_to_df()

    Returns:
        DataFrame with columns:
            ['timestamp', 'uuid', 'mutid', 'action', 'target_body', 'target_sv', 'msg']
    """
    msgs = read_kafka_messages(server, uuid, instance, action_filter, dag_filter, group_id=group_id)
    msgs_df = labelmap_kafka_msgs_to_df(msgs, default_timestamp, completes_only)
    return msgs_df


def labelmap_kafka_msgs_to_df(kafka_msgs, default_timestamp=DEFAULT_TIMESTAMP, completes_only=True, fill_completes=True):
    """
    Convert the kafka messages for a labelmap instance into a DataFrame.

    Args:
        kafka_msgs:
            A list of JSON messages from the kafka log emitted by a labelmap instance.

        default_timestamp:
            Old versions of DVID did not emit a timestamp with each message.
            For such messages, we'll assign a default timestamp, specified by this argument.

        completes_only:
            Some operations are logged in kafka twice: once for the initial request,
            and another when the operation completes.
            (If an operation fails to complete, the completion message is not logged.)
            This argument will filter out the 'trigger' messages, leaving only the completion messages.

        fill_completes:
            In newer versions of DVID, the '*-complete' message includes all the same fields that
            the original (trigger) message included, but older versions of DVID didn't automatically
            include all fields in both messages.
            With this argument, the '*-complete' messages from old kafka logs are modified to include
            all fields, which are obtained by copying them out of the corresponding trigger messsage.
    """
    FINAL_COLUMNS = [
        'timestamp', 'uuid', 'action', 'user', 'mutid',
        'target_body', 'merged',
        'child_cleaved_body', 'child_cleaved_svs',
        'target_sv', 'child_split_sv', 'child_remain_sv',
        'mutation_bodies',  # All bodies mentioned in the mutation (target, merged, cleaved, etc.)
        'msg'
    ]

    # Generic conversion to DataFrame.
    df = kafka_msgs_to_df(kafka_msgs, drop_duplicates=False, default_timestamp=default_timestamp)

    if len(df) == 0:
        return pd.DataFrame([], columns=FINAL_COLUMNS)

    # Append action
    df['action'] = [msg['Action'] for msg in df['msg']]
    df['user'] = [msg['User'] if 'User' in msg else '' for msg in df['msg']]

    KNOWN_ACTIONS = {
        'post-maxlabel', 'post-nextlabel',
        *chain(*((a, f'{a}-complete') for a in ('cleave', 'merge', 'split', 'split-supervoxel', 'renumber')))}

    COMPLETE_ACTIONS = {  # noqa
        'post-maxlabel', 'post-nextlabel',
        'cleave-complete', 'merge-complete',
        'split-complete', 'split-supervoxel-complete',
        'renumber-complete'
    }

    unknown_actions = set(df['action'].unique()) - KNOWN_ACTIONS
    assert not unknown_actions, f"Unknown actions in mutation log: {unknown_actions}"

    target_bodies = []
    child_cleaved_bodies = []
    child_cleaved_svs = []
    target_svs = []
    child_split_svs = []
    child_remain_svs = []
    merged_bodies = []

    # This will be the full list of any body mentioned in the mutation.
    mutation_bodies = []

    # Determine which field to place in the target_body, target_sv, merged fields
    for msg in df['msg'].values:
        action = msg['Action']
        target_body = np.nan
        child_cleaved_body = np.nan
        cleaved_svs = []
        target_sv = np.nan
        child_split_sv = np.nan
        child_remain_sv = np.nan
        merges = []
        _mutation_bodies = []

        if action in ('cleave', 'cleave-complete'):
            target_body = msg.get('OrigLabel', np.nan)
            child_cleaved_body = msg.get('CleavedLabel', np.nan)
            cleaved_svs = msg.get('CleavedSupervoxels', np.nan)
            _mutation_bodies = [target_body, child_cleaved_body]
        elif action in ('merge', 'merge-complete'):
            target_body = msg.get('Target', np.nan)
            merges = msg.get('Labels')
            if merges:
                _mutation_bodies = [target_body, *merges]
            else:
                _mutation_bodies = [target_body]
                merges = np.nan
        elif action in ('split', 'split-complete'):
            target_body = msg.get('Target', np.nan)
            _mutation_bodies = [target_body]
            # For deprecated 'split' commands, there are
            # multiple sv splits listed in a single message.
            target_sv = []
            child_split_sv = []
            child_remain_sv = []
            for t, sr in msg['SVSplits'].items():
                target_sv.append(int(t))
                child_split_sv.append(sr['Split'])
                child_remain_sv.append(sr['Remain'])
            if len(target_sv) == 1:
                # If there's only a single split sv,
                # it's more convenient not to wrap it in a list.
                # Callers will likely be using 'explode' anyway.
                target_sv = target_sv[0]
                child_split_sv = child_split_sv[0]
                child_remain_sv = child_remain_sv[0]
        elif action in ('split-supervoxel', 'split-supervoxel-complete'):
            target_body = msg.get('Body', np.nan)
            target_sv = msg.get('Supervoxel', np.nan)
            child_split_sv = msg.get('SplitSupervoxel', np.nan)
            child_remain_sv = msg.get('RemainSupervoxel', np.nan)
            _mutation_bodies = [target_body]
        elif action in ('renumber', 'renumber-complete'):
            target_body = msg['NewLabel']
            merges = [msg['OrigLabel']]
            _mutation_bodies = [target_body, *merges]

        target_bodies.append(target_body)
        child_cleaved_bodies.append(child_cleaved_body)
        child_cleaved_svs.append(cleaved_svs)
        child_split_svs.append(child_split_sv)
        child_remain_svs.append(child_remain_sv)
        target_svs.append(target_sv)
        merged_bodies.append(merges)
        mutation_bodies.append(_mutation_bodies)

    df['target_body'] = target_bodies
    df['child_cleaved_body'] = child_cleaved_bodies
    df['child_cleaved_svs'] = child_cleaved_svs
    df['merged'] = merged_bodies
    df['mutation_bodies'] = mutation_bodies

    df['target_sv'] = target_svs
    df['child_split_sv'] = child_split_svs
    df['child_remain_sv'] = child_remain_svs

    # Create completely empty columns if necessary to provide the expected output columns.
    for col in FINAL_COLUMNS:
        if col not in df:
            df[col] = np.nan

    # We're usually not filling completes,
    # so this is where most calls return.
    if not fill_completes:
        if completes_only:
            df = df.query('action in @COMPLETE_ACTIONS')

        # Fill ints with zeros and convert to better dtype.
        df['target_body'] = df['target_body'].fillna(0).astype(np.uint64)
        df['child_cleaved_body'] = df['child_cleaved_body'].fillna(0).astype(np.uint64)
        df['mutid'] = df['mutid'].fillna(-1).astype(int)

        df['target_sv'] = df['target_sv'].fillna(0)
        df['child_split_sv'] = df['child_split_sv'].fillna(0)
        df['child_remain_sv'] = df['child_remain_sv'].fillna(0)

        # The 'split' action can result in elements of type 'list',
        # so we can only convert the dtype of the following columns
        # if there are no 'split' actions.
        if not (df['target_sv'].map(type) == list).any():
            df['target_sv'] = df['target_sv'].astype(np.uint64)
            df['child_split_sv'] = df['child_split_sv'].astype(np.uint64)
            df['child_remain_sv'] = df['child_remain_sv'].astype(np.uint64)

        return df[FINAL_COLUMNS]

    # This logic is somewhat more complex than you might think is necessary,
    # but that's because the kafka logs for some repos might (sadly) contain
    # duplicate mutation IDs, i.e. the mutation ID was not unique in our earlier logs.
    # So, we might have two pairs of merge + merge-complete messages that use the same mutation ID.
    # We append a 'transaction_id' to distinguish between those pairs.
    # The transaction_id is just the index of the completion message.
    df['trigger'] = df['action'].map(lambda s: s[:-len('-complete')] if '-complete' in s else s)
    completion_idx = df.query('action in @COMPLETE_ACTIONS').index
    df['transaction_id'] = np.nan
    df.loc[completion_idx, 'transaction_id'] = completion_idx

    # Fill backwards to associate each trigger message with the completion message that follows it.
    df['transaction_id'] = df.groupby(['uuid', 'mutid', 'trigger'])['transaction_id'].bfill()

    # But undo that filling operation for those trigger messages
    # which DON'T HAVE a completion message (because the operation failed).
    df['transaction_component'] = df.iloc[::-1].groupby('transaction_id').cumcount()
    df.loc[df['transaction_component'] > 1, 'transaction_id'] = np.nan

    # Old kafka logs didn't include all fields in the 'complete' message,
    # so fill them in from the trigger message to ensure that 'complete'
    # message JSON contains all the info that the corresponding trigger
    # message contained..
    for _, tx_df in df.groupby('transaction_id'):
        assert len(tx_df) == 2
        first, second = tx_df['msg'].values
        for k, v in first.items():
            second.setdefault(k, v)

    # Also forward-fill in the columns to ensure that the 'complete' rows have everything the 'trigger' rows had.
    fill_cols = list({*df.columns} - {'action', 'uuid', 'mutid', 'timestamp', 'trigger', 'transaction_id', 'transaction_component'})
    df[fill_cols] = df.groupby('transaction_id')[fill_cols].ffill()

    if completes_only:
        df = df.query('action in @COMPLETE_ACTIONS')

    df = df.copy()
    df['target_body'] = df['target_body'].fillna(0).astype(np.uint64)
    df['target_sv'] = df['target_sv'].fillna(0).astype(np.uint64)
    df['mutid'] = df['mutid'].fillna(-1).astype(int)
    return df[FINAL_COLUMNS]


AffectedBodies = namedtuple("AffectedBodies", "new_bodies changed_bodies removed_bodies new_svs deleted_svs")


def compute_affected_bodies(kafka_msgs):
    """
    Given a list of json messages from a labelmap instance (from kafka or from /mutations),
    compute the set of all bodies that are mentioned in the log as either new, changed, or removed.
    Also return the set of new supervoxels from 'supervoxel-split' actions.

    Note:
        The set of 'changed' bodies does NOT include changes due to supervoxel-split events,
        since DVID currently doesn't make it easy to determine which body the supervoxel
        belonged to at the time it was split.

    Note:
        Supervoxels from the deprecated body 'split' action (as opposed to 'split-supervoxel')
        are not included in the results.
        If you're interested in all supervoxel splits, see fetch_supervoxel_splits().

    Note:
        This function analyzes ONLY the '*-complete' messages, and it's assumed that they
        contain ALL fields that the corresponding trigger messages contained.
        For newer versions of DVID, they do contain all fields.
        However, that's not true for older kafka logs.
        Fortunately the labelmap_kafka_msgs_to_df() function will transform the log messages to
        appear as if they had been written by a newer version of DVID.
        Pre-process the kafka log with that function first.

    See also:
        neuclease.dvid.kafka.filter_kafka_msgs_by_timerange()

    Args:
        Kafka log for a labelmap instance, obtained via ``read_kafka_messages()`` or ``read_labelmap_kafka_df()``.
        It's recommended to use the latter, as it will transform old kafka logs to put
        them in the newer format as explained in the note above.

    Returns:
        new_bodies, changed_bodies, removed_bodies, new_svs, deleted_svs

    Example:

        .. code-block:: ipython

            In [1]: vnc_seg = ('emdata5.janelia.org:8400', 'e98a33', 'segmentation')
            ...:
            ...: # This calls /mutations repeatedly for every upstream UUID
            ...: mutations = fetch_mutations(*vnc_seg, dag_filter='leaf-and-parents')
            ...:
            ...: new_bodies, changed_bodies, removed_bodies, new_svs, deleted_svs = compute_affected_bodies(mutations)
    """
    if isinstance(kafka_msgs, pd.DataFrame):
        kafka_msgs = kafka_msgs['msg']

    new_bodies = set()
    changed_bodies = set()
    removed_bodies = set()
    new_svs = set()
    deleted_svs = set()

    for msg in kafka_msgs:
        if not msg['Action'].endswith('complete'):
            continue

        if msg['Action'] == 'cleave-complete':
            changed_bodies.add( msg['OrigLabel'] )
            new_bodies.add( msg['CleavedLabel'] )

        if msg['Action'] == 'merge-complete':
            changed_bodies.add( msg['Target'] )
            labels = set( msg['Labels'] )
            removed_bodies |= labels
            changed_bodies -= labels
            new_bodies -= labels

        # 'renumber' is treated identically to 'merge'
        if msg['Action'] == 'renumber-complete':
            changed_bodies.add( msg['NewLabel'] )
            labels = {msg['OrigLabel']}
            removed_bodies |= labels
            changed_bodies -= labels
            new_bodies -= labels

        if msg['Action'] == 'split-complete':
            changed_bodies.add( msg['Target'] )
            new_bodies.add( msg['NewLabel'] )

        if msg['Action'] == 'split-supervoxel-complete':
            new_svs.add(msg['SplitSupervoxel'])
            new_svs.add(msg['RemainSupervoxel'])
            deleted_svs.add(msg['Supervoxel'])

    new_bodies = np.fromiter(new_bodies, np.uint64)
    changed_bodies = np.fromiter(changed_bodies, np.uint64)
    removed_bodies = np.fromiter(removed_bodies, np.uint64)
    new_svs = np.fromiter(new_svs, np.uint64)
    deleted_svs = np.fromiter(deleted_svs, np.uint64)

    return AffectedBodies(new_bodies, changed_bodies, removed_bodies, new_svs, deleted_svs)


def compute_merge_hierarchies(msgs):
    """
    Using messages from the mutation log, construct an
    nx.DiGraph that encodes the forest of body merge trees,
    i.e. a hierarchy indicating which bodies absorbed which other bodies.
    The root node in each tree is the body that still exists in DVID.
    Conversely, all nodes with parents no longer exist (they've been
    merged into something else).

    Note:
        This result does not make any effort to account for cleaves.

    Returns:
        nx.DiGraph
    """
    if isinstance(msgs, pd.DataFrame):
        msgs = msgs['msg']

    g = nx.DiGraph()
    for msg in msgs:
        if msg['Action'] == 'merge-complete':
            target = msg['Target']
            edges = [(target, label) for label in msg['Labels']]
            g.add_edges_from(edges)
        elif msg['Action'] == 'renumber-complete':
            # 'renumber' is treated the same as 'merge'
            target = msg['NewLabel']
            merged = msg['OrigLabel']
            g.add_edge(target, merged)
    return g


def determine_owners(merge_hierarchy, bodies):
    """
    Use the merge history graph which was obtained from the mutations
    log via ``compute_merge_hierarchies()`` to determine the new
    "owner" of a given body, if it no longer exists.

    Any bodies not mentioned in the graph will be assumed to be "owned" by themselves.
    """
    owners = []
    for body in bodies:
        try:
            r = find_root(merge_hierarchy, body)
        except nx.NetworkXError:
            owners.append(body)
        else:
            owners.append(r)

    return pd.Series(owners, index=bodies, name='current_body').rename_axis('orig_body')


def determine_merge_chains(merge_hierarchy, bodies):
    """
    Use the merge history graph which was obtained from the mutations
    log via ``compute_merge_hierarchies()`` to determine the new
    "owner" of a given body, if it no longer exists, along with the
    intermediate chain of owners.

    Any bodies not mentioned in the graph will be assumed to be "owned" by themselves.

    The returned ``merge_chain`` column includes the starting body,
    and ends with the owning (root) body.
    """
    mc = []
    for body in bodies:
        try:
            a = toposorted_ancestors(merge_hierarchy, body, reversed=True)
        except nx.NetworkXError:
            mc.append([body])
        else:
            mc.append([body, *a])

    owners = [c[-1] for c in mc]
    merge_counts = [len(c)-1 for c in mc]
    return pd.DataFrame({
        'owner': owners,
        'merge_count': merge_counts,
        'merge_chain': mc},
        index=bodies).rename_axis('body')


def find_furthest_point(server, uuid, instance, body, starting_coord_zyx, *, session=None):
    """
    Find the approximate furthest point on a body from a given starting point
    (in euclidean terms, not cable length).

    This function uses the coarse sparsevol to narrow its search,
    which means that the results are not guaranteed to be exact.
    (A point in a different block might be chosen if that block's
    distance is within 64px of the optimal distance.)

    It can be inconvenient to work with and visualize points if they're
    on the surface of a body, so we don't return the exact furthest point.
    Instead, we return a point on the interior of the body near the actual
    furthest point.

    Returns:
        coord_zyx, distance, in scale-0 units.
    """
    # Determine furthest block via sparsevol-coarse
    svc = (2**6) * fetch_sparsevol_coarse(server, uuid, instance, body, format='coords', session=session)
    svc_centers = svc + (2**5)
    i = np.argmax(np.linalg.norm(starting_coord_zyx - svc_centers, axis=1))

    # Pick the center of the segment within that block
    # (Proofreaders don't like it if the point is on the segment edge)
    box = (svc[i], svc[i] + (2**6))
    vol = fetch_labelmap_voxels(server, uuid, instance, box, session=session)
    mask = (vol == body)
    dt = distance_transform(mask.astype(np.uint32), background=False, pad=True)
    c = np.unravel_index(np.argmax(dt), dt.shape)
    c += box[0]
    dist = np.linalg.norm(starting_coord_zyx - c)
    return (c, dist)


def thickest_point_in_body(server, uuid, instance, body, *, session=None):
    """
    Find the approximate "thickest point" in a body,
    i.e. the point furthest from the body edge.

    Returns:
        (point_zyx, radius)
        Both expressed in scale-0 units.

    Note:
        This function really returns a cheap approximation.
        This function starts by finding the thickest point the coarse
        sparsevol of the body, i.e. downsampled by 64x.
        Then it picks a somewhat arbitrary point within that block.
    """
    try:
        mask, mask_box = fetch_sparsevol_coarse(server, uuid, instance, body, format='mask', session=session)
    except HTTPError:
        return (0,0,0), 0

    p, d = thickest_point_in_mask(mask)
    p += mask_box[0]
    p *= 64
    d *= 64

    block_seg = fetch_labelmap_voxels(server, uuid, instance, (p, p + 64))
    p2, d2 = thickest_point_in_mask(block_seg == body)
    return tuple(p + p2), d + d2


def recursive_sv_split_by_grid(server, uuid, instance, sv, init_grid=8192, final_grid=2048):
    """
    Split a huge supervoxel into smaller pieces using a simple, block-aligned strategy.

    For efficiency, the splits are first performed along a coarse grid and then
    recursively split along finer grids.

    Note:
        The resulting pieces will NOT necessarily be contiguous.  This function exists
        solely to make it easier to work with unmanagably large supervoxels by chopping
        them up.
    """
    if not isinstance(init_grid, Collection):
        init_grid = 3 * (init_grid,)
    if not isinstance(final_grid, Collection):
        final_grid = 3 * (final_grid,)

    init_grid = np.asarray(init_grid, int)
    final_grid = np.asarray(final_grid, int)

    assert (init_grid >= final_grid).all(), \
        "Final grid should not be larger than initial grid."
    assert (np.log2(init_grid / final_grid) % 1 == 0).all(), \
        "Final grid must divide into initial grid via a power of 2"

    def _recursive_split(sv, rng, init_grid, new_svs, indent=0):
        # Determine split ranges for each block
        with Timer(f"Computing grid RLEs [{tuple(init_grid)}]", logger):
            block_rng_df = split_ranges_for_grid(rng, init_grid)
            block_ids_and_ranges = list(block_rng_df.groupby(['Bz', 'By', 'Bx']))
            num_blocks = len(block_ids_and_ranges)

        logger.info(f"{' '*indent}Splitting {sv} into {num_blocks} blocks [{tuple(init_grid)}]")
        new_svs.setdefault(tuple(init_grid), {})
        remain_sv = sv
        for i, (block_id, rle_df) in enumerate(block_ids_and_ranges):
            block_rng = rle_df[['z', 'y', 'x1', 'x2']].values.astype(np.int32)

            if i < num_blocks - 1:
                payload = construct_rle_payload_from_ranges(block_rng)
                with Timer(f"{' '*(indent+2)}Posting split: {sv}", logger):
                    split_sv, remain_sv = post_split_supervoxel(server, uuid, instance, remain_sv, payload)

                # Before continuing, poll RLEs until split appears complete.
                _wait_for_split_done(server, uuid, instance, sv, split_sv, indent + 2)
                _wait_for_split_done(server, uuid, instance, sv, remain_sv, indent + 2)

                new_svs[tuple(init_grid)][block_id] = split_sv
                if tuple(init_grid) != tuple(final_grid):
                    _recursive_split(split_sv, block_rng, init_grid // 2, new_svs, indent+2)
            else:
                # The final, unprocessed block is the remainder.
                new_svs[tuple(init_grid)][block_id] = remain_sv

                if tuple(init_grid) != tuple(final_grid):
                    _recursive_split(remain_sv, block_rng, init_grid // 2, new_svs, indent+2)

        return new_svs

    with Timer(f"Fetching full sparsevol for sv {sv}", logger):
        full_rng = fetch_sparsevol(server, uuid, instance, sv, supervoxels=True, format='ranges')
    return _recursive_split(sv, full_rng, init_grid, {})


def _wait_for_split_done(server, uuid, instance, parent_sv, child_sv, indent=0, scales=[0, 3, 6]):
    """
    Helper function for recursive_sv_split_by_grid().

    Waits until the given child supervoxel has been completely
    updated in the DVID voxels at the given scales.

    Determines whether a split is complete this by obtaining the set of blocks in which
    the new (child) supervoxel should reside and checks their voxels for the ABSCENCE
    of the PARENT id.  If the parent id is still present in any block, then that block
    hasn't yet been rewritten, and therefore the supervoxel split is not yet complete.

    (We can't check for the presence of the child ID because at low-res scales,
    the child ID isn't guaranteed to exist in the voxels at all, due to downsampling
    effects. But we know the parent MUST NOT be absent.)
    """
    svc = fetch_sparsevol_coarse(server, uuid, instance, child_sv, supervoxels=True, format='coords')

    for scale in scales:
        logger.info(f"{' '*indent}Split {parent_sv}->{child_sv}: Scale {scale}: Checking.")
        incomplete_corners = pd.DataFrame(svc * (2**(6-scale)))
        incomplete_corners = ((incomplete_corners // 64) * 64).drop_duplicates().values
        while len(incomplete_corners):
            incomplete_corners = _determine_incomplete_blocks(server, uuid, instance, incomplete_corners, 0, parent_sv)
            if len(incomplete_corners):
                logger.info(f"{' '*indent}Split {parent_sv}->{child_sv}: Scale {scale}: Not yet complete.")
                time.sleep(1.0)

    logger.info(f"{' '*indent}Split {parent_sv}->{child_sv}: Successfully completed.")


def _determine_incomplete_blocks(server, uuid, instance, corners, scale, parent_sv):
    """
    Helper function for recursive_sv_split_by_grid()
    """
    corner_batches = iter_batches(corners, 20)
    impl_fn = partial(_determine_incomplete_blocks_impl, server, uuid, instance, scale, parent_sv)
    proc = min(16, len(corner_batches))
    if proc == 1:
        proc = 0
    incomplete = compute_parallel(impl_fn, corner_batches, processes=proc)
    return sorted(chain(*incomplete))


def _determine_incomplete_blocks_impl(server, uuid, instance, scale, parent_sv, corner_batch):
    """
    Helper function for recursive_sv_split_by_grid()
    """
    lazyblocks = fetch_labelmap_specificblocks(server, uuid, instance, corner_batch, scale,
                                               supervoxels=True, format='callable-blocks')
    incomplete = []
    for corner, lb in lazyblocks.items():
        if parent_sv in lb().reshape(-1):
            incomplete.append(tuple(corner))
    return incomplete
