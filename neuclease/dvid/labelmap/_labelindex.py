import warnings
from functools import partial
from collections import namedtuple
from collections.abc import Iterable, Collection

import numpy as np
import pandas as pd
from numba import jit
from requests import HTTPError

from ...util import tqdm_proxy, compute_parallel, iter_batches
from .. import dvid_api_wrapper

# $ protoc --python_out=. neuclease/dvid/labelmap/labelops.proto
from .labelops_pb2 import LabelIndex, LabelIndices
from . import fetch_mapping

PandasLabelIndex = namedtuple("PandasLabelIndex", "blocks label last_mutid last_mod_time last_mod_user")


@dvid_api_wrapper
def fetch_labelindex(server, uuid, instance, label, format='protobuf', *, missing='raise', session=None):
    """
    Fetch the LabelIndex for the given label ID from DVID,
    and return it as the native protobuf structure, or as a more-convenient
    structure that encodes all block counts into a single big DataFrame.
    (See convert_labelindex_to_pandas())

    Args:
        server:
            dvid server, e.g. 'emdata4:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid labelmap instance name, e.g. 'segmentation'

        label:
            A body ID

        format:
            How to return the data. Choices are:
              - ``raw`` (raw bytes of the DVID response, i.e. the raw bytes of the protobuf structure)
              - ``protobuf`` (A ``LabelIndex`` protobuf structure.)
              - ``pandas`` (See description in ``convert_labelindex_to_pandas()``)

            The 'pandas' format is slowest, but is most convenient to analyze.

        missing:
            What to do if the label doesn't exist.
            By default, raise an exception, but if 'return-None' is given,
            then return None instead of raising an error.

    Returns:
        See 'format' description.
    """
    assert format in ('protobuf', 'pandas', 'raw')
    assert missing in ('raise', 'return-None')

    try:
        r = session.get(f'{server}/api/node/{uuid}/{instance}/index/{label}')
        r.raise_for_status()
    except HTTPError as ex:
        if ex.response.status_code == 404 and missing == 'return-None':
            return None
        raise

    if format == 'raw':
        return r.content

    labelindex = LabelIndex()
    labelindex.ParseFromString(r.content)

    if format == 'protobuf':
        return labelindex
    elif format == 'pandas':
        return convert_labelindex_to_pandas(labelindex)


def fetch_labelindices_parallel(server, uuid, instance, labels, *, format='single-dataframe', processes=16):
    """
    Like fetch_labelindices(), but fetches and converts format in parallel.

    Note: The 'single-dataframe' format simply omits any
    """
    assert format in ('list-of-protobuf', 'pandas', 'single-dataframe')
    fmt = format
    if fmt == 'single-dataframe':
        fmt = 'pandas'

    _fetch = partial(fetch_labelindex, server, uuid, instance, format=fmt, missing='return-None')
    indexes = compute_parallel(_fetch, labels, processes=processes)

    missing = [label
               for (label, index) in zip(labels, indexes)
               if index is None]
    if missing:
        warnings.warn(f"Could not find an index for bodies {missing}")

    indexes = [*filter(None, indexes)]
    if format != 'single-dataframe':
        return indexes

    if not indexes:
        return pd.DataFrame([], columns=[*'zyx', 'sv', 'count', 'label'])

    for li in indexes:
        li.blocks['label'] = li.label

    return pd.concat([li.blocks for li in indexes], ignore_index=True)


@dvid_api_wrapper
def fetch_labelindices(server, uuid, instance, labels, *, format='protobuf', session=None):
    """
    Fetch a batch of label indexes via a single call to dvid.

    Note:
        No error is raised for bodies which do not exist.
        Instead, an empty labelindex is returned.

    Args:
        labels:
            A list of label IDs to fetch indices for.

    Returns:
        Depends on the 'format' argument:
            'raw':
                The raw bytes from DVID.  Useful if you're just planning to
                copy them to another dvid server.
            'protobuf':
                A LabelIndices (protobuf) object containing all the
                requested LabelIndex (protobuf) objects.
            'list-of-protobuf':
                A list of LabelIndex (protobuf) objects.
            'pandas':
                A list of PandasLabelIndex (tuple) objects,
                which each contain a DataFrame representation of the labelindex.
            'single-dataframe':
                All of the label indexes will be converted to pandas dataframes and
                concatenated into a single dataframe, with an extra column for 'label'.
    """
    assert format in ('raw', 'protobuf', 'list-of-protobuf', 'pandas', 'single-dataframe')
    if isinstance(labels, (np.ndarray, pd.Series)):
        labels = labels.tolist()
    elif not isinstance(labels, list):
        labels = list(labels)

    endpoint = f'{server}/api/node/{uuid}/{instance}/indices'
    r = session.get(endpoint, json=labels)
    r.raise_for_status()

    if format == 'raw':
        return r.content

    labelindices = LabelIndices()
    labelindices.ParseFromString(r.content)

    with warnings.catch_warnings():
        warnings.filterwarnings("once", message=".*contains no block list.*")

        if format == 'protobuf':
            return labelindices
        if format == 'list-of-protobuf':
            return list(labelindices.indices)
        if format == 'pandas':
            return list(map(convert_labelindex_to_pandas, labelindices.indices))
        if format == 'single-dataframe':
            dfs = []
            for idx in labelindices.indices:
                df = convert_labelindex_to_pandas(idx)
                df.blocks['label'] = idx.label
                dfs.append(df.blocks)
            return pd.concat(dfs, ignore_index=True)


@dvid_api_wrapper
def post_labelindex(server, uuid, instance, label, proto_index, *, session=None):
    """
    Post a protobuf LabelIndex object for the given
    label to the specified DVID labelmap instance.

    Args:
        label:
            The label ID corresponding to this index.
            Must match the internal label field within the proto_index structure.
        proto_index:
            A protobuf labelops_pb2.LabelIndex
            Optionally, you may pre-serialize it and provide it as bytes instead.
    """
    payload = None
    assert isinstance(proto_index, (bytes, LabelIndex))
    if isinstance(proto_index, LabelIndex):
        assert proto_index.label == label
        payload = proto_index.SerializeToString()
    elif isinstance(proto_index, bytes):
        payload = proto_index

    r = session.post(f'{server}/api/node/{uuid}/{instance}/index/{label}', data=payload)
    r.raise_for_status()

@dvid_api_wrapper
def post_labelindices(server, uuid, instance, indices, *, session=None):
    """
    Send a batch (list) of LabelIndex objects to dvid.

    Args:
        indices:
            One of the following:
              - A list of LabelIndex (protobuf) objects
              - A pre-loaded LabelIndices protobuf object
              - A list of pre-serialized LabelIndex objects (as bytes)
              - A single pre-serializsed LabelIndices object (as bytes)
    """
    if isinstance(indices, Collection) and len(indices) == 0:
        # This can happen when tombstone_mode == 'only'
        # and a label contained only one supervoxel.
        return

    if not isinstance(indices, (LabelIndices, Collection)):
        assert isinstance(indices, Iterable)
        indices = list(indices)

    payload = None
    if isinstance(indices, bytes):
        payload = indices
    elif isinstance(indices, Collection) and isinstance(indices[0], bytes):
        payload = b''.join(indices)
    elif isinstance(indices, Collection) and isinstance(indices[0], LabelIndex):
        label_indices = LabelIndices()
        label_indices.indices.extend(indices)
        payload = label_indices.SerializeToString()
    elif isinstance(indices, LabelIndices):
        if len(indices.indices) == 0:
            # This can happen when tombstone_mode == 'only'
            # and a label contained only one supervoxel.
            return
        payload = indices.SerializeToString()
    else:
        msg = f"Unhandled input type for posting label indices: {type(indices)}"
        raise AssertionError(msg)

    endpoint = f'{server}/api/node/{uuid}/{instance}/indices'
    r = session.post(endpoint, data=payload)
    r.raise_for_status()


# Deprecated name
post_labelindex_batch = post_labelindices


@dvid_api_wrapper
def delete_labelindices(server, uuid, instance, bodies, *, session=None):
    """
    Delete the label indexes for a list of bodies.
    DVID supports deletion en masse via POST of empty label index prototbuf structures.
    (See the DVID docs for POST .../index and POST .../indices)
    """
    index_list = []
    for body in bodies:
        li = LabelIndex()
        li.label = int(body)
        index_list.append(li)

    indices = LabelIndices()
    indices.indices.extend(index_list)
    payload = indices.SerializeToString()

    endpoint = f'{server}/api/node/{uuid}/{instance}/indices'
    r = session.post(endpoint, data=payload)
    r.raise_for_status()


def copy_labelindices(src_triple, dest_triple, labels, *, batch_size=10_000, threads=None, processes=None):
    """
    Copy many labelindexes from one dvid repo to another, in batches.
    
    Note:
        There is no special error handling in this function.
        If any single labelindex cannot be fetched or posted, the entire copy fails.
    
    Args:
        src_triple:
            tuple (server, uuid, instance) to copy from
        
        dest_triple:
            tuple (server, uuid, instance) to copy to
        
        labels:
            Array of label IDs indicating which indexes to copy.
            The given labels must be body IDs, not supervoxel IDs
            (There are no independent label indexes for supervoxels
            that are part of a body.)
        
        batch_size:
            The labels will be sent in batches, optionally from multiple threads/processes.
            This batch_size indicates how many indexes will be copied per batch.
        
        threads:
            If provided, use multiple threads to fetch/post the indexes.
        
        processes:
            If provided, use multiple processes to fetch/post the indexes.
    """
    labels = np.asarray(labels)
    label_batches = []
    for batch_start in range(0, len(labels), batch_size):
        batch = labels[batch_start:batch_start+batch_size]
        label_batches.append(batch)

    if threads is None and processes is None:
        for batch in tqdm_proxy(label_batches, leave=True):
            _copy_labelindex_batch(src_triple, dest_triple, batch)
    else:
        assert not threads or not processes, "Choose either threads or processes (not both)"
        f = partial(_copy_labelindex_batch, src_triple, dest_triple)
        compute_parallel(f, label_batches, 1, threads, processes, ordered=False, leave_progress=True)


def _copy_labelindex_batch(src_triple, dest_triple, labels_batch):
    """
    Helper for copy_labelindexes(), above.
    Defined here at the module top-level to allow it to be
    pickled when using multiprocessing.
    """
    indexes_batch = fetch_labelindices(*src_triple, labels_batch)
    post_labelindices(*dest_triple, indexes_batch)


def convert_labelindex_to_pandas(labelindex):
    try:
        return _convert_labelindex_to_pandas(labelindex)
    except Exception as ex:
        ex2 = RuntimeError(f"Can't convert labelindex for body {labelindex.label}")
        ex2.labelindex = labelindex
        raise ex2 from ex


def _convert_labelindex_to_pandas(labelindex):
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
        That is, all coordinates in the table are multiples of 64.
    """
    encoded_block_coords = np.fromiter(labelindex.blocks.keys(), np.uint64, len(labelindex.blocks))
    coords_zyx = decode_labelindex_blocks(encoded_block_coords)

    block_svs = []
    block_counts = []
    block_coords = []

    # Convert each block's data into arrays
    for coord_zyx, sv_counts in zip(coords_zyx, labelindex.blocks.values()):
        svs = np.fromiter(sv_counts.counts.keys(), np.uint64, count=len(sv_counts.counts))
        counts = np.fromiter(sv_counts.counts.values(), np.int64, count=len(sv_counts.counts))

        coord_zyx = np.array(coord_zyx, np.int32)
        coords = np.repeat(coord_zyx[None], len(svs), axis=0)
        #coords = np.lib.stride_tricks.as_strided(coord_zyx, shape=(len(svs), 3), strides=(0,4))
        #coords = np.broadcast_to(coord_zyx, (len(svs),3))

        block_svs.append(svs)
        block_counts.append(counts)
        block_coords.append(coords)

    if len(block_coords) == 0:
        # Before editing this message, see filterwarnings, above.
        warnings.warn(f"LabelIndex for label {labelindex.label} contains no block list!")
        cols = ['z', 'y', 'x', 'sv', 'count']
        blocks_df = pd.DataFrame(columns=cols, dtype=int).astype({'sv': np.uint64})
    else:
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


def create_labelindex(pandas_labelindex):
    """
    Create a protobuf LabelIndex structure from a PandasLabelIndex tuple.
    
    In the PandasLabelIndex tuple, the ``blocks`` member is a pd.DataFrame 
    with the following columns: ['z', 'y', 'x', 'sv', 'count'].
    
    Note that the block coordinates are given in VOXEL units.
    That is, all coordinates in the table are multiples of 64.
    (The coordinates will be converted to DVID block coordinates here
    when they are encoded into the LabelIndex protobuf structure.)
    
    Args:
        pandas_labelindex:
            Instance of PandasLabelIndex (a namedtuple)
        
    Returns:
        neuclease.dvid.labelmap.labelops_pb2.LabelIndex
        (a protobuf structure), suitable for ``post_labelindex()``
    """
    pli = pandas_labelindex
    assert isinstance(pli, PandasLabelIndex)
    labelindex = LabelIndex()
    labelindex.label = pli.label
    labelindex.last_mutid = pli.last_mutid
    labelindex.last_mod_time = pli.last_mod_time
    labelindex.last_mod_user = pli.last_mod_user
    
    assert (pli.blocks.columns == ['z', 'y', 'x', 'sv', 'count']).all()

    block_ids = encode_block_coords(pli.blocks[['z', 'y', 'x']].values)
    pli.blocks['block_id'] = block_ids
    
    for block_id, df in pli.blocks.groupby('block_id'):
        labelindex.blocks[block_id].counts.update( zip(df['sv'].values, df['count'].values) )
    
    del pli.blocks['block_id']
    return labelindex


@dvid_api_wrapper
def fetch_sizes_via_labelindex(server, uuid, instance, labels, supervoxels=False, *, batch_size=None, threads=None, processes=None, session=None):
    """
    Equivalent to fetch_sizes(), but uses the raw /labelindex endpoint
    to obtain the sizes, rather than requesting the sizes from dvid.

    In a single thread, this will be slower than simply callying fetch_sizes(),
    but if you have more CPU cores than DVID does (or you want to save DVID a little bit of CPU load),
    then you can use this function, which will compute the sizes from the labelindexes without
    requiring DVID to do it.  Perhaps this is most useful when supervoxels=True.

    The disadvantage is that DVID will have to send the entire label index to the client,
    so the network connection will be more heavily used.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid labelmap instance name, e.g. 'segmentation'

        labels:
            A label ID or a list of label IDs, which will be interpreted as either
            body labels or supervoxel labels depending on the value of ``supervoxels``.

        supervoxels:
            If True, interpret the given labels are supervoxel IDs, otherwise body IDs.
            Note, if supervoxels=True in conjunction with threads (or processes),
            there is a chance that some labelindexes will be fetched more than once
            (if the supervoxels for a particular body happen to span across multiple batches).

        batch_size:
            If using threads or processes, the labels will be analyzed in batches.
            This parameter specifies the size of the batches (in number of labels).

        threads:
            Integer. If provided, use a thread pool to fetch labelindex batches in parallel.

        processes:
            Integer. If provided, use a process pool to fetch labelindex batches in parallel.

    Returns:
        pd.Series, indexed by label (named either 'body' or 'sv',
        depending on the value of ``supervoxels``).
    """
    if batch_size is None:
        assert threads is None and processes is None, \
            "Specify a batch size or don't use multithreading"
        sizes = _fetch_sizes_via_labelindex(server, uuid, instance, labels, supervoxels=supervoxels, session=session)
        return sizes.loc[labels]

    if threads is None and processes is None:
        threads = 1

    f = partial(_fetch_sizes_via_labelindex, server, uuid, instance, supervoxels=supervoxels)
    batches = iter_batches(pd.unique(labels), batch_size)
    batch_sizes = compute_parallel(f, batches, 1, threads, processes, ordered=False, leave_progress=True)
    sizes = pd.concat(batch_sizes)
    return sizes.loc[labels]


def _fetch_sizes_via_labelindex(server, uuid, instance, labels, supervoxels=False, *, session=None):
    """
    Helper for fetch_sizes_via_labelindex()
    """
    if supervoxels:
        mapping = fetch_mapping(server, uuid, instance, labels, as_series=True)
        bodies = pd.unique(mapping.values)
        indices = fetch_labelindices(server, uuid, instance, bodies, format='pandas')
        df = pd.concat([index.blocks for index in indices])

        _supervoxel_set = set(labels)  # noqa
        df = df.query('sv in @_supervoxel_set')
        sizes = df.groupby('sv')['count'].sum()
        sizes.index = sizes.index.astype(np.uint64)
        sizes.name = 'size'

        # Non-existent supervoxels are given 0 size
        if (mapping == 0).any():
            missing_svs = mapping[mapping == 0].index
            missing_sizes = pd.Series(0, index=missing_svs, name='size')
            sizes = pd.concat((sizes, missing_sizes))

        return sizes
    else:
        labelindices = fetch_labelindices(server, uuid, instance, labels, format='protobuf')

        bodies = []
        sizes = []
        for index in labelindices.indices:
            bodies.append(index.label)
            block_sums = (sum(block_svs.counts.values()) for block_svs in index.blocks.values())
            total_sum = sum(block_sums)
            sizes.append(total_sum)

        bodies = np.fromiter(bodies, np.uint64)
        sizes = pd.Series(sizes, index=bodies, name='size')
        sizes.index.name = 'body'
        return sizes


@dvid_api_wrapper
def fetch_sparsevol_coarse_via_labelindex(server, uuid, instance, labels, supervoxels=False, *, method='pandas', session=None):
    """
    Equivalent to fetch_sparsevol_coarse, but uses the raw /labelindex endpoint
    to obtain the coordinate list, rather than requesting sparsevol RLEs from dvid.
    
    You can provide a list of labels to this function,
    and the output will include all blocks that any of the given labels intersect.

    When fetching coarse sparsevols for a list of supervoxels, some of which may
    share the same parent body, this function efficiently fetches each body's
    labelindex only once.
    
    This method of fetching coarse sparsevols minimizes workload on DVID,
    but requires more work on the client side (about 5x more time if not using grouped supervoxels).
    It is well suited for fetching thousands or millions of coarse sparsevols
    in a cluster-computing workflow, in which DVID is a bottleneck,
    and you have more than 5 workers.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid labelmap instance name, e.g. 'segmentation'

        labels:
            A label ID or a list of label IDs, which will be interpreted as either
            body labels or supervoxel labels depending on the value of ``supervoxels``.
        
        supervoxels:
            If True, interpret the given labels are supervoxel IDs, otherwise body IDs.
        
        method:
            This function can extract the block IDs directly from
            protobuf (using Python set operations), or by converting to pandas first.
            This option is here just for testing and performance analysis.
            It may be removed in the future.

    Returns:
        An array of coordinates (at scale 6) of the form:
    
            [[Z,Y,X],
             [Z,Y,X],
             [Z,Y,X],
             ...
            ]
        
        See also: ``fetch_sparsevol_coarse()``
    
    Note:
        The returned coordinates are not necessarily sorted.
    """
    assert method in ('pandas', 'protobuf')
    if np.issubdtype(type(labels), np.integer):
        labels = np.asarray([labels], np.uint64)
    else:
        assert isinstance(labels, Iterable), \
            "Please provide an iterable of labels, or a single label."
        labels = np.asarray(labels, np.uint64)

    block_ids = set()
    if supervoxels:
        bodies = fetch_mapping(server, uuid, instance, labels, as_series=True)
        for body, mapping_df in bodies.reset_index().groupby('body'):
            if body == 0:
                continue
            svs = set(mapping_df['sv'])
            if method == 'pandas':
                labelindex_df = fetch_labelindex(server, uuid, instance, body, 'pandas', session=session).blocks
                coords_zyx = labelindex_df.query('sv in @svs')[['z', 'y', 'x']].drop_duplicates().values
            else:
                labelindex = fetch_labelindex(server, uuid, instance, body, session=session)
                block_ids |= set( block_id for block_id, blockdata in labelindex.blocks.items()
                               if svs & blockdata.counts.keys() )                
                block_ids = np.fromiter(block_ids, np.uint64, len(block_ids))
                coords_zyx = decode_labelindex_blocks(block_ids)
    else:
        for body in labels:
            if method == 'pandas':
                labelindex_df = fetch_labelindex(server, uuid, instance, body, 'pandas', session=session).blocks
                coords_zyx = labelindex_df[['z', 'y', 'x']].drop_duplicates().values
            else:
                labelindex = fetch_labelindex(server, uuid, instance, body, session=session)
                block_ids |= labelindex.blocks.keys()
                block_ids = np.fromiter(block_ids, np.uint64, len(block_ids))
                coords_zyx = decode_labelindex_blocks(block_ids)

    return coords_zyx // (2**6)


def encode_block_coords(coords):
    """
    Encodes a coordinate array into an array of uint64,
    in the format DVID expects.
    
    Args:
        coords:
            2D array with shape (N,3) (each row is Z,Y,X)
            
            Note:
                The coords should be specified in full-resolution units,
                i.e. each coord must be 64-px block-aligned.
                Internally, these will be divided by 64 to generate
                a block ID as DVID expects.

    Returns:
        1D array, dtype uint64, shape (N,)
    """
    coords = np.asarray(coords, np.int32, 'C')
    assert (coords % 64 == 0).all(), \
        "Block coordinates are not 64-px aligned!"

    record_dtype = [('z', np.int32), ('y', np.int32), ('x', np.int32)]
    coord_records = coords.view(record_dtype).reshape(-1)
    return encode_labelindex_block_ids(coord_records)


@jit(nopython=True)
def encode_labelindex_block_ids(coord_records):
    """
    Encodes a coord list (1-D array of structured array elements)
    into an array of uint64, in the format DVID expects.
    
    The dtype of coord_records should be:
    [('z', np.int32), ('y', np.int32), ('x', np.int32)]
    """
    results = np.empty(len(coord_records), np.uint64)
    for i, rec in enumerate(coord_records):
        results[i] = encode_labelindex_block_id(rec)
    return results

@jit(nopython=True)
def encode_labelindex_block_id(coord_record):
    """
    Encodes a coord (structured array of z,y,x)
    into a uint64, in the format DVID expects.
    """
    encoded_block_id = np.uint64(0)
    encoded_block_id |= np.uint64(coord_record.z // 64) << 42
    encoded_block_id |= np.uint64(coord_record.y // 64) << 21
    encoded_block_id |= np.uint64(coord_record.x // 64)
    return encoded_block_id


@jit(nopython=True, nogil=True)
def decode_labelindex_blocks(encoded_blocks):
    """
    Calls decode_labelindex_block() on a 1-D array of encoded coordinates.
    """
    decoded_blocks = np.zeros((len(encoded_blocks), 3), dtype=np.int32)
    for i in range(len(encoded_blocks)):
        encoded = encoded_blocks[i]
        decoded_blocks[i,:] = decode_labelindex_block(encoded)
    return decoded_blocks


@jit(nopython=True, nogil=True)
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
