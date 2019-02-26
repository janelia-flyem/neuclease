from collections import namedtuple

import numpy as np
import pandas as pd
from numba import jit

from .. import dvid_api_wrapper

# $ protoc --python_out=. neuclease/dvid/labelmap/labelops.proto
from .labelops_pb2 import LabelIndex, LabelIndices
from . import fetch_mapping

@dvid_api_wrapper
def fetch_labelindex(server, uuid, instance, label, format='protobuf', *, session=None): # @ReservedAssignment
    """
    Fetch the LabelIndex for the given label ID from DVID,
    and return it as the native protobuf structure, or as a more-convenient
    structure that encodes all block counts into a single big DataFrame.
    (See convert_labelindex_to_pandas())
    
    Note that selecting the 'pandas' format takes ~10x longer.
    """
    assert format in ('protobuf', 'pandas')

    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/index/{label}')
    r.raise_for_status()
    labelindex = LabelIndex()
    labelindex.ParseFromString(r.content)

    if format == 'protobuf':
        return labelindex
    elif format == 'pandas':
        return convert_labelindex_to_pandas(labelindex)


@dvid_api_wrapper
def post_labelindex(server, uuid, instance, label, proto_index, *, session=None):
    """
    Post a protobuf LabelIndex object for the given
    label to the specified DVID labelmap instance.
    """
    assert isinstance(proto_index, LabelIndex)
    assert proto_index.label == label
    
    payload = proto_index.SerializeToString()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/index/{label}', data=payload)
    r.raise_for_status()
    

@dvid_api_wrapper
def post_labelindex_batch(server, uuid, instance, batch_indexes, *, session=None):
    """
    Send a batch (list) of LabelIndex objects to dvid.
    
    Args:
        A list of LabelIndex (protobuf) objects
    """
    label_indices = LabelIndices()
    label_indices.indices.extend(batch_indexes)
    if len(label_indices.indices) == 0:
        # This can happen when tombstone_mode == 'only'
        # and a label contained only one supervoxel.
        return
    payload = label_indices.SerializeToString()

    endpoint = f'{server}/api/node/{uuid}/{instance}/indices'
    r = session.post(endpoint, data=payload)
    r.raise_for_status()


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
        counts = np.fromiter(sv_counts.counts.values(), np.uint32, count=len(sv_counts.counts))

        coord_zyx = np.array(coord_zyx, np.int32)
        coords = np.repeat(coord_zyx[None], len(svs), axis=0)
        #coords = np.lib.stride_tricks.as_strided(coord_zyx, shape=(len(svs), 3), strides=(0,4))
        #coords = np.broadcast_to(coord_zyx, (len(svs),3))
        
        block_svs.append(svs)
        block_counts.append(counts)
        block_coords.append(coords)

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
def fetch_sparsevol_coarse_via_labelindex(server, uuid, instance, label, supervoxels=False, *, session=None):
    """
    Equivalent to fetch_sparsevol_coarse, but uses the raw /labelindex endpoint
    to obtain the coordinate list, rather than requesting sparsevol RLEs from dvid.
    
    This method of fetching coarse sparsevols minimizes workload on DVID,
    but requires more work on the client side (about 5x more time).
    It is well suited for fetching thousands or millions of coarse sparsevols
    in a cluster-computing workflow, in which DVID is a bottleneck,
    and you have more than 5 workers.
    
    Note: The returned coordinates are not necessarily sorted.
    """
    if supervoxels:
        # LabelIndexes are stored by body, so fetch the full label
        # index and find the blocks that contain the supervoxel of interest.
        [body] = fetch_mapping(server, uuid, instance, [label], session=session)
        labelindex = fetch_labelindex(server, uuid, instance, body, session=session)
        filtered_block_ids = (block_id for block_id, blockdata in labelindex.blocks.items()
                              if label in blockdata.counts.keys())
        block_ids = np.fromiter(filtered_block_ids, np.uint64)
    else:
        labelindex = fetch_labelindex(server, uuid, instance, label, session=session)
        block_ids = np.fromiter(labelindex.blocks.keys(), np.uint64, len(labelindex.blocks))

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
