from collections import namedtuple

import numpy as np
import pandas as pd
from numba import jit

from .. import sanitize_server, default_dvid_session

# The labelops_pb2 file was generated with the following commands:
# $ cd neuclease/dvid
# $ protoc --python_out=. labelops.proto
# $ sed -i '' s/labelops_pb2/neuclease.dvid.labelops_pb2/g labelops_pb2.py
from .labelops_pb2 import LabelIndex

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
