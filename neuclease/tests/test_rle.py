import pytest

import numpy as np
import scipy.ndimage

from neuclease.dvid.rle import (runlength_encode_to_ranges, runlength_decode_from_ranges,
                                runlength_encode_to_lengths, runlength_decode_from_lengths,
                                rle_box_dilation)

@pytest.fixture
def sparse_object():
    
    _ = 0
    a = [[[_,_,_,_,_],
          [_,1,1,1,_],
          [1,1,_,1,1],
          [_,1,_,_,_],
          [_,1,1,1,1]]]
    
    a = np.array(a)
    coords = np.transpose(a.nonzero()).astype(np.int32)
    ranges = np.array([[0, 1, 1, 3],
                       [0, 2, 0, 1],
                       [0, 2, 3, 4],
                       [0, 3, 1, 1],
                       [0, 4, 1, 4]]).astype(np.int32)

    lengths = 1 + ranges[:,3] - ranges[:,2]
    lengths_table = ranges.copy()
    lengths_table[:, 3] = lengths
    
    return coords, ranges, lengths_table

def test_runlength_encode_to_ranges(sparse_object):
    coords, ranges, _lengths_table = sparse_object
    assert (runlength_encode_to_ranges(coords) == ranges).all()


def test_runlength_encode_to_lengths(sparse_object):
    coords, _ranges, lengths_table = sparse_object
    assert (runlength_encode_to_lengths(coords) == lengths_table).all()


def test_runlength_decode_from_ranges(sparse_object):
    coords, ranges, _lengths_table = sparse_object
    assert (runlength_decode_from_ranges(ranges) == coords).all()


def test_runlength_decode_from_lengths(sparse_object):
    coords, _ranges, lengths_table = sparse_object

    # Copies necessary because the function requires contiguous input
    assert (runlength_decode_from_lengths( lengths_table[:, :3].copy(),
                                           lengths_table[:, 3].copy()) == coords ).all()


def test_rle_box_dilation(sparse_object):
    coords, _ranges, lengths_table = sparse_object
    assert coords.dtype == np.int32
    assert lengths_table.dtype == np.int32
    
    # offset by 3 to embed in larger image
    a = np.zeros((7,11,11), bool)
    a[(*(coords + 3).transpose(),)] = 1
    
    # Original voxels are located at offset (3,3,3)
    assert (np.transpose(a[3:4, 3:-3, 3:-3].nonzero()) == coords).all()
    
    dilated_start_coords, dilated_lengths = rle_box_dilation( lengths_table[:, :3].copy(),
                                                              lengths_table[:, 3].copy(),
                                                              radius=2 )
    
    dilated_table = np.zeros((len(dilated_start_coords), 4), np.int32)
    dilated_table[:, :3] = dilated_start_coords
    dilated_table[:, 3] = dilated_lengths

#     print()
#     print(lengths_table)
#     print()
#     print(dilated_table)
    
    
    decoded_dilated_coords = runlength_decode_from_lengths( dilated_start_coords.copy(),
                                                            dilated_lengths.copy() )
    
    dilated_vol = scipy.ndimage.morphology.binary_dilation(a, np.ones((5, 5, 5), bool))
    dilated_coords = np.transpose(dilated_vol.nonzero())
    
#     redilated_table = runlength_encode_to_lengths(dilated_coords.astype(np.int32)) - (3,3,3,0)
#     print()
#     print(redilated_table)
    
    assert (decoded_dilated_coords == (dilated_coords - 3)).all()

    

if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_rle'])
