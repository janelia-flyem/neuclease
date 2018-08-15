import pytest
import numpy as np

from neuclease.util import Grid, SparseBlockMask, upsample

def test_get_fullres_mask():
    coarse_mask = np.random.randint(2, size=(10,10), dtype=bool)
    full_mask = upsample(coarse_mask, 10)
    sbm = SparseBlockMask(coarse_mask, [(0,0), (100,100)], (10,10))
    
    # Try the exact bounding box
    extracted = sbm.get_fullres_mask([(0,0), (100,100)])
    assert (extracted == full_mask).all()

    # Try a bounding box that exceeds the original mask
    # (excess region should be all zeros)
    extracted = sbm.get_fullres_mask([(10,20), (150,150)])
    assert extracted.shape == (140, 130)
    expected = np.zeros((140, 130), dtype=bool)
    expected[:90,:80] = full_mask[10:, 20:]
    assert (extracted == expected).all()
    

def test_sparse_boxes_NO_OFFSET():
    block_mask = np.zeros((5,6,7), dtype=bool)
    
    block_mask[0, 0, 0:5] = True
    
    block_mask[0, 1, 1:4] = True
    
    block_mask_resolution = 10
    
    # MASK STARTS AT ORIGIN (NO OFFSET)
    mask_box_start = np.array([0,0,0])
    mask_box_stop = mask_box_start + 10*np.array(block_mask.shape)
    
    block_mask_box = (mask_box_start, mask_box_stop)
    brick_grid = Grid( (10,10,30) )
    
    sparse_block_mask = SparseBlockMask( block_mask, block_mask_box, block_mask_resolution )
    logical_boxes = sparse_block_mask.sparse_boxes(brick_grid, return_logical_boxes=True)
    assert (logical_boxes == [[[0, 0,   0], [10, 10, 30]],
                              [[0, 0,  30], [10, 10, 60]],
                              [[0, 10,  0], [10, 20, 30]],
                              [[0, 10, 30], [10, 20, 60]]]).all()
    
    physical_boxes = sparse_block_mask.sparse_boxes(brick_grid, return_logical_boxes=False)
    assert (physical_boxes == [[[0, 0,   0], [10, 10, 30]],
                               [[0, 0,  30], [10, 10, 50]],
                               [[0, 10, 10], [10, 20, 30]],
                               [[0, 10, 30], [10, 20, 40]]]).all()


def test_sparse_boxes_WITH_OFFSET():
    block_mask = np.zeros((5,6,7), dtype=bool)
     
    # since mask offset is 20, this spans 3 bricks (physical: 20-70, logical: 0-90)
    block_mask[0, 0, 0:5] = True
     
    # spans a single brick (physical: 30-60, logical: 30-60)
    block_mask[0, 1, 1:4] = True
     
    block_mask_resolution = 10
     
    # MASK STARTS AT OFFSET
    mask_box_start = np.array([0,10,20])
    mask_box_stop = mask_box_start + 10*np.array(block_mask.shape)
     
    block_mask_box = (mask_box_start, mask_box_stop)
    brick_grid = Grid( (10,10,30), (0,0,0) )
     
    sparse_block_mask = SparseBlockMask( block_mask, block_mask_box, block_mask_resolution )
    logical_boxes = sparse_block_mask.sparse_boxes(brick_grid, return_logical_boxes=True)
    
    assert (logical_boxes == [[[0, 10, 0], [10, 20, 30]],
                              [[0, 10, 30], [10, 20, 60]],
                              [[0, 10, 60], [10, 20, 90]],
                              [[0, 20, 30], [10, 30, 60]]]).all()

    physical_boxes = sparse_block_mask.sparse_boxes(brick_grid, return_logical_boxes=False)

    assert (physical_boxes == [[[0, 10, 20], [10, 20, 30]],
                               [[0, 10, 30], [10, 20, 60]],
                               [[0, 10, 60], [10, 20, 70]],
                               [[0, 20, 30], [10, 30, 60]]]).all() 

    
if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_sparse_block_mask'])
