import pytest
import numpy as np
from neuclease.util import Grid, boxes_from_grid, slabs_from_box


def test_boxes_from_grid_0():
    # Simple: bounding_box starts at zero, no offset
    grid = Grid( (10,20), (0,0) )
    bounding_box = [(0,0), (100,300)]
    boxes = np.array(list(boxes_from_grid(bounding_box, grid)))
    assert boxes.shape == (np.prod( np.array(bounding_box[1]) / grid.block_shape ), 2, 2)
    assert (boxes % grid.block_shape == 0).all()
    assert (boxes[:, 1, :] - boxes[:, 0, :] == grid.block_shape).all()


def test_boxes_from_grid_1():
    # Set a non-aligned bounding box
    grid = Grid( (10,20), (0,0) )
    bounding_box = np.array([(15,30), (95,290)])
    
    aligned_bounding_box = (  bounding_box[0]                          // grid.block_shape * grid.block_shape,
                             (bounding_box[1] + grid.block_shape - 1 ) // grid.block_shape * grid.block_shape )
    
    algined_bb_shape = aligned_bounding_box[1] - aligned_bounding_box[0]
    
    boxes = np.array(list(boxes_from_grid(bounding_box, grid)))
    assert boxes.shape == (np.prod( algined_bb_shape / grid.block_shape ), 2, 2)
    assert (boxes % grid.block_shape == 0).all()
    assert (boxes[:, 1, :] - boxes[:, 0, :] == grid.block_shape).all()


def test_boxes_from_grid_2():
    # Use a grid offset
    grid = Grid( (10,20), (2,3) )
    bounding_box = np.array([(5,10), (95,290)])
    
    aligned_bounding_box = (  bounding_box[0]                          // grid.block_shape * grid.block_shape,
                             (bounding_box[1] + grid.block_shape - 1 ) // grid.block_shape * grid.block_shape )
    
    aligned_bb_shape = aligned_bounding_box[1] - aligned_bounding_box[0]
    
    boxes = np.array(list(boxes_from_grid(bounding_box, grid)))
    assert boxes.shape == (np.prod( aligned_bb_shape / grid.block_shape ), 2, 2)
    
    # Boxes should be offset by grid.offset.
    assert ((boxes - grid.offset) % grid.block_shape == 0).all()
    assert (boxes[:, 1, :] - boxes[:, 0, :] == grid.block_shape).all()


def test_slabs_for_box_simple():
    box = [(0,0,0), (100, 200, 300)]
    slabs = list(slabs_from_box( box, 10 ))
    expected = np.array([((a,0,0), (b,200,300)) for (a,b) in zip(range(0,100,10), range(10,101,10))])
    assert (np.array(slabs) == expected).all()

def test_slabs_for_box_scaled():
    box = [(0,0,0), (100, 200, 300)]
    slabs = list(slabs_from_box( box, 10, scale=1 ))
    expected = np.array([((a,0,0), (b,100,150)) for (a,b) in zip(range(0,50,10), range(10,51,10))])
    assert (np.array(slabs) == expected).all()

def test_slabs_for_box_scaled_nonaligned():
    box = [(5,7,8), (99, 200, 299)]
    slabs = list(slabs_from_box( box, 10, scale=1 ))
    
    first_slab = ((2,3,4), (10,100,150))
    expected = [first_slab] + [((a,3,4), (b,100,150)) for (a,b) in zip(range(10,50,10), range(20,51,10))]
    
    assert (np.array(slabs) == expected).all()

def test_slabs_for_box_scaled_nonaligned_round_in():
    box = [(5,7,8), (99, 200, 299)]
    slabs = list(slabs_from_box( box, 10, scale=1, scaling_policy='round-in' ))
    
    first_slab = ((3,4,4), (10,100,149))
    last_slab = ((40,4,4), (49,100,149))
    expected = [first_slab] + [((a,4,4), (b,100,149)) for (a,b) in zip(range(10,40,10), range(20,50,10))] + [last_slab]
    
    assert (np.array(slabs) == expected).all()


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_grid'])
