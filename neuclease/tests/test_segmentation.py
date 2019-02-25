from itertools import product

import pytest
import numpy as np
import pandas as pd

from dvidutils import LabelMapper
from neuclease.util import mask_for_labels, apply_mask_for_labels, contingency_table, split_disconnected_bodies

def test_mask_for_labels():
    volume = [[0,2,3], [4,5,0]]

    volume = np.asarray(volume)

    masked_volume = mask_for_labels(volume, {2,5,9})
    expected = [[0,1,0], [0,1,0]]
    assert (masked_volume == expected).all()
    


def test_apply_mask_for_labels():
    volume = [[0,2,3], [4,5,0]]

    volume = np.asarray(volume)

    masked_volume = apply_mask_for_labels(volume, {2,5,9})
    expected = [[0,2,0], [0,5,0]]
    assert (masked_volume == expected).all()
    
    apply_mask_for_labels(volume, {2,5,9}, inplace=True)
    expected = [[0,2,0], [0,5,0]]
    assert (volume == expected).all()
    

def test_contingency_table_simple():
    left =  np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4]])
    right = np.array([[0,0,5,5,5,6,6,6,7,7,7,8,0]])
    
    table = contingency_table(left, right)
    assert isinstance(table, pd.Series)
    assert (np.array(table.index.values.tolist()) == [(0,0), (0,5), (1,5), (1,6), (2,6), (2,7), (3,7), (3,8), (4,0)]).all()
    assert (table == [2,1,2,1,2,1,2,1,1]).all()


def test_contingency_table_random():
    a = np.random.randint(5,10, size=(20,20), dtype=np.uint32)
    b = np.random.randint(10,15, size=(20,20), dtype=np.uint32)
    table = contingency_table(a,b)

    for (val_a, val_b) in product(range(5,10), range(10,15)):
        expected_overlap = ((a == val_a) & (b == val_b)).sum()
        rows = pd.DataFrame(table).query('left == @val_a and right == @val_b')
        if expected_overlap == 0:
            assert len(rows) == 0
        else:
            assert len(rows) == 1
            assert rows['voxel_count'].iloc[0] == expected_overlap


def test_split_disconnected_bodies():

    _ = 2 # for readability in the array below

    # Note that we multiply these by 10 for this test!
    orig = [[ 1,1,1,1,_,_,3,3,3,3 ],
            [ 1,1,1,1,_,_,3,3,3,3 ],
            [ 1,1,1,1,_,_,3,3,3,3 ],
            [ 1,1,1,1,_,_,3,3,3,3 ],
            [ _,_,_,_,_,_,_,_,_,_ ],
            [ 0,0,_,_,4,4,_,_,0,0 ],  # Note that the zeros here will not be touched.
            [ _,_,_,_,4,4,_,_,_,_ ],
            [ 1,1,1,_,_,_,_,3,3,3 ],
            [ 1,1,1,_,1,1,_,3,3,3 ],
            [ 1,1,1,_,1,1,_,3,3,3 ]]
    
    orig = np.array(orig).astype(np.uint64)
    orig *= 10
    
    split, mapping, split_unique = split_disconnected_bodies(orig)
    
    # New IDs are generated starting after the original max value
    assert (split_unique == [0,10,20,30,40,41,42,43]).all()
    
    assert ((orig == 20) == (split == 20)).all(), \
        "Label 2 is a single component and therefore should remain untouched in the output"

    assert ((orig == 40) == (split == 40)).all(), \
        "Label 4 is a single component and therefore should remain untouched in the output"
    
    assert (split[:4,:4] == 10).all(), \
        "The largest segment in each split object is supposed to keep it's label"
    assert (split[:4,-4:] == 30).all(), \
        "The largest segment in each split object is supposed to keep it's label"

    lower_left_label = split[-1,1]
    lower_right_label = split[-1,-1]
    bottom_center_label = split[-1,5]

    assert lower_left_label != 10, "Split object was not relabeled"
    assert bottom_center_label != 10, "Split object was not relabeled"
    assert lower_right_label != 30, "Split object was not relabeled"

    assert lower_left_label in (41,42,43), "New labels are supposed to be consecutive with the old"
    assert lower_right_label in (41,42,43), "New labels are supposed to be consecutive with the old"
    assert bottom_center_label in (41,42,43), "New labels are supposed to be consecutive with the old"

    assert (split[-3:,:3] == lower_left_label).all()
    assert (split[-3:,-3:] == lower_right_label).all()
    assert (split[-2:,4:6] == bottom_center_label).all()

    assert set(mapping.keys()) == set([10,30,41,42,43]), f"mapping: {mapping}"

    mapper = LabelMapper(np.fromiter(mapping.keys(), np.uint64), np.fromiter(mapping.values(), np.uint64))
    assert (mapper.apply(split, True) == orig).all(), \
        "Applying mapping to the relabeled image did not recreate the original image."


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_segmentation'])
