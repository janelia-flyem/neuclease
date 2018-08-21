import pytest

import numpy as np
import pandas as pd

from neuclease.focused.hotknife import contingency_table, compute_cc, region_coordinates, match_overlaps


def test_contingency_table():
    left =  np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4]])
    right = np.array([[0,0,5,5,5,6,6,6,7,7,7,8,0]])
    
    table = contingency_table(left, right)
    assert isinstance(table, pd.Series)
    assert (np.array(table.index.values.tolist()) == [(0,0), (0,5), (1,5), (1,6), (2,6), (2,7), (3,7), (3,8), (4,0)]).all()
    assert (table == [2,1,2,1,2,1,2,1,1]).all()


def test_compute_cc():
    img = [[0,0,1,1,0,0,3,3,3,0,5]]
    img = np.asarray(img, dtype=np.uint64)
    cc, mapping = compute_cc(img)
    assert (cc[0, 2:4] == 1).all()
    assert (cc[0, 6:9] == 2).all()
    assert (cc[0, 10] == 3)

    assert isinstance(mapping, pd.Series)
    assert mapping.index.name == 'cc'
    assert mapping.name == 'orig'
    assert mapping.loc[0] == 0
    assert mapping.loc[1] == 1
    assert mapping.loc[2] == 3
    assert mapping.loc[3] == 5


def test_region_coordinates():
    _ = 0
    
    #       0 1 2 3 4 5 6 7 8 9
    img = [[_,_,_,_,_,_,_,_,_,_], # 0
           [_,_,1,_,_,_,2,_,_,_], # 1
           [_,1,1,1,_,_,2,_,_,_], # 2
           [_,_,1,_,_,_,2,_,_,_], # 3
           [_,_,_,_,_,_,_,_,_,_], # 4
           [_,3,3,3,_,_,_,4,_,_], # 5
           [_,_,_,_,_,_,4,_,4,_], # 6
           [_,_,_,_,_,_,_,4,_,_]] # 7

    img = np.array(img, np.uint32)
    coords = region_coordinates(img, [4,3,2,1]) # backwards, just cuz we can

    assert coords.shape == (4,2)
    assert (coords[-1] == [2,2]).all()
    assert (coords[-2] == [2,6]).all()
    assert (coords[-3] == [5,2]).all()
    
    # For object 4, the center of mass is not labeled,
    # so coord should revert to first scan-order pixel
    assert (coords[-4] == [5,7]).all()
    

def test_match_overlaps_DEFAULT():
    left =  np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4]], np.uint64)
    right = np.array([[0,0,5,5,5,1,2,2,7,7,7,8,0]], np.uint64)
 
    matched_edges = match_overlaps(left, right, crossover_filter=None)

    expected_edges = [[1, 5],
                      [1, 1],
                      [2, 2],
                      [2, 7],
                      [3, 7],
                      [3, 8]]
    
    overlaps = [2, 1, 2, 1, 2, 1]

    assert (matched_edges[['left', 'right']].values == expected_edges).all()
    assert (matched_edges['overlap'].values == overlaps).all()


def test_match_overlaps_no_crossover_identities():
    left =  np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4]], np.uint64)
    right = np.array([[0,0,5,5,5,1,2,2,7,7,7,8,0]], np.uint64)
 
    matched_edges = match_overlaps(left, right, crossover_filter='exclude-identities')
     
    expected_edges = [[1, 5],
                      #[1, 1],
                      #[2, 2],
                      [2, 7],
                      [3, 7],
                      [3, 8]]
 
    overlaps = [2, 1, 2, 1]
 
    assert (matched_edges[['left', 'right']].values == expected_edges).all()
    assert (matched_edges['overlap'].values == overlaps).all()
 
 
def test_match_overlaps_no_crossover_mentions():
    left =  np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4]], np.uint64)
    right = np.array([[0,0,5,5,5,1,2,2,7,7,7,8,0]], np.uint64)
 
    matched_edges = match_overlaps(left, right, crossover_filter='exclude-all')

    expected_edges = [#[1, 5],
                      #[1, 1],
                      #[2, 2],
                      #[2, 7],
                      [3, 7],
                      [3, 8]]
 
    overlaps = [2, 1]

    assert (matched_edges[['left', 'right']].values == expected_edges).all()
    assert (matched_edges['overlap'].values == overlaps).all()
 
 
def test_match_overlaps_favorites_only():
    left =  np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4]], np.uint64)
    right = np.array([[0,0,5,5,5,1,2,2,7,7,7,8,0]], np.uint64)
 
    matched_edges = match_overlaps(left, right, crossover_filter=None, match_filter='favorites')
     
    expected_edges = [[1, 5],
                      [1, 1],
                      [2, 2],
                      #[2, 7], # nobody's favorite
                      [3, 7],
                      [3, 8]]
 
    overlaps = [2, 1, 2, 2, 1]

    assert (matched_edges[['left', 'right']].values == expected_edges).all()
    assert (matched_edges['overlap'].values == overlaps).all()


def test_match_overlaps_mutual_favorites_only():
    left =  np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4]], np.uint64)
    right = np.array([[0,0,5,5,5,1,2,2,7,7,7,8,0]], np.uint64)

    matched_edges = match_overlaps(left, right, crossover_filter=None, match_filter='mutual-favorites')
    
    expected_edges = [[1, 5],
                      #[1, 1], # Not 1's favorite
                      [2, 2],
                      #[2, 7], # nobody's favorite
                      [3, 7],
                      #[3, 8], # Not 3's favorite
                      ]

    overlaps = [2, 2, 2]

    assert (matched_edges[['left', 'right']].values == expected_edges).all()
    assert (matched_edges['overlap'].values == overlaps).all()


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_hotknife'])
