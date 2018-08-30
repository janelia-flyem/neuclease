import pytest

import numpy as np
import pandas as pd

from neuclease.merge_table import compute_body_sizes

def test_compute_body_sizes():
    sv_sizes = [(1,10),
                (2,20),
                (3,30),
                (4,40),
                (5,50)]
    sv_sizes = np.array(sv_sizes, dtype=np.uint64)
    sv_sizes = pd.Series(index=sv_sizes[:,0], data=sv_sizes[:,1])
    
    mapping = [(1,11),
               (2,11),
               (3,12),
               (4,12)]
    
    mapping = np.asarray(mapping, dtype=np.uint64)
    mapping = pd.Series(index=mapping[:,0], data=mapping[:,1])

    # Without singleton (sv 5 is not in the mapping)
    body_sizes = compute_body_sizes(sv_sizes, mapping, False)
    assert set(body_sizes.index) == {11,12}
    assert body_sizes.loc[11, 'voxel_count'] == 30
    assert body_sizes.loc[12, 'voxel_count'] == 70
    assert body_sizes.loc[11, 'sv_count'] == 2
    assert body_sizes.loc[12, 'sv_count'] == 2
    
    # With singleton (sv 5 is not in the mapping)
    body_sizes = compute_body_sizes(sv_sizes, mapping, True)
    assert set(body_sizes.index) == {11,12,5}
    assert body_sizes.loc[11, 'voxel_count'] == 30
    assert body_sizes.loc[12, 'voxel_count'] == 70
    assert body_sizes.loc[11, 'sv_count'] == 2
    assert body_sizes.loc[12, 'sv_count'] == 2

    # singleton supervoxel was added
    assert body_sizes.loc[5, 'voxel_count'] == 50
    assert body_sizes.loc[5, 'sv_count'] == 1

if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_merge_table'])
