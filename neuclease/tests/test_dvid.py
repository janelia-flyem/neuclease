"""
Test module for the dvid API wrapper functions defined in neuclease.dvid
"""
import sys
import logging

import pytest
import numpy as np
import pandas as pd

from neuclease.dvid import (fetch_supervoxels_for_body, fetch_supervoxel_sizes_for_body,
                            fetch_label_for_coordinate, fetch_mappings, fetch_mutation_id)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

##
## These tests rely on the global setupfunction 'labelmap_setup',
## defined in conftest.py and used here via pytest magic
##

def test_fetch_supervoxels_for_body(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    supervoxels = fetch_supervoxels_for_body(dvid_server, dvid_repo, 'segmentation', 1)
    assert (supervoxels == [1,2,3,4,5]).all()


def test_fetch_supervoxel_sizes_for_body(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    sv_sizes = fetch_supervoxel_sizes_for_body(dvid_server, dvid_repo, 'segmentation', 1)
    assert isinstance(sv_sizes, pd.Series)
    assert (sv_sizes.index == [1,2,3,4,5]).all()
    assert (sv_sizes == 9).all() # see initialization in conftest.py


def test_fetch_label_for_coordinate(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    label = fetch_label_for_coordinate(dvid_server, dvid_repo, 'segmentation', (0, 1, 3), supervoxels=False)
    assert isinstance(label, np.uint64)
    assert label == 1

    label = fetch_label_for_coordinate(dvid_server, dvid_repo, 'segmentation', (0, 1, 3), supervoxels=True)
    assert isinstance(label, np.uint64)
    assert label == 2


def test_fetch_mappings(labelmap_setup):
    """
    Very BASIC test for the wrapper function for the /mappings DVID API.
    Note: This doesn't test the options related to automatic identity mappings.
    """
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    mapping = fetch_mappings(dvid_server, dvid_repo, 'segmentation')
    assert isinstance(mapping, pd.Series)
    assert mapping.index.name == 'sv'
    assert mapping.name == 'body'
    assert (sorted(mapping.index) == [1,2,3,4,5])
    assert (mapping == 1).all() # see initialization in conftest.py


def test_fetch_mutation_id(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    mut_id = fetch_mutation_id(dvid_server, dvid_repo, 'segmentation', 1)
    assert isinstance(mut_id, int)


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_dvid'])
