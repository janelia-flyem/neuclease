"""
Test module for the dvid API wrapper functions defined in neuclease.dvid
"""
import sys
import logging

import pytest
import numpy as np
import pandas as pd

from neuclease.dvid import (dvid_api_wrapper, DvidInstanceInfo, fetch_supervoxels_for_body, fetch_supervoxel_sizes_for_body,
                            fetch_label_for_coordinate, fetch_mappings, fetch_complete_mappings, fetch_mutation_id,
                            generate_sample_coordinate, fetch_labelarray_voxels, post_labelarray_blocks, fetch_maxlabel,
                            fetch_raw, post_raw)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

##
## These tests rely on the global setupfunction 'labelmap_setup',
## defined in conftest.py and used here via pytest magic
##

def test_dvid_api_wrapper():
    f = dvid_api_wrapper(lambda server, uuid, instance, x, *, session=None: (server, uuid, instance, x))
    server, uuid, instance, x = f("http://foo", "bar", "baz", 5)
    assert (server, uuid, instance, x) == ('foo', 'bar', 'baz', 5)

    server, uuid, instance, x = f("foo", "bar", "baz", 5)
    assert (server, uuid, instance, x) == ("foo", "bar", "baz", 5)


def test_fetch_maxlabel(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    maxlabel = fetch_maxlabel(dvid_server, dvid_repo, 'segmentation')
    assert maxlabel == 5


def test_fetch_supervoxels_for_body(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')

    supervoxels = fetch_supervoxels_for_body(*instance_info, 1)
    assert (supervoxels == [1,2,3,4,5]).all()


def test_fetch_supervoxel_sizes_for_body(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')

    sv_sizes = fetch_supervoxel_sizes_for_body(*instance_info, 1)
    assert isinstance(sv_sizes, pd.Series)
    assert (sv_sizes.index == [1,2,3,4,5]).all()
    assert (sv_sizes == 9).all() # see initialization in conftest.py


def test_fetch_label_for_coordinate(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')
    
    label = fetch_label_for_coordinate(*instance_info, (0, 1, 3), supervoxels=False)
    assert isinstance(label, np.uint64)
    assert label == 1

    label = fetch_label_for_coordinate(*instance_info, (0, 1, 3), supervoxels=True)
    assert isinstance(label, np.uint64)
    assert label == 2


def test_fetch_mappings(labelmap_setup):
    """
    Test the wrapper function for the /mappings DVID API.
    """
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')
    
    mapping = fetch_mappings(*instance_info)
    assert isinstance(mapping, pd.Series)
    assert mapping.index.name == 'sv'
    assert mapping.name == 'body'
    assert (sorted(mapping.index) == [2,3,4,5]) # Does not include 'identity' row for SV 1. See docstring.
    assert (mapping == 1).all() # see initialization in conftest.py


def test_fetch_complete_mappings(labelmap_setup):
    """
    Very BASIC test for fetch_complete_mappings().
    Does not verify features related to split supervoxels
    """
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')
    
    mapping = fetch_complete_mappings(*instance_info, kafka_msgs=[])
    assert isinstance(mapping, pd.Series)
    assert mapping.index.name == 'sv'
    assert mapping.name == 'body'
    assert (sorted(mapping.index) == [1,2,3,4,5])
    assert (mapping == 1).all() # see initialization in conftest.py


def test_fetch_mutation_id(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')

    mut_id = fetch_mutation_id(*instance_info, 1)
    assert isinstance(mut_id, int)


def test_generate_sample_coordinate(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')

    coord_zyx = generate_sample_coordinate(*instance_info, 1)
    label = fetch_label_for_coordinate(*instance_info, coord_zyx.tolist())
    assert label == 1

    coord_zyx = generate_sample_coordinate(*instance_info, 2, supervoxels=True)
    label = fetch_label_for_coordinate(*instance_info, coord_zyx.tolist(), supervoxels=True)
    assert label == 2


def test_fetch_labelarray_voxels(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')

    # Test raw supervoxels
    voxels = fetch_labelarray_voxels(*instance_info, [(0,0,0), supervoxel_vol.shape], supervoxels=True)
    assert (voxels == supervoxel_vol).all()
    
    # Test mapped bodies
    voxels = fetch_labelarray_voxels(*instance_info, [(0,0,0), supervoxel_vol.shape], supervoxels=False)
    assert (voxels == 1).all()

    # Test uninflated mode
    voxels_proxy = fetch_labelarray_voxels(*instance_info, [(0,0,0), supervoxel_vol.shape], supervoxels=True, inflate=False)
    assert len(voxels_proxy.content) < supervoxel_vol.nbytes, \
        "Fetched data was apparently not compressed"
    assert (voxels_proxy() == supervoxel_vol).all()


def test_post_labelarray_voxels(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation-scratch')
    
    # Write some random data and read it back.
    blocks = np.random.randint(10, size=(3,64,64,64), dtype=np.uint64)
    corners_zyx = [[0,0,0], [0,64,0], [0,0,64]]

    post_labelarray_blocks(dvid_server, dvid_repo, 'segmentation-scratch', corners_zyx, blocks, 0)
    complete_voxels = fetch_labelarray_voxels(*instance_info, [(0,0,0), (128,128,128)], supervoxels=True)
        
    assert (complete_voxels[0:64,  0:64,  0:64]  == blocks[0]).all()
    assert (complete_voxels[0:64, 64:128, 0:64]  == blocks[1]).all()
    assert (complete_voxels[0:64,  0:64, 64:128] == blocks[2]).all()


def test_fetch_raw(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, supervoxel_vol = labelmap_setup
    instance_info = (dvid_server, dvid_repo, 'segmentation')

    # fetch_raw() returns mapped labels
    voxels = fetch_raw(*instance_info, [(0,0,0), supervoxel_vol.shape], dtype=np.uint64)
    assert (voxels == 1).all()


def test_post_raw(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = (dvid_server, dvid_repo, 'segmentation-scratch')
    
    # Write some random data and read it back.
    data = np.random.randint(10, size=(64,64,64*3), dtype=np.uint64)
    offset_zyx = (0,64,0)

    post_raw(*instance_info, offset_zyx, data)
    complete_voxels = fetch_labelarray_voxels(*instance_info, [(0,0,0), (128,128,192)], supervoxels=True)
        
    assert (complete_voxels[0:64, 64:128,   0:64]  == data[:, :,   0:64]).all()
    assert (complete_voxels[0:64, 64:128,  64:128] == data[:, :,  64:128]).all()
    assert (complete_voxels[0:64, 64:128, 128:192] == data[:, :, 128:192]).all()


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_dvid'])
