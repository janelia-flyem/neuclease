"""
Test module for the dvid API wrapper functions defined in neuclease.dvid
"""
import sys
import time
import logging
import datetime
from multiprocessing.pool import ThreadPool

import pytest
import numpy as np
import pandas as pd

from libdvid import DVIDNodeService

from neuclease.dvid import (dvid_api_wrapper, DvidInstanceInfo, fetch_supervoxels_for_body, fetch_supervoxel_sizes_for_body,
                            fetch_label, fetch_labels, fetch_labels_batched, fetch_mappings, fetch_complete_mappings, post_mappings,
                            fetch_mutation_id, generate_sample_coordinate, fetch_labelmap_voxels, post_labelmap_blocks, post_labelmap_voxels,
                            encode_labelarray_volume, encode_nonaligned_labelarray_volume, fetch_raw, post_raw,
                            fetch_labelindex, post_labelindex, fetch_labelindices, create_labelindex, PandasLabelIndex,
                            copy_labelindices,
                            fetch_maxlabel, post_maxlabel, fetch_nextlabel, post_nextlabel, create_labelmap_instance,
                            post_merge, fetch_sparsevol_coarse, fetch_sparsevol_coarse_via_labelindex, post_branch,
                            post_hierarchical_cleaves, fetch_mapping)

from neuclease.dvid._dvid import default_dvid_session
from neuclease.util import box_to_slicing, extract_subvol, ndrange

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

##
## These tests rely on the global setupfunction 'labelmap_setup',
## defined in conftest.py and used here via pytest magic
##

def test_default_dvid_session():
    """
    Verify that default_dvid_session() really re-uses sessions (one per thread).
    """
    def session_id(_):
        time.sleep(0.01)
        return id(default_dvid_session())
    
    with ThreadPool(2) as pool:
        ids = list(pool.map(session_id, range(20)))
    assert len(set(ids)) == 2


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


def test_fetch_label(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')
    
    label = fetch_label(*instance_info, (0, 1, 3), supervoxels=False)
    assert isinstance(label, np.uint64)
    assert label == 1

    label = fetch_label(*instance_info, (0, 1, 3), supervoxels=True)
    assert isinstance(label, np.uint64)
    assert label == 2


def test_fetch_labels(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')

    coords = [[0,0,0], [0,0,1], [0,0,2],
              [0,0,3], [0,0,4], [0,0,4]]
    
    labels = fetch_labels(*instance_info, coords, supervoxels=False)
    assert labels.dtype == np.uint64
    assert (labels == 1).all() # See init_labelmap_nodes() in conftest.py

    labels = fetch_labels(*instance_info, coords, supervoxels=True)
    assert labels.dtype == np.uint64
    assert (labels == [1,1,1,2,2,2]).all() # See init_labelmap_nodes() in conftest.py


def test_fetch_labels_batched(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')

    coords = [[0,0,0], [0,0,1], [0,0,2],
              [0,0,3], [0,0,4], [0,0,4]]
    
    labels = fetch_labels_batched(*instance_info, coords, supervoxels=False, batch_size=2, threads=2)
    assert labels.dtype == np.uint64
    assert (labels == 1).all() # See init_labelmap_nodes() in conftest.py

    labels = fetch_labels_batched(*instance_info, coords, supervoxels=True, batch_size=2, threads=2)
    assert labels.dtype == np.uint64
    assert (labels == [1,1,1,2,2,2]).all() # See init_labelmap_nodes() in conftest.py

    labels = fetch_labels_batched(*instance_info, coords, supervoxels=False, batch_size=2, processes=2)
    assert labels.dtype == np.uint64
    assert (labels == 1).all() # See init_labelmap_nodes() in conftest.py

    labels = fetch_labels_batched(*instance_info, coords, supervoxels=True, batch_size=2, processes=2)
    assert labels.dtype == np.uint64
    assert (labels == [1,1,1,2,2,2]).all() # See init_labelmap_nodes() in conftest.py


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


def test_post_mappings(labelmap_setup):
    """
    Test the wrapper function for the /mappings DVID API.
    """
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')
    
    # Fetch the original mapping
    orig_mapping = fetch_mappings(*instance_info).sort_index()
    assert (orig_mapping.index == [2,3,4,5]).all()
    assert (orig_mapping == 1).all() # see initialization in conftest.py
    
    # Make sure post_mappings does not REQUIRE the Series to be named
    orig_mapping.index.name = 'barfoo'
    orig_mapping.name = 'foobar'
    
    # Now post a new mapping and read it back.
    new_mapping = orig_mapping.copy()
    new_mapping[:] = 2
    new_mapping.sort_index(inplace=True)
    
    # Post all but sv 5
    post_mappings(*instance_info, new_mapping.iloc[:-1], mutid=1)
    fetched_mapping = fetch_mappings(*instance_info).sort_index()
    assert (fetched_mapping.index == [3,4,5]).all()
    assert (fetched_mapping.iloc[:-1] == 2).all()
    assert (fetched_mapping.iloc[-1:] == 1).all()

    # Now post sv 5, too
    post_mappings(*instance_info, new_mapping.iloc[-1:], mutid=1)
    fetched_mapping = fetch_mappings(*instance_info).sort_index()
    assert (fetched_mapping.index == [3,4,5]).all()
    assert (fetched_mapping == 2).all()

    # Try batched
    new_mapping = pd.Series(index=[1,2,3,4,5], data=[1,1,1,2,2])
    post_mappings(*instance_info, new_mapping, mutid=1, batch_size=4)
    fetched_mapping = fetch_mappings(*instance_info).sort_index()
    assert (fetched_mapping.index == [2,3,4,5]).all()
    assert (fetched_mapping == [1,1,2,2]).all(), f"{fetched_mapping} != {[1,1,2,2]}"
    
    # Restore the original mapping
    post_mappings(*instance_info, orig_mapping, 1)
    fetched_mapping = fetch_mappings(*instance_info).sort_index()
    assert (fetched_mapping.index == [2,3,4,5]).all()
    assert (fetched_mapping == 1).all()
    

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
    label = fetch_label(*instance_info, coord_zyx.tolist())
    assert label == 1

    coord_zyx = generate_sample_coordinate(*instance_info, 2, supervoxels=True)
    label = fetch_label(*instance_info, coord_zyx.tolist(), supervoxels=True)
    assert label == 2


def test_encode_labelarray_volume():
    vol = np.random.randint(1000,2000, size=(128,128,128), dtype=np.uint64)
    vol[:64,:64,:64] = 0
    vol[-64:, -64:, -64:] = 0
    
    encoded = encode_labelarray_volume((512,1024,2048), vol, gzip_level=0)
    encoded2 = encode_labelarray_volume((512,1024,2048), vol, gzip_level=9)
    assert len(encoded) > len(encoded2)

    inflated = DVIDNodeService.inflate_labelarray_blocks3D_from_raw(encoded, (128,128,128), (512,1024,2048))
    assert (inflated == vol).all()

    inflated = DVIDNodeService.inflate_labelarray_blocks3D_from_raw(encoded2, (128,128,128), (512,1024,2048))
    assert (inflated == vol).all()


def test_encode_nonaligned_labelarray_volume():
    nonaligned_box = np.array([(520,1050,2050), (620,1150,2150)])
    nonaligned_vol = np.random.randint(1000,2000, size=(100,100,100), dtype=np.uint64)
    
    aligned_start = np.array((512,1024,2048))
    aligned_box = np.array([aligned_start, aligned_start+128])
    aligned_vol = np.zeros( (128,128,128), dtype=np.uint64 )
    aligned_vol[box_to_slicing(*(nonaligned_box - aligned_box[0]))] = nonaligned_vol
    
    encoded_box, encoded_vol = encode_nonaligned_labelarray_volume(nonaligned_box[0], nonaligned_vol)
    inflated = DVIDNodeService.inflate_labelarray_blocks3D_from_raw(encoded_vol, (128,128,128), aligned_start)
    
    assert (encoded_box == aligned_box).all()
    assert (inflated == aligned_vol).all()


def test_fetch_labelmap_voxels(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')

    # Test raw supervoxels
    voxels = fetch_labelmap_voxels(*instance_info, [(0,0,0), supervoxel_vol.shape], supervoxels=True)
    assert (voxels == supervoxel_vol).all()
    
    # Test mapped bodies
    voxels = fetch_labelmap_voxels(*instance_info, [(0,0,0), supervoxel_vol.shape], supervoxels=False)
    assert (voxels == 1).all()

    # Test uninflated mode
    voxels_proxy = fetch_labelmap_voxels(*instance_info, [(0,0,0), supervoxel_vol.shape], supervoxels=True, format='lazy-array')
    assert len(voxels_proxy.content) < supervoxel_vol.nbytes, \
        "Fetched data was apparently not compressed"
    assert (voxels_proxy() == supervoxel_vol).all()


def test_post_labelmap_blocks(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation-scratch')
    
    # Write some random data and read it back.
    blocks = np.random.randint(10, size=(3,64,64,64), dtype=np.uint64)
    corners_zyx = [[0,0,0], [0,64,0], [0,0,64]]

    post_labelmap_blocks(dvid_server, dvid_repo, 'segmentation-scratch', corners_zyx, blocks, 0)
    complete_voxels = fetch_labelmap_voxels(*instance_info, [(0,0,0), (128,128,128)], supervoxels=True)
        
    assert (complete_voxels[0:64,  0:64,  0:64]  == blocks[0]).all()
    assert (complete_voxels[0:64, 64:128, 0:64]  == blocks[1]).all()
    assert (complete_voxels[0:64,  0:64, 64:128] == blocks[2]).all()


def test_post_labelmap_voxels(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation-scratch')
    
    # Write some random data and read it back.
    vol = np.random.randint(10, size=(128,128,128), dtype=np.uint64)
    offset = (64,64,64)

    post_labelmap_voxels(dvid_server, dvid_repo, 'segmentation-scratch', offset, vol, 0)
    complete_voxels = fetch_labelmap_voxels(*instance_info, [(0,0,0), (256,256,256)], supervoxels=True)
    
    assert (complete_voxels[64:192, 64:192, 64:192] == vol).all()


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
    complete_voxels = fetch_labelmap_voxels(*instance_info, [(0,0,0), (128,128,192)], supervoxels=True)
        
    assert (complete_voxels[0:64, 64:128,   0:64]  == data[:, :,   0:64]).all()
    assert (complete_voxels[0:64, 64:128,  64:128] == data[:, :,  64:128]).all()
    assert (complete_voxels[0:64, 64:128, 128:192] == data[:, :, 128:192]).all()


def test_maxlabel_and_friends(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    
    # Need an unlocked node to test these posts
    uuid = post_branch(dvid_server, dvid_repo, 'test_maxlabel_and_friends', 'test_maxlabel_and_friends')
    instance_info = (dvid_server, uuid, 'segmentation-scratch')

    max_label = fetch_maxlabel(*instance_info)
    next_label = fetch_nextlabel(*instance_info)
    assert max_label+1 == next_label
    
    start, end = post_nextlabel(*instance_info, 5)
    assert start == max_label+1
    assert end == start + 5-1

    max_label = fetch_maxlabel(*instance_info)
    next_label = fetch_nextlabel(*instance_info)
    assert next_label == max_label+1 == end+1

    new_max = next_label+10
    post_maxlabel(*instance_info, new_max)
    
    max_label = fetch_maxlabel(*instance_info)
    assert max_label == new_max

    next_label = fetch_nextlabel(*instance_info)
    assert max_label+1 == next_label


def test_labelindex(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup

    # Need an unlocked node to test these posts
    uuid = post_branch(dvid_server, dvid_repo, 'test_labelindex', 'test_labelindex')
    instance_info = (dvid_server, uuid, 'segmentation-scratch')

    # Write some random data
    sv = 99
    vol = sv*np.random.randint(2, size=(128,128,128), dtype=np.uint64)
    offset = np.array((64,64,64))
    
    # DVID will generate the index.
    post_labelmap_voxels(*instance_info, offset, vol)

    # Compute labelindex table from scratch
    rows = []
    for block_coord in ndrange(offset, offset+vol.shape, (64,64,64)):
        block_coord = np.array(block_coord)
        block_box = np.array((block_coord, block_coord+64))
        block = extract_subvol(vol, block_box - offset)
        
        count = (block == sv).sum()
        rows.append( [*block_coord, sv, count] )
    
    index_df = pd.DataFrame( rows, columns=['z', 'y', 'x', 'sv', 'count'])
    
    # Check DVID's generated labelindex table against expected
    labelindex_tuple = fetch_labelindex(*instance_info, sv, format='pandas')
    assert labelindex_tuple.label == sv

    labelindex_tuple.blocks.sort_values(['z', 'y', 'x', 'sv'], inplace=True)
    labelindex_tuple.blocks.reset_index(drop=True, inplace=True)
    assert (labelindex_tuple.blocks == index_df).all().all()

    # Check our protobuf against DVID's
    index_tuple = PandasLabelIndex(index_df, sv, 1, datetime.datetime.now().isoformat(), 'someuser')
    labelindex = create_labelindex(index_tuple)
    
    # Since labelindex block entries are not required to be sorted,
    # dvid might return them in a different order.
    # Hence this comparison function which sorts them first.
    def compare_proto_blocks(left, right):
        left_blocks = sorted(left.blocks.items())
        right_blocks = sorted(right.blocks.items())
        return left_blocks == right_blocks
    
    dvid_labelindex = fetch_labelindex(*instance_info, sv, format='protobuf')
    assert compare_proto_blocks(labelindex, dvid_labelindex)
    
    # Check post/get roundtrip
    post_labelindex(*instance_info, sv, labelindex)
    dvid_labelindex = fetch_labelindex(*instance_info, sv, format='protobuf')
    assert compare_proto_blocks(labelindex, dvid_labelindex)


def test_fetch_labelindices(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup

    # Need an unlocked node to test these posts
    uuid = post_branch(dvid_server, dvid_repo, 'test_labelindices', 'test_labelindices')
    instance_info = (dvid_server, uuid, 'segmentation-scratch')

    # Write some random data
    vol = np.random.randint(1, 10, size=(128,128,128), dtype=np.uint64)
    offset = np.array((64,64,64))
    
    # DVID will generate the index.
    post_labelmap_voxels(*instance_info, offset, vol)

    labelindices = fetch_labelindices(*instance_info, list(range(1,10)))
    for sv, li in zip(range(1,10), labelindices.indices):
        # This function is already tested elsewhere, so we'll use it as a reference
        li2 = fetch_labelindex(*instance_info, sv)
        assert li == li2

    labelindices = fetch_labelindices(*instance_info, list(range(1,10)), format='list-of-protobuf')
    for sv, li in zip(range(1,10), labelindices):
        # This function is already tested elsewhere, so we'll use it as a reference
        li2 = fetch_labelindex(*instance_info, sv)
        assert li == li2

    labelindices = fetch_labelindices(*instance_info, list(range(1,10)), format='pandas')
    for sv, li in zip(range(1,10), labelindices):
        # This function is already tested elsewhere, so we'll use it as a reference
        li2 = fetch_labelindex(*instance_info, sv, format='pandas')
        li_df = li.blocks.sort_values(['z', 'y', 'x']).reset_index(drop=True)
        li2_df = li2.blocks.sort_values(['z', 'y', 'x']).reset_index(drop=True)
        assert (li_df == li2_df).all().all()

    # Test the copy function (just do a round-trip -- hopefully I didn't swap src and dest anywhere...)
    copy_labelindices(instance_info, instance_info, list(range(1,10)), batch_size=2)
    copy_labelindices(instance_info, instance_info, list(range(1,10)), batch_size=2, processes=2)


def test_fetch_sparsevol_coarse_via_labelindex(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup

    # Create a labelmap volume with 3 blocks.
    #
    # Supervoxels are arranged like this:
    #
    #   | 1 2 | 3 4 | 5 6 |
    # 
    # After merging [2,3,4,5], bodies will be:
    #
    #   | 1 2 | 2 4 | 5 6 |
    #
    vol_shape = (64,64,256)
    sv_vol = np.zeros(vol_shape, np.uint64)
    sv_vol[:,:,0:32] = 1
    sv_vol[:,:,32:64] = 2
    sv_vol[:,:,64:96] = 3
    sv_vol[:,:,96:128] = 4
    sv_vol[:,:,128:160] = 5
    sv_vol[:,:,160:192] = 6

    instance_info = dvid_server, dvid_repo, 'segmentation-test-sparsevol-coarse'
    create_labelmap_instance(*instance_info)
    post_labelmap_voxels(*instance_info, (0,0,0), sv_vol)
    
    post_merge(*instance_info, 2, [3,4,5])
    
    body_svc = fetch_sparsevol_coarse_via_labelindex(*instance_info, 2, method='protobuf')
    expected_body_svc = fetch_sparsevol_coarse(*instance_info, 2)
    assert sorted(body_svc.tolist()) == sorted(expected_body_svc.tolist())

    body_svc = fetch_sparsevol_coarse_via_labelindex(*instance_info, 2, method='pandas')
    expected_body_svc = fetch_sparsevol_coarse(*instance_info, 2)
    assert sorted(body_svc.tolist()) == sorted(expected_body_svc.tolist())

    sv_svc = fetch_sparsevol_coarse_via_labelindex(*instance_info, 3, supervoxels=True, method='protobuf')
    expected_sv_svc = fetch_sparsevol_coarse(*instance_info, 3, supervoxels=True)
    assert sorted(sv_svc.tolist()) == sorted(expected_sv_svc.tolist())

    sv_svc = fetch_sparsevol_coarse_via_labelindex(*instance_info, 3, supervoxels=True, method='pandas')
    expected_sv_svc = fetch_sparsevol_coarse(*instance_info, 3, supervoxels=True)
    assert sorted(sv_svc.tolist()) == sorted(expected_sv_svc.tolist())


def test_post_hierarchical_cleaves(labelmap_setup):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup

    uuid = post_branch(dvid_server, dvid_repo, 'segmentation-post_hierarchical_cleaves', '')    
    instance_info = dvid_server, uuid, 'segmentation-post_hierarchical_cleaves'
    create_labelmap_instance(*instance_info)

    svs    = [1,2,3,4,5,6,7,8,9,10]
    groups = [1,1,2,2,3,3,3,3,3,4]

    svs = np.asarray(svs, np.uint64)

    # Post some supervoxels in multiple blocks, just to prove that post_hierarchical_cleaves()
    # doesn't assume that the labelindex has the same length as the mapping.
    sv_vol = np.zeros((128,64,64), np.uint64)
    sv_vol[0,0,:len(svs)] = svs
    sv_vol[64,0,0:len(svs):2] = svs[::2]
    
    post_labelmap_voxels(*instance_info, (0,0,0), sv_vol)

    post_merge(*instance_info, 1, svs[1:])
    
    group_mapping = pd.Series(index=svs, data=groups)
    final_table = post_hierarchical_cleaves(*instance_info, 1, group_mapping)
    
    assert (fetch_mapping(*instance_info, svs) == final_table['body'].values).all()
    assert (final_table.drop_duplicates(['group']) == final_table.drop_duplicates(['group', 'body'])).all().all()
    assert (final_table.drop_duplicates(['body']) == final_table.drop_duplicates(['group', 'body'])).all().all()
    
    # Since the mapping included all supervoxels in the body,
    # the last group is left with the original label.
    assert final_table.iloc[-1]['body'] == 1

    # Now merge them all together and try again, but leave
    # two supevoxels out of the groups this time.
    merges = set(pd.unique(final_table['body'].values)) - set([1])
    post_merge(*instance_info, 1, list(merges))
    
    group_mapping = pd.Series(index=svs[:-2], data=groups[:-2])
    final_table = post_hierarchical_cleaves(*instance_info, 1, group_mapping)

    assert len(final_table.query('body == 1')) == 0, "Did not expect any of the groups to retain the original body ID!"    
    assert (fetch_mapping(*instance_info, svs[:-2]) == final_table['body'].values).all()
    assert (final_table.drop_duplicates(['group']) == final_table.drop_duplicates(['group', 'body'])).all().all()
    assert (final_table.drop_duplicates(['body']) == final_table.drop_duplicates(['group', 'body'])).all().all()
    assert (fetch_mapping(*instance_info, svs[-2:]) == 1).all()


if __name__ == "__main__":
    #from neuclease import configure_default_logging
    #configure_default_logging()
    args = ['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_dvid']
    #args += ['-k', 'fetch_labelindices']
    pytest.main(args)
