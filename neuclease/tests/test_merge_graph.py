import logging
import pytest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from libdvid import DVIDNodeService

from neuclease.dvid import ( DvidInstanceInfo, post_key, post_branch, create_instance,
                             fetch_mutation_id, post_cleave, post_split_supervoxel, post_merge )
from neuclease.cleave.merge_graph import LabelmapMergeGraph
from neuclease.cleave.merge_table import load_merge_table, MAPPED_MERGE_TABLE_DTYPE

##
## These tests rely on the global setupfunction 'labelmap_setup',
## defined in conftest.py and used here via pytest magic
##


def test_fetch_and_apply_mapping(labelmap_setup):
    dvid_server, dvid_repo, merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')
    
    # Don't give mapping, ensure it's loaded from dvid.
    merge_graph = LabelmapMergeGraph(merge_table_path, no_kafka=True)
    merge_graph.fetch_and_apply_mapping(*instance_info)
    assert (merge_graph.merge_table_df['body'] == 1).all()


def _test_extract_edges(labelmap_setup, force_dirty_mapping):
    """
    Implementation for testing extract_edges(), starting either with a "clean" mapping
    (in which the body column is already correct beforehand),
    or a "dirty" mapping (in which the body column is not correct beforehand).
    """
    dvid_server, dvid_repo, merge_table_path, mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')

    orig_merge_table = load_merge_table(merge_table_path, mapping_path, normalize=True)
    
    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.apply_mapping(mapping_path)

    if force_dirty_mapping:
        # A little white-box manipulation here to ensure that the mapping is dirty
        merge_graph.merge_table_df['body'] = np.uint64(0)
        merge_graph.mapping[:] = np.uint64(0)

    # First test: If nothing has changed in DVID, we get all rows.
    # We should be able to repeat this with the same results
    # (Make sure the cache is repopulated correctly.)
    _mutid, dvid_supervoxels, edges, _scores = merge_graph.extract_edges(*instance_info, 1)
    assert (dvid_supervoxels == [1,2,3,4,5]).all()
    assert (orig_merge_table[['id_a', 'id_b']].values == edges).all().all(), \
        f"Original merge table doesn't match fetched:\n{orig_merge_table}\n\n{edges}\n"

    # Now change the mapping in DVID and verify it is reflected in the extracted rows.
    # For this test, we'll cleave supervoxels [4,5] from the rest of the body.
    uuid = post_branch(dvid_server, dvid_repo, f'extract-rows-test-{force_dirty_mapping}', '')

    cleaved_body = post_cleave(dvid_server, uuid, 'segmentation', 1, [4,5])    
    cleaved_mutid = fetch_mutation_id(dvid_server, uuid, 'segmentation', 1)

    if force_dirty_mapping:
        # A little white-box manipulation here to ensure that the mapping is dirty
        merge_graph.mapping.loc[2] = 0
        merge_graph.merge_table_df['body'].values[0:2] = np.uint64(0)

    mutid, dvid_supervoxels, edges, _scores = merge_graph.extract_edges(dvid_server, uuid, 'segmentation', 1)
    assert (dvid_supervoxels == [1,2,3]).all()
    _cleaved_svs = set([4,5])
    assert (edges == orig_merge_table[['id_a', 'id_b']].query('id_a not in @_cleaved_svs and id_b not in @_cleaved_svs')).all().all()
    assert mutid == cleaved_mutid, "Expected cached mutation ID to match DVID"

    cleaved_mutid = fetch_mutation_id(dvid_server, uuid, 'segmentation', cleaved_body)

    # Check the other body
    mutid, dvid_supervoxels, edges, _scores = merge_graph.extract_edges(dvid_server, uuid, 'segmentation', cleaved_body)

    assert (edges == orig_merge_table[['id_a', 'id_b']].query('id_a in @_cleaved_svs and id_b in @_cleaved_svs')).all().all()
    assert mutid == cleaved_mutid, "Expected cached mutation ID to match DVID"
    
def test_extract_edges_clean_mapping(labelmap_setup):
    _test_extract_edges(labelmap_setup, force_dirty_mapping=False)

def test_extract_edges_dirty_mapping(labelmap_setup):
    _test_extract_edges(labelmap_setup, force_dirty_mapping=True)
    

def test_extract_edges_with_large_gap(labelmap_setup):
    """
    If a large gap exists between a supervoxel and the rest of the body,
    we won't find an edge for it, but there should be no crash.
    """
    dvid_server, dvid_repo, merge_table_path, mapping_path, _supervoxel_vol = labelmap_setup
    orig_merge_table = load_merge_table(merge_table_path, mapping_path, normalize=True)
    
    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.apply_mapping(mapping_path)

    uuid = post_branch(dvid_server, dvid_repo, f'test_extract_edges_large_gap', '')
    
    # Exercise a corner case:
    # Add a new supervoxel to the body, far away from the others.
    # (No edge will be added for that supervoxel.)
    block_99 = 99*np.ones((64,64,64), np.uint64)
    DVIDNodeService(dvid_server, uuid).put_labels3D('segmentation', block_99, (128,0,0))
    post_merge(dvid_server, uuid, 'segmentation', 1, [99])

    root_logger = logging.getLogger()
    oldlevel = root_logger.level
    try:
        # Hide warnings for this call; they're intentional.
        logging.getLogger().setLevel(logging.ERROR)
        _mutid, dvid_supervoxels, edges, _scores = merge_graph.extract_edges(dvid_server, uuid, 'segmentation', 1)
    finally:
        root_logger.setLevel(oldlevel)

    assert (dvid_supervoxels == [1,2,3,4,5,99]).all()
    assert (orig_merge_table[['id_a', 'id_b']].values == edges).all().all(), \
        f"Original merge table doesn't match fetched:\n{orig_merge_table}\n\n{edges}\n"



def _setup_test_append_edges_for_split(labelmap_setup, branch_name):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, supervoxel_vol = labelmap_setup
    uuid = post_branch(dvid_server, dvid_repo, branch_name, '')
    
    # Split supervoxel 3 (see conftest.init_labelmap_nodes)
    # Remove the first column of pixels from it.
    
    # supervoxel 3 starts in column 6
    assert (supervoxel_vol == 3).nonzero()[2][0] == 6

    rle = [[6,0,0,1], # x,y,z,runlength
           [6,1,0,1],
           [6,2,0,1]]

    rle = np.array(rle, np.uint32)

    header = np.array([0,3,0,0], np.uint8)
    voxels = np.array([0], np.uint32)
    num_spans = np.array([len(rle)], np.uint32)
    payload = bytes(header) + bytes(voxels) + bytes(num_spans) + bytes(rle)

    split_sv, remain_sv = post_split_supervoxel(dvid_server, uuid, 'segmentation', 3, payload)
    return uuid, split_sv, remain_sv


@pytest.fixture(params=('drop', 'keep', 'unmap'))#, ids=('drop', 'keep', 'unmap'))
def parent_sv_handling(request):
    yield request.param

def test_append_edges_for_split_supervoxels(labelmap_setup, parent_sv_handling):
    dvid_server, _dvid_repo, merge_table_path, mapping_path, _supervoxel_vol = labelmap_setup
    uuid, split_sv, remainder_sv = \
        _setup_test_append_edges_for_split(labelmap_setup, f'split-test-{parent_sv_handling}')
    
    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.apply_mapping(mapping_path)
    orig_table = merge_graph.merge_table_df.copy()
    
    table_dtype = merge_graph.merge_table_df.to_records(index=False).dtype
    assert table_dtype == MAPPED_MERGE_TABLE_DTYPE, \
        f"Merge table started with wrong dtype: \n{table_dtype}\nExpected:\n{MAPPED_MERGE_TABLE_DTYPE}"

    # Append edges for split supervoxels.
    merge_graph.append_edges_for_split_supervoxels((dvid_server, uuid, 'segmentation'), parent_sv_handling, read_from='dvid')

    table_dtype = merge_graph.merge_table_df.to_records(index=False).dtype
    assert merge_graph.merge_table_df.to_records(index=False).dtype == MAPPED_MERGE_TABLE_DTYPE, \
        f"Merge table has wrong dtype after splits were appended: \n{table_dtype}\nExpected:\n{MAPPED_MERGE_TABLE_DTYPE}"
        
    if parent_sv_handling == 'drop':
        assert merge_graph.merge_table_df.shape[0] == orig_table.shape[0]
        assert set(merge_graph.merge_table_df[['id_a', 'id_b']].values.flat) == set([1,2,4,5,split_sv,remainder_sv])
    elif parent_sv_handling == 'keep':
        assert merge_graph.merge_table_df.shape[0] == orig_table.shape[0] + 2
        assert set(merge_graph.merge_table_df[['id_a', 'id_b']].values.flat) == set([1,2,3,4,5,split_sv,remainder_sv])
    elif parent_sv_handling == 'unmap':
        assert merge_graph.merge_table_df.shape[0] == orig_table.shape[0] + 2
        assert set(merge_graph.merge_table_df[['id_a', 'id_b']].values.flat) == set([1,2,3,4,5,split_sv,remainder_sv])
        assert (merge_graph.merge_table_df.query('id_a == 3 or id_b == 3')['body'] == 0).all()
    
    # SV 3 was originally connected to SV 2 and 4.
    # We should have new rows for those connections, but with the new IDs
    assert len(merge_graph.merge_table_df.query('id_a == 2 and id_b == @split_sv')) == 1, \
        f"Merge graph:\n:{str(merge_graph.merge_table_df)}"
    assert len(merge_graph.merge_table_df.query('id_a == 4 and id_b == @remainder_sv')) == 1, \
        f"Merge graph:\n:{str(merge_graph.merge_table_df)}"

def test_append_edges_for_split_supervoxels_with_bad_edges(labelmap_setup, parent_sv_handling):
    dvid_server, _dvid_repo, merge_table_path, mapping_path, _supervoxel_vol = labelmap_setup
    uuid, split_sv, _remainder_sv = \
        _setup_test_append_edges_for_split(labelmap_setup, f'split-bad-edge-test-{parent_sv_handling}')

    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.apply_mapping(mapping_path)
    orig_table = merge_graph.merge_table_df.copy()

    # Overwrite the coordinate in one of the edges with something incorrect.
    # That edges should not be included in the appended results 
    edge_row = merge_graph.merge_table_df.query('id_a == 3').index
    merge_graph.merge_table_df.loc[edge_row, 'xa'] = np.uint32(0)
    bad_edges = merge_graph.append_edges_for_split_supervoxels((dvid_server, uuid, 'segmentation'), parent_sv_handling, read_from='dvid')
    assert len(bad_edges) == 1
    
    if parent_sv_handling == 'drop':
        assert merge_graph.merge_table_df.shape[0] == orig_table.shape[0] - 1 # bad edge dropped and not replaced
        assert set(merge_graph.merge_table_df[['id_a', 'id_b']].values.flat) == set([1,2,4,5,split_sv])
    elif parent_sv_handling == 'keep':
        assert merge_graph.merge_table_df.shape[0] == orig_table.shape[0] + 2 - 1 # bad edge dropped
        assert set(merge_graph.merge_table_df[['id_a', 'id_b']].values.flat) == set([1,2,3,4,5,split_sv])
    elif parent_sv_handling == 'unmap':
        assert merge_graph.merge_table_df.shape[0] == orig_table.shape[0] + 2 - 1 # bad edge dropped
        assert set(merge_graph.merge_table_df[['id_a', 'id_b']].values.flat) == set([1,2,3,4,5,split_sv])
        assert (merge_graph.merge_table_df.query('id_a == 3 or id_b == 3')['body'] == 0).all()


def test_append_edges_for_focused_merges(labelmap_setup):
    dvid_server, dvid_repo, merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    
    decision_instance = 'segmentation_merged_TEST'
    create_instance(dvid_server, dvid_repo, decision_instance, 'keyvalue')

    # Post a new 'decision' between 1 and 5
    post_key(dvid_server, dvid_repo, decision_instance, '1+5',
             json={'supervoxel ID 1': 1,
                   'supervoxel ID 2': 5,
                   'body ID 1': 1,
                   'body ID 2': 1,
                   'result': 'merge',
                   'supervoxel point 1': [0,0,0],   # xyz
                   'supervoxel point 2': [12,0,0]}) # xyz

    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.append_edges_for_focused_merges(dvid_server, dvid_repo, decision_instance)
    assert len(merge_graph.merge_table_df.query('id_a == 1 and id_b == 5')) == 1


def test_extract_edges_multithreaded(labelmap_setup):
    """
    Make sure the extract_edges() can be used from multiple threads without deadlocking.
    """
    dvid_server, dvid_repo, merge_table_path, mapping_path, _supervoxel_vol = labelmap_setup
    instance_info = DvidInstanceInfo(dvid_server, dvid_repo, 'segmentation')

    orig_merge_table = load_merge_table(merge_table_path, mapping_path, normalize=True)
     
    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.apply_mapping(mapping_path)

    def _test(force_dirty):
        if force_dirty:
            # A little white-box manipulation here to ensure that the cache is out-of-date.
            with merge_graph._edge_cache_main_lock:
                merge_graph._edge_cache.clear()

        # Extraction should still work.
        mutid, dvid_supervoxels, edges, _scores = merge_graph.extract_edges(*instance_info, 1)
        assert (dvid_supervoxels == [1,2,3,4,5]).all()
        assert (orig_merge_table[['id_a', 'id_b']] == edges).all().all(), \
            f"Original merge table doesn't match fetched:\n{orig_merge_table}\n\n{edges}\n"

    # Quickly check the test function before loading it into a pool.
    # (If it's going to fail in a trivial way, let's see it in the main thread.)
    _test(False)
    _test(True)

    with ThreadPoolExecutor(max_workers=11) as executor:
        list(executor.map(_test, 300*[[True], [False], [False]]))

if __name__ == "__main__":
#     import sys
#     import logging
#     handler = logging.StreamHandler(sys.stdout)
#     logging.getLogger().addHandler(handler)
#     logging.getLogger().setLevel(logging.INFO)
    
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_merge_graph'])
