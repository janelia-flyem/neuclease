import pytest
import requests
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from neuclease.merge_graph import LabelmapMergeGraph
from neuclease.merge_table import load_merge_table, MAPPED_MERGE_TABLE_DTYPE

##
## These tests rely on the global setupfunction 'labelmap_setup',
## defined in conftest.py and used here via pytest magic
##

def test_fetch_supervoxels_for_body(labelmap_setup):
    dvid_server, dvid_repo, merge_table_path, mapping_path, _supervoxel_vol = labelmap_setup
    
    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.apply_mapping(mapping_path)
    _mut_id, supervoxels = merge_graph.fetch_supervoxels_for_body(dvid_server, dvid_repo, 'segmentation', 1)
    assert (sorted(supervoxels) == [1,2,3,4,5])


def test_fetch_and_apply_mapping(labelmap_setup):
    dvid_server, dvid_repo, merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    
    # Don't give mapping, ensure it's loaded from dvid.
    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.fetch_and_apply_mapping(dvid_server, dvid_repo, 'segmentation')
    assert (merge_graph.merge_table_df['body'] == 1).all()


def _test_extract_rows(labelmap_setup, force_dirty_mapping):
    """
    Implementation for testing extract_rows(), starting either with a "clean" mapping
    (in which the body column is already correct beforehand),
    or a "dirty" mapping (in which the body column is not correct beforehand).
    """
    dvid_server, dvid_repo, merge_table_path, mapping_path, _supervoxel_vol = labelmap_setup
    orig_merge_table = load_merge_table(merge_table_path, mapping_path, normalize=True)
    
    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.apply_mapping(mapping_path)

    if force_dirty_mapping:
        # A little white-box manipulation here to ensure that the mapping is dirty
        merge_graph.merge_table_df['body'] = np.uint64(0)
        merge_graph._mapping_versions.clear()

    # First test: If nothing has changed in DVID, we get all rows.
    # We should be able to repeat this with the same results
    # (Make sure the cache is repopulated correctly.)
    def _extract():
        subset_df, dvid_supervoxels = merge_graph.extract_rows(dvid_server, dvid_repo, 'segmentation', 1)
        assert (dvid_supervoxels == [1,2,3,4,5]).all()
        assert (orig_merge_table == subset_df).all().all(), \
            f"Original merge table doesn't match fetched:\n{orig_merge_table}\n\n{subset_df}\n"
        assert (orig_merge_table == merge_graph.merge_table_df).all().all(), \
            f"Original merge table doesn't match updated:\n{orig_merge_table}\n\n{merge_graph.merge_table_df}\n"
    _extract()
    _extract()

    # Now change the mapping in DVID and verify it is reflected in the extracted rows.
    # For this test, we'll cleave supervoxels [4,5] from the rest of the body.
    r = requests.post(f'http://{dvid_server}/api/node/{dvid_repo}/branch', json={'branch': f'extract-rows-test-{force_dirty_mapping}'})
    r.raise_for_status()
    uuid = r.json()["child"]

    r = requests.post(f'http://{dvid_server}/api/node/{uuid}/segmentation/cleave/1', json=[4,5])
    r.raise_for_status()
    cleaved_body = r.json()["CleavedLabel"]

    if force_dirty_mapping:
        # A little white-box manipulation here to ensure that the mapping is dirty
        merge_graph.merge_table_df['body'].values[0:2] = np.uint64(0)
        merge_graph._mapping_versions.clear()

    subset_df, dvid_supervoxels = merge_graph.extract_rows(dvid_server, uuid, 'segmentation', 1)
    assert (dvid_supervoxels == [1,2,3]).all()
    cleaved_svs = set([4,5])
    assert (subset_df == orig_merge_table.query('id_a not in @cleaved_svs and id_b not in @cleaved_svs')).all().all()

    # Check the other body
    subset_df, dvid_supervoxels = merge_graph.extract_rows(dvid_server, uuid, 'segmentation', cleaved_body)

    # Checking one body or the other shouldn't invalidate the rest of the body column!
    assert (merge_graph.merge_table_df.iloc[:2]['body'] == 1).all()
    assert merge_graph.merge_table_df.iloc[2]['body'] == 0 # This edge (3,4) was cut by the cleave, and so has no body
    assert merge_graph.merge_table_df.iloc[3]['body'] == cleaved_body
    
def test_extract_rows_clean_mapping(labelmap_setup):
    _test_extract_rows(labelmap_setup, force_dirty_mapping=False)

def test_extract_rows_dirty_mapping(labelmap_setup):
    _test_extract_rows(labelmap_setup, force_dirty_mapping=True)
    

def _setup_test_append_edges_for_split(labelmap_setup, branch_name):
    dvid_server, dvid_repo, _merge_table_path, _mapping_path, supervoxel_vol = labelmap_setup
    r = requests.post(f'http://{dvid_server}/api/node/{dvid_repo}/branch', json={'branch': branch_name})
    r.raise_for_status()
    uuid = r.json()["child"]
    
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

    r = requests.post(f'http://{dvid_server}/api/node/{uuid}/segmentation/split-supervoxel/3', data=payload)
    r.raise_for_status()
    split_response = r.json()
    split_sv = split_response["SplitSupervoxel"]
    remainder_sv = split_response["RemainSupervoxel"]
    split_mapping = np.array([[split_sv, 3],
                              [remainder_sv, 3]], np.uint64)

    return uuid, split_mapping, split_sv, remainder_sv


@pytest.fixture(params=('drop', 'keep', 'unmap'))#, ids=('drop', 'keep', 'unmap'))
def parent_sv_handling(request):
    yield request.param

def test_append_edges_for_split_supervoxels(labelmap_setup, parent_sv_handling):
    dvid_server, _dvid_repo, merge_table_path, mapping_path, _supervoxel_vol = labelmap_setup
    uuid, split_mapping, split_sv, remainder_sv = \
        _setup_test_append_edges_for_split(labelmap_setup, f'split-test-{parent_sv_handling}')
    
    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.apply_mapping(mapping_path)
    orig_table = merge_graph.merge_table_df.copy()
    
    table_dtype = merge_graph.merge_table_df.to_records(index=False).dtype
    assert table_dtype == MAPPED_MERGE_TABLE_DTYPE, \
        f"Merge table started with wrong dtype: \n{table_dtype}\nExpected:\n{MAPPED_MERGE_TABLE_DTYPE}"

    # Append edges for split supervoxels.
    merge_graph.append_edges_for_split_supervoxels(split_mapping, dvid_server, uuid, 'segmentation', parent_sv_handling)

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
    uuid, split_mapping, split_sv, _remainder_sv = \
        _setup_test_append_edges_for_split(labelmap_setup, f'split-bad-edge-test-{parent_sv_handling}')

    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.apply_mapping(mapping_path)
    orig_table = merge_graph.merge_table_df.copy()

    # Overwrite the coordinate in one of the edges with something incorrect.
    # That edges should not be included in the appended results 
    edge_row = merge_graph.merge_table_df.query('id_a == 3').index
    merge_graph.merge_table_df.loc[edge_row, 'xa'] = np.uint32(0)
    bad_edges = merge_graph.append_edges_for_split_supervoxels(split_mapping, dvid_server, uuid, 'segmentation', parent_sv_handling)
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


def test_extract_rows_multithreaded(labelmap_setup):
    """
    Make sure the extract_rows() can be used from multiple threads without deadlocking.
    """
    dvid_server, dvid_repo, merge_table_path, mapping_path, _supervoxel_vol = labelmap_setup
    orig_merge_table = load_merge_table(merge_table_path, mapping_path, normalize=True)
     
    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.apply_mapping(mapping_path)

    def _test(force_dirty):
        if force_dirty:
            # A little white-box manipulation here to ensure that the mapping is dirty
            with merge_graph.rwlock.context(write=True):
                merge_graph.merge_table_df['body'] = np.uint64(0)
                merge_graph._mapping_versions.clear()

        # Extraction should still work.
        subset_df, dvid_supervoxels = merge_graph.extract_rows(dvid_server, dvid_repo, 'segmentation', 1)
        assert (dvid_supervoxels == [1,2,3,4,5]).all()
        assert (orig_merge_table == subset_df).all().all(), f"Original merge table doesn't match fetched:\n{orig_merge_table}\n\n{subset_df}\n"

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
