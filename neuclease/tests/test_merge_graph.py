import sys
import pytest

import numpy as np
import requests

from neuclease.merge_graph import LabelmapMergeGraph
from neuclease.merge_table import load_merge_table

##
## These tests rely on the global setupfunction 'labelmap_setup',
## defined in conftest.py and used here via pytest magic
##

def test_fetch_supervoxels_for_body(labelmap_setup):
    dvid_server, dvid_repo, merge_table_path, mapping_path = labelmap_setup
    
    merge_graph = LabelmapMergeGraph(merge_table_path, mapping_path)
    supervoxels = merge_graph.fetch_supervoxels_for_body(dvid_server, dvid_repo, 'segmentation', 1, 0)
    assert (sorted(supervoxels) == [1,2,3,4,5])


def test_fetch_and_apply_mapping(labelmap_setup):
    dvid_server, dvid_repo, merge_table_path, _mapping_path = labelmap_setup
    
    # Don't give mapping, ensure it's loaded from dvid.
    merge_graph = LabelmapMergeGraph(merge_table_path)
    merge_graph.fetch_and_apply_mapping(dvid_server, dvid_repo, 'segmentation')
    assert (merge_graph.merge_table_df['body'] == 1).all()


def test_extract_rows(labelmap_setup):
    dvid_server, dvid_repo, merge_table_path, mapping_path = labelmap_setup
    orig_merge_table = load_merge_table(merge_table_path, mapping_path, normalize=True)
    
    merge_graph = LabelmapMergeGraph(merge_table_path, mapping_path)

    # First test: If nothing has changed in DVID, we get all rows.
    subset_df, dvid_supervoxels = merge_graph.extract_rows(dvid_server, dvid_repo, 'segmentation', 1)
    assert (dvid_supervoxels == [1,2,3,4,5]).all()
    assert (subset_df == orig_merge_table).all().all()

    # Now change the mapping in DVID and verify it is reflected in the extracted rows.
    # For this test, we'll cleave supervoxel 5 from the rest of the body.
    r = requests.post(f'http://{dvid_server}/api/node/{dvid_repo}/segmentation/cleave/1', json=[5])
    r.raise_for_status()
    cleaved_body = r.json()["CleavedLabel"]

    try:
        subset_df, dvid_supervoxels = merge_graph.extract_rows(dvid_server, dvid_repo, 'segmentation', 1)
        assert (dvid_supervoxels == [1,2,3,4]).all()
        assert (subset_df == orig_merge_table.query('id_a != 5 and id_b != 5')).all().all()
    finally:
        # Undo the cleave, so other tests still work.
        r = requests.post(f'http://{dvid_server}/api/node/{dvid_repo}/segmentation/merge', json=[1,cleaved_body])
        r.raise_for_status()


if __name__ == "__main__":
    pytest.main()
