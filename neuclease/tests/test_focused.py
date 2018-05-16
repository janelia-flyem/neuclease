import pytest

import numpy as np
import pandas as pd

from neuclease.focused import find_paths_from, find_all_paths
from neuclease.util import graph_tool_available

pytestmark = pytest.mark.skipif(not graph_tool_available(),
                                reason='graph-tool not available')

if graph_tool_available():
    import graph_tool as gt


def assert_path_lists_equal(l1, l2):
    # Convert to sets-of-tuples for easy comparison
    s1 = set(map(tuple, l1))
    s2 = set(map(tuple, l2))
    assert s1 == s2, f"Path sets not equal:\n{l1}\n{l2}"

@pytest.fixture
def toy_graph_setup():
    # Set up our test topology
    # 0 == ignored
    # 10,20,30, etc. == 'important' (end points)
    # Body grouping will be determined by the 10's place (e.g. 20,21,22 are all in body 20)    
    _ = 0
    grid = [[10,11,12,13,21,20,22,23,33,32,31,30],
            [ _,14, _, _, _,24, _, _,34, _, _, _],
            [ _,15,61,60, _,25,26,27,35, _, _,90],
            [ _,51, _, _, _,41, _, _, _, _, _,100],
            [ _,52, _, _,40,71,72,81,82, _, _, _],
            [ _, _, _, _, _, _,73, _, _, _, _,110]]

    # Note that there are multiple paths between 20 and 40.
    expected_paths = {
        10:  [[10, 11, 12, 13, 21, 20], [10, 11, 14, 15, 61, 60]],
        20:  [[20, 22, 23, 33, 32, 31, 30], [20, 22, 23, 33, 34, 35, 27, 26, 25, 41, 71, 40], [20, 24, 25, 26, 27, 35, 34, 33, 32, 31, 30], [20, 24, 25, 41, 71, 40], [20, 21, 13, 12, 11, 14, 15, 61, 60], [20, 21, 13, 12, 11, 10]],
        30:  [[30, 31, 32, 33, 34, 35, 27, 26, 25, 41, 71, 40], [30, 31, 32, 33, 34, 35, 27, 26, 25, 24, 20], [30, 31, 32, 33, 23, 22, 20]],
        40:  [[40, 71, 41, 25, 26, 27, 35, 34, 33, 32, 31, 30], [40, 71, 41, 25, 26, 27, 35, 34, 33, 23, 22, 20], [40, 71, 41, 25, 24, 20]],
        60:  [[60, 61, 15, 14, 11, 12, 13, 21, 20], [60, 61, 15, 14, 11, 10]],
        90:  [[90, 100]],
        100: [[100, 90]],
        110: []
    }

    grid = np.array(grid, np.int64)
    
    # No repeated vertex IDs in grid (except 0, which isn't a vertex)
    assert grid.nonzero()[0].shape[0] + 1 == np.unique(grid).shape[0]
    
    def get_horizontal_edges(grid):
        edges = []
        for row in grid:
            for (a,b) in zip(row[:-1], row[1:]):
                if a != 0 and b != 0:
                    edges.append((a,b))
        return edges

    edges = get_horizontal_edges(grid)
    edges += get_horizontal_edges(grid.transpose())
    
    edges = np.array(edges, np.int64)
    
    max_vertex = grid.max()
    
    all_verts = np.unique(edges)
    important_verts = all_verts[all_verts % 10 == 0]
    important_verts = pd.Index(important_verts)

    return edges, max_vertex, expected_paths, important_verts

    
def test_find_paths_from(toy_graph_setup):
    edges, max_vertex, expected_paths, important_verts = toy_graph_setup

    g = gt.Graph(directed=False)
    g.add_edge_list(edges)
    g.add_vertex(max_vertex)

    all_paths = {}

    def _find_from(v):
        all_paths[v] = []
        find_paths_from(v, important_verts, g, v, all_paths[v], [v], 0)

    starts = [10, 20, 30, 40, 60, 90, 100, 110]
    for start in starts:
        _find_from(start)

    for start in expected_paths.keys():
        assert_path_lists_equal(all_paths[start], expected_paths[start])


def test_find_all_paths(toy_graph_setup):
    edges, _max_vertex, expected_paths, important_verts = toy_graph_setup

    # Body grouping is determined by the 10's place (e.g. 20,21,22 are all in body 20)
    edge_verts = np.unique(edges)
    grouping = pd.DataFrame({'sv': edge_verts, 'group': edge_verts // 10 * 10})
    
    group_to_body = grouping.groupby('group').agg('min')
    group_to_body.columns = ['body']
    grouping = grouping.merge(group_to_body, left_on='group', right_index=True)
    body_mapping = grouping[['sv', 'body']].set_index('sv')['body']

    # Compute
    all_paths = find_all_paths(edges, body_mapping, important_verts)

    # Verify
    for start in expected_paths.keys():
        if start in edge_verts:
            assert_path_lists_equal(all_paths[start], expected_paths[start])


if __name__ == "__main__":
#     import sys
#     import logging
#     handler = logging.StreamHandler(sys.stdout)
#     logging.getLogger().addHandler(handler)
#     logging.getLogger().setLevel(logging.INFO)

    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_focused'])
