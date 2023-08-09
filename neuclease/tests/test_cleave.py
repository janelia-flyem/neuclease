import pytest
import numpy as np
from neuclease.cleave.cleave import cleave, CleaveResults


@pytest.fixture(params=('seeded-mst', 'agglomerative-clustering')) # skipping 'seeded-watershed',
def cleave_method(request):
    yield request.param
    

def test_simple_cleave(cleave_method):
    # Simple graph (a line of 10 adjacent nodes)
    node_ids = np.arange(10, dtype=np.uint64)
    edges = list(zip(range(0,9), range(1,10)))
    edges = np.array(edges, dtype=np.uint32)
    
    # Edges are uniform, except the middle edge, which is more costly
    edge_weights = np.ones((len(edges),), dtype=np.float32) / 2
    edge_weights[len(edges)//2] *= 2

    # Seeds at both ends
    seeds_dict = { 1: [0], 2: [9] }
    
    cleave_results = cleave(edges, edge_weights, seeds_dict, node_ids, method=cleave_method)
    assert isinstance(cleave_results, CleaveResults)
    output_labels, disconnected_components, contains_unlabeled_components = cleave_results
    
    assert (node_ids == np.arange(10)).all()
    assert not disconnected_components
    assert not contains_unlabeled_components
    assert (output_labels == [1,1,1,1,1,2,2,2,2,2]).all()


def test_discontiguous_components(cleave_method):
    # Simple graph (a line of 10 adjacent nodes)
    node_ids = np.arange(10, dtype=np.uint64)
    edges = list(zip(range(0,9), range(1,10)))
    
    edges = np.array(edges, dtype=np.uint32)
    
    # Edges are uniform
    edge_weights = np.ones((len(edges),), dtype=np.float32) / 2

    # Seeds on both sides
    seeds_dict = { 1: [3,6], 2: [4,5] }
    
    cleave_results = cleave(edges, edge_weights, seeds_dict, node_ids, method=cleave_method)
    assert isinstance(cleave_results, CleaveResults)
    output_labels, disconnected_components, contains_unlabeled_components = cleave_results
    
    assert (output_labels == [1,1,1,1,2,2,1,1,1,1]).all()
    assert disconnected_components
    assert not contains_unlabeled_components


def test_unlabeled_components(cleave_method):
    # Simple graph (a line of 10 adjacent nodes)
    node_ids = np.arange(10, dtype=np.uint64)
    edges = list(zip(range(0,9), range(1,10)))
     
    # Sever the last two nodes from the rest.
    del edges[-2]
     
    edges = np.array(edges, dtype=np.uint32)
     
    # Edges are uniform, except the middle edge, which is more costly
    edge_weights = np.ones((len(edges),), dtype=np.float32) / 2
    edge_weights[len(edges)//2] *= 2
 
    # Seeds on both sides
    seeds_dict = { 1: [0], 2: [7] }
     
    cleave_results = cleave(edges, edge_weights, seeds_dict, node_ids, method=cleave_method)
    assert isinstance(cleave_results, CleaveResults)
    output_labels, disconnected_components, contains_unlabeled_components = cleave_results
    
    assert not disconnected_components
    assert contains_unlabeled_components
    assert (output_labels == [1,1,1,1,1,2,2,2,0,0]).all()


def test_discontiguous_components_and_unlabeled_components(cleave_method):
    # Simple graph (a line of 10 adjacent nodes)
    node_ids = np.arange(10, dtype=np.uint64)
    edges = list(zip(range(0,9), range(1,10)))

    # Sever the last two nodes from the rest.
    del edges[-2]
    
    edges = np.array(edges, dtype=np.uint32)

    # Edges are uniform
    edge_weights = np.ones((len(edges),), dtype=np.float32) / 2

    # Seeds on both sides
    seeds_dict = { 1: [3,6], 2: [4,5] }
    
    cleave_results = cleave(edges, edge_weights, seeds_dict, node_ids, method=cleave_method)
    assert isinstance(cleave_results, CleaveResults)
    output_labels, disconnected_components, contains_unlabeled_components = cleave_results
    
    assert (output_labels == [1,1,1,1,2,2,1,1,0,0]).all()
    assert disconnected_components
    assert contains_unlabeled_components


def test_discontiguous_unlabeled_components(cleave_method):
    """
    If all seeded components are contiguous, but the unseeded components are discontiguous,
    don't list any discontiguous components, just return contains_unlabeled_components=True,
    as usual.
    """
    # Simple graph (a line of 10 adjacent nodes)
    node_ids = np.arange(10, dtype=np.uint64)
    edges = list(zip(range(0,9), range(1,10)))

    # Sever the end nodes from the rest.
    del edges[0]
    del edges[-1]
    
    edges = np.array(edges, dtype=np.uint32)

    # Edges are uniform, except the middle edge, which is more costly
    edge_weights = np.ones((len(edges),), dtype=np.float32) / 2
    edge_weights[len(edges)//2] *= 2

    # Seeds on both sides
    seeds_dict = { 1: [1], 2: [8] }
    
    cleave_results = cleave(edges, edge_weights, seeds_dict, node_ids, method=cleave_method)
    assert isinstance(cleave_results, CleaveResults)
    output_labels, disconnected_components, contains_unlabeled_components = cleave_results
    
    assert not disconnected_components
    assert contains_unlabeled_components
    assert (output_labels == [0,1,1,1,1,2,2,2,2,0]).all()


@pytest.mark.xfail(reason='None of our cleaving methods are stable.')
def test_stability(cleave_method):
    """
    Cleaving with the same inputs should always produce
    the same outputs, even if many weights are tied.
    """
    N = 1000
    E = 10_000

    # Randomly-generated graph.
    node_ids = np.arange(N, dtype=np.uint32)
    edges = np.random.randint(0,N, size=(E,2), dtype=np.uint32)
    edge_weights = (np.random.randint(0,4, size=(edges.shape[0],)) / 5 + 0.2).astype(np.float32)

    seeds = { 1: np.random.randint(N, size=(20,)),
              2: np.random.randint(N, size=(20,)) }
    
    first_results = cleave(edges, edge_weights, seeds, node_ids, method=cleave_method)

    # Repeat the cleave.  Should get the same results every time.
    for _ in range(5):
        repeat_results = cleave(edges, edge_weights, seeds, node_ids)
        assert (repeat_results.output_labels == first_results.output_labels).all()
        assert (repeat_results.disconnected_components == first_results.disconnected_components)
        assert (repeat_results.contains_unlabeled_components == first_results.contains_unlabeled_components)


def test_empty_cleave(cleave_method):
    # Simple graph (a line of 10 adjacent nodes)
    node_ids = 10*np.arange(10, dtype=np.uint64)
    
    # No edges
    edges = np.zeros((0,2), dtype=np.uint64)
    edge_weights = np.zeros((0,2), dtype=np.float32)

    # Seeds at both ends
    seeds_dict = { 1: [0], 2: [90] }
    
    cleave_results = cleave(edges, edge_weights, seeds_dict, node_ids, method=cleave_method)
    assert isinstance(cleave_results, CleaveResults)
    output_labels, disconnected_components, contains_unlabeled_components = cleave_results
    
    assert disconnected_components == set(seeds_dict.keys())
    assert contains_unlabeled_components
    assert (output_labels == [1,0,0,0,0,0,0,0,0,2]).all()


def test_empty_cleave_complete_seeds(cleave_method):
    """
    Test proper outputs of "empty cleave' (i.e. no edges in the graph)
    when all nodes have been manually seeded.
    """
    # Trivial graph: 2 nodes, no edges
    node_ids = np.asarray([10,20], np.uint64)
    edges = np.zeros((0,2), dtype=np.uint64)
    edge_weights = np.zeros((0,2), dtype=np.float32)

    # Both nodes seeded
    seeds_dict = { 1: [10], 2: [20] }
    
    cleave_results = cleave(edges, edge_weights, seeds_dict, node_ids, method=cleave_method)
    assert isinstance(cleave_results, CleaveResults)
    output_labels, disconnected_components, contains_unlabeled_components = cleave_results
    
    assert disconnected_components == set(seeds_dict.keys())
    assert not contains_unlabeled_components
    assert (output_labels == [1,2]).all()


def test_echo_seeds():
    # Simple graph (a line of 10 adjacent nodes)
    node_ids = np.arange(10, dtype=np.uint64)
    edges = list(zip(range(0,9), range(1,10)))
    edges = np.array(edges, dtype=np.uint32)
    
    # Edge weights don't matter for this test.
    edge_weights = np.ones((len(edges),), dtype=np.float32) / 2

    seeds_dict = { 1: [0,2], 2: [9] }
    
    cleave_results = cleave(edges, edge_weights, seeds_dict, node_ids, method='echo-seeds')
    assert isinstance(cleave_results, CleaveResults)
    output_labels, disconnected_components, contains_unlabeled_components = cleave_results
    
    assert disconnected_components == {1}
    assert contains_unlabeled_components
    assert (output_labels == [1,0,1,0,0,0,0,0,0,2]).all()


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_cleave'])
