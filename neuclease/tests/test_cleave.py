import pytest
import numpy as np
from neuclease.cleave import cleave


def test_simple_cleave():
    # Simple graph (a line of 10 adjacent nodes)
    edges = list(zip(range(0,9), range(1,10)))
    edges = np.array(edges, dtype=np.uint32)
    
    # Edges are uniform, except the middle edge, which is more costly
    edge_weights = np.ones((len(edges),), dtype=np.float32) / 2
    edge_weights[len(edges)//2] *= 2

    # Seeds at both ends
    seeds_dict = { 1: [0], 2: [9] }
    
    (node_ids, output_labels, disconnected_components, contains_unlabeled_components) = cleave(edges, edge_weights, seeds_dict)
    
    assert (node_ids == np.arange(10)).all()
    assert not disconnected_components
    assert not contains_unlabeled_components
    assert (output_labels == [1,1,1,1,1,2,2,2,2,2]).all()


def test_discontiguous_components():
    # Simple graph (a line of 10 adjacent nodes)
    edges = list(zip(range(0,9), range(1,10)))
    
    edges = np.array(edges, dtype=np.uint32)
    
    # Edges are uniform
    edge_weights = np.ones((len(edges),), dtype=np.float32) / 2

    # Seeds on both sides
    seeds_dict = { 1: [3,6], 2: [4,5] }
    
    (node_ids, output_labels, disconnected_components, contains_unlabeled_components) = cleave(edges, edge_weights, seeds_dict, np.arange(10, dtype=np.uint64))
    
    assert (node_ids == np.arange(10)).all()
    assert (output_labels == [1,1,1,1,2,2,1,1,1,1]).all()
    assert disconnected_components
    assert not contains_unlabeled_components


def test_unlabeled_components():
    # Simple graph (a line of 10 adjacent nodes)
    edges = list(zip(range(0,9), range(1,10)))
     
    # Sever the last two nodes from the rest.
    del edges[-2]
     
    edges = np.array(edges, dtype=np.uint32)
     
    # Edges are uniform, except the middle edge, which is more costly
    edge_weights = np.ones((len(edges),), dtype=np.float32) / 2
    edge_weights[len(edges)//2] *= 2
 
    # Seeds on both sides
    seeds_dict = { 1: [0], 2: [7] }
     
    (node_ids, output_labels, disconnected_components, contains_unlabeled_components) = cleave(edges, edge_weights, seeds_dict, np.arange(10, dtype=np.uint64))
     
    assert (node_ids == np.arange(10)).all()
    assert not disconnected_components
    assert contains_unlabeled_components
    assert (output_labels == [1,1,1,1,1,2,2,2,0,0]).all()


def test_discontiguous_components_and_unlabeled_components():
    # Simple graph (a line of 10 adjacent nodes)
    edges = list(zip(range(0,9), range(1,10)))

    # Sever the last two nodes from the rest.
    del edges[-2]
    
    edges = np.array(edges, dtype=np.uint32)

    # Edges are uniform
    edge_weights = np.ones((len(edges),), dtype=np.float32) / 2

    # Seeds on both sides
    seeds_dict = { 1: [3,6], 2: [4,5] }
    
    (node_ids, output_labels, disconnected_components, contains_unlabeled_components) = cleave(edges, edge_weights, seeds_dict, np.arange(10, dtype=np.uint64))
    
    assert (node_ids == np.arange(10)).all()
    assert (output_labels == [1,1,1,1,2,2,1,1,0,0]).all()
    assert disconnected_components
    assert contains_unlabeled_components


def test_discontiguous_unlabeled_components():
    """
    If all seeded components are contiguous, but the unseeded components are discontiguous,
    don't list any discontiguous components, just return contains_unlabeled_components=True,
    as usual.
    """
    # Simple graph (a line of 10 adjacent nodes)
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
    
    (node_ids, output_labels, disconnected_components, contains_unlabeled_components) = cleave(edges, edge_weights, seeds_dict, np.arange(10, dtype=np.uint64))
    
    assert (node_ids == np.arange(10)).all()
    assert not disconnected_components
    assert contains_unlabeled_components
    assert (output_labels == [0,1,1,1,1,2,2,2,2,0]).all()


def test_stability():
    """
    Cleaving with the same inputs should always produce
    the same outputs, even if many weights are tied.
    """
    N = 1000
    E = 10_000

    # Randomly-generated graph.
    nodes = np.arange(N, dtype=np.uint32)
    edges = np.random.randint(0,N, size=(E,2), dtype=np.uint32)
    edge_weights = (np.random.randint(0,4, size=(edges.shape[0],)) / 5 + 0.2).astype(np.float32)

    seeds = { 1: np.random.randint(N, size=(20,)),
              2: np.random.randint(N, size=(20,)) }
    
    first_results = cleave(edges, edge_weights, seeds, nodes)

    # Repeat the cleave.  Should get the same results every time.
    for _ in range(5):
        repeat_results = cleave(edges, edge_weights, seeds, nodes)
        assert (repeat_results.node_ids == first_results.node_ids).all()
        assert (repeat_results.output_labels == first_results.output_labels).all()
        assert (repeat_results.disconnected_components == first_results.disconnected_components)
        assert (repeat_results.contains_unlabeled_components == first_results.contains_unlabeled_components)


if __name__ == "__main__":
    pytest.main()
