from collections import namedtuple
import numpy as np
import pandas as pd
import vigra.graphs as vg
import nifty.graph

from dvidutils import LabelMapper

from .util import connected_components

class InvalidCleaveMethodError(Exception):
    pass

def get_cleave_method(method_string):
    """
    Given the name of a cleave method algorithm, return the function to
    call for that algorithm, along with a bool indicating whether or not
    the algorithm expects valid node sizes as input.
    
    Returns (func, requires_sizes)
    """
    if method_string == 'seeded-watershed':
        return edge_weighted_watershed, False
    if method_string == 'agglomerative-clustering':
        return agglomerative_clustering, False
    if method_string == 'agglomerative-clustering-with-ward':
        return agglomerative_clustering, True
    if method_string == 'echo-seeds':
        return echo_seeds, False
    
    raise RuntimeError(f"Invalid cleave method name: {method_string}")


CleaveResults = namedtuple("CleaveResults", "output_labels disconnected_components contains_unlabeled_components")
def cleave(edges, edge_weights, seeds_dict, node_ids, node_sizes=None, method='seeded-watershed'):
    """
    Cleave the graph with the given edges and edge weights.
    
    Args:
        
        edges:
            array, (E,2), uint32
        
        edge_weights:
            array, (E,), float32
        
        seeds_dict:
            dict, { seed_class : [node_id, node_id, ...] }
        
        node_ids:
            The complete list of node IDs in the graph. Must contain a superset of the ids given in edges.
            Extra ids in node_ids (i.e. not mentioned in 'edges') will be included
            in the results as disconnected components.
        
        method:
            One of: 'seeded-watershed', 'agglomerative-clustering', 'echo-seeds'

    Returns:
    
        CleaveResults, namedtuple with fields:
        (node_ids, output_labels, disconnected_components, contains_unlabeled_components)
        
        Where:
            node_ids:
                The graph node_ids.
                
            output_labels:
                array (N,), uint32
                Agglomerated node labeling, in the same order as node_ids.
                
            disconnected_components:
                A set of seeds which ended up with more than one component in the result.
            
            contains_unlabeled_components:
                True if the input contains one or more disjoint components that were not seeded
                and thus not labeled during agglomeration. False otherwise.
        
    """
    assert isinstance(node_ids, np.ndarray)
    assert node_ids.dtype in (np.uint32, np.uint64)
    assert node_ids.ndim == 1
    assert node_sizes is None or node_sizes.shape == node_ids.shape

    cleave_func, requires_sizes = get_cleave_method(method)
    assert not requires_sizes or node_sizes is not None, \
        f"The specified cleave method ({method}) requires node sizes but none were provided."
    
    # Relabel node ids consecutively
    cons_node_ids = np.arange(len(node_ids), dtype=np.uint32)
    mapper = LabelMapper(node_ids, cons_node_ids)

    # Initialize sparse seed label array
    seed_labels = np.zeros_like(cons_node_ids)
    for seed_class, seed_nodes in seeds_dict.items():
        seed_nodes = np.asarray(seed_nodes, dtype=np.uint64)
        mapper.apply_inplace(seed_nodes)
        seed_labels[seed_nodes] = seed_class
    
    if len(edges) == 0:
        # No edges: Return empty results (just seeds)
        return CleaveResults(seed_labels, set(seeds_dict.keys()), not seed_labels.all())

    # Clean the edges (normalized form, no duplicates, no loops)
    edges.sort(axis=1)
    edges_df = pd.DataFrame({'u': edges[:,0], 'v': edges[:,1], 'weight': edge_weights})
    edges_df.drop_duplicates(['u', 'v'], keep='last', inplace=True)
    edges_df = edges_df.query('u != v')
    edges = edges_df[['u', 'v']].values
    edge_weights = edges_df['weight'].values

    # Relabel edges for consecutive nodes
    cons_edges = mapper.apply(edges)
    assert cons_edges.dtype == np.uint32

    cleave_results = cleave_func(cons_edges, edge_weights, seed_labels, node_sizes)
    assert isinstance(cleave_results, CleaveResults)
    return cleave_results


def agglomerative_clustering(cleaned_edges, edge_weights, seed_labels, node_sizes=None, num_classes=None):
    """
    Run vigra.graphs.agglomerativeClustering() on the given graph with N nodes and E edges.
    The graph node IDs must be consecutive, starting with zero, dtype=np.uint32
    
    
    Args:
        cleaned_edges:
            array, (E,2), uint32
            Node IDs should be consecutive (more-or-less).
            To avoid segfaults:
                - Must not contain duplicates.
                - Must not contain 'loops' (no self-edges).
        
        edge_weights:
            array, (E,), float32
        
        seed_labels:
            array (N,), uint32
            All un-seeded nodes should be marked as 0.
        
    Returns:
        (output_labels, disconnected_components, contains_unlabeled_components)
        
        Where:
        
            output_labels:
                array (N,), uint32
                Agglomerated node labeling.
                
            disconnected_components:
                A set of seeds which ended up with more than one component in the result.
            
            contains_unlabeled_components:
                True if the input contains one or more disjoint components that were not seeded
                and thus not labeled during agglomeration. False otherwise.
    """
    #
    # Notes:
    # 
    # vigra.graphs.agglomerativeClustering() is somewhat sophisticated.
    #
    # During agglomeration, edges are selected for 'contraction' and the corresponding nodes are merged.
    # The newly merged node contains the superset of the edges from its constituent nodes, with duplicate
    # edges combined via weighted average according to their relative 'edgeLengths'.
    # 
    # The edge weights used in the optimization are adjusted dynamically after every merge.
    # The dynamic edge weight is computed as a weighted average of it's original 'edgeWeight'
    # and the similarity of its two nodes (by distance between 'nodeFeatures',
    # using the distance measure defined by 'metric').
    #
    # The relative importances of the original edgeWeight and the node similarity is determined by 'beta'.
    # To ignore node feature similarity completely, use beta=0.0.  To ignore edgeWeights completely, use beta=1.0.
    #
    # After computing that weighted average, the dynamic edge weight is then scaled by a 'Ward factor',
    # which seems to give priority to edges that connect smaller components.
    # The importance of the 'Ward factor' is determined by 'wardness'. To disable it, set wardness=0.0.
    #
    # 
    # For reference, here are the relevant lines from vigra/hierarchical_clustering.hxx:
    #
    #    ValueType getEdgeWeight(const Edge & e){
    #        ...
    #        const ValueType wardFac = 2.0 / ( 1.0/std::pow(sizeU,wardness_) + 1/std::pow(sizeV,wardness_) );
    #        const ValueType fromEdgeIndicator = edgeIndicatorMap_[ee];
    #        ValueType fromNodeDist = metric_(nodeFeatureMap_[uu],nodeFeatureMap_[vv]);
    #        ValueType totalWeight = ((1.0-beta_)*fromEdgeIndicator + beta_*fromNodeDist)*wardFac;
    #        ...
    #    }
    #        
    #
    # To achieve the "most naive" version of hierarchical clustering,
    # i.e. based purely on pre-computed edge weights (and no node features),
    # use beta=0.0, wardness=0.0.
    #
    # (Ideally, we would also set nodeSizes=[0,...], but unfortunately,
    # setting nodeSizes of 0.0 seems to result in strange bugs.
    # Therefore, we can't avoid the affect of using cumulative node size during the agglomeration.)

    assert cleaned_edges.dtype == np.uint32
    assert cleaned_edges.ndim == 2
    assert cleaned_edges.shape[1] == 2
    assert edge_weights.shape == (len(cleaned_edges),)
    assert seed_labels.ndim == 1
    assert cleaned_edges.max() < len(seed_labels)
    
    # Initialize graph
    # (These params merely reserve RAM in advance. They don't initialize actual graph state.)
    g = vg.AdjacencyListGraph(len(seed_labels), len(cleaned_edges))
    
    # Make sure there are the correct number of nodes.
    # (Internally, AdjacencyListGraph ensures contiguous nodes are created
    # up to the max id it has seen, so adding the max node is sufficient to
    # ensure all nodes are present.) 
    g.addNode(len(seed_labels)-1)

    # Insert edges.
    g.addEdges(cleaned_edges)
    
    if num_classes is None:
        num_classes = len(set(pd.unique(seed_labels)) - set([0]))
    
    output_labels = vg.agglomerativeClustering( graph=g,
                                                edgeWeights=edge_weights,
                                                #edgeLengths=...,
                                                #nodeFeatures=...,
                                                #nodeSizes=...,
                                                nodeLabels=seed_labels,
                                                nodeNumStop=num_classes,
                                                beta=0.0,
                                                #metric='l1',
                                                wardness=0.0 )
    
    # For some reason, the output labels do not necessarily
    # have the same values as the seed labels. We have to relabel them ourselves.
    #
    # Furthermore, there are some special cases to consider:
    # 
    # 1. It is possible that some seeds will map to disconnected components,
    #    if one of the following is true:
    #      - The input contains disconnected components with identical seeds
    #      - The input contains no disconnected components, but it failed to
    #        connect two components with identical seeds (some other seeded
    #        component ended up blocking the path between the two disconnected
    #        components).
    #    In those cases, we should ensure that the disconnected components are
    #    still labeled with the right input seed, but add the seed to the returned
    #    'disconnected components' set.
    #
    # 2. If the input contains any disconnected components that were NOT seeded,
    #    we should relabel those as 0, and return contains_unlabeled_components=True

    # Get mapping of seeds -> corresponding agg values.
    # (There might be more than one agg value for a given seed, as explained in point 1 above)
    df = pd.DataFrame({'seed': seed_labels, 'agg': output_labels})
    df.drop_duplicates(inplace=True)

    # How many unique agg values are there for each seed class?
    seed_mapping_df = df.query('seed != 0')
    seed_component_counts = seed_mapping_df.groupby(['seed']).agg({'agg': 'size'})
    seed_component_counts.columns = ['component_count']

    # More than one agg value for a seed class implies that it wasn't fully agglomerated.
    disconnected_components = set(seed_component_counts.query('component_count > 1').index)

    # If there are 'extra' agg values (not corresponding to seeds),
    # then some component(s) are unlabeled. (Point 2 above.)
    _seeded_agg_ids = set(seed_mapping_df['agg'])
    nonseeded_agg_ids = df.query('agg not in @_seeded_agg_ids')['agg']
    contains_unlabeled_components = (len(nonseeded_agg_ids) > 0)

    # Map from output agg values back to original seed classes.
    agg_values = seed_mapping_df['agg'].values
    seed_values = seed_mapping_df['seed'].values
    if len(nonseeded_agg_ids) > 0:
        nonseeded_agg_ids = np.fromiter(nonseeded_agg_ids, np.uint32)
        agg_values = np.concatenate((agg_values, nonseeded_agg_ids))
        seed_values = np.concatenate((seed_values, np.zeros((len(nonseeded_agg_ids),), np.uint32)))

    mapper = LabelMapper(agg_values, seed_values)
    mapper.apply_inplace(output_labels)
    
    return CleaveResults(output_labels, disconnected_components, contains_unlabeled_components)


def edge_weighted_watershed(cleaned_edges, edge_weights, seed_labels, _node_sizes=None):
    """
    Run nifty.graph.edgeWeightedWatershedsSegmentation() on the given graph with N nodes and E edges.
    The graph node IDs must be consecutive, starting with zero, dtype=np.uint32
    
    
    Args:
        cleaned_edges:
            array, (E,2), uint32
            Node IDs should be consecutive (more-or-less).
            To avoid segfaults:
                - Must not contain duplicates.
                - Must not contain 'loops' (no self-edges).
        
        edge_weights:
            array, (E,), float32
        
        seed_labels:
            array (N,), uint32
            All un-seeded nodes should be marked as 0.
        
    Returns:
        (output_labels, disconnected_components, contains_unlabeled_components)
        
        Where:
        
            output_labels:
                array (N,), uint32
                Agglomerated node labeling.
                
            disconnected_components:
                A set of seeds which ended up with more than one component in the result.
            
            contains_unlabeled_components:
                True if the input contains one or more disjoint components that were not seeded
                and thus not labeled during agglomeration. False otherwise.
    """
    assert cleaned_edges.dtype == np.uint32
    assert cleaned_edges.ndim == 2
    assert cleaned_edges.shape[1] == 2
    assert edge_weights.shape == (len(cleaned_edges),)
    assert seed_labels.ndim == 1
    assert cleaned_edges.max() < len(seed_labels)
    
    # Run the watershed
    g = nifty.graph.UndirectedGraph(len(seed_labels))
    g.insertEdges(cleaned_edges)
    output_labels = nifty.graph.edgeWeightedWatershedsSegmentation(g, seed_labels, edge_weights)
    assert len(output_labels) == len(seed_labels)
    contains_unlabeled_components = not output_labels.all()
    disconnected_components = _find_disconnected_components(cleaned_edges, output_labels)
    return CleaveResults(output_labels, disconnected_components, contains_unlabeled_components)


def echo_seeds(cleaned_edges, _edge_weights, seed_labels, _node_sizes):
    """
    A dummy 'cleave' stand-in that merely echoes the seeds back,
    with correctly reported disconnected_components.
    """
    output_labels = seed_labels # echo seeds input
    disconnected_components = _find_disconnected_components(cleaned_edges, output_labels)
    contains_unlabeled_components = not output_labels.all()
    return CleaveResults(output_labels, disconnected_components, contains_unlabeled_components)


def _find_disconnected_components(cleaned_edges, output_labels):
    """
    Given a graph defined by cleaned_edges and a node labeling in output_labels,
    Check if any output labels are split among discontiguous groups,
    and return the set of output label IDs for such objects.
    """
    # Figure out which edges were 'cut' (endpoints got different labels)
    # and which were preserved
    mapper = LabelMapper(np.arange(output_labels.shape[0], dtype=np.uint32), output_labels)
    labeled_edges = mapper.apply(cleaned_edges)
    preserved_edges = cleaned_edges[labeled_edges[:,0] == labeled_edges[:,1]]
    
    # Compute CC on the graph WITHOUT cut edges (keep only preserved edges)
    component_labels = connected_components(preserved_edges, len(output_labels))
    assert len(component_labels) == len(output_labels)
    
    # Align node output labels to their connected component labels
    cc_df = pd.DataFrame({'label': output_labels, 'cc': component_labels})
    
    # How many unique connected component labels are associated with each output label?
    cc_counts = cc_df.groupby('label').nunique()['cc']

    # Any output labels that map to multiple CC labels are 'disconnected components' in the output.
    disconnected_cc_counts = cc_counts[cc_counts > 1]
    disconnected_components = set(disconnected_cc_counts.index) - set([0])
    
    return disconnected_components


