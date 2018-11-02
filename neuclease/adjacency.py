from itertools import combinations

import numpy as np
import pandas as pd

from neuclease.dvid.labelmap import fetch_labelindex, fetch_labelarray_voxels
from neuclease.util.graph import connected_components, connected_components_nonconsecutive

def find_missing_adjacencies(server, uuid, instance, body, known_edges, svs=None):
    """
    Given a body and an intra-body merge graph defined by the given
    list of "known" supervoxel-to-supervoxel edges within that body,
    
    1. Determine whether or not all supervoxels in the body are
       connected by a single component within the given graph.
       If so, return immediately.
    
    2. Attempt to augment the graph with additional edges based on
       supervoxel adjacencies in the segmentation from DVID.
       This is done by downloading the DVID labelindex to determine
       which blocks might contain adjacent supervoxels that could unify
       the graph, and then downloading those blocks (only) to search
       for the adjacencies.
    
    Notes:
        - This function does not attempt to find ALL adjacencies between supervoxels;
          it stops looking as soon as they form a single connected component.

        - This function only considers two supervoxels "adjacent" if they are
          literally touching each other in the scale-0 segmentation. If there is
          a small gap between them, then they are not considered adjacent.
        
        - This function does not attempt to find inter-block adjacencies;
          only adjacencies within each block are detected.
          So, in pathological cases where a supervoxel is only adjacent to the
          rest of the body on a block-aligned edge, the adjacency will not be
          detected by this funciton.
        
    Args:
        server, uuid, instance:
            DVID segmentation labelmap instance
        
        body:
            ID of the body to inspect
        
        known_edges:
            ndarray (N,2), array of supervoxel pairs;
            known edges of the intra-body merge graph
        
        svs:
            Optional. The complete list of supervoxels
            that belong to this body, according to DVID.
            Providing this enhances performance in one important case:
            If the known_edges ALREADY constitute a single connected component
            which covers all supervoxels in the body, there is no need to
            download the labelindex.
    
    Returns:
        (new_edges, orig_num_cc, final_num_cc),
        Where new_edges are the new edges found via inspection of supervoxel adjacencies,
        orig_num_cc is the number of disjoint components in the given merge graph before
        this function runs, and final_num_cc is the number of disjoint components after
        adding the new_edges.
        Ideally, final_num_cc == 1, but in some cases the body's supervoxels may not be
        directly adjacent, or the adjacencies were not detected.  (See notes above.)
    """
    li = None
    
    if svs is None:
        li = fetch_labelindex(server, uuid, instance, body, format='pandas')
        svs = pd.unique( li.blocks['sv'].values )

    cc = connected_components_nonconsecutive(known_edges, svs)
    orig_num_cc = final_num_cc = cc.max()+1

    if orig_num_cc == 1:
        return np.zeros((0,2), np.uint64), orig_num_cc, final_num_cc

    if li is None:
        # FIXME: This would be faster if we just used the protobuf format directly
        li = fetch_labelindex(server, uuid, instance, body, format='pandas')

    cc_mapping = pd.DataFrame({'sv': svs, 'cc': cc})
    blocks_df = li.blocks.merge(cc_mapping, how='left', on='sv')
    svs_set = set(svs)
    
    sv_adj_found = []
    cc_adj_found = set()
    for coord, df in blocks_df.groupby(['z', 'y', 'x']):
        # Given the supervoxels in this block, what CC adjacencies
        # MIGHT we find if we were to inspect the segmentation?
        possible_adjacencies = set(combinations( set(df['cc']), 2 ))
        
        # We only aim to find (at most) a single link between each CC pair.
        # That is, we don't care about adjacencies between CC that we've already linked so far.
        possible_adjacencies -= cc_adj_found
        if not possible_adjacencies:
            continue

        block_box = coord + np.array([[0,0,0], [64,64,64]])
        block_vol = fetch_labelarray_voxels(server, uuid, instance, block_box, supervoxels=True, scale=0)
        
        # Drop supervoxels that don't belong to this body
        block_flat = block_vol.reshape(-1)
        in_body = pd.Series(block_vol.reshape(-1)).isin(svs_set)
        block_flat[(~in_body).values] = 0
        block_vol = block_flat.reshape((64,64,64))

        sv_adjacencies = compute_label_adjacencies(block_vol)
        sv_adjacencies = sv_adjacencies.merge(cc_mapping, how='left', left_on='sv_a', right_on='sv')
        sv_adjacencies = sv_adjacencies.merge(cc_mapping, how='left', left_on='sv_b', right_on='sv', suffixes=['_a', '_b'])
        
        for row in sv_adjacencies.itertuples(index=False):
            if (row.cc_a != row.cc_b) and (row.cc_a, row.cc_b) not in cc_adj_found:
                cc_adj_found.add( (row.cc_a, row.cc_b) )
                sv_adj_found.append( (row.sv_a, row.sv_b) )
        
        # If we've unified all components, then we're done.
        final_num_cc = connected_components(np.array(list(cc_adj_found), np.uint64), orig_num_cc).max()+1
        if final_num_cc == 1:
            break
    
    return np.array(sv_adj_found, np.uint64), orig_num_cc, final_num_cc
        

def compute_label_adjacencies(vol, exclude_zero=True):
    adjacencies = []
    for axis in range(vol.ndim):
        adjacencies.append( compute_label_adjacencies_for_axis(vol, axis) )

    adj_df = pd.DataFrame(np.concatenate(adjacencies), columns=['sv_a', 'sv_b']).drop_duplicates()
    
    if exclude_zero:
        adj_df.query('sv_a != 0 and sv_b != 0', inplace=True)

    return adj_df
        

def compute_label_adjacencies_for_axis(vol, axis=0):
    up_slicing = ((slice(None),) * axis) + (np.s_[:-1],)
    down_slicing = ((slice(None),) * axis) + (np.s_[1:],)

    up_vol = vol[up_slicing]
    down_vol = vol[down_slicing]
    
    edge_mask = (up_vol != down_vol)
    
    edges = np.array( (up_vol[edge_mask], down_vol[edge_mask]) ).transpose()
    edges.sort(axis=1)
    
    return pd.DataFrame( edges ).drop_duplicates().values


if __name__ == "__main__":
    test_node = ('emdata3:9400', '9e0d')
    test_seg = (*test_node, 'segmentation')
    
    edges = np.array([[1250441100, 1250441119],
                      [1219406368, 1250441119],
                      [1250441100, 1250445613],
                      [1250441100, 1250445520],
                      [1250441088, 1250441100],
                      [1219406368, 1250441100],
                      [1250445520, 1250445524],
                      [1250441100, 1250445988]], dtype=np.uint64)
    
    missing_adj = find_missing_adjacencies(*test_seg, 1219406368, edges[:6])
    print(missing_adj)

