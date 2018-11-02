from itertools import combinations

import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from neuclease.dvid.labelmap import fetch_labelindex, fetch_labelarray_voxels, decode_labelindex_blocks
from neuclease.util.graph import connected_components, connected_components_nonconsecutive
from neuclease.dvid.labelmap._labelmap import fetch_supervoxels_for_body

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
    labelindex = None
    
    if svs is None:
        # We could compute the supervoxel list ourselves from the labelindex, but dvid can do it faster.
        svs = fetch_supervoxels_for_body(server, uuid, instance, body)
        labelindex = fetch_labelindex(server, uuid, instance, body, format='protobuf')

    cc = connected_components_nonconsecutive(known_edges, svs)
    orig_num_cc = final_num_cc = cc.max()+1

    if orig_num_cc == 1:
        return np.zeros((0,2), np.uint64), orig_num_cc, final_num_cc

    if labelindex is None:
        labelindex = fetch_labelindex(server, uuid, instance, body, format='protobuf')

    encoded_block_coords = np.fromiter(labelindex.blocks.keys(), np.uint64, len(labelindex.blocks))
    coords_zyx = decode_labelindex_blocks(encoded_block_coords)

    cc_mapper = LabelMapper(svs, cc)
    svs_set = set(svs)
    
    sv_adj_found = []
    cc_adj_found = set()
    for coord_zyx, sv_counts in zip(coords_zyx, labelindex.blocks.values()):
        # Given the supervoxels in this block, what CC adjacencies
        # MIGHT we find if we were to inspect the segmentation?
        block_svs = np.fromiter(sv_counts.counts.keys(), np.uint64)
        block_ccs = cc_mapper.apply(block_svs)
        possible_adjacencies = set(combinations( set(block_ccs), 2 ))
        
        # We only aim to find (at most) a single link between each CC pair.
        # That is, we don't care about adjacencies between CC that we've already linked so far.
        possible_adjacencies -= cc_adj_found
        if not possible_adjacencies:
            continue

        block_box = coord_zyx + np.array([[0,0,0], [64,64,64]])
        block_vol = fetch_labelarray_voxels(server, uuid, instance, block_box, supervoxels=True, scale=0)
        
        # Drop supervoxels that don't belong to this body
        block_flat = block_vol.reshape(-1)
        in_body = pd.Series(block_vol.reshape(-1)).isin(svs_set)
        block_flat[(~in_body).values] = 0
        block_vol = block_flat.reshape((64,64,64))

        sv_adjacencies = compute_label_adjacencies(block_vol)
        sv_adjacencies['cc_a'] = cc_mapper.apply(sv_adjacencies['sv_a'].values)
        sv_adjacencies['cc_b'] = cc_mapper.apply(sv_adjacencies['sv_b'].values)
        
        for row in sv_adjacencies.itertuples(index=False):
            if (row.cc_a != row.cc_b) and (row.cc_a, row.cc_b) not in cc_adj_found:
                cc_adj_found.add( (row.cc_a, row.cc_b) )
                sv_adj_found.append( (row.sv_a, row.sv_b) )
        
        # If we've unified all components, then we're done.
        final_num_cc = connected_components(np.array(list(cc_adj_found), np.uint64), orig_num_cc).max()+1
        if final_num_cc == 1:
            break
    
    return np.array(sv_adj_found, np.uint64), int(orig_num_cc), int(final_num_cc)
        

def compute_label_adjacencies(vol):
    adjacencies = []
    for axis in range(vol.ndim):
        adjacencies.append( _compute_label_adjacencies_for_axis(vol, axis) )

    adj = np.concatenate(adjacencies)
    adj_df = pd.DataFrame(adj, columns=['sv_a', 'sv_b'])
    adj_df = adj_df.drop_duplicates().copy()
    return adj_df
        

def _compute_label_adjacencies_for_axis(vol, axis=0):
    up_slicing = ((slice(None),) * axis) + (np.s_[:-1],)
    down_slicing = ((slice(None),) * axis) + (np.s_[1:],)

    up_vol = vol[up_slicing]
    down_vol = vol[down_slicing]

    # Edges    
    keep = (up_vol != down_vol)
    
    # Exclude edges to label 0
    keep &= (up_vol != 0)
    keep &= (down_vol != 0)
    
    edges = np.zeros((keep.sum(), 2), vol.dtype)
    edges[:,0] = up_vol[keep]
    edges[:,1] = down_vol[keep]
    edges.sort(axis=1)

    return edges
    


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

