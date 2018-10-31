from itertools import combinations

import numpy as np
import pandas as pd

from neuclease.dvid.labelmap import fetch_labelindex, fetch_labelarray_voxels
from neuclease.util.graph import connected_components, connected_components_nonconsecutive

def find_missing_adjacencies(server, uuid, instance, body, known_edges, scale=0):
    li = fetch_labelindex(server, uuid, instance, body, format='pandas')
    blocks_df = li.blocks

    svs = pd.unique( blocks_df['sv'].values )
    svs_set = set(svs)
    cc = connected_components_nonconsecutive(known_edges, svs)
    if cc.max() == 0:
        return np.zeros((0,2), np.uint64)

    cc_mapping = pd.DataFrame({'sv': svs, 'cc': cc})
    blocks_df = blocks_df.merge(cc_mapping, how='left', on='sv')

    sv_adj_found = []
    
    num_cc = cc.max()+1
    
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

        coord = np.array(coord) // (2**scale)
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
        if not connected_components(np.array(list(cc_adj_found), np.uint64), num_cc).any():
            break
    
    return np.array(sv_adj_found, np.uint64)
        

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

