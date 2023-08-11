from itertools import combinations

import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from ..dvid.labelmap import fetch_labelindex, fetch_labelarray_voxels, decode_labelindex_blocks
from ..util.graph import connected_components, connected_components_nonconsecutive
from ..dvid.labelmap._labelmap import fetch_supervoxels


def find_missing_adjacencies(server, uuid, instance, body, known_edges, svs=None, search_distance=1, connect_non_adjacent=False):
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
        - Requires scikit-image (which, currently, is not otherwise
          listed as a dependency of neuclease's conda-recipe).

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

        search_distance:
            If > 1, supervoxels are considered adjacent if they are within
            the given distance from each other, even if they aren't directly adjacent.

        connect_non_adjacent:
            If searching by adjacency failed to fully connect all supervoxels in the
            body into a single connected component, generate edges for supervoxels
            that are not adjacent, but merely are in the same block (if it helps
            unify the body).

    Returns:
        (new_edges, orig_num_cc, final_num_cc, block_tables),
        Where:
            new_edges are the new edges found via inspection of supervoxel adjacencies,

            orig_num_cc is the number of disjoint components in the given merge graph before
                this function runs,

            final_num_cc is the number of disjoint components after adding the new_edges,

            block_tables contains debug information about the adjacencies found in each
                block of analyzed segmentation

        Ideally, final_num_cc == 1, but in some cases the body's supervoxels may not be
        directly adjacent, or the adjacencies were not detected.  (See notes above.)
    """
    from skimage.morphology import dilation

    BLOCK_TABLE_COLS = ['z', 'y', 'x', 'sv_a', 'sv_b', 'cc_a', 'cc_b', 'detected', 'applied']
    known_edges = np.asarray(known_edges, np.uint64)
    if svs is None:
        # We could compute the supervoxel list ourselves from
        # the labelindex, but dvid can do it faster.
        svs = fetch_supervoxels(server, uuid, instance, body)

    cc = connected_components_nonconsecutive(known_edges, svs)
    orig_num_cc = final_num_cc = cc.max()+1

    if orig_num_cc == 1:
        return np.zeros((0,2), np.uint64), orig_num_cc, final_num_cc, pd.DataFrame(columns=BLOCK_TABLE_COLS)

    labelindex = fetch_labelindex(server, uuid, instance, body, format='protobuf')
    encoded_block_coords = np.fromiter(labelindex.blocks.keys(), np.uint64, len(labelindex.blocks))
    coords_zyx = decode_labelindex_blocks(encoded_block_coords)

    cc_mapper = LabelMapper(svs, cc)
    sv_adj_found = []
    cc_adj_found = set()
    block_tables = {}

    searched_block_svs = {}

    for coord_zyx, sv_counts in zip(coords_zyx, labelindex.blocks.values()):
        # Given the supervoxels in this block, what CC adjacencies
        # MIGHT we find if we were to inspect the segmentation?
        block_svs = np.fromiter(sv_counts.counts.keys(), np.uint64)
        block_ccs = cc_mapper.apply(block_svs)
        possible_cc_adjacencies = set(combinations(sorted(set(block_ccs)), 2))

        # We only aim to find (at most) a single link between each CC pair.
        # That is, we don't care about adjacencies between CC that we've already linked so far.
        possible_cc_adjacencies -= cc_adj_found
        if not possible_cc_adjacencies:
            continue

        searched_block_svs[(*coord_zyx,)] = block_svs

        # Not used in the search; only returned for debug purposes.
        block_adj_table = _init_adj_table(coord_zyx, block_svs, cc_mapper)

        block_vol = fetch_block_vol(server, uuid, instance, coord_zyx, block_svs)
        if search_distance > 0:
            # It would be nice to do a proper spherical dilation,
            # but apparently dilation() is special-cased to be WAY
            # faster with a square structuring element, and we prefer
            # speed over cleaner dilation.
            # footprint = skimage.morphology.ball(dilation)
            radius = search_distance // 2
            footprint = np.ones(3*(1+2*radius,), np.uint8)
            dilated_block_vol = dilation(block_vol, footprint)

            # Since dilation is a max-filter, we might have accidentally
            # erased small, low-valued supervoxels, erasing the adjacendies.
            # Overlay the original volume to make sure they still count.
            block_vol = np.where(block_vol, block_vol, dilated_block_vol)

        sv_adjacencies = compute_label_adjacencies(block_vol)
        sv_adjacencies['cc_a'] = cc_mapper.apply( sv_adjacencies['sv_a'].values )
        sv_adjacencies['cc_b'] = cc_mapper.apply( sv_adjacencies['sv_b'].values )

        # Normalize
        # Note: This could swap only cc (or sv) without swapping sv (or cc),
        #       but that doesn't matter here.
        swap_cc = sv_adjacencies.eval('cc_a > cc_b')
        swap_sv = sv_adjacencies.eval('sv_a > sv_b')
        sv_adjacencies.loc[swap_cc, ['cc_a', 'cc_b']] = sv_adjacencies.loc[swap_cc, ['cc_b', 'cc_a']]
        sv_adjacencies.loc[swap_sv, ['sv_a', 'sv_b']] = sv_adjacencies.loc[swap_sv, ['sv_b', 'sv_a']]

        found_new_adj = False
        for row in sv_adjacencies.query('cc_a != cc_b').itertuples(index=False):
            sv_adj = (row.sv_a, row.sv_b)
            cc_adj = (row.cc_a, row.cc_b)

            block_adj_table.loc[sv_adj, 'detected'] = True
            if cc_adj in cc_adj_found:
                continue

            found_new_adj = True
            cc_adj_found.add( cc_adj )
            sv_adj_found.append( sv_adj )
            block_adj_table.loc[sv_adj, 'applied'] = True

        block_tables[(*coord_zyx,)] = block_adj_table

        # If we made at least one change and we've
        # finally unified all components, then we're done.
        if found_new_adj:
            final_num_cc = connected_components(np.array(list(cc_adj_found), np.uint64), orig_num_cc).max()+1
            if final_num_cc == 1:
                break

    # If we couldn't connect everything via direct adjacencies,
    # we can just add edges for any supervoxels that share a block.
    if final_num_cc > 1 and connect_non_adjacent:
        for coord_zyx, block_svs in searched_block_svs.items():
            block_ccs = cc_mapper.apply(block_svs)

            # We only need one SV per connected component,
            # so load them into a dict.
            selected_svs = dict(zip(block_ccs, block_svs))
            for (sv_a, sv_b) in combinations(sorted(selected_svs.values()), 2):
                (cc_a, cc_b) = cc_mapper.apply(np.array([sv_a, sv_b], np.uint64))
                if cc_a > cc_b:
                    cc_a, cc_b = cc_b, cc_a

                if (cc_a, cc_b) not in cc_adj_found:
                    if sv_a > sv_b:
                        sv_a, sv_b = sv_b, sv_a

                    cc_adj_found.add( (cc_a, cc_b) )
                    sv_adj_found.append( (sv_a, sv_b) )

                    block_tables[(*coord_zyx,)].loc[(sv_a, sv_b), 'applied'] = True

        final_num_cc = connected_components(np.array(list(cc_adj_found), np.uint64), orig_num_cc).max()+1

    if len(block_tables) == 0:
        block_table = pd.DataFrame(columns=BLOCK_TABLE_COLS)
    else:
        block_table = pd.concat(block_tables.values(), sort=False).reset_index()
        block_table = block_table[BLOCK_TABLE_COLS]

    new_edges = np.array(sv_adj_found, np.uint64)
    return new_edges, int(orig_num_cc), int(final_num_cc), block_table


def fetch_block_vol(server, uuid, instance, coord_zyx, svs_set=None):
    """
    Fetch a block of segmentation starting at the given coordinate.
    Optionally filter out (set to 0) any supervoxels that do not belong to the given svs_set.
    """
    block_box = coord_zyx + np.array([[0,0,0], [64,64,64]])
    block_vol = fetch_labelarray_voxels(server, uuid, instance, block_box, supervoxels=True, scale=0)

    if svs_set is None:
        return block_vol

    apply_mask_for_labels(block_vol, svs_set, inplace=True)
    return block_vol


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


def _init_adj_table(coord_zyx, block_svs, cc_mapper):
    block_adj_table = pd.DataFrame(list(combinations( sorted(set(block_svs)), 2 )), columns=['sv_a', 'sv_b'], dtype=np.uint64)
    block_adj_table['cc_a'] = cc_mapper.apply(block_adj_table['sv_a'].values)
    block_adj_table['cc_b'] = cc_mapper.apply(block_adj_table['sv_b'].values)
    block_adj_table.query('cc_a != cc_b', inplace=True)

    block_adj_table['detected'] = False
    block_adj_table['applied'] = False
    block_adj_table.set_index(['sv_a', 'sv_b'], inplace=True)
    block_adj_table = block_adj_table.assign(**dict(zip('zyx', coord_zyx)))
    return block_adj_table


def export_debug_volumes(server, uuid, instance, body, block_table, outdir='/tmp'):
    import os
    from tqdm import tqdm
    svs = fetch_supervoxels(server, uuid, instance, body)
    for coord_zyx, df in tqdm(block_table.groupby(['z', 'y', 'x'])):
        block_vol = fetch_block_vol(server, uuid, instance, coord_zyx, svs)
        
        first_index = df.index[0]
        z, y, x = coord_zyx
        block_dir = f'{outdir}/{first_index:03d}-z{z:05d}-y{y:05d}-x{x:05d}'
        os.makedirs(block_dir, exist_ok=True)
        
        np.save(f'{block_dir}/block.npy', block_vol)
        for row in df.itertuples():
            filtered_vol = apply_mask_for_labels(block_vol, {row.sv_a, row.sv_b})
            np.save(f'{block_dir}/filtered-{row.Index}.npy', filtered_vol)
        

if __name__ == "__main__":
#     test_node = ('emdata3:9400', '9e0d')
#     test_seg = (*test_node, 'segmentation')
#     
#     edges = np.array([[1250441100, 1250441119],
#                       [1219406368, 1250441119],
#                       [1250441100, 1250445613],
#                       [1250441100, 1250445520],
#                       [1250441088, 1250441100],
#                       [1219406368, 1250441100],
#                       [1250445520, 1250445524],
#                       [1250441100, 1250445988]], dtype=np.uint64)
#     
#     missing_adj = find_missing_adjacencies(*test_seg, 1219406368, edges[:6])
#     print(missing_adj)

    pd.set_option('display.max_columns', 10)
    
    test_node = ('emdata4:8900', 'a0df')
    test_seg = (*test_node, 'segmentation')
    body = 1138501508

    #edges = np.load(f'/tmp/edges-{body}.npy').astype(np.uint64)

    # Find edges from scratch
    edges, orig_num_cc, final_num_cc, block_table = \
        find_missing_adjacencies(*test_seg, body, np.zeros((0,2), dtype=np.uint64), svs=None, search_distance=10, connect_non_adjacent=True)

    # Try again, but leave out most of the edges
    missing_edges, orig_num_cc, final_num_cc, block_table = \
        find_missing_adjacencies(*test_seg, body, edges[:10], svs=None, search_distance=10, connect_non_adjacent=True)

    print(orig_num_cc, final_num_cc)
    print(block_table)
