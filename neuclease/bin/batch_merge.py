import numpy as np
import pandas as pd
import networkx as nx
from neuclease.dvid import fetch_mapping, fetch_sizes, post_merge

def post_merges(instance_info, sv_merges):
    """
    Given a list of SV pairs to merge, map the SVs to their
    corresponding bodies, and then issue the minimum number
    merge commands to DVID, grouping "chained" merges together
    as needed.
    
    Args:
        instance_info:
            (server, uuid, segmentation_name)
        sv_merges:
            list/array of supervoxel pairs to merge

    Returns:
        If any merged sets contained multple named bodies whose names did not match,
        the merge is posted anyway, but a list of such merge sets is returned.
    """
    sv_merges = np.asarray(sv_merges)
    bodies_a = fetch_mapping(instance_info, sv_merges[:,0])
    bodies_b = fetch_mapping(instance_info, sv_merges[:,1])

    assert 0 not in bodies_a, "Some supervoxels no longer exist"
    assert 0 not in bodies_b, "Some supervoxels no longer exist"
    
    # Build graph of bodies to merge
    g = nx.Graph()
    g.add_edges_from(zip(bodies_a, bodies_b))
    
    # Extract bodies in connected groups
    flagged_merge_sets = []
    for body_set in nx.connected_components(g):
        if len(body_set) == 1:
            continue # sv_merges might contain self-loops, which are no-ops.
        
        body_list = list(body_set)
        sizes = fetch_sizes(instance_info, body_list)
        merge_df = pd.DataFrame(sizes, columns=['size'], index=body_list)
        
        # FIXME: How do I fetch body names?
        merge_df['name'] = ''
        
        rows_with_name = merge_df.query(" name != '' ")    
        if len(rows_with_name) == 0:
            # No names; choose biggest body
            main_body = merge_df['size'].idxmax()
        else:
            # Choose biggest NAMED body
            main_body = rows_with_name['size'].idxmax()

            # Are there multiple names in this set?
            if rows_with_name['name'].nunique() > 1:
                flagged_merge_sets.append(body_set)
    
        # Send to DVID
        other_bodies = body_set - {main_body}
        post_merge(instance_info, main_body, list(other_bodies))
        
    return flagged_merge_sets


if __name__ == "__main__":
    # Exmaple usage.
    #
    # FIXME: Load actual SV merge table from completed assignments
    sv_merges = [[1,2],
                 [1,3],
                 [3,4],
                 [5,6],
                 [7,8],
                 [7,8],
                 [8,9]]

    instance_info = ('emdata3:8900', '7e52', 'segmentation')
    flagged_sets = post_merges(instance_info, sv_merges)
