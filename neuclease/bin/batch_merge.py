import numpy as np
import pandas as pd
import networkx as nx
from neuclease.dvid import fetch_mapping, fetch_sizes, post_merge

def batch_merge(instance_info, sv_merges):
    """
    Given a list of SV pairs to merge, map the SVs to their
    corresponding bodies, and then merge their bodies via
    the minimum number merge commands to DVID, grouping 
    "chained" merges together into a single merge command.
    
    Args:
        instance_info:
            (server, uuid, segmentation_name)
        sv_merges:
            list/array of supervoxel pairs to merge

    Returns:
        It's suspicious when two bodies with different names are merged.
        When we detect a merge set with multiple unique names, the merge
        is posted anyway, but the list of those suspicious merge sets is
        returned.
    """
    sv_merges = np.asarray(sv_merges)
    assert sv_merges.ndim == 2 and sv_merges.shape[1] == 2

    bodies_a = fetch_mapping(instance_info, sv_merges[:,0])
    bodies_b = fetch_mapping(instance_info, sv_merges[:,1])

    assert 0 not in bodies_a, "Some supervoxels no longer exist"
    assert 0 not in bodies_b, "Some supervoxels no longer exist"
    
    # Build graph of bodies to merge
    g = nx.Graph()
    g.add_edges_from( zip(bodies_a, bodies_b) )
    
    # Extract bodies in connected groups
    flagged_merge_sets = []
    for body_set in nx.connected_components(g):
        if len(body_set) == 1:
            continue # sv_merges might contain self-loops, which are no-ops.
        
        body_list = list(body_set)
        body_df = pd.DataFrame(index=body_list)
        body_df['size'] = fetch_sizes(instance_info, body_list)
        body_df['name'] = '' # FIXME: How do I fetch body names?

        # If no named bodies, choose the biggest overall.
        # Otherwise, choose the biggest body that has a name.
        named_body_df = body_df.query(" name != '' ")
        if len(named_body_df) == 0:
            main_body = body_df['size'].idxmax()
        else:
            main_body = named_body_df['size'].idxmax()

            # Are there multiple names in this set? Flag it.
            if named_body_df['name'].nunique() > 1:
                flagged_merge_sets.append(body_set)
    
        # Send merge group to DVID
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
    flagged_sets = batch_merge(instance_info, sv_merges)
