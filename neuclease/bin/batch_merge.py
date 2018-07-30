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
    """
    df = pd.DataFrame(sv_merges, columns=['id_a', 'id_b'])
    df['body_a'] = fetch_mapping(instance_info, sv_merges['id_a'].values)
    df['body_b'] = fetch_mapping(instance_info, sv_merges['id_b'].values)
    
    # Build graph of bodies to merge
    g = nx.Graph()
    g.add_edges_from(df[['body_a', 'body_b']].values)
    
    # Pull out bodies in connected groups
    for body_set in nx.connected_components(g):
        bodies = list(body_set)
        sizes = fetch_sizes(instance_info, bodies)
        merge_df = pd.DataFrame(sizes, columns=['size'], index=bodies)
        
        # FIXME: How do I fetch body names?
        merge_df['name'] = ''
        
        rows_with_name = merge_df.query(" name != '' ")
    
        if len(rows_with_name) > 0:
            # Choose biggest NAMED body
            main_body = rows_with_name['size'].idxmax()
        else:
            # Choose biggest body
            main_body = merge_df['size'].idxmax()
    
        other_bodies = body_set - {main_body}
    
        # Send to DVID    
        post_merge(instance_info, main_body, list(other_bodies))

if __name__ == "__main__":
    # FIXME: Load actual SV merge table from completed assignments
    sv_merges = [[1,2],
                 [1,3],
                 [3,4],
                 [5,6],
                 [7,8],
                 [7,8],
                 [8,9]]

    instance_info = ('emdata3:8900', '7e52', 'segmentation')
    post_merges(instance_info, sv_merges)
