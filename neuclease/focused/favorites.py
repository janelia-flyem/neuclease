import numpy as np
import pandas as pd

def compute_favorites(edge_table_df):
    """
    Given a table with at least columns 'body_a', 'body_b', and 'score',
    
    Compute a table of each body's "favorite" edge in the table, (i.e. the miniumum score).
    
    Returns DataFrame indexed by body, with columns:
        - 'min_score': The score of the favorite edge
        - 'min_index': Where to find the favorite edge within your input table
        = 'side': Which side of the favorite edge the body is on.  Either 'a' or 'b'.
    """
    body_a_stats = edge_table_df.groupby('body_a').agg({'score': ['min', 'idxmin', 'size']})
    body_a_stats.columns = ['min_score', 'min_index', 'edge_count']
    
    body_b_stats = edge_table_df.groupby('body_b').agg({'score': ['min', 'idxmin', 'size']})
    body_b_stats.columns = ['min_score', 'min_index', 'edge_count']
    
    merged_body_stats = pd.DataFrame(body_a_stats).merge(body_b_stats, 'outer',
                                                         left_index=True, right_index=True,
                                                         suffixes=['_a', '_b'])
    
    # Combine scores/indexes and edge counts, and record which
    # position (a or b) the body was in for its best edge.
    min_scores = merged_body_stats[['min_score_a', 'min_score_b']].min(axis=1)
    edge_counts = merged_body_stats[['edge_count_a', 'edge_count_b']].sum(axis=1).astype(np.uint32)
    
    min_index = merged_body_stats['min_index_b'].copy()
    pos_a = (merged_body_stats['min_score_a'] < merged_body_stats['min_score_b']) | merged_body_stats['min_score_b'].isnull()
    min_index.loc[pos_a] = merged_body_stats['min_index_a']
    
    sides = pd.Series('b', index=merged_body_stats.index)
    sides.loc[pos_a] = 'a'
    sides.loc[min_index.isnull()] = 'x' # Can't normally happen. I don't know why this is here. 
    sides = sides.astype('category')
    
    # body_favorites is indexed by body
    body_favorites = pd.DataFrame({'min_score': min_scores,
                                   'min_index': min_index,
                                   'edge_count': edge_counts,
                                   'side': sides})
    
    body_favorites = body_favorites.drop(0, errors='ignore').query('side != "x"')
    return body_favorites


def mark_favorites(edge_table_df, body_favorites):
    favorite_flags_df = pd.DataFrame(columns=['is_favorite_of_a', 'is_favorite_of_b'], index=edge_table_df.index)
    favorite_flags_df['is_favorite_of_a'] = False
    favorite_flags_df['is_favorite_of_b'] = False
    
    body_a_favorite_edges = body_favorites.query("side == 'a'")['min_index'].astype(np.int64)
    body_b_favorite_edges = body_favorites.query("side == 'b'")['min_index'].astype(np.int64)
    
    favorite_flags_df.loc[body_a_favorite_edges.values, 'is_favorite_of_a'] = True
    favorite_flags_df.loc[body_b_favorite_edges.values, 'is_favorite_of_b'] = True
    
    return favorite_flags_df


def extract_favorites(edge_table_df, favorite_flags_df, only_mutual=False):
    if only_mutual:
        selected = favorite_flags_df.query('is_favorite_of_a and is_favorite_of_b')
    else:
        selected = favorite_flags_df.query('is_favorite_of_a or is_favorite_of_b')
    
    return edge_table_df.loc[selected.index]

