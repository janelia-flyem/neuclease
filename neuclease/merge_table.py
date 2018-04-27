import os
import csv
import numpy as np
import pandas as pd

# TODO: Move these funtions into neuclease, and make
# DVIDSparkServies depend on neuclease, not the other way around.
from DVIDSparkServices.graph_comparison import normalize_merge_table, MERGE_TABLE_DTYPE

def load_merge_table(path, normalize=True, set_multiindex=True, scores_only=True):
    ext = os.path.splitext(path)[1]
    assert ext in ('.npy', '.csv'), f"Invalid file extension: {ext}"
    if ext == '.npy':
        merge_table_df = load_ffn_merge_table(path, normalize)
    elif ext == '.csv':
        merge_table_df = load_celis_csv(path, normalize)

    if scores_only:
        merge_table_df = merge_table_df[['id_a', 'id_b', 'score']]

    if set_multiindex:
        idx_columns = (merge_table_df['id_a'], merge_table_df['id_b'])
        merge_table_df.index = pd.MultiIndex.from_arrays(idx_columns, names=['idx_a', 'idx_b'])
        merge_table_df.sort_index(ascending=True, inplace=True)
    return merge_table_df

def load_celis_csv(csv_path, normalize=True):
    """
    Jeremy's CELIS exports are given in CSV format, with the following columns:
    segment_a,segment_b,score,x,y,z
    
    This isn't sufficient for every use-case because we
    would prefer to have TWO representative coordinates for the merge,
    on both sides of the merge boundary.
    
    But for testing purposes, we'll just duplicate the coordinate
    columns to provide the same columns that an FFN merge table provides.
    
    Returns a DataFrame with columns:
        ['id_a', 'id_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb', 'score']
    """
    assert os.path.splitext(csv_path)[1] == '.csv'
    with open(csv_path, 'r') as csv_file:
        # Is there a header?
        has_header = csv.Sniffer().has_header(csv_file.read(1024))
        if not has_header:
            raise RuntimeError(f"{csv_path} has no header row")

    df = pd.read_csv(csv_path, header=0, usecols=['segment_a,segment_b,score,x,y,z'], engine='c')
    df = df[['segment_a', 'segment_b', 'x', 'y', 'z', 'x', 'y', 'z', 'score']]
    df.columns = ['id_a', 'id_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb', 'score']

    if normalize:
        mt = df.to_records(index=False)
        mt = normalize_merge_table(mt)
        df = pd.DataFrame(mt)
    return df

def load_ffn_merge_table(npy_path, normalize=True):
    """
    Load the FFN merge table from the given .npy file,
    and return it as a DataFrame.
    
    If normalize=True, ensure the following:
    - no 'loops', i.e. id_a != id_b for all edges
    - no duplicate edges
    - id_a < id_b for all edges
    
    Returns a DataFrame with columns:
        ['id_a', 'id_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb', 'score']
    """
    assert os.path.splitext(npy_path)[1] == '.npy'
    merge_table = np.load(npy_path)
    assert merge_table.dtype == MERGE_TABLE_DTYPE
    
    if normalize:
        merge_table = normalize_merge_table(merge_table)
    
    return pd.DataFrame(merge_table)

def extract_rows(merge_table_df, supervoxels):
    """
    Extract all edges involving the given supervoxels from the given merge table.
    """
    if merge_table_df.index.names == ['idx_a', 'idx_b']:
        supervoxels = np.asarray(supervoxels, dtype=np.uint64)
        supervoxels = np.sort(supervoxels)
        subset_a = merge_table_df.loc[(supervoxels, slice(None)), :]
        subset_b = merge_table_df.loc[(slice(None), supervoxels), :]
        subset_df = pd.concat((subset_a, subset_b))
    else:
        _sv_set = set(supervoxels)
        subset_df = merge_table_df.query('id_a in @_sv_set or id_b in @_sv_set')
    return subset_df

