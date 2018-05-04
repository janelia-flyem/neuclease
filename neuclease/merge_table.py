import os
import csv
import logging

import numpy as np
import pandas as pd

# TODO: Move these funtions into neuclease, and make
# DVIDSparkServies depend on neuclease, not the other way around.
from DVIDSparkServices.graph_comparison import normalize_merge_table, MERGE_TABLE_DTYPE
from DVIDSparkServices.io_util.labelmap_utils import load_edge_csv

from dvidutils import LabelMapper
from .util import Timer

logger = logging.getLogger(__name__)


def load_merge_table(path, mapping=None, normalize=True, set_multiindex=False, scores_only=False):
    """
    Load the merge table from the given path (preferably '.npy' in FFN format),
    and return it as a DataFrame, with an appended a 'body' column according to the given mapping.
    Args:
        path:
            Either .npy (with FFN-style columns) or .csv (with CELIS-style columns)
        
        mapping:
            Either None, a pd.Series (index is SV, value is body), or a path from which one can be loaded.
            Assign 'body' column according to the given mapping of SV->body.
            Only id_a is considered when applying the mapping to each edge.
            If None, the returned 'body' column will be zero for all rows.
    
        normalize:
            If True, ensure that id_a < id_b for all edges (and ensure no self-edges)
        
        set_multiindex:
            If True, copy (id_a,id_b) to the index, and sort by the index.
            Allows pandas MultiIndex-based selection.
        
        scores_only:
            If True, discard coordinate columns
        
    Returns:
        DataFrame.
        If scores_only=True: columns=['id_a', 'id_b', 'score', 'body']
        If scores_only=False, columns=['id_a', 'id_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb', 'score', 'body']
    """
    ext = os.path.splitext(path)[1]
    assert ext in ('.npy', '.csv'), f"Invalid file extension: {ext}"
    
    sort_by = None
    if set_multiindex:
        # MultiIndex selection requires a sorted index
        # It's faster to sort the array in-place now, before converting to DataFrame
        sort_by = ['id_a', 'id_b']
    
    if ext == '.npy':
        merge_table_df = load_ffn_merge_table(path, normalize, sort_by)
    elif ext == '.csv':
        merge_table_df = load_celis_csv(path, normalize, sort_by)

    if scores_only:
        merge_table_df = merge_table_df[['id_a', 'id_b', 'score']].copy()

    if set_multiindex:
        # (Note that the table is already sorted by now)
        idx_columns = (merge_table_df['id_a'], merge_table_df['id_b'])
        merge_table_df.index = pd.MultiIndex.from_arrays(idx_columns, names=['idx_a', 'idx_b'])
    
    if mapping is None:
        merge_table_df['body'] = np.zeros((len(merge_table_df),), dtype=np.uint64)
    else:
        if isinstance(mapping, str):
            with Timer("Loading preloaded mapping to merge table", logger):
                mapping = load_mapping(mapping)

        assert isinstance(mapping, pd.Series), "Mapping must be a pd.Series"        
        with Timer("Loading preloaded mapping to merge table", logger):
            mapper = LabelMapper(mapping.index.values, mapping.values)
            merge_table_df['body'] = mapper.apply(merge_table_df['id_a'].values, allow_unmapped=True)

    return merge_table_df

def load_celis_csv(csv_path, normalize=True, sort_by=None):
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

    if sort_by:
        mt.sort_values(sort_by, inplace=True)
    
    return df


def load_ffn_merge_table(npy_path, normalize=True, sort_by=None):
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

    if sort_by:
        merge_table.sort(0, order=sort_by)
    
    return pd.DataFrame(merge_table)


def load_mapping(path):
    ext = os.path.splitext(path)[1]
    assert ext in ('.csv', '.npy')
    if ext == '.csv':
        mapping = load_edge_csv(path)
    elif ext == '.npy':
        mapping = np.load(path)
    
    mapping_series = pd.Series(index=mapping[:,0], data=mapping[:,1])
    mapping_series.index.name = 'sv'
    mapping_series.name = 'body'
    return mapping_series
