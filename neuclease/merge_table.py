import os
import csv
import logging

from tqdm import tqdm

import h5py
import ujson
import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from .util import Timer, read_csv_header, tqdm_proxy
from .dvid import (fetch_complete_mappings, fetch_split_supervoxel_sizes, read_kafka_messages,
                   fetch_supervoxel_splits, split_events_to_mapping, fetch_label)

logger = logging.getLogger(__name__)

# Note: cleave() expects scores to be "costs"
MERGE_TABLE_DTYPE = [('id_a', '<u8'),
                     ('id_b', '<u8'),
                     ('xa', '<u4'),
                     ('ya', '<u4'),
                     ('za', '<u4'),
                     ('xb', '<u4'),
                     ('yb', '<u4'),
                     ('zb', '<u4'),
                     ('score', '<f4')]

MAPPED_MERGE_TABLE_DTYPE = MERGE_TABLE_DTYPE + [('body', '<u8')]

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
        with open(path, 'r') as f:
            header = f.readline()
            if len(header) == 6:
                merge_table_df = load_celis_csv(path, normalize, sort_by)
            else:
                merge_table_df = load_ffn_merge_table(path, normalize, sort_by)

    if scores_only:
        merge_table_df = merge_table_df[['id_a', 'id_b', 'score']].copy()

    if set_multiindex:
        # (Note that the table is already sorted by now)
        idx_columns = (merge_table_df['id_a'], merge_table_df['id_b'])
        merge_table_df.index = pd.MultiIndex.from_arrays(idx_columns, names=['idx_a', 'idx_b'])
    
    if mapping is None:
        merge_table_df['body'] = np.zeros((len(merge_table_df),), dtype=np.uint64)
    else:
        apply_mapping_to_mergetable(merge_table_df, mapping)
    return merge_table_df


def apply_mapping_to_mergetable(merge_table_df, mapping):
    """
    Set the 'body' column of the given merge table (append one if it didn't exist)
    by applying the given SV->body mapping to the merge table's id_a column.
    """
    if isinstance(mapping, str):
        with Timer("Loading mapping", logger):
            mapping = load_mapping(mapping)

    assert isinstance(mapping, pd.Series), "Mapping must be a pd.Series"        
    with Timer("Applying mapping to merge table", logger):
        mapper = LabelMapper(mapping.index.values, mapping.values)
        body_a = mapper.apply(merge_table_df['id_a'].values, allow_unmapped=True)
        body_b = mapper.apply(merge_table_df['id_b'].values, allow_unmapped=True)

        # Cut edges that span across bodies
        body_a[body_a != body_b] = 0
        merge_table_df['body'] = body_a

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
    if read_csv_header(csv_path) is None:
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


def load_ffn_merge_table(path, normalize=True, sort_by=None):
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
    ext = os.path.splitext(path)[1]
    assert ext in ('.npy', '.csv'), f"Invalid merge table file extension: {ext}"

    if ext == '.npy':
        merge_table = np.load(path)
        assert merge_table.dtype == MERGE_TABLE_DTYPE
    elif ext == '.csv':
        header = read_csv_header(path)
        if not header:
            raise RuntimeError(f"Can't load merge table. CSV has no header: {path}")
        
        if len(header) == len(MERGE_TABLE_DTYPE):
            assert header == [name for (name, _type) in MERGE_TABLE_DTYPE]
            merge_table_df = pd.read_csv(path, dtype=dict(MERGE_TABLE_DTYPE))
            merge_table = merge_table_df.to_records(index=False)
        elif len(header) == len(MAPPED_MERGE_TABLE_DTYPE):
            assert header == [name for (name, _type) in MAPPED_MERGE_TABLE_DTYPE]
            merge_table_df = pd.read_csv(path, dtype=dict(MAPPED_MERGE_TABLE_DTYPE))
            del merge_table_df['body']
            merge_table = merge_table_df.to_records(index=False)
        else:
            raise RuntimeError("CSV file is not a merge table.  (Wrong column set/order.)")
    
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


def load_edge_csv(csv_path):
    """
    Load and return the given edge list CSV file as a numpy array.
    
    Each row represents an edge. For example:
    
        123,456
        123,789
        789,234
    
    The CSV file may optionally contain a header row.
    Also, it may contain more than two columns, but only the first two columns are used.
    
    Returns:
        ndarray with shape (N,2)
    """
    with open(csv_path, 'r') as csv_file:
        # Is there a header?
        has_header = csv.Sniffer().has_header(csv_file.read(1024))
        csv_file.seek(0)
        rows = iter(csv.reader(csv_file))
        if has_header:
            # Skip header
            _header = next(rows)
        
        # We only care about the first two columns
        df = pd.read_csv(csv_file, usecols=[0,1], header=None, names=['u', 'v'], dtype=np.uint64, engine='c')
        edges = df.values
        assert edges.dtype == np.uint64
        assert edges.shape[1] == 2

    return edges


def swap_cols(table, rows, name_a, name_b):
    """
    Swap two columns of a structured array, in-place.
    """
    col_a = table[name_a][rows]
    col_b = table[name_b][rows]
    
    # Swap dtypes to avoid assignment error
    col_a, col_b = col_a.view(col_b.dtype), col_b.view(col_a.dtype)

    table[name_a][rows] = col_b
    table[name_b][rows] = col_a


def normalize_merge_table(merge_table, drop_duplicate_edges=True, sort=None):
    """
    'Normalize' the given merge table by ensuring that id_a <= id_b for all rows,
    swapping fields as needed.
    
    If drop_duplicate_edges=True, duplicate edges will be dropped,
    without regard to any of the other columns (e.g. two rows with
    identical edges but different scores are still considered duplicates).
    """
    expected_dtype_names = tuple(name for name, _t in MERGE_TABLE_DTYPE)
    assert merge_table.dtype.names[:9] == expected_dtype_names, \
        f"Table has wrong dtype: {merge_table.dtype.names[:9]}\nExpected: {expected_dtype_names}"

    # Group the A coords and the B coords so they can be swapped together
    grouped_dtype = [('id_a', '<u8'),
                     ('id_b', '<u8'),
                     ('loc_a', [('xa', '<u4'), ('ya', '<u4'), ('za', '<u4')]),
                     ('loc_b', [('xb', '<u4'), ('yb', '<u4'), ('zb', '<u4')]),
                     ('score', '<f4')]

    grouped_dtype += list((k,v[0]) for k,v in merge_table.dtype.fields.items())[9:]

    swap_rows = merge_table['id_a'] > merge_table['id_b']
    merge_table_grouped = merge_table.view(grouped_dtype)
    
    swap_cols(merge_table_grouped, swap_rows, 'id_a', 'id_b')
    swap_cols(merge_table_grouped, swap_rows, 'loc_a', 'loc_b')

    assert (merge_table['id_a'] <= merge_table['id_b']).all()

    if drop_duplicate_edges:
        edge_df = pd.DataFrame( {'id_a': merge_table['id_a'], 'id_b': merge_table['id_b']} )
        dupe_rows = edge_df.duplicated(keep='last')
        if dupe_rows.sum() > 0:
            merge_table = merge_table[~dupe_rows]
    
    if sort is not None:
        merge_table.sort(order=sort)
    
    return merge_table

def normalize_recarray_inplace(table, ref_col_a, ref_col_b, columns_a, columns_b):
    columns_a = list(columns_a)
    columns_b = list(columns_b)
    
    assert len(columns_a) == len(columns_b)
    
    assert ref_col_a in columns_a
    assert ref_col_b in columns_b
    
    swap_rows = (table[ref_col_a] > table[ref_col_b])
    for col_a, col_b in zip(columns_a, columns_b):
        swap_cols(table, swap_rows, col_a, col_b)


def apply_mappings(supervoxels, mappings):
    assert isinstance(mappings, dict)
    df = pd.DataFrame(index=supervoxels.astype(np.uint64, copy=False))
    df.index.name = 'sv'

    for name, mapping in mappings.items():
        assert isinstance(mapping, pd.Series)
        index_values = mapping.index.values.astype(np.uint64, copy=False)
        mapping_values = mapping.values.astype(np.uint64, copy=False)
        mapper = LabelMapper(index_values, mapping_values)
        df[name] = mapper.apply(df.index.values, allow_unmapped=True)

    return df


def load_supervoxel_sizes(h5_path):
    """
    Load the stored supervoxel size table from hdf5 and return the result as a pd.Series, with sv as the index.
    
    Args:
        h5_path:
            A file with two datasets: sv_ids and sv_sizes
    
    Returns:
        pd.Series, indexed by supervoxel ID
    """
    with h5py.File(h5_path, 'r') as f:
        sv_sizes = pd.Series(index=f['sv_ids'][:], data=f['sv_sizes'][:])
    sv_sizes.name = 'voxel_count'
    sv_sizes.index.name = 'sv'

    logger.info(f"Volume contains {len(sv_sizes)} supervoxels and {sv_sizes.values.sum()/1e12:.1f} Teravoxels in total")    

    # Sorting by supervoxel ID may give better performance during merges later
    sv_sizes.sort_index(inplace=True)
    return sv_sizes


def load_all_supervoxel_sizes(server, uuid, instance, root_sv_sizes, split_source='kafka'):
    """
    Given the original SV sizes (or a path to the .h5 file that contains it)
    and a DVID node/instance, compute the set of ALL supervoxel sizes for the node.
    
    Note:
        Result will contain all supervoxels that exist in the given node,
        but it may ALSO include some supervoxels that no longer exist.
    
    Args:
        server, uuid, instance:
            DVID instance
        
        root_sv_sizes:
            Either a pd.Series (indexed by sv) or a path to an hdf5
            file that can be passed to load_supervoxel_sizes()
        
        split_source:
            Either 'kafka' or 'dvid'.
            Specifies where to fetch the supervoxel history from.
    
    Returns:
        pd.Series, indexed by supervoxel ID
    """
    if isinstance(root_sv_sizes, str):
        root_sv_sizes = load_supervoxel_sizes(root_sv_sizes)
    assert isinstance(root_sv_sizes, pd.Series)
    
    split_fragment_sizes = fetch_split_supervoxel_sizes(server, uuid, instance, split_source=split_source)
    
    all_sv_sizes = pd.concat((root_sv_sizes, split_fragment_sizes))
    all_sv_sizes.index.name = 'sv'
    all_sv_sizes.name = 'voxel_count'
    return all_sv_sizes
    
    

def compute_body_sizes(sv_sizes, mapping, include_unmapped_singletons=False):
    """
    Given a Series of supervoxel sizes and an sv-to-body mapping,
    compute the size of each body in the mapping, and its count of supervoxels.
    
    Any supervoxels in the mapping that are missing from sv_sizes will be ignored.

    Body 0 will be excluded from the results, even if it is present in the mapping.
    
    Args:
        sv_sizes:
            pd.Series, indexed by sv, or a path to an hdf5 file which
            can be loaded via load_supervoxel_sizes()
        
        mapping:
            pd.Series, indexed by sv, with body as value,
            or a path to a file which can be loaded by load_mapping()
       
        include_unmapped_singletons:
            If True, then the result will also include all
            supervoxels from sv_sizes that weren't mentioned in the mapping.
            (They are presumed to be singleton-supervoxel bodies.)
    
    Returns:
        pd.DataFrame, indexed by body, with columns ['voxel_count', 'sv_count'],
        and sorted by decreasing voxel_count.

    Example:
    
        >>> mesh_job_dir = '/groups/flyem/data/scratchspace/copyseg-configs/labelmaps/hemibrain/8nm'
        >>> sv_sizes_path = f'{mesh_job_dir}/compute-8nm-extended-fixed-STATS-ONLY-20180402.192015/supervoxel-sizes.h5'
        >>> sv_sizes = load_supervoxel_sizes(sv_sizes_path)
        >>> mapping = fetch_complete_mappings('emdata3:8900', '52f9', 'segmentation')
        >>> body_sizes = compute_body_sizes(sv_sizes, mapping)
    """
    if isinstance(sv_sizes, str):
        logger.info("Loading supervoxel sizes")
        assert os.path.splitext(sv_sizes)[1] == '.h5'
        sv_sizes = load_supervoxel_sizes(sv_sizes)
    
    if isinstance(mapping, str):
        logger.info("Loading mapping")
        mapping = load_mapping(mapping)
    
    assert isinstance(sv_sizes, pd.Series)
    assert isinstance(mapping, pd.Series)
    
    assert sv_sizes.index.dtype == np.uint64
    
    sv_sizes = sv_sizes.astype(np.uint64, copy=False)
    size_mapper = LabelMapper(sv_sizes.index.values, sv_sizes.values)

    # Just drop SVs that we don't have sizes for.
    logger.info("Dropping unknown supervoxels")
    mapping = mapping.loc[mapping.index.isin(sv_sizes.index)]

    logger.info("Applying sizes to mapping")
    df = pd.DataFrame({'body': mapping})
    df['voxel_count'] = size_mapper.apply(mapping.index.values)

    logger.info("Aggregating sizes by body")
    body_stats = df.groupby('body').agg({'voxel_count': ['sum', 'size']})
    body_stats.columns = ['voxel_count', 'sv_count']
    body_stats['sv_count'] = body_stats['sv_count'].astype(np.uint32)
    #body_sizes = body_stats['voxel_count']
    
    if include_unmapped_singletons:
        logger.info("Appending singleton sizes")
        nonsingleton_rows = sv_sizes.index.isin(mapping.index)
        singleton_sizes = sv_sizes[~nonsingleton_rows]
        singleton_stats = pd.DataFrame({'voxel_count': singleton_sizes})
        singleton_stats['sv_count'] = np.uint32(1)
        body_stats = pd.concat((body_stats, singleton_stats))
    
    if 0 in body_stats.index:
        body_stats.drop(0, inplace=True)
    
    logger.info("Sorting sizes")
    body_stats.index.name = 'body'
    body_stats.sort_values(['voxel_count', 'sv_count'], inplace=True, ascending=False)
    return body_stats


def extract_important_merges(speculative_merge_tables, important_bodies, body_mapping=None, mapping_instance_info=None, drop_duplicate_body_pairs=False):
    assert (body_mapping is None) ^ (mapping_instance_info is None), \
        "You must set either body_mapping or mapping_instance_info (but not both)"

    if mapping_instance_info is not None:
        body_mapping = fetch_complete_mappings(mapping_instance_info)

    assert isinstance(body_mapping, pd.Series)
    mapper = LabelMapper(body_mapping.index.values, body_mapping.values)

    # pd.Index is faster than builtin set for large sets
    important_bodies = pd.Index(important_bodies)
    
    results = []
    for spec_merge_table_df in tqdm(speculative_merge_tables):
        logger.info(f"Processing table with {len(spec_merge_table_df)} rows")

        with Timer("Applying mapping", logger):
            spec_merge_table_df['body_a'] = mapper.apply(spec_merge_table_df['id_a'].values, allow_unmapped=True)
            spec_merge_table_df['body_b'] = mapper.apply(spec_merge_table_df['id_b'].values, allow_unmapped=True)

        with Timer("Dropping identity merges", logger):
            orig_size = len(spec_merge_table_df)
            spec_merge_table_df.query('body_a != body_b', inplace=True)
            logger.info(f"Dropped {orig_size-len(spec_merge_table_df)}/{orig_size} edges.")

        with Timer("Normalizing edges", logger):
            # Normalize for body ID, not SV ID
            # (This involves a lot of copying, but you've got plenty of RAM, right?)
            a_cols = list(filter(lambda s: s[-1] == 'a', spec_merge_table_df.columns))
            b_cols = list(filter(lambda s: s[-1] == 'b', spec_merge_table_df.columns))
            spec_merge_table = spec_merge_table_df.to_records(index=False)
            normalize_recarray_inplace(spec_merge_table, 'body_a', 'body_b', a_cols, b_cols)
            spec_merge_table_df = pd.DataFrame(spec_merge_table)

        with Timer("Filtering edges", logger):
            q = 'body_a in @important_bodies and body_b in @important_bodies'
            orig_len = len(spec_merge_table_df)
            spec_merge_table_df = spec_merge_table_df.query(q, inplace=True)
            logger.info(f"Filtered out {orig_len - len(spec_merge_table_df)} non-important edges.")

        if drop_duplicate_body_pairs:
            with Timer("Dropping duplicate body pairs", logger):
                orig_len = len(spec_merge_table_df)
                spec_merge_table_df.drop_duplicates(['body_a', 'body_b'], inplace=True)
                logger.info(f"Dropped {orig_len - len(spec_merge_table_df)} duplicate body pairs")

        results.append(spec_merge_table_df)
    
    return results


def update_merge_table(server, uuid, instance, table_df, complete_mapping=None, split_mapping=None):
    """
    Given a merge table (such as a focused proofreading decision table),
    find rows whose supervoxels no longer exist in the given instance (due to splits).

    For those invalid rows, determine the new supervoxel and body ID at the given coordinates
    to determine the updated supervoxel/body IDs.
    
    Updates (in-place) the supervoxel and body columns.
    
    Note: If any coordinate appears to be misplaced (i.e. the supervoxel ID at
    the coordinate is not a descendant of the listed supervoxel), the supervoxel is
    left unchanged and the body is mapped to 0.
    
    Args:
        server, uuid, instance:
            Table will be updated with respect to the given segmentation instance info
        
        table_df:
            DataFrame with SV columns and coordinate columns ('xa', 'ya', 'za', etc.)
        
        complete_mapping:
            Optional.  Will be fetched if not provided.
            Must be the complete mapping as returned by fetch_complete_mappings(..., include_retired=True)
        
        split_mapping:
            Optional.  Will be fetched if not provided.
            A mapping from supervoxel fragments to root supervoxel IDs.

    Returns:
        None. (The table is modified in place.)
    """
    seg_info = server, uuid, instance

    # Ensure proper table columns/dtypes
    if 'id_a' in table_df.columns:
        col_sv_a, col_sv_b = 'id_a', 'id_b'
    elif 'sv_a' in table_df.columns:
        col_sv_a, col_sv_b = 'sv_a', 'sv_b'
    else:
        raise RuntimeError("table has no sv columns")
    
    assert set([col_sv_a, col_sv_b, 'xa', 'ya', 'za', 'xb', 'yb', 'zb']).issubset(table_df.columns)
    for col in ['xa', 'ya', 'za', 'xb', 'yb', 'zb']:
        table_df[col] = table_df[col].fillna(0).astype(np.int32)

    # Construct mappings if necessary
    kafka_msgs = None
    if complete_mapping is None or split_mapping is None:
        kafka_msgs = read_kafka_messages(*seg_info)

    if complete_mapping is None:
        complete_mapping = fetch_complete_mappings(*seg_info, include_retired=True, kafka_msgs=kafka_msgs)
    complete_mapper = LabelMapper(complete_mapping.index.values, complete_mapping.values)

    if split_mapping is None:
        split_events = fetch_supervoxel_splits(*seg_info)
        split_mapping = split_events_to_mapping(split_events, leaves_only=False)
    split_mapper = LabelMapper(split_mapping.index.values, split_mapping.values)

    # Apply up-to-date body mapping
    # (Retired supervoxels will map to body 0)
    table_df['body_a'] = complete_mapper.apply(table_df[col_sv_a].values, True)
    table_df['body_b'] = complete_mapper.apply(table_df[col_sv_b].values, True)

    successfully_updated = 0
    failed_update = 0

    # Update the rows with invalid bodies.    
    for index in tqdm_proxy(table_df.query('body_a == 0 or body_b == 0').index):
        
        def update_row(index, col_body, col_sv, col_z, col_y, col_x):
            nonlocal successfully_updated, failed_update
            
            # Extract from table
            coord = table_df.loc[index, [col_z, col_y, col_x]]
            old_sv = table_df.loc[index, col_sv]
            
            # Check current SV in the volume
            new_sv = fetch_label(*seg_info, coord, supervoxels=True)

            # The old/new SV must have come from the same root SV.
            # If not, the coordinate must be misplaced and can't be used here.
            svs = np.asarray([new_sv, old_sv], np.uint64)
            mapped_svs = split_mapper.apply(svs, True)
            if mapped_svs[0] != mapped_svs[1]:
                failed_update += 1
            else:
                body = complete_mapper.apply(np.array([new_sv], np.uint64))[0]
                table_df.loc[index, col_body] = body
                table_df.loc[index, col_sv] = new_sv
                successfully_updated +=1

        # id_a/body_a
        if table_df.loc[index, 'body_a'] == 0:
            update_row(index, 'body_a', col_sv_a, 'za', 'ya', 'xa')

        # id_b/body_b
        if table_df.loc[index, 'body_b'] == 0:
            update_row(index, 'body_b', col_sv_b, 'zb', 'yb', 'xb')
    
    logger.info(f"Updated {successfully_updated}, failed to update {failed_update}")





