import os
import sys
import glob
import ujson

import numpy as np
import pandas as pd

from ..util import tqdm_proxy, swap_df_cols

def generate_focused_assignment(merge_table, output_path=None):
    """
    Generate a single focused-proofreading assignment from the given table of proposed edges.
    
    Args:
        merge_table:
            pd.DataFrame with at least the following columns:
            ['sv_a', 'xa', 'ya', 'za', 'sv_b', 'xb', 'yb', 'zb']

            (For backwards compatibility, 'id_a/id_b' can be given instead of sv_a/sv_b.
        
        output_path:
            If provided, the assignment is written as JSON to the given path.
    
    Returns:
        The assignment data as Python object, ready for JSON serialization.
    """
    if isinstance(merge_table, np.ndarray):
        merge_table = pd.DataFrame(merge_table)
    
    if (('id_a' in merge_table and 'sv_a' in merge_table) or 
        ('id_b' in merge_table and 'sv_b' in merge_table)):
        raise RuntimeError("Please pass either id_a or sv_a columns, not both.")
    
    if ('id_a' in merge_table and 'id_b' in merge_table):
        merge_table = merge_table.rename(columns={'id_a': 'sv_a', 'id_b': 'sv_b'})
    
    assert isinstance(merge_table, pd.DataFrame)
    REQUIRED_COLUMNS = ['sv_a', 'xa', 'ya', 'za', 'sv_b', 'xb', 'yb', 'zb']
    assert set(merge_table.columns).issuperset( REQUIRED_COLUMNS ), \
        "Table does not have the required columns to generate a focused proofreading assignment"
    
    merge_table = merge_table.copy()
    swap_df_cols(merge_table, None, merge_table.eval('sv_a > sv_b'), ('a', 'b'))
    
    tasks = []
    for row in merge_table.itertuples():
        coord_a = list(map(int, [row.xa, row.ya, row.za]))
        coord_b = list(map(int, [row.xb, row.yb, row.zb]))
        task = { "task type": "body merge",
                 "supervoxel ID 1": int(row.sv_a), "supervoxel point 1": coord_a,
                 "supervoxel ID 2": int(row.sv_b), "supervoxel point 2": coord_b }
        tasks.append(task)
        
    assignment = { "file type": "Neu3 task list",
                   "file version": 1,
                   "task list": tasks }
    if output_path:
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(output_path, 'w') as f:
            ujson.dump(assignment, f, indent=2)
    return assignment


def generate_focused_assignments(merge_table, approximate_assignment_size, output_dir, prefix='assignment-'):
    """
    Generate a set of focused-proofreading assignments from the given table of proposed edges.
    
    Args:
        merge_table:
            pd.DataFrame with at least the following columns:
            ['sv_a', 'xa', 'ya', 'za', 'sv_b', 'xb', 'yb', 'zb']

        approximate_assignment_size:
            Split the table into roughly equal-sized assignments of this size.
        
        output_dir:
            Where to dump the generated assignment files.
        
        prefix:
            Assignment files will be named {prefix}{N}.json, where N is the assignment number.
    """
    if isinstance(merge_table, np.ndarray):
        merge_table = pd.DataFrame(merge_table)

    total_size = len(merge_table)
    num_assignments = max(1, total_size // approximate_assignment_size)
    assignment_size = total_size // num_assignments

    partitions = list(range(0, assignment_size*num_assignments, assignment_size))
    if partitions[-1] < total_size:
        partitions.append( total_size )

    os.makedirs(output_dir, exist_ok=True)
    for i, (start, stop) in tqdm_proxy(enumerate(zip(partitions[:-1], partitions[1:])), total=len(partitions)-1, leave=False):
        output_path = output_dir + f'/{prefix}{i:03d}.json'
        generate_focused_assignment(merge_table.iloc[start:stop], output_path)

def load_focused_assignment(path):
    with open(path, 'r') as f:
        assignment = ujson.load(f)
    
    raw_df = pd.DataFrame(assignment['task list'])
    
    df = raw_df[['supervoxel ID 1', 'supervoxel ID 2']].rename(columns={'supervoxel ID 1': 'sv_a', 'supervoxel ID 2': 'sv_b'})
    df['xa'] = df['ya'] = df['za'] = 0
    df['xb'] = df['yb'] = df['zb'] = 0
    
    # Concatenate all [x,y,z] lists and convert to a 2D array
    df[['xa', 'ya', 'za']] = np.asarray(raw_df['supervoxel point 1'].sum()).reshape(-1,3)
    df[['xb', 'yb', 'zb']] = np.asarray(raw_df['supervoxel point 2'].sum()).reshape(-1,3)

    # Copy any other fields without renaming them.
    for col in raw_df.drop(columns=['supervoxel ID 1', 'supervoxel ID 2', 'supervoxel point 1', 'supervoxel point 2']):
        df[col] = raw_df[col]
    
    return df

def load_focused_assignments(directory):
    paths = list(map(sys.intern, sorted(glob.glob(f"{directory}/*.json"))))

    dfs = []
    for path in tqdm_proxy(paths):
        df = load_focused_assignment(path)
        df['path'] = path
        dfs.append(df)
    
    return pd.concat(dfs)
