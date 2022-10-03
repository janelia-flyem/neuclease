import os
import sys
import glob
import logging
import subprocess
from math import ceil, log10

import ujson
import numpy as np
import pandas as pd

from ..util import tqdm_proxy, swap_df_cols, iter_batches

logger = logging.getLogger(__name__)

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


def define_task_batches(tasks,
                        total_tasks_per_assignment=100,
                        qc_tasks_per_assignment=10,
                        assignments_per_batch=10):
    """
    Given a dataframe where each row represents a proofreading task,
    this function will assign each task to an 'assignment' and each assignment to a 'batch'.
    All assignments within a batch will share a certain number of QC (quality control) tasks,
    i.e. the QC tasks will be duplicated so they can be included within every assignment of the batch.
    The placement of the QC tasks within each assignment will be randomized.

    Returns:
        DataFrame, with 'batch' and 'assignment' columns (in addition to the columns of the input).
    """
    regular_tasks_per_assignment = total_tasks_per_assignment - qc_tasks_per_assignment
    unique_tasks_per_batch = regular_tasks_per_assignment * assignments_per_batch + qc_tasks_per_assignment

    tasks = tasks.reset_index(drop=True)
    tasks = tasks.copy()
    tasks['batch'] = tasks.index // unique_tasks_per_batch
    tasks['assignment'] = -1

    batch_dfs = []
    for _, batch_df in tasks.groupby('batch'):
        qc_df = batch_df.sample(min(qc_tasks_per_assignment, len(batch_df)))
        reg_df = batch_df.loc[~(batch_df.index.isin(qc_df.index))].sample(frac=1.0)
        if len(reg_df) == 0:
            # The batch is so small that every task is a QC task.
            # That's okay, we'll still emit an assignment (just one, though).
            assign_df = qc_df
            assign_df['assignment'] = 0
            batch_dfs.append(assign_df)
        else:
            for i, assign_df in enumerate(iter_batches(reg_df, regular_tasks_per_assignment)):
                assign_df = pd.concat((assign_df, qc_df), ignore_index=True).sample(frac=1.0)
                assign_df['assignment'] = i
                batch_dfs.append(assign_df)

    final_df = pd.concat(batch_dfs, ignore_index=True)

    cols = tasks.columns.tolist()
    cols.remove('batch')
    cols.remove('assignment')
    cols = ['batch', 'assignment', *cols]
    return final_df[cols]


def upload_batched_assignments(tasks, bucket_path, campaign='focused'):
    output_dir = f'{campaign}-assignments'
    os.makedirs(output_dir)

    batch_digits = int(ceil(log10(tasks['batch'].max())))
    assignment_digits = int(ceil(log10(tasks['assginment'].max())))

    num_assignments = len(tasks.drop_duplicates(['batch', 'assignment']))
    files = []
    for (batch, assignment), assign_df in tqdm_proxy(tasks.groupby(['batch', 'assignment']), total=num_assignments):
        name = f"{output_dir}/{campaign}-batch-{batch:0{batch_digits}d}-assignment-{assignment:0{assignment_digits}d}.json"
        generate_focused_assignment(assign_df, name)
        files.append((batch, assignment, name))

    # Explicitly *unset* content type, to trigger browsers to download the file, not display it as JSON.
    # Also, forbid caching.
    logging.info(f"Uploading {len(files)} files to gs://{bucket_path}/{output_dir}")
    cmd = f"gsutil -m -h 'Cache-Control:public, no-store' -h 'Content-Type' cp -r {output_dir} gs://{bucket_path}/"
    _ = subprocess.run(cmd, shell=True, check=True, capture_output=True)

    tracking_df = pd.DataFrame(files, columns=['batch', 'assignment', 'file'])
    tracking_df['file'] = f'https://storage.googleapis.com/{bucket_path}/' + tracking_df['file']
    tracking_df['user'] = ''
    tracking_df['hidden_col'] = ''
    tracking_df['date started'] = ''
    tracking_df['date completed'] = ''
    tracking_df['notes'] = ''
    p = f'{campaign}-tracking-template.csv'
    tracking_df.to_csv(p, index=False, header=True)
    logger.info(f"Wrote {p}")
