import os
import json
import logging

import h5py
import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from ..util import read_csv_header, Timer, swap_df_cols
from ..util.csv import read_csv_col
from ..merge_table import load_all_supervoxel_sizes, compute_body_sizes
from ..dvid import (fetch_keys, post_keyvalues, fetch_complete_mappings, fetch_keyvalues,
                    fetch_labels_batched, fetch_mapping, find_master, fetch_body_annotations)
from ..dvid.annotation import load_synapses, body_synapse_counts
from ..dvid.labelmap import fetch_supervoxel_fragments, fetch_labels

from .assignments import generate_focused_assignment

# Load new table. Normalize.

# Optionally: Score and choose body-wise favorites.

# Load previously assigned. Normalize.

logger = logging.getLogger(__name__)

CSV_DTYPES = { 'body': np.uint64,
               'sv': np.uint64,
               'id_a': np.uint64, 'id_b': np.uint64, # We usually use'id_a', and 'id_b' for consistency with our other code.
               'sv_a': np.uint64, 'sv_b': np.uint64,
               'body_a': np.uint64, 'body_b': np.uint64,
               'xa': np.int32, 'ya': np.int32, 'za': np.int32,
               'xb': np.int32, 'yb': np.int32, 'zb': np.int32,
               'caa': np.float32, 'cab': np.float32, 'cba': np.float32, 'cbb': np.float32,
               'iou': np.float32,
               'da': np.float32, 'db': np.float32,
               'score': np.float32,
               'score_a': np.float32, 'score_b': np.float32,
               'jaccard': np.float32,
               'overlap': np.uint32,
               'x': np.int32, 'y': np.int32, 'z': np.int32 }


def update_synapse_table(server, uuid, instance, synapse_df, output_path=None, split_source='kafka'):
    """
    Give a dataframe (or a CSV file) with at least columns ['sv', 'x', 'y', 'z'],
    identify any 'retired' supervoxels and query DVID to
    replace them with their split descendents.
    
    Optionally write the updated table to CSV.
    """
    seg_info = (server, uuid, instance)
    
    if isinstance(synapse_df, str):
        synapse_df = load_synapses(synapse_df)

    # Get the set of all retired supervoxel IDs
    _leaves, _retired = fetch_supervoxel_fragments(*seg_info, split_source)

    # Which rows in the table have a retired ID?
    retired_df = synapse_df.query('sv in @_retired')[['z', 'y', 'x']]

    if len(retired_df) > 0:
        with Timer(f"Updating {len(retired_df)} rows with retired SVs", logger):
            # Fetch the new label for those IDs
            coords_zyx = retired_df.values
            updated_svs = fetch_labels(*seg_info, coords_zyx, supervoxels=True)
            
            # Update the table and write to CSV
            synapse_df.loc[retired_df.index, 'sv'] = updated_svs
    
    if output_path:
        with Timer(f"Writing to {output_path}", logger):
            synapse_df.to_csv(output_path, header=True, index=False)

    return synapse_df


def load_focused_table(path):
    """
    Load an edge table from the given path.
    Must have at least the required columns for an edge table (id_a, id_b, and coordinates),
    but may include extra.  All columns are loaded and included in the result.
    """
    REQUIRED_COLUMNS = ['sv_a', 'sv_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb']
    ext = os.path.splitext(path)[1]
    assert ext in ('.csv', '.npy')
    if ext == '.csv':
        header = read_csv_header(path)
        if header is None:
            raise RuntimeError(f"CSV has no header: {path}")
        df = pd.read_csv(path, header=0, dtype=CSV_DTYPES)

    if ext == '.npy':
        df = pd.DataFrame(np.load(path))

    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        raise RuntimeError(f"file ({path}) does not contain the required columns: {REQUIRED_COLUMNS}")

    return df


def fetch_focused_decisions(server, uuid, instance='segmentation_merged', normalize_pairs=None, subset_pairs=None, drop_invalid=True, update_with_instance='segmentation'):
    """
    Load focused decisions from a given keyvalue instance
    (e.g. 'segmentation_merged') and return them as a DataFrame,
    with slight modifications to use standard column names.
    
    By default, loads all decisions from the instance.
    use subset_pairs or subset_slice to select fewer decisions.
    
    Args:
        server, uuid, instance
            Exmaple: 'emdata3:8900', '7254', 'segmentation_merged'
        
        normalize_pairs:
            Either None, 'sv', or 'body'
            If not None, swap A/B columns so that either sv_a < sv_b or body_a < body_b.
        
        subset_pairs:
            If provided, don't fetch all focused decisions;
            only select those whose key matches the given SV ID pairs
            (The left/right order within the pair doesn't matter.)
            Alternatively, the pairs can be given as exact key strings to retrieve
            (in the format '{id_1}+{id_2}', e.g. '123+456')
            
            Note:
                If any of the listed pairs don't exist in the keyvalue instance,
                they'll be silently dropped from the results.

        drop_invalid:
            If True, remove any rows with missing/invalid fields for sv id or coordinates.
        
        update_with_instance:
            Focused decisions are recorded with the supervoxel and body IDs that existed
            under their coordinates at the time the decision was made, but those IDs may
            have changed in the meantime.
            If ``update_with_instance`` is provided, update the supervoxel IDs and body IDs
            using the coordinates in each row and fetching the corresponding IDs from the
            given segmentation instance.
            Should be provided as an instance name (in which case the server and uuid
            are assumed to be the same as the given keyvalue instance),
            or a complete tuple of ``(server, uuid, instance)`` to use instead.

    Returns:
        DataFrame with columns:
        ['body_a', 'body_b', 'result', 'sv_a', 'sv_b',
        'time', 'time zone', 'user',
        'xa', 'xb', 'ya', 'yb', 'za', 'zb', ...]
    """
    assert normalize_pairs in (None, 'sv', 'body')
    
    
    if subset_pairs is not None:
        subset_pairs = list(subset_pairs)
        if isinstance(subset_pairs[0], str):
            keys = subset_pairs
        else:
            subset_keys1 = [f'{a}+{b}' for a,b in subset_pairs]
            subset_keys2 = [f'{b}+{a}' for a,b in subset_pairs]
            keys = list({*subset_keys1, *subset_keys2})
    else:
        with Timer(f"Fetching keys from '{instance}'", logger):
            # FIXME: Unless subset_pairs is used, it's probably faster
            #        to just fetch all the key/values at once,
            #        rather than fetching the keys, then the values.
            #        (On the server side, fetching keys in leveldb is just as
            #        expensive as fetching the values too.)
            keys = fetch_keys(server, uuid, instance)

    with Timer(f"Fetching values from '{instance}'"):
        task_values = fetch_keyvalues(server, uuid, instance, keys, as_json=True, batch_size=100_000).values()
        
        # fetch_keyvalues() returns None for values that don't exist.
        # Drop those ones.
        task_values = [*filter(None, task_values)]

    # Flatten coords before loading into dataframe
    for value in task_values:
        if 'supervoxel point 1' in value:
            p1 = value['supervoxel point 1']
            p2 = value['supervoxel point 2']
            del value['supervoxel point 1']
            del value['supervoxel point 2']
        else:
            p1 = p2 = [0,0,0]
            
        for name, coord in zip(['xa', 'ya', 'za'], p1):
            value[name] = coord
        for name, coord in zip(['xb', 'yb', 'zb'], p2):
            value[name] = coord

    df = pd.DataFrame(task_values)
    df.rename(inplace=True, columns={'body ID 1': 'body_a', 'body ID 2': 'body_b',
                                     'supervoxel ID 1': 'sv_a', 'supervoxel ID 2': 'sv_b' })

    # Converting to category saves some RAM
    if 'status' in df:
        df['status'] = pd.Series(df['status'], dtype='category')
    if 'result' in df:
        df['result'] = pd.Series(df['result'], dtype='category')
    if 'user' in df:
        df['user'] = pd.Series(df['user'], dtype='category')
    if 'time zone' in df:
        df['time zone'] = pd.Series(df['time zone'], dtype='category')

    # Convert time to proper timestamp
    if 'time' in df:
        df['time'] = pd.to_datetime(df['time'])

    if drop_invalid:
        invalid = df['sv_a'].isnull()
        for col in ['sv_a', 'sv_b', 'body_a', 'body_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb']:
            invalid |= df[col].isnull()
        df = df[~invalid]

        for col in ['sv_a', 'sv_b', 'body_a', 'body_b']:
            df[col] = df[col].astype(np.uint64)
        for col in ['xa', 'ya', 'za', 'xb', 'yb', 'zb']:
            df[col] = df[col].astype(np.int32)

    if update_with_instance:
        if isinstance(update_with_instance, str):
            update_with_instance = (server, uuid, update_with_instance)
        else:
            assert len(update_with_instance) == 3

        coords_a = df[['za', 'ya', 'xa']].values
        coords_b = df[['zb', 'yb', 'xb']].values
        svs_a = fetch_labels_batched(*update_with_instance, coords_a, True, 0, 10_000, threads=8)
        svs_b = fetch_labels_batched(*update_with_instance, coords_b, True, 0, 10_000, threads=8)
        bodies_a = fetch_mapping(*update_with_instance, svs_a)
        bodies_b = fetch_mapping(*update_with_instance, svs_b)
        
        df['sv_a'] = svs_a
        df['sv_b'] = svs_b
        df['body_a'] = bodies_a
        df['body_b'] = bodies_b

    assert normalize_pairs in (None, 'sv', 'body')
    if normalize_pairs == 'sv':
        swap_df_cols(df, None, (df['sv_a'] > df['sv_b']), ['a', 'b'])
    elif normalize_pairs == 'body':
        swap_df_cols(df, None, (df['body_a'] > df['body_b']), ['a', 'b'])

    # Return in chronological order
    if 'time' in df:
        return df.sort_values('time').reset_index(drop=True)
    else:
        return df.reset_index(drop=True)


def drop_previously_reviewed(df, previous_focused_decisions_df):
    """
    Given a DataFrame of speculative focused decisions and 
    a DataFrame of previously reviewed focused decisions,
    drop all previous decisions from the speculative set,
    regardless of review results.
    """
    cols = [*df.columns]
    
    if df.eval('sv_a > sv_b').any():
        raise RuntimeError("Some rows of the input (df) are not in normal form.")

    if previous_focused_decisions_df.eval('sv_a > sv_b').any():
        raise RuntimeError("Some rows of the input (previous_focused_decisions_df) are not in normal form.")
    
    comparison_df = previous_focused_decisions_df[['sv_a', 'sv_b']].drop_duplicates()
    in_prev = df[['sv_a', 'sv_b']].merge(comparison_df,
                                         how='left',
                                         on=['sv_a', 'sv_b'],
                                         indicator='side')

    keep_rows = (in_prev['side'] == 'left_only')
    return df.loc[keep_rows.values, cols]


DEFAULT_IMPORTANT_STATUSES = {'Leaves', 'Prelim Roughly traced', 'Roughly traced', 'Traced'}
DEFAULT_FOCUS_STATUSES = {'Orphan', 'Orphan hotknife', '0.5assign'}
DEFAULT_BAD_BODIES = '/nrs/flyem/bergs/complete-ffn-agglo/bad-bodies-2019-02-26.csv'
def filter_merge_tasks(server, uuid, focused_df=None, mr_df=None, mr_endpoint_df=None,
                       body_annotations_df=None, previous_focused_df=None,
                       connecting_statuses_1=DEFAULT_FOCUS_STATUSES,
                       connecting_statuses_2=DEFAULT_FOCUS_STATUSES,
                       at_least_one_of_statuses=DEFAULT_IMPORTANT_STATUSES,
                       at_most_one_of_statuses=DEFAULT_IMPORTANT_STATUSES,
                       bad_bodies=DEFAULT_BAD_BODIES,
                       min_psds=1, min_tbars=1):
    """
    Filter the given focused and/or merge-review tasks to remove tasks that should not be assigned.
    
    The following criteria are used for filtering:
        - synapse counts: one or both bodies must have the requisite PSDs or T-bars
        - bad bodies (e.g. frankenbodies): no task should involve a bad body
        - previous assignments: Don't re-assign tasks that have already been reviewed.
          Note: At the moment, only previous focused proofreading assignments are checked.
          
        - body status: Tasks will be filtered according to the given sets of statuses.
          Any task that has one body status from `connecting_statuses_1` and one from `connecting_statuses_2`
          will be included, as will any task which has one body status from `at_least_one_of_statuses`.
          Finally, no tasks will be included if they have more than one body status from `at_most_one_of_statuses`.
          To skip status filtering altogether, pass ``None`` for all of the status sets.
    
    Returns:
        (focused_df, mr_df, mr_endpoint_df)
    """
    dvid_node = (server, uuid)

    if isinstance(focused_df, str):
        focused_df = pd.DataFrame(np.load(focused_df, allow_pickle=True))

    if isinstance(mr_df, str):
        mr_df = pd.DataFrame(np.load(mr_df, allow_pickle=True))
    
    if isinstance(mr_endpoint_df, str):
        mr_endpoint_df = pd.DataFrame(np.load(mr_endpoint_df, allow_pickle=True))

    assert not ((mr_df is None) ^ (mr_endpoint_df is None)), \
        "If you have MR tasks, you must supply both mr_df and mr_endpoint_df"

    for df in (focused_df, mr_df, mr_endpoint_df):
        assert df is None or isinstance(df, pd.DataFrame), "bad input type"

    assert bool(connecting_statuses_1) == bool(connecting_statuses_2), \
        "If supplying connecting statuses, you must supply both."

    if focused_df is not None:
        focused_df = focused_df.copy()
        swap_df_cols(focused_df, None, focused_df.eval('sv_a > sv_b'), ('a', 'b'))
        num_focused = len(focused_df)
        logger.info(f"Starting with {num_focused} focused tasks")

    if mr_df is not None:
        mr_df = mr_df.copy()
        mr_endpoint_df = mr_endpoint_df.copy()
        num_mr_tasks = len(mr_endpoint_df)
        logger.info(f"Starting with {num_mr_tasks} merge-review tasks")
        num_mr_edges = len(mr_df)
        logger.info(f"Starting with {num_mr_edges} merge-review edges")

    with Timer("Filtering for synapse counts", logger):
        if min_psds > 0 or min_tbars > 0:
            if focused_df is not None:
                assert {'PostSyn_a', 'PreSyn_a', 'PostSyn_b', 'PreSyn_b'} < set(focused_df.columns), \
                    "tasks are missing synapse count columns"
            if mr_df is not None:
                assert {'PostSyn_a', 'PreSyn_a', 'PostSyn_b', 'PreSyn_b'} < set(mr_df.columns), \
                    "tasks are missing synapse count columns"
            if mr_endpoint_df is not None:
                assert {'PostSyn_a', 'PreSyn_a', 'PostSyn_b', 'PreSyn_b'} < set(mr_endpoint_df.columns), \
                    "tasks are missing synapse count columns"
    
            _q = '(PostSyn_a >= @min_psds or PreSyn_a >= @min_tbars) and (PostSyn_b >= @min_psds or PreSyn_b >= @min_tbars)'

            if focused_df is not None:
                focused_df = focused_df.query(_q).copy()
                logger.info(f"Dropped {num_focused - len(focused_df)} focused tasks")
                num_focused = len(focused_df)
            
            if mr_df is not None:
                mr_endpoint_df = mr_endpoint_df.query(_q).copy()
                mr_df = mr_df.merge(mr_endpoint_df[['group_cc', 'cc_task']], 'right', ['group_cc', 'cc_task'])

                logger.info(f"Dropped {num_mr_tasks - len(mr_endpoint_df)} merge-review tasks"
                            f" ({num_mr_edges - len(mr_df)} edges)")
                num_mr_tasks = len(mr_endpoint_df)
                num_mr_edges = len(mr_df)

    with Timer("Dropping bad bodies", logger):
        if isinstance(bad_bodies, str):
            bad_bodies = pd.read_csv(bad_bodies)['body']
    
        # Drop bad bodies (e.g. "frankenbodies")
        if bad_bodies is not None:
            bad_bodies = set(bad_bodies)
            
            if focused_df is not None:
                # Discarding from focused is easy.        
                focused_df = focused_df.query('(body_a not in @bad_bodies) and (body_b not in @bad_bodies)').copy()
                logger.info(f"Dropped {num_focused - len(focused_df)} focused tasks")
                num_focused = len(focused_df)
    
            if mr_df is not None:
                # Discarding from the two MR tables is a little trickier.
                # Figure out which tasks (group_cc, cc_task) are bad, and then
                # merge with an 'indicator' column so we can drop the rows that were NOT found in the bad tasks.
                bad_mr_tasks = mr_df.query('(body_a in @bad_bodies) or (body_b in @bad_bodies)')[['group_cc', 'cc_task']].drop_duplicates()
                marked_mr = (mr_df[['group_cc', 'cc_task']]
                                .merge(bad_mr_tasks, how='left', on=['group_cc', 'cc_task'], indicator='side'))
                keep_rows = (marked_mr['side'] == 'left_only')
                mr_df = mr_df.loc[keep_rows]
            
                marked_mr_endpoints = (mr_endpoint_df[['group_cc', 'cc_task']]
                                        .merge(bad_mr_tasks, how='left', on=['group_cc', 'cc_task'], indicator='side'))
                keep_rows = (marked_mr_endpoints['side'] == 'left_only')
                mr_endpoint_df = mr_endpoint_df.loc[keep_rows]

                logger.info(f"Dropped {num_mr_tasks - len(mr_endpoint_df)} merge-review tasks"
                            f" ({num_mr_edges - len(mr_df)} edges)")
                num_mr_tasks = len(mr_endpoint_df)
                num_mr_edges = len(mr_df)


    with Timer("Dropping previously-reviewed focused tasks", logger):
        # Drop any focused tasks that have already been assigned in the past.
        if previous_focused_df is None:
            previous_focused_df = fetch_focused_decisions(*dvid_node, normalize_pairs='sv')
        drop_previously_reviewed(focused_df, previous_focused_df)
        logger.info(f"Dropped {num_focused - len(focused_df)} focused tasks")
        num_focused = len(focused_df)

    # TODO: Drop previously-reviewed merge-review tasks

    # Are any of the inputs missing their status columns?
    # If so, update them all.
    dfs = [*filter(lambda df: df is not None, (focused_df, mr_df, mr_endpoint_df))]
    if not all(({'status_a', 'status_b'} < {*df.columns}) for df in dfs):
        with Timer("Updating status columns", logger):
            if body_annotations_df is None:
                body_annotations_df = fetch_body_annotations(*dvid_node)
    
            if focused_df is not None:
                focused_df = focused_df.drop(columns=['status_a', 'status_b'], errors='ignore')
                focused_df = focused_df.merge(body_annotations_df['status'], 'left', left_on='body_a', right_index=True)
                focused_df = focused_df.merge(body_annotations_df['status'], 'left', left_on='body_b', right_index=True, suffixes=['_a', '_b'])
    
            if mr_df is not None:
                mr_df = mr_df.drop(columns=['status_a', 'status_b'], errors='ignore')
                mr_df = mr_df.merge(body_annotations_df['status'], 'left', left_on='body_a', right_index=True)
                mr_df = mr_df.merge(body_annotations_df['status'], 'left', left_on='body_b', right_index=True, suffixes=['_a', '_b'])
        
                mr_endpoint_df = mr_endpoint_df.drop(columns=['status_a', 'status_b'], errors='ignore')
                mr_endpoint_df = mr_endpoint_df.merge(body_annotations_df['status'], 'left', left_on='body_a', right_index=True)
                mr_endpoint_df = mr_endpoint_df.merge(body_annotations_df['status'], 'left', left_on='body_b', right_index=True, suffixes=['_a', '_b'])

    if any([at_least_one_of_statuses, connecting_statuses_1, connecting_statuses_2, at_most_one_of_statuses]):
        with Timer("Filtering by body status", logger):
            query = ''
            if at_least_one_of_statuses:
                clause = '(status_a in @at_least_one_of_statuses) or (status_b in @at_least_one_of_statuses)'
                if query:
                    query = f'({query} or {clause})'
                else:
                    query = clause
            
            if connecting_statuses_1 and connecting_statuses_2:
                clause = ('(    (status_a in @connecting_statuses_1 and status_b in @connecting_statuses_2)'
                          '  or (status_a in @connecting_statuses_2 and status_b in @connecting_statuses_1))')
                if query:
                    query = f'({query} or {clause})'
                else:
                    query = clause
    
            if at_most_one_of_statuses:
                clause = '(not ((status_a in @at_most_one_of_statuses) and (status_b in @at_most_one_of_statuses)))'

                # This one is ANDed
                if query:
                    query = f'({query} and {clause})'
                else:
                    query = clause

            if focused_df is not None:
                focused_df = focused_df.query(query).copy()
                logger.info(f"Dropped {num_focused - len(focused_df)} focused tasks")
                num_focused = len(focused_df)
    
            if mr_df is not None:
                mr_endpoint_df = mr_endpoint_df.query(query).copy()
                mr_df = mr_df.merge(mr_endpoint_df[['group_cc', 'cc_task']], 'right', on=['group_cc', 'cc_task'])
                logger.info(f"Dropped {num_mr_tasks - len(mr_endpoint_df)} merge-review tasks"
                            f" ({num_mr_edges - len(mr_df)} edges)")
                num_mr_tasks = len(mr_endpoint_df)
                num_mr_edges = len(mr_df)

    if focused_df is not None:
        logger.info(f"Remaining focused tasks: {len(focused_df)}")

    if mr_df is not None:
        logger.info(f"Remaining merge review tasks: {len(mr_endpoint_df)} ({len(mr_df)} edges)")

    return (focused_df, mr_df, mr_endpoint_df)


def upload_focused_tasks(assignment, comment, server, uuid=None, instance='focused_assign', *, repo_uuid=None, overwrite_existing=False):
    """
    Upload a set of focused proofreading assignments to a dvid key-value instance.
    
    Args:
        assignment:
            One of the following:
            - A pre-loaded focused proofreading JSON assignment (i.e. a dict)
            - A filepath to a JSON assignment
            - A DataFrame of focused proofreading tasks with columns:
                ['sv_a', 'sv_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb']
            - A filepath to a CSV file that can be loaded as a DataFrame.

        comment:
            What comment to insert into the tasks, to distinguish
            this task set from other task sets.
        
        server:
            DVID server, e.g. 'emdata4:8900'

        uuid:
            The uuid to upload the tasks to.
            If not provided, the current master node will be used.
        
        instance:
            The keyvalue instance to upload to.
        
        repo_uuid:
            If uuid is not provided and the server contains more than one DVID repo,
            you must provide a repo_uuid in order for the master node to be
            automatically identified.
        
        overwrite_existing:
            If any of the given tasks already exist in the keyvalue instance,
            they will not be uploaded unless this flag is True.
    
    Returns:
        (written_pairs, skipped_pairs)
        The lists of SV ID pairs that were uploaded and skipped, respectively.
    """
    # If the user passed a filepath, load it.
    if isinstance(assignment, str):
        if assignment.endswith('.csv'):
            assignment = load_focused_table(assignment)
        elif assignment.endswith('.json'):
            with open(assignment, 'r'):
                assignment = json.load()
    
    # If the loaded data is a DataFrame, convert to JSON.
    if isinstance(assignment, pd.DataFrame):
        assignment = generate_focused_assignment(assignment)

    assert isinstance(assignment, dict)
    assert assignment["file type"] == "Neu3 task list"

    if uuid is None:
        uuid = find_master(server, repo_uuid)

    all_keys = fetch_keys(server, uuid, instance)
    all_keys = set(all_keys)
    tasks = assignment['task list']

    skipped_pairs = []
    written_pairs = []

    upload_tasks = {}
    for task in tasks:
        sv_a = task['supervoxel ID 1']
        sv_b = task['supervoxel ID 2']
        focused_ID = f"{sv_a}_{sv_b}"

        if focused_ID in all_keys and not overwrite_existing:
            # already loaded
            skipped_pairs.append( (sv_a, sv_b) )
        else:
            task['focused ID'] = focused_ID
            task['status'] = "Not examined"
            task['comment'] = comment
            upload_tasks[focused_ID] = task
            written_pairs.append( (sv_a, sv_b) )

    logger.info(f"Uploading {len(upload_tasks)} tasks, skipping {len(skipped_pairs)}")
    post_keyvalues(server, uuid, instance, upload_tasks, batch_size=10_000)
    return written_pairs, skipped_pairs


def compute_focused_bodies(server, uuid, instance, synapse_samples, min_tbars, min_psds, root_sv_sizes, min_body_size, sv_classifications=None, marked_bad_bodies=None, return_table=False, kafka_msgs=None):
    """
    Compute the complete set of focused bodies, based on criteria for
    number of tbars, psds, or overall size, and excluding explicitly
    listed bad bodies.
    
    This function takes ~20 minutes to run on hemibrain inputs, with a ton of RAM.
    
    The procedure is:

    1. Apply synapse-based criteria
      a. Load synapse CSV file
      b. Map synapse SVs -> bodies (if needed)
        b2. If any SVs are 'retired', update those synapses to use the new IDs.
      c. Calculate synapses (tbars, psds) per body
      d. Initialize set with bodies that have enough synapses
    
    2. Apply size-based criteria
      a. Calculate body sizes (based on supervoxel sizes and current mapping)
      b. Add "big" bodies to the set
    
    3. Apply "bad body" criteria
      a. Read the list of "bad bodies"
      b. Remove bad bodies from the set
    
    Example:

        server = 'emdata3:8900'
        uuid = '7254'
        instance = 'segmentation'

        synapse_samples = '/nrs/flyem/bergs/complete-ffn-agglo/sampled-synapses-ef1d-locked.csv'

        min_tbars = 2
        min_psds = 10

        # old repo supervoxels (before server rebase)
        #
        # Note: This was taken from node 5501ae83e31247498303a159eef824d8, which is from a different repo.
        #       But that segmentation was eventually copied to the production repo as root node a776af.
        #       See /groups/flyem/data/scratchspace/copyseg-configs/labelmaps/hemibrain/8nm/copy-fixed-from-emdata2-to-emdata3-20180402.214505
        #
        #root_sv_sizes_dir = '/groups/flyem/data/scratchspace/copyseg-configs/labelmaps/hemibrain/8nm/compute-8nm-extended-fixed-STATS-ONLY-20180402.192015'
        #root_sv_sizes = f'{root_sv_sizes_dir}/supervoxel-sizes.h5'
        
        root_sv_sizes_dir = '/groups/flyem/data/scratchspace/copyseg-configs/labelmaps/hemibrain/flattened/compute-stats-from-corrupt-20181016.203848'
        root_sv_sizes = f'{root_sv_sizes_dir}/supervoxel-sizes-2884.h5'
        
        min_body_size = int(10e6)

        sv_classifications = '/nrs/flyem/bergs/sv-classifications.h5'
        marked_bad_bodies = '/nrs/flyem/bergs/complete-ffn-agglo/bad-bodies-2019-02-26.csv'
        
        table_description = f'{uuid}-{min_tbars}tbars-{min_psds}psds-{min_body_size / 1e6:.1f}Mv'
        focused_table = compute_focused_bodies(server, uuid, instance, synapse_samples, min_tbars, min_psds, root_sv_sizes, min_body_size, sv_classifications, marked_bad_bodies, return_table=True)

        # As npy:
        np.save(f'focused-{table_description}.npy', focused_table.to_records(index=True))

        # As CSV:
        focused_table.to_csv(f'focused-{table_description}.npy', index=True, header=True)
    
    Args:

        server, uuid, instance:
            labelmap instance

        root_sv_sizes:
            mapping of supervoxel sizes from the root node, as returned by load_supervoxel_sizes(),
            or a path to an hdf5 file from which it can be loaded
        
        synapse_samples:
            A DataFrame with columns 'body' (or 'sv') and 'kind', or a path to a CSV file with those columns.
            The 'kind' column is expected to have only 'PreSyn' and 'PostSyn' entries.
            If the table has an 'sv' column, any "retired" supervoxel IDs will be updated before
            continuing with the analysis.
        
        min_tbars:
            The minimum number pf PreSyn entries to pass the filter.
            Bodies with fewer tbars may still be included if they satisfy the min_psds criteria.
        
        min_psds:
            The minimum numer of PostSyn entires to pass the filter.
            Bodies with fewer psds may still pass the filter if they satisfy the min_tbars criteria.

        min_body_size:
            Determines which bodies are included on the basis of their size alone,
            regardless of synapse count.
        
        sv_classifications:
            Optional. Path to an hdf5 file containing supervoxel classifications.
            Must have datasets: 'supervoxel_ids', 'classifications', and 'class_names'.
            Used to exclude known-bad supervoxels. The file need not include all supervoxels.
            Any supervoxels MISSING from this file are not considered 'bad'.

        marked_bad_bodies:
            Optional. A list of known-bad bodies to exclude from the results,
            or a path to a .csv file with that list (in the first column),
            or a keyvalue instance name from which the list can be loaded as JSON.
        
        return_table:
            If True, return the focused bodies in a DataFrame, indexed by body,
            with columns for body size and synapse counts.
            If False, simply return the list of bodies (saves about 4 minutes).
    
    Returns:
        A list of body IDs that passed all filters, or a DataFrame with columns:
            ['voxel_count', 'PreSyn', 'PostSyn']
        (See return_table option.)
    """
    split_source = 'dvid'
    
    # Load full mapping. It's needed for both synapses and body sizes.
    mapping = fetch_complete_mappings(server, uuid, instance, include_retired=True, kafka_msgs=kafka_msgs)
    mapper = LabelMapper(mapping.index.values, mapping.values)

    ##
    ## Synapses
    ##
    if isinstance(synapse_samples, str):
        synapse_samples = load_synapses(synapse_samples)

    assert set(['sv', 'body']).intersection(set(synapse_samples.columns)), \
        "synapse samples must have either 'body' or 'sv' column"

    # If 'sv' column is present, use it to create (or update) the body column
    if 'sv' in synapse_samples.columns:
        synapse_samples = update_synapse_table(server, uuid, instance, synapse_samples, split_source=split_source)
        assert synapse_samples['sv'].dtype == np.uint64
        synapse_samples['body'] = mapper.apply(synapse_samples['sv'].values, True)

    with Timer("Filtering for synapses", logger):
        synapse_body_table = body_synapse_counts(synapse_samples)
        synapse_bodies = synapse_body_table.query('PreSyn >= @min_tbars or PostSyn >= @min_psds').index.values
    logger.info(f"Found {len(synapse_bodies)} with sufficient synapses")

    focused_bodies = set(synapse_bodies)
    
    ##
    ## Body sizes
    ##
    with Timer("Filtering for body size", logger):    
        sv_sizes = load_all_supervoxel_sizes(server, uuid, instance, root_sv_sizes, split_source=split_source)
        body_stats = compute_body_sizes(sv_sizes, mapping, True)
        big_body_stats = body_stats.query('voxel_count >= @min_body_size')
        big_bodies = big_body_stats.index
    logger.info(f"Found {len(big_bodies)} with sufficient size")
    
    focused_bodies |= set(big_bodies)

    ##
    ## SV classifications
    ##
    if sv_classifications is not None:
        with Timer(f"Filtering by supervoxel classifications"), h5py.File(sv_classifications, 'r') as f:
            sv_classes = pd.DataFrame({'sv': f['supervoxel_ids'][:],
                                       'klass': f['classifications'][:].astype(np.uint8)})
    
            # Get the set of bad supervoxels
            all_class_names = list(map(bytes.decode, f['class_names'][:]))
            bad_class_names = ['unknown', 'blood vessels', 'broken white tissue', 'glia', 'oob']
            _bad_class_ids = set(map(all_class_names.index, bad_class_names))
            bad_svs = sv_classes.query('klass in @_bad_class_ids')['sv']
            
            # Add column for sizes
            bad_sv_sizes = pd.DataFrame(index=bad_svs).merge(pd.DataFrame(sv_sizes), how='left', left_index=True, right_index=True)
    
            # Append body
            bad_sv_sizes['body'] = mapper.apply(bad_sv_sizes.index.values, True)
            
            # For bodies that contain at least one bad supervoxel,
            # compute the total size of the bad supervoxels they contain
            body_bad_voxels = bad_sv_sizes.groupby('body').agg({'voxel_count': 'sum'})
            
            # Append total body size for comparison
            body_bad_voxels = body_bad_voxels.merge(body_stats[['voxel_count']], how='left', left_index=True, right_index=True, suffixes=('_bad', '_total'))
            
            bad_bodies = body_bad_voxels.query('voxel_count_bad > voxel_count_total//2').index
            
            bad_focused_bodies = focused_bodies & set(bad_bodies)
            logger.info(f"Dropping {len(bad_focused_bodies)} bodies with more than 50% bad supervoxels")
    
            focused_bodies -= bad_focused_bodies

    ##
    ## Marked Bad bodies
    ##
    if marked_bad_bodies is not None:
        if isinstance(marked_bad_bodies, str):
            if marked_bad_bodies.endswith('.csv'):
                marked_bad_bodies = read_csv_col(marked_bad_bodies, 0)
            else:
                # If it ain't a CSV, maybe it's a key-value instance and key to read from.
                raise AssertionError("FIXME: Need convention for specifying key-value instance and key")
                #marked_bad_bodies = fetch_key(server, uuid, marked_bad_bodies, as_json=True)
        
        with Timer(f"Dropping {len(marked_bad_bodies)} bad bodies (from {len(focused_bodies)})"):
            focused_bodies -= set(marked_bad_bodies)

    ##
    ## Prepare results
    ##
    logger.info(f"Found {len(focused_bodies)} focused bodies")
    focused_bodies = np.fromiter(focused_bodies, dtype=np.uint64)
    focused_bodies.sort()

    if not return_table:
        return focused_bodies

    with Timer("Computing full focused table", logger):
        # Start with an empty DataFrame (except index)
        focus_table = pd.DataFrame(index=focused_bodies)

        # Merge size/sv_count
        focus_table = focus_table.merge(body_stats, how='left', left_index=True, right_index=True, copy=False)
        
        # Add synapse columns
        focus_table = focus_table.merge(synapse_body_table, how='left', left_index=True, right_index=True, copy=False)

        focus_table.fillna(0, inplace=True)
        focus_table['voxel_count'] = focus_table['voxel_count'].astype(np.uint64)
        focus_table['sv_count'] = focus_table['sv_count'].astype(np.uint32)
        focus_table['PreSyn'] = focus_table['PreSyn'].astype(np.uint32)
        focus_table['PostSyn'] = focus_table['PostSyn'].astype(np.uint32)

        # Sort biggest..smallest
        focus_table.sort_values('voxel_count', ascending=False, inplace=True)
        focus_table.index.name = 'body'

    return focus_table
