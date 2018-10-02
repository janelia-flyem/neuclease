import os
import logging

import h5py
import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from ..util import read_csv_header, Timer
from ..util.csv import read_csv_col
from ..merge_table import load_all_supervoxel_sizes, compute_body_sizes
from ..dvid import fetch_keys, fetch_complete_mappings, fetch_keyvalues
from ..dvid.annotation import load_synapses_from_csv
from ..dvid.labelmap import fetch_supervoxel_fragments, fetch_labels

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
               'jaccard': np.float32,
               'overlap': np.uint32,
               'x': np.int32, 'y': np.int32, 'z': np.int32 }


def update_synapse_table(server, uuid, instance, synapse_df, output_path=None):
    """
    Give a dataframe (or a CSV file) with at least columns ['sv', 'x', 'y', 'z'],
    identify any 'retired' supervoxels and query DVID to
    replace them with their split descendents.
    
    Optionally write the updated table to CSV.
    """
    seg_info = (server, uuid, instance)
    
    if isinstance(synapse_df, str):
        synapse_df = load_synapses_from_csv(synapse_df)

    # Get the set of all retired supervoxel IDs
    _leaves, _retired = fetch_supervoxel_fragments(*seg_info, 'dvid')

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
    REQUIRED_COLUMNS = ['id_a', 'id_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb']
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


def fetch_focused_decisions(server, uuid, instance, normalize_pairs=None, subset_pairs=None, subset_slice=None):
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
        
        subset_slice:
            If provided, only select from the given slice of the full key set.
            Must be a slice object, e.g. slice(1000,2000), or np.s_[1000:2000]

    Returns:
        DataFrame with columns:
        ['body_a', 'body_b', 'result', 'sv_a', 'sv_b',
        'time', 'time zone', 'user',
        'xa', 'xb', 'ya', 'yb', 'za', 'zb', ...]
    """
    assert normalize_pairs in (None, 'sv', 'body')
    
    keys = fetch_keys(server, uuid, instance)
    if subset_pairs is not None:
        subset_keys1 = [f'{a}+{b}' for a,b in subset_pairs]
        subset_keys2 = [f'{a}+{b}' for a,b in subset_pairs]
        
        keys = list(set(subset_keys1).intersection(keys) | set(subset_keys2).intersection(keys))

    if subset_slice:
        keys = keys[subset_slice]
    
    task_values = list(fetch_keyvalues(server, uuid, instance, keys, as_json=True).values())

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

    # Replace NaN and fix dtypes
    for col in ['sv_a', 'sv_b', 'body_a', 'body_b']:
        if col in df:
            df[col].fillna(0.0, inplace=True)
            df[col] = df[col].astype(np.uint64)

    if normalize_pairs is None:
        return df
    
    if normalize_pairs == 'sv':
        swap_rows = (df['sv_a'] > df['sv_b'])
    elif normalize_pairs == 'body':
        swap_rows = (df['body_a'] > df['body_b'])
    else:
        raise AssertionError(f"bad 'normalize_pairs' setting: {normalize_pairs}")

    # Swap A/B cols to "normalize" id pairs
    cols = ['sv_a', 'sv_b', 'body_a', 'body_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb']
    cols = filter(lambda col: col in df, cols)
    df_copy = df[list(cols)].copy()
    for name in ['sv_', 'body_', 'x', 'y', 'z']:
        if f'{name}a' in df:
            df.loc[swap_rows, f'{name}a'] = df_copy.loc[swap_rows, f'{name}b']
            df.loc[swap_rows, f'{name}b'] = df_copy.loc[swap_rows, f'{name}a']

    return df


def drop_previously_reviewed(df, previous_focused_decisions_df):
    """
    Given a DataFrame of speculative focused decisions and 
    a DataFrame of previously reviewed focused decisions,
    drop all previous decisions from the speculative set,
    regardless of review results.
    """
    comparison_df = previous_focused_decisions_df[['id_a', 'id_b']].drop_duplicates()
    in_prev = df[['id_a', 'id_b']].merge(comparison_df,
                                         how='left',
                                         on=['id_a', 'id_b'],
                                         indicator='side')

    keep_rows = (in_prev['side'] == 'left_only')
    return df[keep_rows.values]


def body_synapse_counts(synapse_samples):
    """
    Given a DataFrame of sampled synapses (or a path to a CSV file),
    Tally synapse totals (by kind) for each body.
    
    Returns:
        DataFrame with columns: ['body', 'PreSyn', 'PostSyn'],
        (The PreSyn/PostSyn columns are synapse counts.)
    """
    if isinstance(synapse_samples, str):
        synapse_samples = pd.read_csv(synapse_samples)
    
    assert 'body' in synapse_samples.columns, "Samples must have a 'body' col."
    assert 'kind' in synapse_samples.columns, "Samples must have a 'kind' col"
    
    synapse_samples = synapse_samples[['body', 'kind']]
    synapse_counts = synapse_samples.pivot_table(index='body', columns='kind', aggfunc='size')
    synapse_counts.fillna(0.0, inplace=True)

    if 0 in synapse_counts.index:
        logger.warning("*** Synapse table includes body 0 and was therefore probably generated from out-of-date data. ***")
    
    return synapse_counts


def compute_focused_bodies(server, uuid, instance, synapse_samples, min_tbars, min_psds, root_sv_sizes, min_body_size, sv_classifications=None, marked_bad_bodies=None, return_table=False):
    """
    Compute the complete set of focused bodies, based on criteria for
    number of tbars, psds, or overall size, and excluding explicitly
    listed bad bodies.
    
    This function takes ~20 minutes to run on hemibrain inputs, with a ton of RAM.
    
    The procedure is:

    1. Apply synapse-based criteria
      a. Load synapse CSV file
      b. Map synapse SVs -> bodies (if needed)
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
        synapse_samples = '/nrs/flyem/bergs/complete-ffn-agglo/sampled-synapses-d585.csv'
        min_tbars = 2
        min_psds = 10
        root_sv_sizes_dir = '/groups/flyem/data/scratchspace/copyseg-configs/labelmaps/hemibrain/8nm/compute-8nm-extended-fixed-STATS-ONLY-20180402.192015'
        root_sv_sizes = f'{root_sv_sizes_dir}/supervoxel-sizes.h5'
        min_body_size = int(10e6)
        sv_classifications = '/nrs/flyem/bergs/sv-classifications.h5'
        marked_bad_bodies = '/nrs/flyem/bergs/complete-ffn-agglo/bad-bodies-2018-10-01.csv'
        
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
    # Load full mapping. It's needed for both synapses and body sizes.
    mapping = fetch_complete_mappings(server, uuid, instance, include_retired=True)
    mapper = LabelMapper(mapping.index.values, mapping.values)

    ##
    ## Synapses
    ##
    if isinstance(synapse_samples, str):
        synapse_samples = load_synapses_from_csv(synapse_samples)

    assert set(['sv', 'body']) - set(synapse_samples.columns), \
        "synapse samples must have either 'body' or 'sv' column"

    # If 'sv' column is present, use it to create (or update) the body column
    if 'sv' in synapse_samples.columns:
        synapse_samples = update_synapse_table(server, uuid, instance, synapse_samples)
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
        sv_sizes = load_all_supervoxel_sizes(server, uuid, instance, root_sv_sizes)
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
