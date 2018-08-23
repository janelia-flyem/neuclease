import os
import logging

from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from ..util import read_csv_header, Timer
from ..util.csv import read_csv_col
from ..merge_table import load_all_supervoxel_sizes, compute_body_sizes
from ..dvid import fetch_keys, fetch_key, fetch_complete_mappings
from ..dvid.annotation import load_synapses_from_csv

# Load new table. Normalize.

# Optionally: Score and choose body-wise favorites.

# Load previously assigned. Normalize.

logger = logging.getLogger(__name__)

CSV_DTYPES = { 'id_a': np.uint64, 'id_b': np.uint64, # Use'id_a', and 'id_b' for consistency with our other code.
               'xa': np.int32, 'ya': np.int32, 'za': np.int32,
               'xb': np.int32, 'yb': np.int32, 'zb': np.int32,
               'caa': np.float32, 'cab': np.float32, 'cba': np.float32, 'cbb': np.float32,
               'iou': np.float32,
               'da': np.float32, 'db': np.float32,
               'score': np.float32,
               'jaccard': np.float32,
               'overlap': np.uint32,
               'body_a': np.uint64, 'body_b': np.uint64,
               'sv_a': np.uint64, 'sv_b': np.uint64,
               'body': np.uint64,
               'sv': np.uint64,
               'x': np.int32, 'y': np.int32, 'z': np.int32 }


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


def fetch_focused_decisions(server, uuid, instance, normalize_pairs=None, show_progress=False):
    """
    Load all focused decisions from a given keyvalue instance
    (e.g. 'segmentation_merged') and return them as a DataFrame,
    with slight modifications to use standard column names.
    
    Args:
        server, uuid, instance
            Exmaple: 'emdata3:8900', '7254', 'segmentation_merged'
        
        normalize_pairs:
            Either None, 'sv', or 'body'
            If not None, swap A/B columns so that either sv_a < sv_b or body_a < body_b.
        
        show_progress:
            If True, show a tqdm progress bar while the data is downloading.
    
    Returns:
        DataFrame with columns:
        ['body_a', 'body_b', 'result', 'sv_a', 'sv_b',
        'time', 'time zone', 'user',
        'xa', 'xb', 'ya', 'yb', 'za', 'zb', ...]
    """
    assert normalize_pairs in (None, 'sv', 'body')
    
    keys = fetch_keys(server, uuid, instance)
    task_values = [fetch_key(server, uuid, instance, key, as_json=True) for key in tqdm(keys, disable=not show_progress)]

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
    df['result'] = pd.Series(df['result'], dtype='category')
    df['user'] = pd.Series(df['user'], dtype='category')
    df['time zone'] = pd.Series(df['time zone'], dtype='category')

    if normalize_pairs is None:
        return df
    
    if normalize_pairs == 'sv':
        swap_rows = (df['sv_a'] > df['sv_b'])
    elif normalize_pairs == 'body':
        swap_rows = (df['body_a'] > df['body_b'])
    else:
        raise AssertionError(f"bad 'normalize_pairs' setting: {normalize_pairs}")

    # Swap A/B cols to "normalize" id pairs
    df_copy = df[['sv_a', 'sv_b', 'body_a', 'body_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb']].copy()
    for name in ['sv_', 'body_', 'x', 'y', 'z']:
        df.loc[swap_rows, f'{name}a'] = df_copy.loc[swap_rows, f'{name}b']
        df.loc[swap_rows, f'{name}b'] = df_copy.loc[swap_rows, f'{name}a']

    return df


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
        marked_bad_bodies = '/nrs/flyem/bergs/complete-ffn-agglo/bad-bodies-2018-08-21.csv'
        
        focused_table = compute_focused_bodies(server, uuid, instance, synapse_samples, min_tbars, min_psds, root_sv_sizes, min_body_size, sv_classifications, marked_bad_bodies, return_table=True)
    
    Args:

        server, uuid, instance:
            labelmap instance

        root_sv_sizes:
            mapping of supervoxel sizes from the root node, as returned by load_supervoxel_sizes(),
            or a path to an hdf5 file from which it can be loaded
        
        synapse_samples:
            A DataFrame with columns 'body' and 'kind', or a path to a CSV file with those columns.
            The 'kind' column is expected to have only 'PreSyn' and 'PostSyn' entries.
        
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
    mapping = fetch_complete_mappings(server, uuid, instance)
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
        body_sizes = compute_body_sizes(sv_sizes, mapping, True)
        big_body_sizes = body_sizes[body_sizes >= min_body_size]
        big_bodies = big_body_sizes.index
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
            body_bad_voxels = body_bad_voxels.merge(pd.DataFrame(body_sizes), how='left', left_index=True, right_index=True, suffixes=('_bad', '_total'))
            
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
                # If it ain't a CSV, maybe it's a key-value instance to read from.
                marked_bad_bodies = fetch_key(server, uuid, marked_bad_bodies, as_json=True)
        
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

        # Add size column (voxel_count)
        body_sizes_df = pd.DataFrame({'voxel_count': body_sizes})
        focus_table = focus_table.merge(body_sizes_df, how='left', left_index=True, right_index=True, copy=False)
        
        # Add synapse columns
        focus_table = focus_table.merge(synapse_body_table, how='left', left_index=True, right_index=True, copy=False)

        # Sort biggest..smallest
        focus_table.sort_values('voxel_count', ascending=False, inplace=True)
        focus_table.index.name = 'body'

    return focus_table
