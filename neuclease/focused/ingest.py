import os
import logging

import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from ..util import read_csv_header, Timer
from ..util.csv import read_csv_col
from ..merge_table import load_all_supervoxel_sizes, compute_body_sizes
from ..dvid import fetch_key, fetch_complete_mappings
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


def compute_focused_bodies(server, uuid, instance, synapse_samples, min_tbars, min_psds, root_sv_sizes, min_body_size, bad_bodies, return_table=False):
    """
    Compute the complete set of focused bodies, based on criteria for
    number of tbars, psds, or overall size, and excluding explicitly
    listed bad bodies.
    
    This function takes ~12 minutes to run on hemibrain inputs, with a ton of RAM.
    
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
        bad_bodies = '/nrs/flyem/bergs/complete-ffn-agglo/bad-bodies-2018-08-21.csv'
        
        focused_table = compute_focused_bodies(server, uuid, instance, synapse_samples, min_tbars, min_psds, root_sv_sizes, min_body_size, bad_bodies, return_table=True)
    
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
        
        bad_bodies:
            A list of known-bad bodies to exclude from the results,
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
        mapper = LabelMapper(mapping.index.values, mapping.values)
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
        big_bodies = body_sizes[body_sizes >= min_body_size].index
    logger.info(f"Found {len(big_bodies)} with sufficient size")
    
    focused_bodies |= set(big_bodies)

    ##
    ## Bad bodies
    ##
    if isinstance(bad_bodies, str):
        if bad_bodies.endswith('.csv'):
            bad_bodies = read_csv_col(bad_bodies, 0)
        else:
            # If it ain't a CSV, maybe it's a key-value instance to read from.
            bad_bodies = fetch_key(server, uuid, bad_bodies, as_json=True)
    
    with Timer(f"Dropping {len(bad_bodies)} bad bodies (from {len(focused_bodies)})"):
        focused_bodies -= set(bad_bodies)

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
        focus_table.index.name = 'body'

        # Add size column (voxel_count)
        body_sizes_df = pd.DataFrame({'voxel_count': body_sizes})
        focus_table = focus_table.merge(body_sizes_df, how='left', left_index=True, right_index=True, copy=False)
        
        # Add synapse columns
        focus_table = focus_table.merge(synapse_body_table, how='left', left_index=True, right_index=True, copy=False)

        # Sort biggest..smallest
        focus_table.sort_values('voxel_count', ascending=False, inplace=True)

    return focus_table
