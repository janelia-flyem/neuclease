import os
import logging

import numpy as np
import pandas as pd

from ..util import read_csv_header

from ..merge_table import load_all_supervoxel_sizes, compute_body_sizes

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
               'overlap': np.uint32 }


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


def filter_bodies_for_synapses(synapse_samples, min_tbars, min_psds): # @UnusedVariable
    if isinstance(synapse_samples, str):
        synapse_samples = pd.read_csv(synapse_samples)
    
    assert 'body' in synapse_samples.columns, "Samples must have a 'body' col."
    assert 'kind' in synapse_samples.columns, "Samples must have a 'kind' col"
    
    synapse_samples = synapse_samples[['body', 'kind']]
    synapse_counts = synapse_samples.pivot_table(index='body', columns='kind', aggfunc='size')
    filtered_bodies = synapse_counts.query('PreSyn >= @min_tbars and PostSyn > @min_psds')
    return filtered_bodies.fillna(0.0)

#def compute_focused_bodies(server, uuid, instance, root_sv_sizes, synapse_samples, min_tbars, min_psds):
    
    