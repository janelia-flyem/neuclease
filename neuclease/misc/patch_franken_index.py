"""
This is a one-off script for patching the frankenbody labelindex and
mapping after the its segmentation voxels have already been updated.
If you want to use this again, you'll need to edit the constants below.
"""
import os
import logging
import argparse
import datetime

import numpy as np
import pandas as pd

from neuclease import configure_default_logging
from neuclease.dvid import fetch_repo_info, fetch_labelindex, convert_labelindex_to_pandas, PandasLabelIndex, create_labelindex, post_labelindex, post_mappings
from neuclease.util import Timer

# This script patches exactly one "franken-supervoxel"
# The body and supervoxel IDs happen to be the same in this example,
# but they don't have to be.
FRANKENBODY = 106979579
FRANKENBODY_SV = 106979579

PATCH_BOX_XYZ = np.array([[13312, 25728, 31488], [18432, 36480, 38272]])
PATCH_BOX_ZYX = PATCH_BOX_XYZ[:, ::-1]
del PATCH_BOX_XYZ

NEW_SV_THRESHOLD = 6_000_000_000 # new supervoxels start at 6e9

logger = logging.getLogger(__name__)


def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('instance')
    parser.add_argument('block_stats')
    args = parser.parse_args()
    
    seg_instance = (args.server, args.uuid, args.instance)
    
    from flyemflows.bin.ingest_label_indexes import load_stats_h5_to_records
    with Timer("Loading block stats", logger):
        (block_sv_stats, _presorted_by, _agglo_path) = load_stats_h5_to_records('block-statistics.h5')
        stats_df = pd.DataFrame(block_sv_stats)
        stats_df = stats_df[['z', 'y', 'x', 'segment_id', 'count']]
        stats_df = stats_df.rename(columns={'segment_id': 'sv'})
        
        # Keep only the new supervoxels.
        stats_df = stats_df.query('sv > @NEW_SV_THRESHOLD').copy()
    
    with Timer("Fetching old labelindex", logger):
        labelindex = fetch_labelindex(*seg_instance, 106979579, format='protobuf')

    with Timer("Extracting labelindex table", logger):
        old_df = convert_labelindex_to_pandas(labelindex).blocks

    with Timer("Patching labelindex table", logger):
        # Discard old supervoxel stats within patched area
        in_patch  = (old_df[['z', 'y', 'x']].values >= PATCH_BOX_ZYX[0]).all(axis=1)
        in_patch &= (old_df[['z', 'y', 'x']].values  < PATCH_BOX_ZYX[1]).all(axis=1)
        
        old_df['in_patch'] = in_patch
        unpatched_df = old_df.query('not (in_patch and sv == @FRANKENBODY_SV)').copy()
        del unpatched_df['in_patch']
        
        # Append new stats
        new_df = pd.concat((unpatched_df, stats_df), ignore_index=True)
        new_df = new_df.sort_values(['z', 'y', 'x', 'sv'])

        np.save('old_df.npy', old_df.to_records(index=False))
        np.save('new_df.npy', new_df.to_records(index=False))

        if old_df['count'].sum() != new_df['count'].sum():
            logger.warning("Old and new indexes do not have the same total counts.  See old_df.npy and new_df.npy")

    with Timer("Constructing new labelindex", logger):    
        last_mutid = fetch_repo_info(*seg_instance[:2])["MutationID"]
        mod_time = datetime.datetime.now().isoformat()
        new_li = PandasLabelIndex(new_df, FRANKENBODY_SV, last_mutid, mod_time, os.environ.get("USER", "unknown"))
        new_labelindex = create_labelindex(new_li)

    with Timer("Posting new labelindex", logger):
        post_labelindex(*seg_instance, FRANKENBODY_SV, new_labelindex)

    with Timer("Posting updated mapping", logger):
        new_mapping = pd.Series(FRANKENBODY_SV, index=new_df['sv'].unique(), dtype=np.uint64, name='body')
        post_mappings(*seg_instance, new_mapping, last_mutid)

    logger.info("DONE")

    
if __name__ == "__main__":
    main()