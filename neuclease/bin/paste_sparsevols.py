"""
Copy set of exported sparsevol objects to a segmentation instance,
optionally excluding voxels inside (or outside) an ROI.
"""
import os
import sys
import json
import logging
import argparse

import numpy as np

from neuclease.util import Timer, lexsort_columns, groupby_presorted
from neuclease.dvid import fetch_labelarray_voxels, post_labelarray_blocks, fetch_maxlabel
from neuclease.dvid.rle import parse_rle_response

logger = logging.getLogger(__name__)


def main():
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dvid_server')
    parser.add_argument('uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('sparsevol_files', nargs='+')
    args = parser.parse_args()

    instance_info = (args.dvid_server, args.uuid, args.labelmap_instance)

    maxlabel = fetch_maxlabel(args.dvid_server, args.uuid, args.labelmap_instance)
    for path in args.sparsevol_files:
        maxlabel += 1
        name = os.path.split(path)[1]
        with Timer(f"Pasting {name} as {maxlabel}", logger):
            overwritten_labels = overwrite_sparsevol(*instance_info, maxlabel, path)

        results_path = os.path.splitext(path)[0] + '.json'
        with open(results_path, 'w') as f:
            results = { 'new-label': maxlabel,
                        'overwritten_labels': sorted(overwritten_labels) }
            json.dump(results, f, indent=2)

    logger.info(f"Done.")


def overwrite_sparsevol(server, uuid, instance, new_label, sparsevol_filepath):
    sorted_path = sparsevol_filepath + '.sorted_blocktable.npy'
    if os.path.exists(sorted_path):
        logger.info("Loading presorted sparse coordinates")
        sorted_table = np.load(sorted_path)
    else:
        with open(sparsevol_filepath, 'rb') as f:
            with Timer("Parsing sparsevol coordinates", logger):
                coords = parse_rle_response(f.read(), np.int16)
            with Timer("Sorting sparsevol coordiantes", logger):
                table = np.concatenate( (coords // 64, coords) )
                sorted_table = lexsort_columns(table)
                del table

            with Timer("Saving sparsevol sorted blocktable"):           
                np.save(sorted_path, sorted_table)

    overwritten_labels = set()
    
    BLOCK_GROUP_SIZE = 400
    next_block_set = []
    for coord_group in groupby_presorted(sorted_table[:,3:], sorted_table[:, :3]):
        block_corner = coord_group[0] // 64 * 64
        block_voxels = fetch_labelarray_voxels(server, uuid, instance, (block_corner, 64 + block_corner), supervoxels=True)
        
        overwritten_labels |= set(np.unique(block_voxels[tuple(coord_group.transpose())]))
        block_voxels[tuple(coord_group.transpose())] = new_label
        
        next_block_set.append((block_corner, block_voxels))
        
        # Flush the blocks periodically
        if len(next_block_set) == BLOCK_GROUP_SIZE:
            logger.info("Sending block set...")
            post_labelarray_blocks(server, uuid, instance, *zip(*next_block_set), downres=True)
            next_block_set = []

    logger.info("Sending last block set...")
    post_labelarray_blocks(server, uuid, instance, *zip(*next_block_set), downres=True)
    next_block_set = []
    
    return overwritten_labels
        

if __name__ == "__main__":
    main()
