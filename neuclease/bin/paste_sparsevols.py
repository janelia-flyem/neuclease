"""
Copy set of exported sparsevol objects to a segmentation instance,
optionally excluding voxels inside (or outside) an ROI.
"""
import os
import json
import logging
import argparse

import numpy as np
import pandas as pd

from neuclease import configure_default_logging
from neuclease.util import Timer, lexsort_columns, groupby_presorted, SparseBlockMask, NumpyConvertingEncoder, PrefixedLogger
from neuclease.dvid import fetch_labelarray_voxels, post_labelarray_blocks, fetch_roi, parse_rle_response, fetch_instance_info

logger = logging.getLogger(__name__)


def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-downres', action='store_true')
    parser.add_argument('--only-within-roi')
    parser.add_argument('--not-within-roi')
    parser.add_argument('dvid_server')
    parser.add_argument('uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('sparsevol_files', nargs='+')
    args = parser.parse_args()

    instance_info = (args.dvid_server, args.uuid, args.labelmap_instance)

    assert not args.only_within_roi or not args.not_within_roi, \
        "Can't supply both --only-within-roi and --not-within-roi.  Pick one or the other (or neither)."

    roi = args.only_within_roi or args.not_within_roi
    invert_roi = (args.not_within_roi is not None)
    
    if roi:
        roi_mask, mask_box = fetch_roi(args.dvid_server, args.uuid, roi, format='mask')
        roi_sbm = SparseBlockMask(roi_mask, mask_box*(2**5), 2**5) # ROIs are provided at scale 5
    else:
        roi_sbm = None

    # Ideally, we would choose the max label for the node we're writing to,
    # but the /maxlabel endpoint doesn't work for all nodes
    # instead, we'll use the repo-wide maxlabel from the /info JSON.
    #maxlabel = fetch_maxlabel(args.dvid_server, args.uuid, args.labelmap_instance)
    maxlabel = fetch_instance_info(args.dvid_server, args.uuid, args.labelmap_instance)["Extended"]["MaxRepoLabel"]
    
    for i, path in enumerate(args.sparsevol_files):
        maxlabel += 1
        name = os.path.split(path)[1]
        prefix_logger = PrefixedLogger(logger, f"Vol #{i:02d} {name}: ")
        
        with Timer(f"Pasting {name} as {maxlabel}", logger):
            overwritten_labels = overwrite_sparsevol(*instance_info, maxlabel, path, roi_sbm, invert_roi, args.no_downres, prefix_logger)

        results_path = os.path.splitext(path)[0] + '.json'
        with open(results_path, 'w') as f:
            results = { 'new-label': maxlabel,
                        'overwritten_labels': sorted(overwritten_labels) }
            json.dump(results, f, indent=2, cls=NumpyConvertingEncoder)

    logger.info(f"Done.")


def overwrite_sparsevol(server, uuid, instance, new_label, sparsevol_filepath, roi_sbm, invert_roi, no_downres, logger):
    """
    Given a sparsevol (and an optional ROI mask), download all blocks
    intersecting the sparsevol, and overwrite the supervoxels in each
    block that are covered by the sparsevol (and ROI).
    
    Pseudo-code:
        
        1. Parse the sparsevol RLE data into a complete list of coordinates.
           (Note: For large bodies, this requires a lot of RAM.)
           
           Results in a large array:
           
               [[z,y,x],
                [z,y,x],
                ...
               ]

        2. Append columns to the coordinate array for the block index:
        
               [[z,y,x,bz,by,bx],
                [z,y,x,bz,by,bx],
                ...
               ]

        3. Sort the coordinate array by BLOCK index (bz,by,bx).
        
        4. Divide the coordinates into groups, by block index.
           Now each group of coordinates corresponds to a single
           block that needs to be patched.
           
        5. For each group:
             a. Construct a 3D mask from the coordinates in this group
                AND the intersection of the given ROI mask (if provided).
             b. Download the corresponding labelmap block.
             c. Overwrite the masked voxels with new_label.
             d. Do not post the patched block data immediately.
                Instead, save it to a temporary queue.
             e. If the queue has 400 blocks in it, post them all,
                and then clear the queue.

        6. After the above loop runs, clear the queue one last time.


    Args:
    
        server:
            dvid server, e.g. emdata1:8000

        uuid:
            uuid to read/write.  Must be unlocked.

        instance:
            labelmap instance to read/write blocks
        
        new_label:
            The supervoxel ID to use when overwriting the voxel data.

        sparsevol_filepath:
            path to a binary file containing a sparsevol,
            exactly as downloaded from DVID's /sparsevol endpoint.

        roi_sbm:
            Optional. An ROI mask, loaded into a SparseBlockMask object.
        
        invert_roi:
            If False, only overwrite voxels that overlap the given ROI.
            If True, only overwrite voxels that DON'T overlap the given ROI.
            (If you don't want the ROI to be used at all, set roi_sbm=None.)
        
        no_downres:
            If True, tell DVID not to update the labelmap downscale pyramids in
            response to each block write.  In that case, you're responsible
            for updating the downscale pyramids yourself.
        
        logger:
            A Python logger object for writing misc. status messages.
    
    Returns:
        The set of supervoxels that were at least partially overwritten by the new label.
    """
    sorted_path = sparsevol_filepath + '.sorted_blocktable.npy'
    if os.path.exists(sorted_path):
        logger.info("Loading presorted sparse coordinates")
        sorted_table = np.load(sorted_path)
    else:
        with open(sparsevol_filepath, 'rb') as f:
            with Timer("Parsing sparsevol coordinates", logger):
                coords = parse_rle_response(f.read(), np.int16)
            with Timer("Sorting sparsevol coordiantes", logger):
                table = np.concatenate( (coords // 64, coords), axis=1 )
                sorted_table = lexsort_columns(table)
                del table

            with Timer("Saving sparsevol sorted blocktable"):           
                np.save(sorted_path, sorted_table)

    overwritten_labels = set()
    
    BLOCK_GROUP_SIZE = 400
    next_block_set = []
    for coord_group in groupby_presorted(sorted_table[:,3:], sorted_table[:, :3]):
        block_corner = coord_group[0] // 64 * 64
        block_box = (block_corner, 64 + block_corner)
        block_voxels = fetch_labelarray_voxels(server, uuid, instance, block_box, supervoxels=True)
        
        block_mask = np.zeros_like(block_voxels, dtype=bool)
        mask_coords = coord_group - block_corner
        block_mask[tuple(mask_coords.transpose())] = True

        if roi_sbm is not None:
            roi_mask = roi_sbm.get_fullres_mask(block_box)
            if invert_roi:
                roi_mask = ~roi_mask
            block_mask[:] &= roi_mask

        if not block_mask.any():
            continue

        overwritten_labels |= set(pd.unique(block_voxels[block_mask]))
        block_voxels[block_mask] = new_label
        
        next_block_set.append((block_corner, block_voxels))
        
        # Flush the blocks periodically
        if len(next_block_set) == BLOCK_GROUP_SIZE:
            with Timer(f"Sending block set (N={len(next_block_set)})", logger):
                post_labelarray_blocks(server, uuid, instance, *zip(*next_block_set), downres=not no_downres)
            next_block_set = []

    with Timer(f"Sending last block set (N={len(next_block_set)})", logger):
        post_labelarray_blocks(server, uuid, instance, *zip(*next_block_set), downres=not no_downres)

    return overwritten_labels
        

if __name__ == "__main__":
    main()
