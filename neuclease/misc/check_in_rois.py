"""
If called as a script, this file takes a table of labeled synapse points and a list of ROIs,
and checks each synapse to see if it falls within ROI mask(s).
A final count is displayed of the synapses and bodies which intersect the ROI.
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

from neuclease import configure_default_logging
from neuclease.util import box_to_slicing
from neuclease.dvid import fetch_roi, load_synapses

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('synapse_table')
    parser.add_argument('rois', nargs='+')
    args = parser.parse_args()

    configure_default_logging()

    syn_ext = os.path.splitext(args.synapse_table)[1]
    assert syn_ext in ('.npy', '.csv')

    logging.info(f"Reading {args.synapse_table}")
    synapse_df = load_synapses(args.synapse_table)
    check_in_rois(args.server, args.uuid, synapse_df, args.rois)
    
    logging.info("DONE")

def check_in_rois(server, uuid, synapse_df, rois):
    """
    Adds a column 'in_roi' to the given points dataframe
    indicating whether or not each point is covered by a ROI in the given list of ROIs.
    Adds the column (IN-PLACE).
    
    Args:
        server:
            dvid server
        uuid:
            Where to pull the rois from
        synapse_df:
            A DataFrame with at least columns ['x', 'y', 'z', 'body']
        rois:
            list of strings (roi instance names)

    Returns:
        None (Operates in-place)
    """
    num_bodies = len(pd.unique(synapse_df['body']))
    logging.info(f"Checking in {len(rois)} ROIs for {len(synapse_df)} synapses from {num_bodies} bodies")
    
    masks_and_boxes = []
    for roi in rois:
        logger.info(f"Fetching ROI '{roi}'")
        mask, box = fetch_roi(server, uuid, roi, format='mask')
        masks_and_boxes.append((mask, box))
    
    _masks, boxes = zip(*masks_and_boxes)
    boxes = np.array(boxes)
    
    # box/shape is in scale-5 coordinates
    logger.info("Combining ROIs into a single mask")
    combined_box = (boxes[:,0,:].min(axis=0), boxes[:,1,:].max(axis=0))
    combined_shape = (combined_box[1] - combined_box[0])
    combined_mask = np.zeros(combined_shape, dtype=bool)
    
    for mask, box in masks_and_boxes:
        offset_box = box - combined_box[0]
        combined_mask[box_to_slicing(*offset_box)] |= mask

    # Rescale points to scale 5 (ROIs are given at scale 5)
    logger.info("Scaling points")
    downsampled_coords_zyx = synapse_df[['z', 'y', 'x']] // (2**5)

    # Drop everything outside the combined_box
    logger.info("Excluding OOB points")
    min_z, min_y, min_x = combined_box[0]
    max_z, max_y, max_x = combined_box[1]
    q = 'z >= @min_z and y >= @min_y and x >= @min_x and z < @max_z and y < @max_y and x < @max_x'
    downsampled_coords_zyx.query(q, inplace=True)

    logging.info("Extracting mask values")
    synapse_df['in_roi'] = False
    downsampled_coords_zyx -= combined_box[0]
    synapse_df.loc[downsampled_coords_zyx.index, 'in_roi'] = combined_mask[tuple(downsampled_coords_zyx.values.transpose())]

    roi_synapses = synapse_df['in_roi'].sum()
    roi_bodies = len(pd.unique(synapse_df['body'][synapse_df['in_roi']]))

    logging.info(f"Found {roi_synapses} synapses in the ROI from {roi_bodies} bodies")

if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        sys.argv += "emdata2:7900 9e0d synapses-9e0d-unlocked-SMALL-focused-only.npy PB-lm FB-lm EB-lm NO-lm LAL-lm".split()
    main()

