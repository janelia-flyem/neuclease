"""
Applies saved split RLEs to a new DVID node.

Example usage:

    python apply_saved_splits.py emdata2:8700 d8d22b segmentation split_8c245.log

Note: It is assumed that the saved split RLEs exist
      in the current directory, with names like 5812984611.rle.
"""
import os
import sys
import logging
import argparse

import ujson
from tqdm import tqdm

import numpy as np

from neuclease.dvid import fetch_label_for_coordinate, split_supervoxel
from neuclease.util import Timer

logger = logging.getLogger(__name__)


def main():
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('kafka_log')
    args = parser.parse_args()

    apply_splits_from_saved_rles(args.kafka_log, args.server, args.uuid, args.labelmap_instance, args.verbose)
    logger.info("Done.")
    

def apply_splits_from_saved_rles(kafka_log, server, uuid, labelmap_instance, verbose=False):
    new_ids = parse_new_ids(kafka_log)
    with Timer(f"Applying {len(new_ids)} split RLEs"):
        for new_id in tqdm(new_ids):
            rle_path = f'{new_id}.rle'
            if not os.path.exists(rle_path):
                raise RuntimeError(f"Can't find .rle file: {rle_path}")
            
            if verbose:
                with tqdm.external_write_mode():
                    logger.info(f"Loading {rle_path}")

            with open(rle_path, 'rb') as f:
                rle_payload = f.read()
        
            if len(rle_payload) == 0:
                logger.error(f"Error: {rle_path} has no content!")
                continue

            apply_split(rle_payload, server, uuid, labelmap_instance, verbose)

    
def parse_new_ids(kafka_log_path):
    """
    Each line of the exported kafka log looks like this:
    {"Action":"split","MutationID":937,"NewLabel":5812984527,"Split":"WT-yxGFLh0NLFVnUX46Xcg==","Target":1291237038,"UUID":"8c245f16fe7a46c1808c55b4bc0230b1"}
    """
    new_ids = []
    with open(kafka_log_path, 'r') as f:
        for line in f:
            entry = ujson.loads(line)
            new_ids.append(entry["NewLabel"])
    return new_ids
        

def apply_split(rle_payload_bytes, server, uuid, instance, verbose=False):
    rles = np.frombuffer(rle_payload_bytes, dtype=np.uint32)[3:]
    rles = rles.reshape(-1, 4)
    first_coord_xyz = rles[0, :3]
    first_coord_zyx = first_coord_xyz[::-1]
    supervoxel = fetch_label_for_coordinate(server, uuid, instance, first_coord_zyx, supervoxels=True)
    voxel_count = rles[:, 3].sum()
    
    if verbose:
        with tqdm.external_write_mode():
            logger.info(f"Applying split to {supervoxel} ({voxel_count} voxels)")

    split_supervoxel(server, uuid, instance, supervoxel, rle_payload_bytes)
    
    
if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        os.chdir('/tmp/neu_tu_split_rle')
        sys.argv += ['-v', 'emdata2:8700', 'd8d22ba8a1c64c8c9b422226adba17cc', 'segmentation', 'split_8c245.log']
    main()
