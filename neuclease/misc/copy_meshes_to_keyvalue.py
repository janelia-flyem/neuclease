import logging
import argparse

from tqdm import tqdm

import numpy as np
import pandas as pd

from neuclease import configure_default_logging
from neuclease.util.csv import read_csv_col
from neuclease.dvid import post_key, fetch_tarfile

logger = logging.getLogger(__name__)

keyEncodeLevel0 = np.uint64(10000000000000)

def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('body_list_csv')
    parser.add_argument('src_server')
    parser.add_argument('src_uuid')
    parser.add_argument('src_tarsupervoxels_instance')
    parser.add_argument('dest_server')
    parser.add_argument('dest_uuid')
    parser.add_argument('dest_keyvalue_instance')
    
    args = parser.parse_args()

    body_list = read_csv_col(args.body_list_csv)

    src_info = (args.src_server, args.src_uuid, args.src_tarsupervoxels_instance)
    dest_info = (args.dest_server, args.dest_uuid, args.dest_keyvalue_instance )

    logger.info(f"Copying {len(body_list)}")    
    failed_bodies = copy_meshes_to_keyvalue(src_info, dest_info, body_list)
    
    if failed_bodies:
        logger.warning(f"Writing {len(failed_bodies)} to failed-bodies.txt")
        pd.Series(failed_bodies).to_csv('failed-bodies.txt', index=False)
        
    logger.info("DONE.")

def copy_meshes_to_keyvalue(src_info, dest_info, body_list):
    failed_bodies = []
    for body_id in tqdm(body_list):
        try:
            tar_bytes = fetch_tarfile(*src_info, body_id)
        except:
            logger.error(f"Failed to copy {body_id}")
            failed_bodies.append(body_id)
            continue

        encoded_body = np.uint64(keyEncodeLevel0 + body_id)
        assert isinstance(encoded_body, np.uint64)
        post_key(*dest_info, f'{encoded_body}.tar', tar_bytes)
    
    return failed_bodies

if __name__ == "__main__":
    main()
