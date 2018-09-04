import logging
import argparse
import requests
from tqdm import tqdm

import numpy as np
import pandas as pd

from neuclease import configure_default_logging
from neuclease.util import read_csv_col
from neuclease.dvid import fetch_missing, fetch_sizes

logger = logging.getLogger(__name__)

def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='tsv-status.csv')
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('seg_instance')
    parser.add_argument('tsv_instance')
    parser.add_argument('bodies_csv')
    args = parser.parse_args()
    
    bodies = read_csv_col(args.bodies_csv, 0, np.uint64)
    status_df = check_tarsupervoxels_status(args.server, args.uuid, args.tsv_instance, args.seg_instance, bodies)
    
    logger.info(f"Writing to {args.output}")
    status_df.to_csv(args.output, index=True, header=True)
    logging.info("DONE")


def check_tarsupervoxels_status(server, uuid, tsv_instance, seg_instance, bodies):
    """
    For the given bodies, query the given tarsupervoxels instance and return a
    DataFrame indicating which supervoxels are 'missing' from the instance,
    along with their sizes.
    
    Bodies that no longer exist in the segmentation instance are ignored.
    """
    body_sv_sizes = []
    
    try:
        for body in tqdm(bodies):
            try:
                missing_svs = fetch_missing(server, uuid, tsv_instance, body)
            except requests.RequestException as ex:
                if 'has no supervoxels' in ex.args[0]:
                    continue
                else:
                    raise
                
            if len(missing_svs) == 0:
                continue
    
            sizes = fetch_sizes(server, uuid, seg_instance, missing_svs, supervoxels=True)
            body_sv_sizes += [(body, sv, size) for (sv, size) in zip(missing_svs, sizes)]

    except KeyboardInterrupt:
        logger.warning("Interrupted. Returning results so far.  Interrupt again to kill.")

    df = pd.DataFrame(body_sv_sizes, columns=['body', 'sv', 'voxel_count'], dtype=np.uint64)
    df.set_index('sv', inplace=True)
    return df

###
### Useful follow-up:
### Write empty files for all missing supervoxels below a certain size.
###
# from tqdm import tqdm
# bio = BytesIO()
# tf = tarfile.TarFile('empty-svs.tar', 'w', bio)
# for sv in tqdm(missing_svs.query('voxel_count <= 100')['sv']):
#     tf.addfile(tarfile.TarInfo(f'{sv}.drc'), BytesIO())

if __name__ == "__main__":
    main()
