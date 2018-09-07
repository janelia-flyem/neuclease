import logging
import argparse
import requests

import numpy as np
import pandas as pd

from neuclease import configure_default_logging
from neuclease.util import read_csv_col, tqdm_proxy
from neuclease.dvid import fetch_missing, fetch_exists, fetch_sizes
from neuclease.dvid.labelmap._labelmap import fetch_complete_mappings

logger = logging.getLogger(__name__)

def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-mapping', action='store_true')
    parser.add_argument('--output', '-o', default='tsv-status.csv')
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('seg_instance')
    parser.add_argument('tsv_instance')
    parser.add_argument('bodies_csv')
    args = parser.parse_args()
    
    bodies = read_csv_col(args.bodies_csv, 0, np.uint64)
    if args.use_mapping:
        status_df = check_tarsupervoxels_status_via_exists(args.server, args.uuid, args.tsv_instance, args.seg_instance, bodies)
    else:
        status_df = check_tarsupervoxels_status_via_missing(args.server, args.uuid, args.tsv_instance, args.seg_instance, bodies)
    
    logger.info(f"Writing to {args.output}")
    status_df.to_csv(args.output, index=True, header=True)
    logging.info("DONE")


def check_tarsupervoxels_status_via_missing(server, uuid, tsv_instance, seg_instance, bodies):
    """
    For the given bodies, query the given tarsupervoxels instance and return a
    DataFrame indicating which supervoxels are 'missing' from the instance,
    along with their sizes.
    
    Bodies that no longer exist in the segmentation instance are ignored.

    This function uses the /missing endpoint, which incurs a disk read in DVID 
    for the LabelIndex of each body.
    """
    body_sv_sizes = []
    
    try:
        for body in tqdm_proxy(bodies):
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


def check_tarsupervoxels_status_via_exists(server, uuid, tsv_instance, seg_instance, bodies):
    """
    For the given bodies, query the given tarsupervoxels instance and return a
    DataFrame indicating which supervoxels are 'missing' from the instance,
    along with their sizes.
    
    Bodies that no longer exist in the segmentation instance are ignored.

    This function downloads the complete mapping in advance and uses it to determine
    which supervoxels belong to each body.  Then uses the /exists endpoint to
    query for missing supervoxels, rather than /missing, which incurs a disk
    read in DVID.
    """
    body_sv_sizes = []
    
    try:
        mapping = fetch_complete_mappings(server, uuid, seg_instance)
        
        _bodies = set(bodies)
        mapping = pd.DataFrame(mapping).query('body in @_bodies')['body'].copy()
        unmapped_bodies = _bodies - set(mapping)
        unmapped_bodies = np.fromiter(unmapped_bodies, np.uint64)

        singleton_mapping = pd.Series(index=unmapped_bodies, data=unmapped_bodies)
        mapping = pd.concat((mapping, singleton_mapping))

        BATCH_SIZE = 1000
        for start in tqdm_proxy(range(0, len(mapping), BATCH_SIZE)):
            svs = mapping.index[start:start+BATCH_SIZE]
            statuses = fetch_exists(server, uuid, tsv_instance, svs)
            missing_svs = statuses[~statuses].index

            if len(missing_svs) == 0:
                continue

            missing_bodies = mapping.loc[missing_svs]
            sizes = fetch_sizes(server, uuid, seg_instance, missing_svs, supervoxels=True)
            body_sv_sizes += list(zip(missing_bodies, missing_svs, sizes))
        
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
