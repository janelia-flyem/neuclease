"""
Checks a DVID tarsupervoxels instance (tsv_instance) for
missing supervoxels, given a list of bodies to check.
Outputs the list of missing supervoxels as a CSV file.

If you are checking a lot of bodies, try the --use-mapping option,
which fetches the entire in-memory mapping for the segmentation instance
and then uses the /exists endpoint (instead of the /missing) endpoint.
Fetching the entire mapping incurs an initial overhead of ~1 minute or so,
but the /exists endpoint is MUCH faster than /missing.

Examples:

    # Check a couple bodies
    check_tarsupervoxels_status emdata3:8900 0716 segmentation segmentation_sv_meshes bodies.csv

    # Check a lot of bodies
    check_tarsupervoxels_status --use-mapping emdata3:8900 0716 segmentation segmentation_sv_meshes bodies.csv
    
"""
import logging
import argparse
import requests

import numpy as np
import pandas as pd

from neuclease import configure_default_logging
from neuclease.util import read_csv_col, tqdm_proxy
from neuclease.dvid import fetch_missing, fetch_exists
from neuclease.dvid.labelmap._labelmap import fetch_complete_mappings

logger = logging.getLogger(__name__)

def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--use-mapping', action='store_true',
                        help='Use in-memory map + /exists instead of /missing, as described in the general help text above.')
    parser.add_argument('--output', '-o', default='missing-from-tsv.csv', help='Where to write the output CSV (default: missing-from-tsv.csv)')
    parser.add_argument('server', help='dvid server, e.g. emdata3:8900')
    parser.add_argument('uuid', help='dvid node')
    parser.add_argument('seg_instance', help='name of a labelmap instance, e.g. segmentation')
    parser.add_argument('tsv_instance', help='name of a tarsupervoxels instance, e.g. segmentation_sv_meshes')
    parser.add_argument('bodies_csv', help='CSV file whose first column is a list of body IDs. Other columns ignored.')
    args = parser.parse_args()
    
    bodies = read_csv_col(args.bodies_csv, 0, np.uint64)
    if args.use_mapping:
        missing_entries = check_tarsupervoxels_status_via_exists(args.server, args.uuid, args.tsv_instance, args.seg_instance, bodies)
    else:
        missing_entries = check_tarsupervoxels_status_via_missing(args.server, args.uuid, args.tsv_instance, args.seg_instance, bodies)
    
    logger.info(f"Writing to {args.output}")
    missing_entries.to_csv(args.output, index=True, header=True)
    logging.info("DONE")


def check_tarsupervoxels_status_via_missing(server, uuid, tsv_instance, seg_instance, bodies):
    """
    For the given bodies, query the given tarsupervoxels instance and return a
    DataFrame indicating which supervoxels are 'missing' from the instance.
    
    Bodies that no longer exist in the segmentation instance are ignored.

    This function uses the /missing endpoint, which incurs a disk read in DVID 
    for the LabelIndex of each body.
    """
    sv_body = []
    
    try:
        for body in tqdm_proxy(bodies):
            try:
                missing_svs = fetch_missing(server, uuid, tsv_instance, body)
            except requests.RequestException as ex:
                if 'has no supervoxels' in ex.args[0]:
                    continue
                else:
                    raise
            
            sv_body += [(sv, body) for sv in missing_svs]

    except KeyboardInterrupt:
        logger.warning("Interrupted. Returning results so far.  Interrupt again to kill.")

    df = pd.DataFrame(sv_body, columns=['sv', 'body'], dtype=np.uint64)
    df.set_index('sv', inplace=True)
    return df['body']


def check_tarsupervoxels_status_via_exists(server, uuid, tsv_instance, seg_instance, bodies):
    """
    For the given bodies, query the given tarsupervoxels instance and return a
    DataFrame indicating which supervoxels are 'missing' from the instance.
    
    Bodies that no longer exist in the segmentation instance are ignored.

    This function downloads the complete mapping in advance and uses it to determine
    which supervoxels belong to each body.  Then uses the /exists endpoint to
    query for missing supervoxels, rather than /missing, which incurs a disk
    read in DVID.
    """
    sv_body = []
    
    try:
        mapping = fetch_complete_mappings(server, uuid, seg_instance)
        
        # Filter out bodies we don't care about,
        # and append unmapped (singleton/identity) bodies
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
            sv_body += list(zip(missing_svs, missing_bodies))
        
    except KeyboardInterrupt:
        logger.warning("Interrupted. Returning results so far.  Interrupt again to kill.")

    df = pd.DataFrame(sv_body, columns=['sv', 'body'], dtype=np.uint64)
    df.set_index('sv', inplace=True)
    return df['body']


###
### Useful follow-up:
### Write empty files for all missing supervoxels below a certain size.
###
# import tarfile
# from io import BytesIO
# from tqdm import tqdm
# 
# bio = BytesIO()
# tf = tarfile.TarFile('empty-svs.tar', 'w', bio)
# for sv in tqdm(missing_svs):
#     tf.addfile(tarfile.TarInfo(f'{sv}.drc'), BytesIO())
# 
# post_load(*master, 'segmentation_sv_meshes', bio.getvalue())

if __name__ == "__main__":
    main()
