"""
Checks a DVID tarsupervoxels instance (tsv_instance) for
missing supervoxels, given a list of bodies to check.
Outputs the list of missing supervoxels as a CSV file.

Alternatively, instead of listing your input bodies via CSV,
this script can check the kafka log for any bodies affected
after a given starting timestamp.  See usage examples below.

If you are checking a lot of bodies, try the --use-mapping option,
which fetches the entire in-memory mapping for the segmentation instance
and then uses the /exists endpoint (instead of the /missing endpoint).
Although fetching the entire mapping incurs an initial overhead of
~1 minute or so, it pays of for a large list of bodies because
the /exists endpoint is MUCH faster than /missing.

Examples:

    # Check a couple thousand bodies (from your own list)
    check_tarsupervoxels_status emdata3:8900 54f7 segmentation segmentation_sv_meshes bodies.csv

    # Check 1,000,000 bodies (from your own list)
    check_tarsupervoxels_status --use-mapping emdata3:8900 54f7 segmentation segmentation_sv_meshes bodies.csv

    # Check bodies that have changed since Nov 22 (Don't provide bodies.csv)
    check_tarsupervoxels_status --kafka-timestamp="2018-11-22 00:00:00" emdata3:8900 54f7 segmentation segmentation_sv_meshes
"""
import sys
import logging
import argparse
import requests

import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from neuclease import configure_default_logging
from neuclease.util import read_csv_header, read_csv_col, tqdm_proxy, parse_timestamp
from neuclease.dvid import fetch_missing, fetch_exists, fetch_mapping
from neuclease.dvid.kafka import read_kafka_messages, filter_kafka_msgs_by_timerange
from neuclease.dvid.labelmap import fetch_complete_mappings, compute_affected_bodies

logger = logging.getLogger(__name__)

def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--use-mapping', action='store_true',
                        help='Use in-memory map + /exists instead of /missing, as described in the general help text above.')
    parser.add_argument('--output', '-o', default='missing-from-tsv.csv', help='Where to write the output CSV (default: missing-from-tsv.csv)')
    
    parser.add_argument('--kafka-timestamp', '-k', type=str,
                        help='Alternative to providing your own bodies list.\n'
                             'Use the kafka log automatically determine the list of bodies that have changed after the given timestamp.\n'
                             'Examples: -k="2018-11-22" -k="2018-11-22 17:34:00"')
    
    parser.add_argument('server', help='dvid server, e.g. emdata3:8900')
    parser.add_argument('uuid', help='dvid node')
    parser.add_argument('seg_instance', help='name of a labelmap instance, e.g. segmentation')
    parser.add_argument('tsv_instance', help='name of a tarsupervoxels instance, e.g. segmentation_sv_meshes')
    parser.add_argument('bodies_csv', nargs='?', help='CSV containin a column named "body", which will be read.\n'
                                                      'If no "body" column exists, the first column is used, regardless of the name.')
    args = parser.parse_args()
    
    if not (bool(args.kafka_timestamp) ^ bool(args.bodies_csv)):
        print("You must provide either --kafka-timestamp or a bodies list (not both)", file=sys.stderr)
        sys.exit(1)

    kafka_msgs = None
    if args.bodies_csv:
        if 'body' in read_csv_header(args.bodies_csv):
            bodies = pd.read_csv(args.bodies_csv)['body'].drop_duplicates()
        else:
            # Just read the first column, no matter what it's named
            bodies = read_csv_col(args.bodies_csv, 0, np.uint64).drop_duplicates()
    elif args.kafka_timestamp:
        # Validate timestamp format before fetching kafka log, which takes a while.
        parse_timestamp(args.kafka_timestamp)
        
        kafka_msgs = read_kafka_messages(args.server, args.uuid, args.seg_instance)
        filtered_kafka_msgs = filter_kafka_msgs_by_timerange(kafka_msgs, min_timestamp=args.kafka_timestamp)
        
        new_bodies, changed_bodies, _removed_bodies, new_supervoxels = compute_affected_bodies(filtered_kafka_msgs)
        sv_split_bodies = set(fetch_mapping(args.server, args.uuid, args.seg_instance, new_supervoxels)) - set([0])
        
        bodies = list(new_bodies | changed_bodies | sv_split_bodies)
        bodies = np.asarray(bodies, np.uint64)
    else: 
        raise AssertionError("Shouldn't get here.")
    
    if args.use_mapping:
        missing_entries = check_tarsupervoxels_status_via_exists(args.server, args.uuid, args.tsv_instance, args.seg_instance, bodies, kafka_msgs=kafka_msgs)
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


def check_tarsupervoxels_status_via_exists(server, uuid, tsv_instance, seg_instance, bodies, mapping=None, kafka_msgs=None):
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
        if mapping is None:
            mapping = fetch_complete_mappings(server, uuid, seg_instance, kafka_msgs=kafka_msgs)
        
        # Filter out bodies we don't care about,
        # and append unmapped (singleton/identity) bodies
        _bodies = set(bodies)
        mapping = pd.DataFrame(mapping).query('body in @_bodies')['body'].copy()
        unmapped_bodies = _bodies - set(mapping)
        unmapped_bodies = np.fromiter(unmapped_bodies, np.uint64)
        singleton_mapping = pd.Series(index=unmapped_bodies, data=unmapped_bodies)
        mapping = pd.concat((mapping, singleton_mapping))
        
        # Faster than mapping.loc[], apparently
        mapper = LabelMapper(mapping.index.values.astype(np.uint64),
                             mapping.values.astype(np.uint64))

        BATCH_SIZE = 10_000
        for start in tqdm_proxy(range(0, len(mapping), BATCH_SIZE)):
            svs = mapping.index[start:start+BATCH_SIZE]
            statuses = fetch_exists(server, uuid, tsv_instance, svs)
            missing_svs = statuses[~statuses].index.values.astype(np.uint64)

            if len(missing_svs) == 0:
                continue

            missing_bodies = mapper.apply(missing_svs, True)
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
### from neuclease.bin.check_tarsupervoxels_status import post_empty_meshes
###
def post_empty_meshes(server, uuid, instance='segmentation_sv_meshes', svs=[], permit_large=False):
    """
    Given a list of supervoxel ids (presumably for SMALL supervoxels),
    post an empty .drc file to the tarsupervoxels instance for each one.
    
    (By convention, we do not generally store meshes for very tiny meshes.
    Instead, we store empty mesh files (i.e. 0 bytes) in their place, and
    our proofreading tools understand this convention.)
    
    Since this function is generally supposed to be used with only small supervoxels,
    it will refuse to write empty files for any supervoxels larger than 100 voxels,
    unless you pass permit_large=True.
    """
    import tarfile
    from io import BytesIO
    from tqdm import tqdm
    from neuclease.dvid import fetch_sizes, post_load, fetch_instance_info

    # Determine segmentation instance
    info = fetch_instance_info(server, uuid, instance)
    segmentation_instance = info["Base"]["Syncs"][0]

    sizes = fetch_sizes(server, uuid, segmentation_instance, svs, supervoxels=True)
    if sizes.any() > 1000:
        msg = "Some of those supervoxels are large ({sizes.max()} voxels)."
        if not permit_large:
            logger.warning(msg)
        else:
            msg = f"Error: {msg} Pass permit_large=True if you really mean it."
            raise RuntimeError(msg)

    bio = BytesIO()
    tf = tarfile.TarFile('empty-svs.tar', 'w', bio)
    for sv in tqdm(svs):
        tf.addfile(tarfile.TarInfo(f'{sv}.drc'), BytesIO())
     
    post_load(server, uuid, instance, bio.getvalue())
    return sizes

if __name__ == "__main__":
    main()
