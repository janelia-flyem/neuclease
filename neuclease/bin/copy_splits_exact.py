import sys
import time
import logging
import argparse

import ujson
import requests
import networkx as nx

from neuclease import configure_default_logging
from neuclease.util import tqdm_proxy, write_json_list, parse_timestamp
from neuclease.dvid.rle import combine_sparsevol_rle_responses, extract_rle_size_and_first_coord
from neuclease.dvid.kafka import read_kafka_messages, filter_kafka_msgs_by_timerange
from neuclease.dvid.labelmap import (
    fetch_mutations, fetch_supervoxel_splits_from_kafka,
    split_events_to_dataframe, split_events_to_graph, fetch_label_for_coordinate, post_split_supervoxel, fetch_sparsevol_rles)

logger = logging.getLogger(__name__)


def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mutation-log', action='store_true')
    parser.add_argument('--kafka-log')
    parser.add_argument('--kafka-servers')
    parser.add_argument('--min-timestamp')
    parser.add_argument('--max-timestamp')
    parser.add_argument('--min-mutid', type=int)
    parser.add_argument('--max-mutid', type=int)
    parser.add_argument('--pause-between-splits', type=float, default=0.0)
    parser.add_argument('src_server')
    parser.add_argument('src_uuid')
    parser.add_argument('src_labelmap_instance')
    parser.add_argument('dest_server')
    parser.add_argument('dest_uuid')
    parser.add_argument('dest_labelmap_instance')
    args = parser.parse_args()
    
    src_seg = (args.src_server, args.src_uuid, args.src_labelmap_instance)
    dest_seg = (args.dest_server, args.dest_uuid, args.dest_labelmap_instance)
    
    # Fetch kafka log from src if none was provided from the command line
    if args.mutation_log:
        kafka_msgs = fetch_mutations(*src_seg)
    elif args.kafka_log is not None:
        with open(args.kafka_log, 'r') as f:
            kafka_msgs = ujson.load(f)
    else:
        if args.kafka_servers:
            args.kafka_servers = args.kafka_servers.split(',')
        kafka_msgs = read_kafka_messages(*src_seg, kafka_servers=args.kafka_servers)
        
        # Cache for later
        path = f'kafka-msgs-{args.src_uuid[:4]}-{args.src_labelmap_instance}.json'
        logger.info(f"Writing {path}")
        with open(path, 'w') as f:
            write_json_list(kafka_msgs, f)
    
    copy_splits_exact(*src_seg, *dest_seg, kafka_msgs, args.min_timestamp, args.max_timestamp, args.min_mutid, args.max_mutid, args.pause_between_splits)
    

def copy_splits_exact(src_server, src_uuid, src_instance, dest_server, dest_uuid, dest_instance,
                      kafka_msgs, min_timestamp=None, max_timestamp=None, min_mutid=None, max_mutid=None, pause_between_splits=0.0):
    src_seg = (src_server, src_uuid, src_instance)
    dest_seg = (dest_server, dest_uuid, dest_instance)

    if min_timestamp is not None:
        min_timestamp = parse_timestamp(min_timestamp)
    if max_timestamp is not None:
        max_timestamp = parse_timestamp(max_timestamp)

    kafka_msgs = filter_kafka_msgs_by_timerange(kafka_msgs, min_timestamp, max_timestamp, min_mutid, max_mutid)
    
    split_events = fetch_supervoxel_splits_from_kafka(*src_seg, kafka_msgs=kafka_msgs)
    split_df = split_events_to_dataframe(split_events)
    
    # We assume that supervoxel ID values are toposorted, so sorting by
    # new 'split' ID is sufficient to ensure in-order splits.
    # (supervoxel splits already appear in the log in-order, but arbitrary splits
    # contain a batch of splits with identical mutation IDs, whose split IDs do
    # not necessarily appear in sorted order.)
    split_df.sort_values('split', inplace=True)

    split_forest = split_events_to_graph(split_events)
 
    def get_combined_leaf_sparsevol(sv):
        descendents = nx.descendants(split_forest, sv)
        descendents.add(sv)
        leaves = list(filter(lambda d: split_forest.out_degree(d) == 0, descendents))
        combined_sparsevol = fetch_and_combine_sparsevols(*src_seg, leaves, supervoxels=True)
        return leaves, combined_sparsevol
    
    for row in tqdm_proxy(split_df.itertuples(index=False), total=len(split_df)):
        logger.info(f"Fetching sparsevols for leaves of {row.split}")
        split_leaves, split_payload = get_combined_leaf_sparsevol(row.split)
        size, coord_zyx = extract_rle_size_and_first_coord(split_payload)
        
        logger.info(f"Posting mutation {row.mutid}: {size}-voxel split of {row.old} into {row.split} and {row.remain}, from sparsevols of {split_leaves}")

        # Check the destination -- is it the supervoxel we expected to split?
        dest_sv = fetch_label_for_coordinate(*dest_seg, coord_zyx, supervoxels=True)
        if dest_sv != row.old:
            raise RuntimeError(f"Unexpected supervoxel at the destination: Expected {row.old}, found {dest_sv}")
        
        post_split_supervoxel(*dest_seg, row.old, split_payload, split_id=row.split, remain_id=row.remain)
        time.sleep(pause_between_splits)

    logger.info("DONE.")


def fetch_and_combine_sparsevols(server, uuid, instance, labels, supervoxels=False):
    sparsevols = []
    for label in labels:
        try:
            sparsevol = fetch_sparsevol_rles(server, uuid, instance, label, supervoxels=supervoxels)
            sparsevols.append(sparsevol)
        except requests.RequestException as ex:
            if ex.response.status_code != 404:
                raise
    
    combined_payload = combine_sparsevol_rle_responses(sparsevols)
    return combined_payload


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        sys.argv += ['--kafka-log=kafka-msgs-2884-segmentation.json',
                     '--kafka-servers=kafka.int.janelia.org:9092,kafka2.int.janelia.org:9092,kafka3.int.janelia.org:9092',
                     #'--min-mutid=1000057100',
                     '--min-mutid=1000065000',
                     '--max-timestamp=2018-10-12 14:00:00',
                     'emdata3:9400', '2884', 'segmentation',
                     'emdata3:9400', 'b09e', 'segmentation']

    main()
