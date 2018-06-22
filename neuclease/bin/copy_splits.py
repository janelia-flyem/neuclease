import sys
import logging
import argparse
from collections import namedtuple

from tqdm import tqdm

import requests
import numpy as np
import pandas as pd

from neuclease.dvid import (fetch_label_for_coordinate, fetch_sparsevol_rles, split_supervoxel, fetch_mapping,
                            generate_sample_coordinate, fetch_body_size, fetch_body_sizes, extract_rle_size_and_first_coord, read_kafka_messages)
from neuclease.util import read_csv_col
from neuclease.dvid.dvid import perform_cleave

logger = logging.getLogger(__name__)

def main():
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-output-log', '-o', default='split-copy-results-log.csv')
    parser.add_argument('--src-supervoxels-csv', required=False)
    parser.add_argument('--src-supervoxels-from-kafka', action='store_true')
    parser.add_argument('src_server')
    parser.add_argument('src_uuid')
    parser.add_argument('src_labelmap_instance')
    parser.add_argument('dest_server')
    parser.add_argument('dest_uuid')
    parser.add_argument('dest_labelmap_instance')
    args = parser.parse_args()

    src_info = InstanceInfo(args.src_server, args.src_uuid, args.src_labelmap_instance)
    dest_info = InstanceInfo(args.dest_server, args.dest_uuid, args.dest_labelmap_instance)

    if not ((args.src_supervoxels_csv is not None) ^ args.src_supervoxels_from_kafka):
        print("You must select either CSV or Kafka (not both)", file=sys.stderr)
        sys.exit(1)

    if args.src_supervoxels_csv:
        src_supervoxels = read_csv_col(args.src_supervoxels_csv, col=0, dtype=np.uint64)
    else:
        src_supervoxels = read_src_supervoxels_from_kafka(src_info)

    if len(src_supervoxels) == 0:
        logger.error("Error: No source supervoxels provided!")
        sys.exit(1)

    copy_results = copy_splits(src_supervoxels, src_info, dest_info)
    df = pd.DataFrame(np.array(copy_results, dtype=np.uint64), columns=['src_sv', 'overwritten_sv', 'split_sv', 'remain_sv'])
    df.to_csv(args.results_output_log, index=False, header=True)
    print(f"Saved results log to {args.results_output_log}")
    
    logger.info("Done.")


def read_src_supervoxels_from_kafka(src_instance_info, types=['body', 'supervoxel']):
    """
    Check Kafka to determine which supervoxels have been split in the source node,
    and return the list of new IDs resulting from splits, including both 'split' and 'remain' IDs.
    
    Then filter out the IDs that don't exist any more (if they've been split again).
    
    TODO: Replace this function with one that uses the /supervoxel-splits dvid endpoint.
    
    Args:
        src_instance_info:
            server, uuid, instance

        types:
            Supervoxels can become split via two different types operation: body split and supervoxel split.
            This function can check for 
    
    Returns:
        Array of supervoxel IDs, all of which were created in the source node and still exist.
    """
    messages = read_kafka_messages(src_instance_info)
    body_split_msgs = filter(lambda msg: msg["Action"] == 'split', messages)
    sv_split_msgs = filter(lambda msg: msg["Action"] == 'split-supervoxel', messages)

    split_sv_ids = []
    
    if 'body'in types:
        for msg in body_split_msgs:
            if msg["SVSplits"] is None:
                print(f"WTF for body {msg['Target']}")
                continue
            for _old_sv, split_info in msg["SVSplits"].items():
                split_sv_ids.append( split_info["Split"] )
                split_sv_ids.append( split_info["Remain"] )

    if 'supervoxel' in types:
        for msg in sv_split_msgs:
            split_sv_ids.append( msg["SplitSupervoxel"] )
            split_sv_ids.append( msg["RemainSupervoxel"] )
    
    # Filter out supervoxels that don't exist any more.
    sizes = fetch_body_sizes(src_instance_info, split_sv_ids, supervoxels=True)
    valid_svs = np.array(sizes).astype(bool)
    split_sv_ids = np.array(split_sv_ids, dtype=np.uint64)
    split_sv_ids = split_sv_ids[valid_svs]
    
    return np.asarray(split_sv_ids, dtype=np.uint64)


InstanceInfo = namedtuple('InstanceInfo', 'server uuid instance')
SplitCopyInfo = namedtuple("SplitCopyInfo", "src_sv overwritten_sv split_sv remain_sv")
def copy_splits(src_supervoxels, src_info, dest_info):
    copy_infos = []
    for i, src_sv in enumerate(tqdm(src_supervoxels)):
        try:
            src_sv_size = fetch_body_size(src_info, src_sv, supervoxels=True)
            first_coord_zyx = generate_sample_coordinate(src_info, src_sv, supervoxels=True)
            dest_sv = fetch_label_for_coordinate(dest_info, first_coord_zyx, supervoxels=True)
            dest_sv_size = fetch_body_size(dest_info, dest_sv, supervoxels=True)
            
            if src_sv_size == dest_sv_size:
                with tqdm.external_write_mode():
                    logger.info(f"SV {src_sv} (#{i}) appears to be already copied at the destination, where it has ID {dest_sv}. Skipping.")
                    split_info = SplitCopyInfo(src_sv, dest_sv, dest_sv, 0)
            elif src_sv_size > dest_sv_size:
                with tqdm.external_write_mode():
                    logger.error(f"Refusing to copy SV {src_sv} (#{i}): It is too big for the destination supervoxel (SV {dest_sv})!")
                    split_info = SplitCopyInfo(src_sv, dest_sv, 0, 0)
            else:
                # Fetch RLEs and apply to destination
                rle_payload = fetch_sparsevol_rles(src_info, src_sv, supervoxels=True)
                rle_size, first_coord_zyx = extract_rle_size_and_first_coord(rle_payload)
                assert rle_size == src_sv_size                
                split_sv, remain_sv = split_supervoxel(dest_info, dest_sv, rle_payload)
                split_info = SplitCopyInfo(src_sv, dest_sv, split_sv, remain_sv)
        except Exception as ex:
            with tqdm.external_write_mode():
                logger.error(f"Error copying SV {src_sv} (#{i}): {ex}")
            split_info = SplitCopyInfo(src_sv, 0, 0, 0)

        copy_infos.append(split_info)

    return copy_infos


def copy_to_scratch(src_info, dest_info, coords_xyz):
    """
    Determine a list of supervoxels from the given src instance by sampling it at the given coordinates.
    Then copy the supervoxels from source to destination.
    
    If any coordinates fell on duplicate supervoxels, they are dropped from the output.
    
    Returns:
        (coords_xyz, copy_infos), where copy_infos is a list of SplitCopyInfo tuples
    """
    logger.info("Sampling coordinates")
    src_svs = []
    for coord_xyz in tqdm(coords_xyz):
        coord_zyx = coord_xyz[::-1]
        src_sv = fetch_label_for_coordinate(src_info, coord_zyx, supervoxels=True)
        src_svs.append(src_sv)

    df = pd.DataFrame(coords_xyz, columns=list('xyz'))
    df['sv'] = src_svs
    df.drop_duplicates(['sv'], inplace=True)

    logger.info("Copying RLEs")
    copy_infos = copy_splits(df['sv'].values, src_info, dest_info)

    return df[['x', 'y', 'z']].values, copy_infos
    

def cleave_supervoxels_as_isolated_bodies(instance_info, sv_ids):
    """
    Separate the given supervoxels from their enclosing bodies by
    cleaving them out as their own independent, single-supervoxel bodies.
    """
    logger.info("Fetching mapping for each SV")
    body_ids = fetch_mapping(instance_info, sv_ids)

    logger.info("Performing cleaves")
    cleaved_ids = []
    for sv_id, body_id in tqdm(list(zip(sv_ids, body_ids))):
        try:
            cleaved_body = perform_cleave(instance_info, body_id, [sv_id])
        except requests.RequestException as ex:
            if 'cannot cleave all supervoxels from the label' in ex.response.content.decode():
                # Body has only one supervoxel to begin with
                cleaved_body = body_id
            else:
                sys.stderr.write(ex.response.content.decode())
                raise

        cleaved_ids.append( cleaved_body )
    
    return list(zip(sv_ids, body_ids, cleaved_ids))


def thursday_madness(coords_csv):
    """
    I used this function to copy splits from the production server
    to a scratch server in the final stages of our phase-0.2 work.
    
    Samples a set of coordinates from the production server to determine
    a set of supervoxels to copy over to the scratch server, and prepares
    them for split assignments by cleaving them away from their parent bodies.
    """
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    coords_xyz = pd.read_csv(coords_csv, header=None, names=['x', 'y', 'z'], dtype=np.int32).values
    production_info = ('emdata3:8900', '25dc', 'segmentation')
    scratch_info = ('emdata2:8700', '029b834695bb457eb0cf43b848ae3461', 'segmentation')
    
    coords_xyz, copy_infos = copy_to_scratch(production_info, scratch_info, coords_xyz)
    scratch_split_svs = np.array([info.split_sv for info in copy_infos])
    cleave_info = cleave_supervoxels_as_isolated_bodies( scratch_info, scratch_split_svs )

    logger.info("Preparing output CSV")    
    table = []
    for coord_xyz, copy_info, cleave_info in zip(coords_xyz, copy_infos, cleave_info):
        x, y, z = coord_xyz
        production_sv = copy_info.src_sv
        scratch_sv = cleave_info[0]
        scratch_body = cleave_info[1]
        scratch_cleaved_body = cleave_info[2]
        table.append( (x,y,z,production_sv,scratch_sv,scratch_body,scratch_cleaved_body) )
    
    df = pd.DataFrame(table, columns=['x','y','z','production_sv','scratch_sv','scratch_body','scratch_cleaved_body'])
    df.to_csv('/tmp/thursday-02-cleanup-split-table.csv', index=False)
    logger.info("DONE!")

# def read_ting_split_info(ting_split_result_log, ting_uuid='194db260e1ee4edcbda0b592bf40d926'):
#     """
#     Read Ting's split results log, which is in this form:
#     
#         >>>> 1 task__http-++emdata2.int.janelia.org-8700+api+node+4c4f+segmentation+sparsevol+1001134218
#         >>>> 2 task__http-++emdata2.int.janelia.org-8700+api+node+4c4f+segmentation+sparsevol+1002735917
#         >>>> 3 task__http-++emdata2.int.janelia.org-8700+api+node+4c4f+segmentation+sparsevol+1004548999
#         ...
#     
#     and query his special key-value instance to learn the new supervoxels he created via the splits.
#     
#     Note: This function isn't used above.  It's mostly here so I can remember
#           how I generated the src_supervoxels CSV in case I need to do it again.
# 
#     Returns:
#         dict of body ids -> new ids
#         The new IDs may be supervoxel IDs or may be body IDs, depending on the type of split that was initiated.
#     """
#     
#     with open(ting_split_result_log, 'r') as f:
#         session = default_dvid_session('focused-finish02')
#         saved_ids = {}
#         for line in tqdm(f):
#             if not line.startswith('>>>>'):
#                 continue
#             task = line.split()[2]
# 
#             url = f'http://zhaot-ws1:9000/api/node/{ting_uuid}/result_split/key/{task}'
#             r = session.get(url)
#             r.raise_for_status()
# 
#             # Response looks like this:
#             # {“committed”: [5812994702, 5812994704, 5812994705], “timestamp”: 1528521108}
# 
#             body_id = int(task.split('+')[-1])
#             saved_ids[body_id] = r.json()["committed"]
#     return saved_ids

# def determine_supervoxels_from_body_splits(ting_split_result_log, ting_uuid='194db260e1ee4edcbda0b592bf40d926'):
#     # Assume that all lines in the log have the same DVID server/uuid/instance
#     with open(ting_split_result_log, 'r') as f:
#         lines = filter(lambda s: s.startswith('>>>>'), f.readlines())
#         first_line = next(lines)
#         _, info_str = first_line.split('++')[1]
#         server, _api, _node, uuid, instance, _sparsevol, _body_id = info_str.split('+')
#         server = server.replace('-', ':')
#         
#     saved_ids = read_ting_split_info(ting_split_result_log, ting_uuid='194db260e1ee4edcbda0b592bf40d926')
#     for orig_body, new_bodies in saved_ids.items():
#         all_bodies = [orig_body] + new_bodies
#         all_split_svs = []
#         for body in all_bodies:
#             all_split_svs += fetch_supervoxels_for_body( (server, uuid, instance), orig_body )


# def affected_bodies(src_supervoxels, src_info, dest_info, bodies=None):
#     """
#     Given a set of source supervoxels on one server, figure out which bodies they touch on a destination server.
# 
#     Note: Assumes that each supervoxel does not span across multiple bodies.
# 
#     Note: This function isn't used above.  It's here as a useful debugging function.
#     """
#     bodies = bodies or set()
#     for i, src_sv in enumerate(tqdm(src_supervoxels)):
#         try:
#             rle_payload = fetch_sparsevol_rles(src_info, src_sv, supervoxels=True)
#             voxel_count, first_coord_zyx = extract_rle_size_and_first_coord(rle_payload)
#             dest_body = fetch_label_for_coordinate(dest_info, first_coord_zyx, supervoxels=False)
#             bodies.add(dest_body)
#         except Exception as ex:
#             with tqdm.external_write_mode():
#                 print(f"Error reading body for SV {src_sv} (#{i}): {ex}")
# 
#     return bodies

 
if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        import os
        os.chdir("/Users/bergs/Downloads")
        #sys.argv += ["HEAD-focused-finish02-src-split-svs.csv"]
        #sys.argv += ['emdata2:8700', '4c4f3cbb51a042d88b7403eb8347d1d6', 'segmentation']
        #sys.argv += ['emdata2:8700', '2b7f805c6eef46e4bd63e028f5ec55e6', 'segmentation']
  
#         # HB splits
#         sys.argv += ['--results-output-log=hb-e09c-split-copy-results-log.csv']
#         sys.argv += ["--src-supervoxels-from-kafka"]
#         sys.argv += ['emdata2:8700', 'e09c', 'segmentation']
#         sys.argv += ['emdata3:8900', '25dcdb44cd934376999d93f1aa4d4b5f', 'segmentation']
  
#         # Erika unagglomerated SV splits, but performed as body splits:
#         sys.argv += ['--results-output-log=unagglo-8189-split-copy-results-log.csv']
#         sys.argv += ["--src-supervoxels-from-kafka"]
#         sys.argv += ['emdata2:8700', '8189', 'segmentation']
#         sys.argv += ['emdata3:8900', '25dcdb44cd934376999d93f1aa4d4b5f', 'segmentation']
 
#         # Erika unagglomerated SV splits, but performed as body splits:
#         sys.argv += ['--results-output-log=unagglo-8940-split-copy-results-log.csv']
#         sys.argv += ["--src-supervoxels-from-kafka"]
#         sys.argv += ['emdata2:8700', '8940', 'segmentation']
#         sys.argv += ['emdata3:8900', '25dcdb44cd934376999d93f1aa4d4b5f', 'segmentation']
 
    #copy_splits([5812995723, 5812995725], ['emdata2:8700', 'e09c', 'segmentation'], ['emdata3:8900', '25dcdb44cd934376999d93f1aa4d4b5f', 'segmentation'])
 
    main()

    