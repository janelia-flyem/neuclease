import sys
import logging
import argparse
from collections import namedtuple

from tqdm import tqdm

import numpy as np
import pandas as pd

from neuclease.dvid import default_dvid_session, fetch_label_for_coordinate, fetch_sparsevol_rles, split_supervoxel, extract_first_rle_coord
from neuclease.util import read_csv_col

logger = logging.getLogger(__name__)

def main():
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-output-log', '-o', default='split-copy-results-log.csv')
    parser.add_argument('src_supervoxels_csv')
    parser.add_argument('src_server')
    parser.add_argument('src_uuid')
    parser.add_argument('src_labelmap_instance')
    parser.add_argument('dest_server')
    parser.add_argument('dest_uuid')
    parser.add_argument('dest_labelmap_instance')
    args = parser.parse_args()

    src_supervoxels = read_csv_col(args.src_supervoxels_csv, col=0, dtype=np.uint64)
    src_info = InstanceInfo(args.src_server, args.src_uuid, args.src_labelmap_instance)
    dest_info = InstanceInfo(args.dest_server, args.dest_uuid, args.dest_labelmap_instance)

    copy_results = copy_splits(src_supervoxels, src_info, dest_info)
    df = pd.DataFrame(np.array(copy_results, dtype=np.uint64), columns=['src_sv', 'overwritten_sv', 'split_sv', 'remain_sv'])
    df.to_csv(args.results_output_log, index=False, header=True)
    print(f"Saved results log to {args.results_output_log}")
    
    logger.info("Done.")


InstanceInfo = namedtuple('InstanceInfo', 'server uuid instance')
SplitCopyInfo = namedtuple("SplitCopyInfo", "src_sv overwritten_sv split_sv remain_sv")
def copy_splits(src_supervoxels, src_info, dest_info):
    copy_infos = []
    for i, src_sv in enumerate(tqdm(src_supervoxels)):
        try:
            rle_payload = fetch_sparsevol_rles(*src_info, src_sv, supervoxels=True)
            first_coord_zyx = extract_first_rle_coord(rle_payload)
            dest_sv = fetch_label_for_coordinate(*dest_info, first_coord_zyx, supervoxels=True)
            split_sv, remain_sv = split_supervoxel(*dest_info, dest_sv, rle_payload)
        except BaseException as ex:
            raise RuntimeError(f"Error copying SV {src_sv} (#{i})") from ex

        split_info = SplitCopyInfo(src_sv, dest_sv, split_sv, remain_sv)
        copy_infos.append(split_info)
    return copy_infos


def read_ting_split_info(ting_split_result_log, uuid='194db260e1ee4edcbda0b592bf40d926'):
    """
    Read Ting's split results log, which is in this form:
    
        >>>> 1 task__http-++emdata2.int.janelia.org-8700+api+node+4c4f+segmentation+sparsevol+1001134218
        >>>> 2 task__http-++emdata2.int.janelia.org-8700+api+node+4c4f+segmentation+sparsevol+1002735917
        >>>> 3 task__http-++emdata2.int.janelia.org-8700+api+node+4c4f+segmentation+sparsevol+1004548999
        ...
    
    and query his special key-value instance to learn the new supervoxels he created via the splits.
    
    Note: This function isn't used above.  It's mostly here so I can remember
          how I generated the src_supervoxels CSV in case I need to do it again.

    Returns:
        List of supervoxel IDs. (The new SVs created after the splits occurred.)    
    """
    with open(ting_split_result_log, 'r') as f:
        session = default_dvid_session('focused-finish02')
        all_svs = []
        for line in tqdm(f):
            if not line.startswith('>>>>'):
                continue
            task = line.split()[2]
            
            url = f'http://zhaot-ws1:9000/api/node/{uuid}/result_split/key/{task}'
            r = session.get(url)
            r.raise_for_status()

            # Response looks like this:
            # {“committed”: [5812994702, 5812994704, 5812994705], “timestamp”: 1528521108}
            all_svs.extend(r.json()["committed"])
    return all_svs


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        import os
        os.chdir("/Users/bergs/Downloads")
        sys.argv += ["HEAD-focused-finish02-src-split-svs.csv"]
        sys.argv += ['emdata2:8700', '4c4f3cbb51a042d88b7403eb8347d1d6', 'segmentation']
        sys.argv += ['emdata2:8700', '2b7f805c6eef46e4bd63e028f5ec55e6', 'segmentation']

    main()
