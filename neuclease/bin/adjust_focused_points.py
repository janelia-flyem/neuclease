"""
Load a focused proofreading assignment file, and adjust the coordinates in
each task such that the starting viewpoint for the task probably shows a
better view of the putative merge site.

Additionally, each task is augmented with an extra json key/value:

    "coordinate-status": "adjusted"

In some cases, the original coordinates may be so badly misplaced that they
cannot be used as a starting point for finding better, adjusted coordinates.
In those cases, the task is marked with:

    "coordinate-status": "misplaced"
"""
import os
import json
import copy
import logging
import argparse

import numpy as np

from neuclease import configure_default_logging
from neuclease.util import tqdm_proxy, Timer
from neuclease.dvid import fetch_mapping, fetch_labelarray_voxels
from neuclease.misc import find_best_plane

logger = logging.getLogger(__name__)

def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=False)
    parser.add_argument('dvid_server')
    parser.add_argument('uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('assignment_json')
    args = parser.parse_args()

    if args.output is None:
        name, ext = os.path.splitext(args.assignment_json)
        args.output = name + '-adjusted' + ext

    instance_info = (args.dvid_server, args.uuid, args.labelmap_instance)

    with Timer(f"Processing {args.assignment_json}", logger):
        with open(args.assignment_json, 'r') as f:
            assignment_data = json.load(f)
    
        new_assignment_data = adjust_focused_points(*instance_info, assignment_data)
    
        with open(args.output, 'w') as f:
            json.dump(new_assignment_data, f, indent=2)

    logger.info(f"Wrote to {args.output}")


def adjust_focused_points(server, uuid, instance, assignment_json_data, supervoxels=True, max_search_scale=3):
    new_assignment_data = copy.deepcopy(assignment_json_data)
    new_tasks = new_assignment_data["task list"]

    for task in tqdm_proxy(new_tasks):
        sv_1 = task["supervoxel ID 1"]
        sv_2 = task["supervoxel ID 2"]
        
        coord_1 = np.array(task["supervoxel point 1"])
        coord_2 = np.array(task["supervoxel point 2"])
        
        if supervoxels:
            label_1 = sv_1
            label_2 = sv_2
        else:
            label_1, label_2 = fetch_mapping(server, uuid, instance, [sv_1, sv_2])
        
        avg_coord = (coord_1 + coord_2) // 2
        
        # Search until we find a scale in which the two touch, or give up.
        for scale in range(1+max_search_scale):
            box_xyz = ( avg_coord // (2**scale) - 64,
                        avg_coord // (2**scale) + 64 )
            box_zyx = np.array(box_xyz)[:,::-1]
            seg_vol = fetch_labelarray_voxels(server, uuid, instance, box_zyx, scale, supervoxels=supervoxels)
            
            adjusted_coords_zyx = find_best_plane(seg_vol, label_1, label_2)
            adjusted_coords_zyx = np.array(adjusted_coords_zyx)

            if not (adjusted_coords_zyx == -1).all():
                # Found it.
                adjusted_coords_zyx += box_zyx[0]
                adjusted_coords_zyx *= (2**scale)
                break

        if (adjusted_coords_zyx == -1).all():
            task["coordinate-status"] = "misplaced"
        else:
            task["supervoxel point 1"] = adjusted_coords_zyx[0, ::-1].tolist()
            task["supervoxel point 2"] = adjusted_coords_zyx[1, ::-1].tolist()
            task["coordinate-status"] = f"adjusted-at-scale-{scale}"
    
    return new_assignment_data
    

if __name__ == "__main__":
    main()
