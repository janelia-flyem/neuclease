import logging
from itertools import chain
from functools import partial

import numpy as np
import pandas as pd

from neuclease.util import Timer, Grid, boxes_from_grid, compute_parallel
from neuclease.dvid import find_master, fetch_labelmap_voxels, fetch_mapping, fetch_labelindex, create_labelindex, post_labelindex

from neuclease import configure_default_logging
configure_default_logging()

logger = logging.getLogger(__name__)

def main():
    # Hard-coded parameters
    prod = 'emdata4:8900'
    master = (prod, find_master(prod))
    master_seg = (*master, 'segmentation')
    
    # I accidentally corrupted the labelindex of bodies in this region
    patch_box = 20480 + np.array([[0,0,0], [1024,1024,1024]])

    with Timer("Fetching supervoxels", logger):
        boxes = boxes_from_grid(patch_box, Grid((64,64,6400)), clipped=True)
        sv_sets = compute_parallel( partial(_fetch_svs, master_seg), boxes,
                                    processes=32, ordered=False, leave_progress=True )
        svs = set(chain(*sv_sets)) - set([0])

    bodies = set(fetch_mapping(*master_seg, svs))

    with Timer(f"Repairing {len(bodies)} labelindexes", logger):
        compute_parallel( partial(_repair_index, master_seg),
                          bodies, processes=32, ordered=False, leave_progress=True )    

    print("DONE.")


def _fetch_svs(master_seg, box):
    vol = fetch_labelmap_voxels(*master_seg, box, supervoxels=True)
    return set(pd.unique(vol.reshape(-1)))


def _repair_index(master_seg, body):
    pli = fetch_labelindex(*master_seg, body, format='pandas')
    
    # Just drop the blocks below coordinate 1024
    # (That's where the bad blocks were added, and
    # there isn't supposed to be segmentation in that region.)
    pli.blocks.query('z >= 1024 and y >= 1024 and x >= 1024', inplace=True)
    
    li = create_labelindex(pli)
    post_labelindex(*master_seg, pli.label, li)


if __name__ == "__main__":
    main()
