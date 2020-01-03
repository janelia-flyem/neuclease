"""
Copies all synapses from one location to another.

If the destination instance doesn't exist yet, creates it and
adds a sync to ensure the block size is correct before ingestion.

BEFORE RUNNING:
    Set the src and dst global variables below!

Note:
    Before the sync will actually work properly, you must perform a
    ``POST .../reload`` on the destination instance yourself, after
    the copy has completed.

Note:
    Does not add any labelsz instances (or syncs).
    Add those yourself if you need them.

Note:
    Not intended to OVERWRITE destinations that already have synapse data in them.
    For instance, empty blocks are skipped, so pre-existing synapses would not be
    deleted, even if the source has no synapses for a particular block.
"""
import logging

from neuclease.util import boxes_from_grid, compute_parallel
from neuclease.dvid import fetch_repo_instances, create_instance, fetch_instance_info, fetch_volume_box
from neuclease.dvid.annotation import post_sync, fetch_blocks, post_blocks

from neuclease import configure_default_logging
configure_default_logging()

logger = logging.getLogger(__name__)

PROCESSES = 8

# Source to copy from
src_node = ('emdata4:8900', '52a13328874c4bb7b15dc4280da26576')
src_syn = 'synapses'
src_seg = 'segmentation' # Needed to determine the bounding box of all synapses to fetch

# Destination to copy to, and the segmentation instance to sync to.
dst_node = ('emdata2:7900', '23400dc115d84f0f92db068c7ab84b65')
dst_syn = 'upload_production_synapses_2020-01-03'
dst_seg = 'segmentation'


def copy_syn_blocks(box):
    blocks = fetch_blocks(*src_node, src_syn, box)
    if blocks:
        post_blocks(*dst_node, dst_syn, blocks)

def main():
    # Create the destination instance if necessary.
    dst_instances = fetch_repo_instances(*dst_node, 'annotation')
    if dst_syn not in dst_instances:
        logger.info(f"Creating instance '{dst_syn}'")
        create_instance(*dst_node, dst_syn, 'annotation')

    # Check to see if the sync already exists; add it if necessary
    syn_info = fetch_instance_info(*dst_node, dst_syn)
    if len(syn_info["Base"]["Syncs"]) == 0:
        logger.info(f"Adding a sync to '{dst_syn}' from '{dst_seg}'")
        post_sync(*dst_node, dst_syn, [dst_seg])
    elif syn_info["Base"]["Syncs"][0] != dst_seg:
        other_seg = syn_info["Base"]["Syncs"][0]
        raise RuntimeError(f"Can't create a sync to '{dst_seg}'. "
                           f"Your instance is already sync'd to a different segmentation: {other_seg}")

    # Fetch segmentation extents
    bounding_box_zyx = fetch_volume_box(*src_node, src_seg).tolist()

    # Break into block-aligned chunks (boxes) that are long in the X direction
    # (optimal access pattern for dvid read/write)
    boxes = boxes_from_grid(bounding_box_zyx, (256,256,6400), clipped=True)
    
    # Use a process pool to copy the chunks in parallel.
    compute_parallel(copy_syn_blocks, boxes, processes=PROCESSES, ordered=False)

if __name__ == "__main__":
    main()
