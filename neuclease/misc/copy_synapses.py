from itertools import chain

from neuclease.util import boxes_from_grid, compute_parallel
from neuclease.dvid import fetch_instance_info, fetch_volume_box
from neuclease.dvid.annotation import fetch_blocks, post_elements

PROCESSES = 32

src_syn = ('emdata4:8950', '52a13328874c4bb7b15dc4280da26576', 'synapses')
dst_syn = ('emdata4:8950', '52a13328874c4bb7b15dc4280da26576', 'synapses_orig')

def copy_syn_blocks(box):
    blocks = fetch_blocks(*src_syn, box)
    if len(blocks) > 0:
        elements = list(chain(*blocks.values()))
        post_elements(*dst_syn, elements)

def main():
    # Fetch segmentation extents
    syn_info = fetch_instance_info(*src_syn)
    seg_name = syn_info["Base"]["Syncs"][0]
    src_seg = (*src_syn[:2], seg_name)
    bounding_box_zyx = fetch_volume_box(*src_seg).tolist()
    
    # Break into block-aligned chunks (boxes) that are long in the X direction
    # (optimal access pattern for dvid read/write)
    boxes = boxes_from_grid(bounding_box_zyx, (64,64,6400), clipped=True)
    
    # Use a process pool to copy the chunks in parallel.
    compute_parallel(copy_syn_blocks, boxes, processes=PROCESSES, ordered=False)

if __name__ == "__main__":
    main()
