"""
A little throw-away script to copy grayscale and segmentation
from our remote-backed VNC volumes to our local server.

As of 2020-06-19, this is hard-coded to use emdata4:8200.

This script carefully selects the endpoints that happen to work
for fetching and posting data to/from the instances involved.

Example Usage:

    copy_vnc_subvolume -e "25300,42100,48500" "200,200,200"
"""
import logging
import argparse

logger = logging.getLogger(__name__)

# Notes:
#
# - In the VNC groundtruth repo, 'grayscalejpeg' is actually backed
#   by a ng-precomputed source, and dvid doesn't support GET /raw for it.
#   We have to use GET /subvolblocks
#
# - The 'segmentation' instance is a 'googlevoxels' instance.
#   It supports, GET /raw, but not /blocks, etc.


def copy_vnc_subvolume(box_zyx, copy_grayscale=True, copy_segmentation=True, chunk_shape=(64,64,2048)):
    assert not (box_zyx % 64).any(), \
        "Only 64px block-aligned volumes can be copied."

    import numpy as np
    from neuclease.util import boxes_from_grid, tqdm_proxy, round_box
    from neuclease.dvid import find_master, fetch_raw, post_raw, fetch_subvol, post_labelmap_voxels

    vnc_master = ('emdata4:8200', find_master('emdata4:8200'))

    NUM_SCALES = 8
    num_voxels = np.prod(box_zyx[1] - box_zyx[0])

    if copy_grayscale:
        logger.info(f"Copying grayscale from box {box_zyx[:,::-1].tolist()} ({num_voxels/1e6:.1f} Mvox) for {NUM_SCALES} scales")
        for scale in tqdm_proxy(range(NUM_SCALES)):
            if scale == 0:
                input_name = 'grayscalejpeg'
                output_name = 'local-grayscalejpeg'
            else:
                input_name = f'grayscalejpeg_{scale}'
                output_name = f'local-grayscalejpeg_{scale}'

            scaled_box_zyx = np.maximum(box_zyx // 2**scale, 1)
            scaled_box_zyx = round_box(scaled_box_zyx, 64, 'out')

            for chunk_box in tqdm_proxy(boxes_from_grid(scaled_box_zyx, chunk_shape, clipped=True), leave=False):
                chunk = fetch_subvol(*vnc_master, input_name, chunk_box, progress=False)
                post_raw(*vnc_master, output_name, chunk_box[0], chunk)

    if copy_segmentation:
        logger.info(f"Copying segmentation from box {box_zyx[:,::-1].tolist()} ({num_voxels/1e6:.2f} Mvox)")
        for chunk_box in tqdm_proxy(boxes_from_grid(box_zyx, chunk_shape, clipped=True)):
            chunk = fetch_raw(*vnc_master, 'segmentation', chunk_box, dtype=np.uint64)
            post_labelmap_voxels(*vnc_master, 'local-segmentation', chunk_box[0], chunk, downres=True)

        # TODO: Update label indexes?

    logger.info("DONE")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--expand-for-alignment', '-e', action='store_true',
                        help='Auto-expand the given subvolume bounds to make it aligned.')
    parser.add_argument('--grayscale', '-g', action='store_true', help='Copy grayscale only')
    parser.add_argument('--segmentation', '-s', action='store_true', help='Copy segmentation only')

    parser.add_argument('offset_xyz', help="Starting offset for the subvolume, e.g. '25280, 42048, 48448'")
    parser.add_argument('shape_xyz', help="Shape of the subvolume, e.g. '512, 512, 512'")
    args = parser.parse_args()

    if not args.grayscale and not args.segmentation:
        args.grayscale = True
        args.segmentation = True

    from neuclease import configure_default_logging
    configure_default_logging()

    import re
    import numpy as np
    from neuclease.util import round_box

    args.offset_xyz = re.sub(r'\D', ' ', args.offset_xyz)
    args.shape_xyz = re.sub(r'\D', ' ', args.shape_xyz)

    offset_xyz = np.array([*map(int, args.offset_xyz.split())])
    shape_xyz = np.array([*map(int, args.shape_xyz.split())])

    box_xyz = np.array([offset_xyz, offset_xyz + shape_xyz])

    box_zyx = box_xyz[:, ::-1]
    del box_xyz

    if args.expand_for_alignment:
        box_zyx = round_box(box_zyx, 64, 'out')
        shape_zyx = box_zyx[1] - box_zyx[0]
        logger.info(f"Expanded box to {box_zyx[:, ::-1].tolist()} (shape = {shape_zyx[::-1].tolist()})")
    elif (box_zyx % 64).any():
        raise RuntimeError("Only 64px block-aligned volumes can be copied.\n"
                           "Adjust your offset/shape or try the --expand-for-alignment option.")

    copy_vnc_subvolume(box_zyx, args.grayscale, args.segmentation)


if __name__ == "__main__":
    main()
