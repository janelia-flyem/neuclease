"""
Example script for downloading a subvolume of
grayscale and segmentation from the hemibrain.
"""
import numpy as np
import h5py
from tqdm import tqdm

from neuclease.dvid import fetch_labelmap_voxels, fetch_raw
from neuclease.util import boxes_from_grid, overwrite_subvol

SEGMENTATION = ('emdata4:8900', 'a0df', 'segmentation')
GRAYSCALE = ('emdata3:8600', 'a89e', 'grayscale')


def download(bounding_box_zyx, output_path):
    shape = bounding_box_zyx[1] - bounding_box_zyx[0]

    with h5py.File(output_path, 'w') as f:
        gray_dset = f.create_dataset('grayscale', shape=shape, dtype=np.uint8, chunks=True)
        seg_dset = f.create_dataset('segmentation', shape=shape, dtype=np.uint64, chunks=True, compression='gzip')
    
        print("Downloading grayscale...")
        block_shape = (256,256,256)
        block_boxes = boxes_from_grid(bounding_box_zyx, block_shape, clipped=True)
        for block_box in tqdm(block_boxes):
            relative_box = block_box - bounding_box_zyx[0]
            block_gray = fetch_raw(*GRAYSCALE, block_box)
            overwrite_subvol(gray_dset, relative_box, block_gray)

        print("")
        print("Downloading segmentation...")
        block_boxes = boxes_from_grid(bounding_box_zyx, block_shape, clipped=True)
        for block_box in tqdm(block_boxes):
            relative_box = block_box - bounding_box_zyx[0]
            block_seg = fetch_labelmap_voxels(*SEGMENTATION, block_box)
            overwrite_subvol(seg_dset, relative_box, block_seg)

    print("")
    print("DONE")

if __name__ == "__main__":
    # Example box
    start_corner_zyx = np.array([20480, 20480, 20480])
    stop_corner_zyx = start_corner_zyx+512
    bounding_box_zyx = (start_corner_zyx, stop_corner_zyx)

    download(bounding_box_zyx, '/tmp/downloaded-vol.h5')
