import os
import vigra
import numpy as np
import pandas as pd
from tqdm import tqdm

from neuclease.dvid import fetch_roi, fetch_volume_box
from libdvid import DVIDNodeService

def export_roi(server, uuid, roi_name, scale, scaled_shape_zyx, parent_output_dir):
    """
    Export the ROI to a PNG stack (as binary images) at the requested scale.
    
    Args:
        server, uuid, roi_name:
            ROI instance to read
        
        scale:
            What scale to export as, relative to the full-res grayscale.
            Must be no greater than 5.
            (ROIs are natively at scale=5, so using that scale will result in no upscaling.)
        
        scaled_shape_zyx:
            The max shape of the exported volume, in scaled coordinates.
            The PNG stack files always start at (0,0,0), and extend to this shape.
            Any ROI blocks below 0 or above this shape are silently ignored.
        
        parent_output_dir:
            Where to write the directory of PNG images.
            (A child directory will be created here and named after the ROI instance.)        
    """
    from skimage.util import view_as_blocks

    assert not ((scaled_shape_zyx * 2**scale) % 64).any(), \
        "The code below assumes that the volume shape is block aligned"

    # Fetch the ROI-block coords (always scale 5)
    roi_coords = fetch_roi((server, uuid, roi_name), format='coords')
    if len(roi_coords) == 0:
        return

    output_dir = f'{parent_output_dir}/{roi_name}-mask-scale-{scale}'
    os.makedirs(output_dir, exist_ok=True)

    # Create a mask for the scale we're using (hopefully it fits in RAM...)
    scaled_mask = np.zeros(scaled_shape_zyx, np.uint8)

    # Create a view of the scaled mask that allows us to broadcast on a per-block basis,
    # indexed as follows: scaled_mask_view[Bz,By,Bx,f,f,f],
    # where f = scale_diff_factor = 32 / (2**SCALE)
    # (ROIs are returned at scale 5!!)
    scale_diff_factor = (2**5) // (2**scale)
    scaled_mask_view = view_as_blocks(scaled_mask, 3*(scale_diff_factor,))

    roi_box = np.array([roi_coords.min(axis=0), 1+roi_coords.max(axis=0)])
    if (roi_box[0] < 0).any() or (roi_box[1] > scaled_mask_view.shape[:3]).any():
        # Drop coordinates outside the volume.
        # (Some ROIs extend beyond our sample.)
        (Z, Y, X) = scaled_mask_view.shape[:3] #@UnusedVariable
        roi_coords_df = pd.DataFrame(roi_coords, columns=list('zyx'))
        roi_coords_df.query('x >= 0 and y >= 0 and z >= 0 and x < @X and y < @Y and z < @Z', inplace=True)

        roi_coords = roi_coords_df.values
        roi_box = np.array([roi_coords.min(axis=0), 1+roi_coords.max(axis=0)])

    # Apply to the mask
    scaled_mask_view[tuple(roi_coords.transpose())] = 1
    scaled_mask = vigra.taggedView(scaled_mask, 'zyx')
    for z, z_slice in enumerate(tqdm(scaled_mask, leave=False)):
        vigra.impex.writeImage(z_slice, f'{output_dir}/{z:05d}.png', 'UINT8')


def export_downsampled_grayscale(instance_info, scale, parent_output_dir):
    if scale > 0:
        instance_name = f'grayscale_{scale}'
        instance_info = (instance_info[0], instance_info[1], instance_name)

    output_dir = f'{parent_output_dir}/grayscale-scale-{scale}'
    os.makedirs(output_dir, exist_ok=True)

    scaled_shape_zyx = fetch_volume_box(instance_info)[1]

    ns = DVIDNodeService(*instance_info[:2])
    z_slab_bounds = list(range(0, scaled_shape_zyx[0] // 64 * 64 + 1, 64))

    y_stop, x_stop = scaled_shape_zyx[1:3]
    for z_start, z_stop in tqdm(list(zip(z_slab_bounds[:-1], z_slab_bounds[1:]))):
        slab_shape = (z_stop - z_start, y_stop, x_stop)
        slab_vol = ns.get_gray3D(instance_name, slab_shape, (z_start, 0, 0), False, False)
        slab_vol = vigra.taggedView(slab_vol, 'zyx')
        for z, z_slice in enumerate(tqdm(slab_vol, leave=False), start=z_start):
            vigra.impex.writeImage(z_slice, f'{output_dir}/{z:05d}.tiff', 'UINT8')
