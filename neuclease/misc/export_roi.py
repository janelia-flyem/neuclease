import os
import logging
import collections

import h5py
import vigra
import numpy as np
import pandas as pd
from tqdm import tqdm

import neuprint
from neuclease.util import Timer, tqdm_proxy, round_box, boxes_from_grid, extract_subvol
from neuclease.dvid import ( fetch_repo_instances, fetch_roi, fetch_volume_box,
                             fetch_combined_roi_volume, create_labelmap_instance, post_labelmap_voxels )

logger = logging.getLogger(__name__)

def load_roi_label_volume(server, uuid, rois_or_neuprint, box_s5=[None, None], export_path=None, export_labelmap=None):
    """
    Fetch several ROIs from DVID and combine them into a single label volume or mask.
    The label values in the returned volume correspond to the order in which the ROI
    names were passed in, starting at label 1.
    
    This function is essentially a convenience function around fetch_combined_roi_volume(),
    but in this case it will optionally auto-fetch the ROI list, and auto-export the volume.
    
    Args:
        server:
            DVID server

        uuid:
            DVID uuid

        rois_or_neuprint:
            Either a list of ROIs or a neuprint server from which to obtain the roi list.

        box_s5:
            If you want to restrict the ROIs to a particular subregion,
            you may pass your own bounding box (at scale 5).
            Alternatively, you may pass the name of a segmentation
            instance from DVID whose bounding box will be used.

        export_path:
            If you want the ROI volume to be exported to disk,
            provide a path name ending with .npy or .h5.
        
        export_labelmap:
            If you want the ROI volume to be exported to a DVID labelmap instance,
            Provide the instance name, or a tuple of (server, uuid, instance).
    
    Returns:
        (roi_vol, roi_box), containing the fetched label volume and the
        bounding box it corresponds to, in DVID scale-5 coordinates.

    Note:
      If you have a list of (full-res) points to extract from the returned volume,
      pass a DataFrame with columns ['z','y','x'] to the following function.
      If you already downloaded the roi_vol (above), provide it.
      Otherwise, leave out those args and it will be fetched first.
      Adds columns to the input DF (in-place) for 'roi' (str) and 'roi_label' (int).
    
        >>> from neuclease.dvid import determine_point_rois
        >>> determine_point_rois(*master, rois, point_df, roi_vol, roi_box)
    """
    if isinstance(box_s5, str):
        # Assume that this is a segmentation instance whose dimensions should be used
        # Fetch the maximum extents of the segmentation,
        # and rescale it for scale-5.
        seg_box = fetch_volume_box(server, uuid, box_s5)
        box_s5 = round_box(seg_box, (2**5), 'out') // 2**5
        box_s5[0] = (0,0,0)

    if export_labelmap:
        assert isinstance(box_s5, np.ndarray)
        assert not (box_s5 % 64).any(), \
            ("If exporting to a labelmap instance, please supply "
             "an explicit box and make sure it is block-aligned.")
    
    if isinstance(rois_or_neuprint, (str, neuprint.Client)):
        if isinstance(rois_or_neuprint, str):
            npclient = neuprint.Client(rois_or_neuprint)
        else:
            npclient = rois_or_neuprint
        
        # Fetch ROI names from neuprint
        q = "MATCH (m: Meta) RETURN m.superLevelRois as rois"
        rois = npclient.fetch_custom(q)['rois'].iloc[0]
        rois = sorted(rois)
        # # Remove '.*ACA' ROIs. Apparently there is some
        # # problem with them. (They overlap with other ROIs.)
        # rois = [*filter(lambda r: 'ACA' not in r, rois)]
    else:
        assert isinstance(rois_or_neuprint, collections.abc.Iterable)
        rois = rois_or_neuprint

    # Fetch each ROI and write it into a volume
    with Timer(f"Fetching combined ROI volume for {len(rois)} ROIs", logger):
        roi_vol, roi_box, overlap_stats = fetch_combined_roi_volume(server, uuid, rois, box_zyx=box_s5)
    
    if len(overlap_stats) > 0:
        logger.warn(f"Some ROIs overlap! Here's an incomplete list of overlapping pairs:\n{overlap_stats}")
    
    # Export to npy/h5py for external use
    if export_path:
        with Timer(f"Exporting to {export_path}", logger):
            if export_path.endswith('.npy'):
                np.save(export_path, roi_vol)
            elif export_path.endswith('.h5'):
                with h5py.File(export_path, 'w') as f:
                    f.create_dataset('rois_scale_5', data=roi_vol, chunks=True)

    if export_labelmap:
        if isinstance(export_labelmap, str):
            export_labelmap = (server, uuid, export_labelmap)
        
        assert len(export_labelmap) == 3
        with Timer(f"Exporting to {export_labelmap[2]}", logger):
            if export_labelmap[2] not in fetch_repo_instances(server, uuid, 'labelmap'):
                create_labelmap_instance(*export_labelmap, voxel_size=8*(2**5), max_scale=6) # FIXME: hard-coded voxel size
            
            boxes = boxes_from_grid(roi_box, (64,64,2048), clipped=True)
            for box in tqdm_proxy(boxes):
                block = extract_subvol(roi_vol, box - roi_box[0])
                post_labelmap_voxels(*export_labelmap, box[0], block, scale=0, downres=True)

    return roi_vol, roi_box, rois
    

def export_roi_slices(server, uuid, roi_name, scale, scaled_shape_zyx, parent_output_dir):
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
    from neuclease.util import view_as_blocks

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

    scaled_shape_zyx = fetch_volume_box(*instance_info)[1]

    ns = DVIDNodeService(*instance_info[:2])
    z_slab_bounds = list(range(0, scaled_shape_zyx[0] // 64 * 64 + 1, 64))

    y_stop, x_stop = scaled_shape_zyx[1:3]
    for z_start, z_stop in tqdm(list(zip(z_slab_bounds[:-1], z_slab_bounds[1:]))):
        slab_shape = (z_stop - z_start, y_stop, x_stop)
        slab_vol = ns.get_gray3D(instance_name, slab_shape, (z_start, 0, 0), False, False)
        slab_vol = vigra.taggedView(slab_vol, 'zyx')
        for z, z_slice in enumerate(tqdm(slab_vol, leave=False), start=z_start):
            vigra.impex.writeImage(z_slice, f'{output_dir}/{z:05d}.tiff', 'UINT8')
