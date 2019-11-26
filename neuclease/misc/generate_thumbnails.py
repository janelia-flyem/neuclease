import argparse
from itertools import chain
from functools import partial

import vigra
import numpy as np
from skimage.transform import rescale


from ..util import downsample_mask, compute_nonzero_box, extract_subvol, round_box, iter_batches, compute_parallel
from ..dvid import fetch_sparsevol_coarse, find_master, fetch_roi, fetch_volume_box

def main():
    assert False, "TODO"

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    bad_bodies = generate_traced_body_thumbnails()

def generate_roi_thumbnail(seg_instance, background_roi):
    # Round out to block boundary before downscaling box
    seg_box_s0 = fetch_volume_box(*seg_instance)
    seg_box_s0 = round_box(seg_box_s0, 64*(2**6))
    seg_box_s5 = seg_box_s0 // 2**5
    seg_box_s5[0] = (0,0,0)

    # Download scale 5
    roi_vol_s5, _roi_box = fetch_roi(*seg_instance, background_roi, format='mask', mask_box=seg_box_s5)
    
    # Downsample to scale 6
    roi_vol_s6 = downsample_mask(roi_vol_s5, 2)
    
    # Project through Y
    roi_thumbnail = np.logical_or.reduce(roi_vol_s6, axis=1)
    return roi_thumbnail


def generate_body_thumbnail(seg_instance, body, roi_thumbnail, final_scale=7, format='rgb', neuron_color=[255,255,0], roi_color=[100,100,100]):
    assert final_scale >= 6
    
    mask, _mask_box = fetch_sparsevol_coarse(*seg_instance, body, format='mask', mask_box=[(0,0,0), roi_thumbnail])

    # Project through Y
    mask_thumbnail = np.logical_or.reduce(mask, axis=1)

    if format == 'labels':
        # If returning labels, can't anti-alias, so just downsample before combining
        mask_thumbnail = downsample_mask(mask_thumbnail, 2**(final_scale - 6))
        roi_thumbnail = downsample_mask(roi_thumbnail, 2**(final_scale - 6))

    combined_thumbnail = roi_thumbnail.copy().astype(np.uint8)
    combined_thumbnail[mask_thumbnail] = 2

    box = compute_nonzero_box(combined_thumbnail)
    combined_thumbnail = extract_subvol(combined_thumbnail, box)
    
    if format == 'labels':
        return combined_thumbnail
    
    if format == 'rgb':
        colortable = np.array([[0,0,0], roi_color, neuron_color])
        rgb_thumbnail = np.zeros(combined_thumbnail.shape + (3,), dtype=np.uint8)
        for c in range(3):
            rgb_thumbnail[...,c] = colortable[:,c][combined_thumbnail]

        if final_scale != 6:
            rgb_thumbnail = rescale(rgb_thumbnail, 1/(2**(final_scale - 6)), anti_aliasing=True, multichannel=True)

        return rgb_thumbnail


def thumbnail_batch(seg_instance, roi_thumbnail, final_scale, output_dir, bodies, neuron_color=[255,255,0], roi_color=[100,100,100]):
    bad_bodies = []
    for body in bodies:
        try:
            rgb = generate_body_thumbnail(seg_instance, body, roi_thumbnail, final_scale,
                                          format='rgb', neuron_color=neuron_color, roi_color=roi_color)
        except Exception:
            bad_bodies.append(body)
            continue
        vigra.impex.writeImage(rgb.transpose(1,0,2), f'{output_dir}/{body}.png')

    return bad_bodies


def generate_traced_body_thumbnails(npclient, seg_instance, background_roi, final_scale,
                                    output_dir, neuron_color=[255,255,0], roi_color=[100,100,100],
                                    processes=32):
    traced_statuses = ["Leaves", "Prelim Roughly traced", "Roughly traced", "Traced"]

    q = f"""\
        MATCH (n:`hemibrain-Neuron`)
        WHERE n.status in {traced_statuses}
        RETURN n.bodyId as body, n.status as status
    """
    traced_bodies = npclient.fetch_custom(q)['body'].values

    batch_size = 20
    body_batches = iter_batches(traced_bodies, batch_size)

    roi_thumbnail = generate_roi_thumbnail(seg_instance, background_roi)
    batch_fn = partial(thumbnail_batch, roi_thumbnail, final_scale, output_dir, neuron_color=neuron_color, roi_color=roi_color)
    bad_bodies = compute_parallel(batch_fn, body_batches, processes=processes, ordered=False)
    bad_bodies = [*chain(*bad_bodies)]
    return bad_bodies


def show_label_thumbnail(combined_thumbnail):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    from matplotlib.colors import LinearSegmentedColormap

    assert combined_thumbnail.max() == 2
    cmap = LinearSegmentedColormap.from_list('thumbnail_colors', ['black', 'gray', 'yellow'], 3)
    figure(figsize=(5,5))
    p = plt.imshow(combined_thumbnail, cmap=cmap);
    return p


if __name__ == "__main__":
    main()
