#!/usr/bin/env python3
"""
Given a set of binary masks which correspond to specific Z slices,
produce a complete volume in which the unlabeled intermediate slices
are filled in with labels. The new label positions are determined by
interpolating between the signed distance transforms of the masks in
the labeled slices.

Usage:

    python binterp.py labeled_slices

See --help for other options.

"""
#
# See also:
# - https://forum.image.sc/t/searching-for-a-3d-binary-interpolation-plugin/20384/4
# - https://imagej.nih.gov/ij/developer/source/ij/process/BinaryInterpolator.java.html
# - https://github.com/saalfeldlab/label-utilities/blob/3af2f0e/src/main/kotlin/org/janelia/saalfeldlab/labels/InterpolateBetweenSections.kt
#
import os
import re
import glob
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import scipy.ndimage
import scipy.interpolate
import skimage.io


def main():
    args = parse_args()

    labeled_slices = read_labeled_slices(args.slice_directory)
    distance_vol = signed_distance_interpolation(labeled_slices, args.minz, args.maxz)

    mask_vol = (distance_vol < 0).astype(np.uint8)
    mask_vol[:] *= args.out_label
    write_slices(mask_vol, args.output_directory, args.minz)

    if args.export_distance_visualization:
        d = args.output_directory + '_distance_viz'
        if args.minz is None:
            args.minz = min(labeled_slices.keys())
        export_distance_visualization(distance_vol, d, args.minz)

    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minz", type=int, help='First output slice to produce')
    parser.add_argument("--maxz", type=int, help='Last output slice to produce')
    parser.add_argument("--out-label", '-l', type=int, default=255, help='The value to assign for labeled output pixels. (e.g. 255)')
    parser.add_argument("--export-distance-visualization", "-e", action="store_true",
                        help='Debugging feature. Export the raw distance transform volume as png slices, '
                        'after taking the absolute value of the signed distance transform and '
                        'renormalizing to the range 0-255')
    parser.add_argument('slice_directory', help='Directory of input PNG slices. Unlabeled slices can be omitted.')
    parser.add_argument('output_directory', nargs='?', help='Directory to place the output slices. Must not exist yet.')
    args = parser.parse_args()

    if not args.output_directory:
        args.output_directory = os.path.normpath(args.slice_directory) + '_interpolated'

    args.output_directory = os.path.normpath(args.output_directory)
    if os.path.exists(args.output_directory):
        raise RuntimeError(f"Output directory already exists: {args.output_directory}")

    return args


def read_labeled_slices(slice_directory):
    """
    Read all images in the given directory,
    and return a dictionary of {z: img} for only
    those slices which are non-empty.
    """
    slice_files = sorted(glob.glob(f"{slice_directory}/*.*"))
    print(f"Reading {len(slice_files)} input files")

    labeled_slices = {}
    for p in tqdm(slice_files):
        img = skimage.io.imread(p)

        # Ignore slices with no labels at all
        if not img.any():
            continue

        # The last number in the filename indicates the Z-index
        numbers = re.split('[^0-9]+', p)
        numbers = [*filter(None, numbers)]
        if len(numbers) == 0:
            raise RuntimeError(f"Couldn't find a slice number in the file name: {p}")
        z = int(numbers[-1])
        labeled_slices[z] = img

    print(f"Labels found on slices: {[*labeled_slices.keys()]}")
    if len(labeled_slices) < 2:
        raise RuntimeError("Not enough labeled slices to interpolate!")

    return labeled_slices


def signed_distance_interpolation(labeled_slices, z_min=None, z_max=None):
    """
    Given a set of binary masks corresponding to specific
    Z-positions in a volume, compute their signed distnace
    transforms and return an interpolated volume for slices
    in the given Z-range.

    Args:
        labeled_slices:
            A dictionary of {z: img}, where img is a binary
            mask slice and z is the z-index it corresponds to.
        z_min:
            The first slice to produce output for.
            Default is the first labeled slice.
        z_max:
            The last slice to produce output for
            Default is the last labeled slice.

    Returns:
        3D ndarray of shape (Z,Y,X), where Z = 1 + z_max - z_min
        The volume is a float32 image, corresponding
        to the slices from z_min to z_max.
        Distances outside the range of first and last labeled slices
        will be set to np.inf.
    """
    if z_min is None:
        z_min = min(labeled_slices.keys())

    if z_max is None:
        z_max = max(labeled_slices.keys())

    assert z_min < z_max, f"Invalid output slice index range: {z_min}..{z_max}"
    print(f"Interpolating {z_min}..{z_max}")

    # Compute signed distance transform slices
    distances = {}
    for z, img in tqdm(labeled_slices.items()):
        img = img.astype(bool)

        # distances will be negative inside the mask, positive outside
        distances[z] = scipy.ndimage.distance_transform_edt(~img)
        distances[z] -= scipy.ndimage.distance_transform_edt(img)

    x = list(distances.keys())
    y = list(distances.values())
    interpolator = scipy.interpolate.interp1d(
        x, y, axis=0, bounds_error=False, fill_value=np.inf, assume_sorted=True)

    distance_vol = interpolator(np.arange(z_min, z_max+1))
    return distance_vol.astype(np.float32)


def write_slices(vol, out_dir, z_min=0):
    """
    Write the given volume to disk as a sequence of PNG files.
    The slice file numbering will start with z_min.
    """
    z_max = z_min + len(vol)
    os.makedirs(out_dir)
    print(f"Writing {len(vol)} slices to {out_dir}/")
    with warnings.catch_warnings():
        # Hide warnings from scikit-image for empty slices
        warnings.filterwarnings("ignore", message='.*is a low contrast image.*', category=UserWarning)

        digits = int(np.ceil(np.log10(z_max)))
        for z, img in enumerate(tqdm(vol), start=z_min):
            p = (out_dir + '/{' + f':0{digits}' + '}.png').format(z)
            skimage.io.imsave(p, img)


def export_distance_visualization(distance_vol, out_dir, z_min):
    """
    For debugging and analysis.
    Convert the given float volume to uint8 by taking its
    absolute value and renormalizing to the range 0-255.
    Then export as PNG slices.
    """
    viz = np.abs(distance_vol.astype(np.float32))
    viz[np.isinf(viz)] = 0
    viz = 255 * viz / viz.max()
    viz = viz.astype(np.uint8)
    write_slices(viz, out_dir, z_min)


if __name__ == "__main__":
    main()
