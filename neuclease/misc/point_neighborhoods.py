"""
This script will take a list of points and generate a "neighborhood" segment
for each point, consisting of the ball around the point, restricted to a
particular neuron. The neighborhood segments are written to a separate
labelmap instance.

Additionally, multiple files are emitted:

  - A CSV file of the points and corresponding neighborhood IDs
  - An html file with the same information, but also neuroglancer links to each point
  - An assingment file for FlyEM's neuroglancer-based protocol for inspecting such
    neighborhoods

To use this script, provide a config file to specify the input and output segmentation
info, and other settings. To specify the points to create neighborhoods for,
you may provide your own CSV of points, OR you can let this script select a
set of random points, subject to various criteria.

Example usage:

        # Process a custom list of points
        point_neighborhoods config.yaml -p my-points.csv

        # Auto-select 100 random points from anywhere in the segmentation volume
        point_neighborhoods config.yaml -c=100

        # Auto-select 100 random points from anywhere within the FB
        point_neighborhoods config.yaml -c=100 --roi=FB

        # Auto-select 100 random points from anywhere within body 1071121755
        point_neighborhoods config.yaml -c=100 --body=1071121755

        # Auto-select 100 random skeleton nodes from body 1071121755
        point_neighborhoods config.yaml -c=100 --body=1071121755 --skeleton

        # Auto-select 100 random tbars from body 1071121755
        point_neighborhoods config.yaml -c=100 --body=1071121755 --tbars

        # Auto-select 100 random tbars from body 1071121755, restricted to the FB roi
        point_neighborhoods config.yaml -c=100 --body=1071121755 --tbars --roi=FB

See the --help text for other options.
"""
import os
import sys
import csv
import json
import logging
import argparse
import urllib.parse
from textwrap import dedent
from collections.abc import Iterable

import numpy as np
from numpy.random import default_rng
import pandas as pd
import scipy.spatial
from vol2mesh import Mesh
from confiddler import load_config, dump_default_config

from neuclease import configure_default_logging
from neuclease.util import round_box, box_to_slicing as b2s, tqdm_proxy, swc_to_dataframe, mask_centroid, sphere_mask
from neuclease.dvid import (post_key, determine_point_rois, fetch_sparsevol, fetch_roi_roi,
                            runlength_encode_mask_to_ranges, fetch_volume_box, fetch_instance_info,
                            fetch_annotation_label, fetch_labelmap_voxels, post_labelmap_voxels,
                            create_labelmap_instance, fetch_key, create_instance)

logger = logging.getLogger(__name__)

RANDOM_SEED = 0

LabelmapSchema = {
    "description": "dvid labelmap location",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "required": ["server", "uuid", "instance"],
    "properties": {
        "server": {
            "description": "DVID server",
            "type": "string"
        },
        "uuid": {
            "description": "DVID node",
            "type": "string"
        },
        "instance": {
            "description": "Name of the labelmap instance",
            "type": "string"
        }
    }
}

ConfigSchema = {
    "description": "Point neighborhoods segmentation sources and settings configuration",
    "default": {},
    "required": ["radius", "input", "output", "neuroglancer"],
    "additionalProperties": False,
    "properties": {
        "radius": {
            "description": "Radius (in voxels) of the neighborhood to create.\n"
                           "Hint: In the hemibrain, 1 micron = 125 voxels.\n",
            "type": "number",
            "default": 125
        },
        "input": {**LabelmapSchema, "description": "Input neuron segmentation instance info"},
        "output": {**LabelmapSchema, "description": "Where to write neighborhood segmentation.\n"
                                                    "Instance will be created if necessary."},
        "neuroglancer": {
            "description": "Neuroglancer links will be output in html (and CSV with the --ng-links option).\n"
                           "Specify the neuroglancer server and view settings here.\n",
            "properties": {
                "server": {
                    "type": "string",
                    "default": "http://emdata4.int.janelia.org:8900"
                },
                "settings": {
                    "type": "object",
                    "default": {
                        "showSlices": False,
                        "layout": "4panel",
                        "layers": [
                            {
                                "type": "image",
                                "source": {
                                    "url": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
                                    "subsources": {
                                        "default": True
                                    },
                                    "enableDefaultSubsources": False
                                },
                                "blend": "default",
                                "name": "emdata"
                            }
                        ]
                    }
                }
            }
        }
    }
}


def write_point_neighborhoods(seg_src, seg_dst, points_zyx, radius=125, src_bodies=None, dst_bodies=None):
    """
    For each point in the given list, create a mask of the portion
    of a particular body that falls within a given distance of the
    point.

    Args:
        seg_src:
            tuple (server, uuid, instance) specifying where to fetch neuron segmentation from.
        seg_dst:
            tuple (server, uuid, instance) specifying where to write neighborhood segmentation to.
        points_zyx:
            Array of coordinates to create neighborhoods around
        radius:
            Radius (in voxels) of the neighborhood to create.
            Hint: In the hemibrain, 1 micron = 125 voxels.
        src_bodies:
            Either a single body ID or a list of body IDs (corresponding to the list of points_zyx).
            Specifies which body the neighborhood around each point should be constructed for.
            If not provided, the body for each neighborhood will be chosen automatically,
            by determining which body each point in points_zyx falls within.
        dst_bodies:
            List of new body IDs.
            Specifies the IDs to use as the 'body ID' for the neighborhood segments when writing to
            the destination instance.  If no list is given, then new body IDs are automatically
            generated with a formula that uses the coordinate around which the neighborhood was created.
            Note that the default formula does not take the source body into account,
            so if there are duplicate points provided in points_zyx, the destination body IDs will
            be duplicated too, unless you supply your own destination body IDs here.

    Returns:
        In addition to writing the neighborhood segments to the seg_dst instance,
        this function returns a dataframe with basic stats about the neighborhoods
        that were written.
    """
    if isinstance(points_zyx, pd.DataFrame):
        points_zyx = points_zyx[[*'zyx']].values
    else:
        points_zyx = np.asarray(points_zyx)

    results = []
    for i, point in enumerate(tqdm_proxy(points_zyx)):
        if isinstance(src_bodies, Iterable):
            src_body = src_bodies[i]
        else:
            src_body = src_bodies

        if isinstance(dst_bodies, Iterable):
            dst_body = dst_bodies[i]
        else:
            dst_body = dst_bodies

        point, centroid, top_point, src_body, dst_body, dst_voxels = \
            process_point(seg_src, seg_dst, point, radius, src_body, dst_body)

        results.append( (*point, *centroid, *top_point, src_body, dst_body, dst_voxels) )

    cols = ['z', 'y', 'x']
    cols += ['cz', 'cy', 'cx']
    cols += ['tz', 'ty', 'tx']
    cols += ['src_body', 'dst_body', 'dst_voxels']
    return pd.DataFrame(results, columns=cols)


def process_point(seg_src, seg_dst, point, radius, src_body, dst_body):
    """
    Generate a neighborhood segment around a particular point.
    Upload the voxels for the segment and the corresponding mesh.
    """
    r = radius
    src_box = np.asarray(( point - r, point + r + 1 ))
    src_vol = fetch_labelmap_voxels(*seg_src, src_box)

    if src_body is None:
        src_body = src_vol[r,r,r]

    if dst_body is None:
        # Generate a neighborhood segment ID from the coordinate.
        # Divide by 4 to ensure the coordinates fit within 2^53.
        # (The segment ID will not retain the full resolution of
        # the coordinate, but that's usually OK for our purposes.)
        dst_body = encode_point_to_uint64(point // 4, 17)

    mask = (src_vol == src_body) & sphere_mask(r)

    dst_box = round_box(src_box, 64, 'out')
    dst_vol = fetch_labelmap_voxels(*seg_dst, dst_box)

    dst_view = dst_vol[b2s(*(src_box - dst_box[0]))]
    dst_view[mask] = dst_body

    post_labelmap_voxels(*seg_dst, dst_box[0], dst_vol, downres=True)

    # Mesh needs to be written in nm, hence 8x
    mesh = Mesh.from_binary_vol(mask, 8*src_box, smoothing_rounds=2)
    mesh.simplify(0.05, in_memory=True)
    post_key(*seg_dst[:2], f'{seg_dst[2]}_meshes', f'{dst_body}.ngmesh', mesh.serialize(fmt='ngmesh'))

    centroid = src_box[0] + mask_centroid(mask, True)
    top_z = mask.sum(axis=(1,2)).nonzero()[0][0]
    top_coords = np.transpose(mask[top_z].nonzero())
    top_point = src_box[0] + (top_z, *top_coords[len(top_coords)//2])

    return point, centroid, top_point, src_body, dst_body, mask.sum()


def encode_point_to_uint64(point_zyx, bitwidth):
    """
    Encode the given point (z,y,x) to as a uint64, giving
    each dimension the specified number of bits.
    """
    point_zyx = point_zyx.astype(np.uint64)
    assert 3*bitwidth <= 53, "You shouldn't use body values greater than 2^53"
    encoded = np.uint64(0)
    encoded |= point_zyx[0] << np.uint64(2*bitwidth)
    encoded |= point_zyx[1] << np.uint64(bitwidth)
    encoded |= point_zyx[2]
    return encoded


def autogen_points(input_seg, count, roi, body, tbars, use_skeleton):
    """
    Generate a list of points within the input segmentation, based on the given criteria.
    See the main help text below for details.
    """
    if tbars and not body:
        sys.exit("If you want to auto-generate tbar points, please specify a body.")

    if not tbars and not count:
        sys.exit("You must supply a --count unless you are generating all tbars of a body.")

    if use_skeleton:
        if not body:
            sys.exit("You must supply a body ID if you want to use a skeleton.")
        if tbars:
            sys.exit("You can't select both tbar points and skeleton points.  Pick one or the other.")
        if not count and not roi:
            logger.warning("You are using all nodes of a skeleton without any ROI filter! Is that what you meant?")

    rng = default_rng(RANDOM_SEED)

    if tbars:
        logger.info(f"Fetching synapses for body {body}")
        syn_df = fetch_annotation_label(*input_seg[:2], 'synapses', body, format='pandas')
        tbars = syn_df.query('kind == "PreSyn"')[[*'zyx']]

        if roi:
            logger.info(f"Filtering tbars for roi {roi}")
            determine_point_rois(*input_seg[:2], [roi], tbars)
            tbars = tbars.query('roi == @roi')[[*'zyx']]

        if count:
            count = min(count, len(tbars))
            logger.info(f"Sampling {count} tbars")
            choices = rng.choice(tbars.index, size=count, replace=False)
            tbars = tbars.loc[choices]

        return tbars

    elif use_skeleton:
        assert body
        logger.info(f"Fetching skeleton for body {body}")
        skeleton_instance = f'{input_seg[2]}_skeletons'
        swc = fetch_key(*input_seg[:2], skeleton_instance, f'{body}_swc')
        skeleton_df = swc_to_dataframe(swc)
        skeleton_df['x'] = skeleton_df['x'].astype(int)
        skeleton_df['y'] = skeleton_df['y'].astype(int)
        skeleton_df['z'] = skeleton_df['z'].astype(int)

        if roi:
            logger.info(f"Filtering skeleton for roi {roi}")
            determine_point_rois(*input_seg[:2], [roi], skeleton_df)
            skeleton_df = skeleton_df.query('roi == @roi')[[*'zyx']]

        if count:
            count = min(count, len(skeleton_df))
            logger.info(f"Sampling {count} skeleton points")
            choices = rng.choice(skeleton_df.index, size=count, replace=False)
            skeleton_df = skeleton_df.loc[choices]

        return skeleton_df

    elif body:
        if roi:
            # TODO: intersect the ranges with the ROI.
            raise NotImplementedError("Sorry, I haven't yet implemented support for "
                                      "body+roi filtering.  Pick one or the other, "
                                      "or ask Stuart to fix this.")

        logger.info(f"Fetching sparsevol for body {body}")
        ranges = fetch_sparsevol(*input_seg, body, format='ranges')
        logger.info("Sampling from sparsevol")
        points = sample_points_from_ranges(ranges, count, rng)
        return pd.DataFrame(points, columns=[*'zyx'])

    elif roi:
        logger.info(f"Fetching roi {roi}")
        roi_ranges = fetch_roi_roi(*input_seg[:2], roi, format='ranges')
        logger.info("Sampling from ranges")
        points_s5 = sample_points_from_ranges(roi_ranges, count, rng)
        corners_s0 = points_s5 * (2**5)
        points_s0 = rng.integers(corners_s0, corners_s0 + (2**5))
        return pd.DataFrame(points_s0, columns=[*'zyx'])
    else:
        # No body or roi specified, just sample from the whole non-zero segmentation area
        logger.info("Sampling random points from entire input segmentation")
        logger.info("Fetching low-res input volume")
        box_s6 = round_box(fetch_volume_box(*input_seg), 2**6, 'out') // 2**6
        seg_s6 = fetch_labelmap_voxels(*input_seg, box_s6, scale=6)
        mask_s6 = seg_s6.astype(bool)
        logger.info("Encoding segmentation as ranges")
        seg_ranges = runlength_encode_mask_to_ranges(mask_s6, box_s6)
        logger.info("Sampling from ranges")
        points_s6 = sample_points_from_ranges(seg_ranges, count, rng)
        corners_s0 = points_s6 * (2**6)
        points_s0 = rng.integers(corners_s0, corners_s0 + (2**6))
        return pd.DataFrame(points_s0, columns=[*'zyx'])


def sample_points_from_ranges(ranges, count, rng=None):
    """
    Sample coordinates from the given ranges.

    Args:
        ranges:
            A 2D array of RLE ranges as returned by dvid, in the form:

                [[Z,Y,X1,X2],
                 [Z,Y,X1,X2],
                 [Z,Y,X1,X2],
                 ...
                 ]

            Note:
                The interval [X1,X2] is INCLUSIVE, following DVID conventions,
                not Python conventions (i.e. X2 is not one-past-the-end).

        count:
            How many coordinates to sample

        rng:
            numpy random number generator

    Returns:
        ndarray [[z,y,x], [z,y,x], ...]
    """
    if rng is None:
        rng = default_rng(RANDOM_SEED)
    ranges = ranges.copy()
    points = []
    _Z, _Y, X0, X1 = ranges.transpose()
    X1 += 1
    lengths = X1 - X0
    p = lengths / lengths.sum()

    rows = rng.choice(np.arange(len(ranges)), count, True, p)

    points = []
    for row in rows:
        z, y, x0, x1 = ranges[row]
        x = rng.integers(x0, x1)
        points.append((z, y, x))

    return np.array(points, dtype=int)


def write_assignment_file(seg_dst, points, path):
    server, uuid, mask_instance = seg_dst
    if not server.startswith('http'):
        server = f'https://{server}'

    assignment = {
        "file version": 1,
        # FIXME don't hardcode grayscale source
        "grayscale source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
        "mito ROI source": f"dvid://{server}/{uuid}/neighborhood-masks",
        "DVID source": f"{server}/#/repo/{uuid}",
        "task list": []
    }

    for row in points.itertuples():
        task = {
            "task type": "mito count",
            "source body id": row.src_body,
            "focal point": [row.tx, row.ty, row.tz],
            "neighborhood id": row.dst_body,
            "neighborhood size": row.dst_voxels,
            "neighborhood origin": [row.x, row.y, row.z],
            "neighborhood top": [row.tx, row.ty, row.tz],
            "neighborhood centroid": [row.cx, row.cy, row.cz],
        }
        assignment["task list"].append(task)

    with open(path, 'w') as f:
        json.dump(assignment, f, indent=2)


def add_link_col(points, config):
    points['link'] = ""
    ng_server = config["neuroglancer"]["server"]
    ng_settings = config["neuroglancer"]["settings"]
    for i, coord in enumerate(points[[*'xyz']].values):
        ng_settings["position"] = coord.tolist()
        assert points.columns[-1] == 'link'
        link = ng_server + '/#!' + urllib.parse.quote(json.dumps(ng_settings))
        points.iloc[i, -1] = link


def export_as_html(points, path):
    path = os.path.splitext(path)[0] + '.html'
    doc = dedent(f"""\
        <html>
        <title>{os.path.basename(path)}</title>
        <body>
        <table>
    """)

    doc += '<tr>'
    for col in points.columns:
        doc += f'<td>{col}</td>'
    doc += '</tr>\n'

    for row in points.itertuples():
        doc += '<tr>'
        for col in points.columns:
            doc += '<td>'
            if col == 'link':
                doc += f'<a href="{row.link}">link</a>'
            else:
                doc += f'{getattr(row, col)}'
            doc += '</td>'
        doc += '</tr>\n'

    doc += dedent("""\
        </table>
        </body>
        </html>
    """)

    with open(path, 'w') as f:
        f.write(doc)


def main():
    # Early exit if we're dumping the config
    # (Parse it ourselves to allow omission of otherwise required parameters.)
    if ({'--dump-config-template', '-d'} & {*sys.argv}):
        dump_default_config(ConfigSchema, sys.stdout, "yaml-with-comments")
        sys.exit(0)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dump-config-template', '-d', action='store_true',
                        help='Dump out a template yaml config file and exit.')
    parser.add_argument('--count', '-c', type=int, help='How many points to generate.')
    parser.add_argument('--roi', '-r', help='Limit points to the given ROI.')
    parser.add_argument('--body', '-b', type=int, help='Limit points to the given body.')
    parser.add_argument('--tbars', '-t', action='store_true',
                        help='If given, limit points to the tbars of the given body, from the "synapses" instance in the input UUID.')
    parser.add_argument('--skeleton', '-s', action='store_true',
                        help='If given, choose the points from the nodes of the skeleton for the given body.')
    parser.add_argument('--generate-points-only', '-g', action='store_true',
                        help="If given, generate the points list, but don't write neighborhood segmentations")
    parser.add_argument('--points', '-p',
                        help='A CSV file containing the points to use instead of automatically generating them.')
    parser.add_argument('--ng-links', '-n', action='store_true',
                        help='If given, include neuroglancer links in the output CSV.'
                             'Your config should specify the basic neuroglancer view settings; only the "position" will be overwritten in each link.')
    parser.add_argument('config')
    args = parser.parse_args()

    configure_default_logging()

    config = load_config(args.config, ConfigSchema)
    input_seg = [*config["input"].values()]
    output_seg = [*config["output"].values()]
    radius = config["radius"]

    if args.points and any([args.count, args.roi, args.body, args.tbars, args.skeleton]):
        msg = ("If you're providing your own list of points, you shouldn't"
               " specify any of the auto-generation arguments, such as"
               " --count --roi --body --tbars")
        sys.exit(msg)

    if not args.points and not any([args.count, args.roi, args.body, args.tbars, args.skeleton]):
        msg = "You must provide a list of points or specify how to auto-generate them."
        sys.exit(msg)

    if args.points:
        assert args.points.endswith('.csv')
        name, _ = os.path.splitext(args.points)
        output_path = name + '-neighborhoods.csv'
        points = pd.read_csv(args.points)
    else:
        points = autogen_points(input_seg, args.count, args.roi, args.body, args.tbars, args.skeleton)
        output_path = 'neighborhoods-from'

        if not any([args.roi, args.body, args.tbars, args.skeleton]):
            output_path += input_seg[2]
        else:
            if args.roi:
                output_path += f'-{args.roi}'
            if args.body:
                output_path += f'-{args.body}'
            if args.tbars:
                output_path += '-tbars'
            if args.skeleton:
                output_path += '-skeleton'

        assignment_path = output_path + '.json'
        csv_path = output_path + '.csv'

    distances = scipy.spatial.distance.cdist(points[[*'zyx']].values, points[[*'zyx']].values)
    distances[np.eye(len(points)).astype(bool)] = np.inf
    if (distances < radius).any():
        msg = ("Some of the chosen points are closer to each other than the "
               f"configured radius ({radius}). Their neighborhood segments may "
               "be mangled in the output.")
        logger.warning(msg)

    cols = [*'xyz'] + list({*points.columns} - {*'xyz'})
    points = points[cols]

    if args.generate_points_only:
        add_link_col(points, config)
        export_as_html(points, csv_path)
        if not args.ng_links:
            del points['link']
            points.to_csv(csv_path, index=False, header=True, quoting=csv.QUOTE_NONE)
        sys.exit(0)

    try:
        input_info = fetch_instance_info(*input_seg)
    except Exception:
        sys.exit(f"Couldn't find input segmentation instance: {' / '.join(input_seg)}")

    try:
        fetch_instance_info(*output_seg)
    except Exception:
        logger.info(f"Output labelmap not found. Creating new label instance: {' / '.join(output_seg)}")

        # Copy details from input instance.
        # But only provide a single value for each, even though the info provides three.
        # Otherwise, DVID kicks back errors like this:
        # Setting for 'VoxelUnits' was not a string: [nanometers nanometers nanometers]
        settings = {
            'block_size': input_info['Extended']['BlockSize'][0],
            'voxel_size': input_info['Extended']['VoxelSize'][0],
            'voxel_units': input_info['Extended']['VoxelUnits'][0],
            'max_scale': input_info['Extended']['MaxDownresLevel']
        }
        create_labelmap_instance(*output_seg, **settings)

        # Also create keyvalue for meshes
        create_instance(*output_seg[:2], output_seg[2] + '_meshes', 'keyvalue')

    results_df = write_point_neighborhoods(input_seg, output_seg, points, radius, args.body)

    add_link_col(results_df, config)
    export_as_html(results_df, csv_path)
    write_assignment_file(output_seg, results_df, assignment_path)
    if not args.ng_links:
        del results_df['link']
    results_df.to_csv(csv_path, index=False, header=True, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        # sys.argv += ['-g', '--ng-links', '-c=100', '/tmp/neighborhood-config.yaml']
        # sys.argv += ['-g', '--ng-links', '-c=100', '--roi=FB', '/tmp/neighborhood-config.yaml']
        # sys.argv += ['-g', '--ng-links', '-c=100', '--body=1071121755', '/tmp/neighborhood-config.yaml']
        # sys.argv += ['-g', '--ng-links', '-c=100', '--body=1071121755', '--tbars', '/tmp/neighborhood-config.yaml']
        # sys.argv += ['-g', '--ng-links', '-c=100', '--roi=FB', '--body=1071121755', '/tmp/neighborhood-config.yaml']
        # sys.argv += ['-g', '--ng-links', '-c=100', '--roi=FB', '--body=1071121755', '--skeleton', '/tmp/neighborhood-config.yaml']
        sys.argv += ['--ng-links', '-c=3', '--roi=FB', '--body=1071121755', '--skeleton', '/tmp/neighborhood-config.yaml']

    main()
