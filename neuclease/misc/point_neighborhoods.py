"""
This script will take a list of points and generate a "neighborhood" segment
for each point, consisting of the ball around the point, restricted to a
particular neuron. The neighborhood segments are written to a separate
labelmap instance.

Additionally, multiple files are emitted:

  - A CSV file of the points and corresponding neighborhood IDs
  - An html file with the same information, but also neuroglancer links to each point
  - An assignment file for FlyEM's neuroglancer-based protocol for inspecting such
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
import networkx as nx

from vol2mesh import Mesh
from confiddler import load_config, dump_default_config

from neuclease import configure_default_logging
from neuclease.util import round_box, box_to_slicing as b2s, tqdm_proxy, swc_to_dataframe, mask_centroid, sphere_mask
from neuclease.dvid import (post_key, determine_point_rois, fetch_sparsevol, fetch_roi_roi,
                            runlength_encode_mask_to_ranges, fetch_volume_box, fetch_instance_info,
                            fetch_annotation_label, fetch_labelmap_voxels, post_labelmap_voxels,
                            create_labelmap_instance, fetch_key, create_instance)

logger = logging.getLogger(__name__)

this_dir = os.path.dirname(__file__)
ng_settings_path = f'{this_dir}/_point_neighborhoods_ng_settings.json'
DefaultNeuroglancerSettings = json.load(open(ng_settings_path, 'r'))

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
        "enforce-minimum-distance": {
            "description": "If true, do not allow auto-generated points to fall close to\n"
                           "each other (where 'close' is defined by 2x the given radius)\n",
            "type": "boolean",
            "default": True
        },
        "random-seed": {
            "description": "For reproducible results, specify a seed (an integer) to the random number generator.\n"
                           "Otherwise, omit this setting and you'll get different points every time.\n",
            "oneOf": [
                {"type": "integer"},
                {"type": "null"}
            ],
            "default": None
        },
        "grayscale-source": {
            "description": "What neuroglancer source layer to use as the grayscale source\n"
                           "when emitting the assignment file and HTML table. (Not used in the computation.)\n",
            "type": "string",
            "default": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg"
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
                    "default": DefaultNeuroglancerSettings
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


def autogen_points(input_seg, count, roi, body, tbars, use_skeleton, random_seed=None, minimum_distance=0):
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
        if not count and minimum_distance > 0:
            sys.exit("You must supply a --count if you want skeleton point samples to respect the minimum distance.")
        if not count and not roi and minimum_distance == 0:
            logger.warning("You are using all nodes of a skeleton without any ROI filter! Is that what you meant?")

    rng = default_rng(random_seed)

    if tbars:
        logger.info(f"Fetching synapses for body {body}")
        syn_df = fetch_annotation_label(*input_seg[:2], 'synapses', body, format='pandas')
        tbars = syn_df.query('kind == "PreSyn"')[[*'zyx']]

        if roi:
            logger.info(f"Filtering tbars for roi {roi}")
            determine_point_rois(*input_seg[:2], [roi], tbars)
            tbars = tbars.query('roi == @roi')[[*'zyx']]

        if minimum_distance:
            logger.info(f"Pruning close points from {len(tbars)} total tbar points")
            tbars = prune_close_pairs(tbars, minimum_distance, rng)
            logger.info(f"After pruning, {len(tbars)} tbars remain.")

        if count:
            count = min(count, len(tbars))
            logger.info(f"Sampling {count} tbars")
            choices = rng.choice(tbars.index, size=count, replace=False)
            tbars = tbars.loc[choices]

        logger.info(f"Returning {len(tbars)} tbar points")
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

        if minimum_distance:
            assert count
            # Distance-pruning is very expensive on a huge number of close points.
            # If skeleton is large, first reduce the workload by pre-selecting a
            # random sample of skeleton points, and prune more from there.
            if len(skeleton_df) > 10_000:
                # FIXME: random_state can't use rng until I upgrade to pandas 1.0
                skeleton_df = skeleton_df.sample(min(4*count, len(skeleton_df)), random_state=None)
            logger.info(f"Pruning close points from {len(skeleton_df)} skeleton points")
            prune_close_pairs(skeleton_df, minimum_distance, rng)
            logger.info(f"After pruning, {len(skeleton_df)} skeleton points remain.")

        if count:
            count = min(count, len(skeleton_df))
            logger.info(f"Sampling {count} skeleton points")
            choices = rng.choice(skeleton_df.index, size=count, replace=False)
            skeleton_df = skeleton_df.loc[choices]

        logger.info(f"Returning {len(skeleton_df)} skeleton points")
        return skeleton_df

    elif body:
        assert count
        if roi:
            # TODO: intersect the ranges with the ROI.
            raise NotImplementedError("Sorry, I haven't yet implemented support for "
                                      "body+roi filtering.  Pick one or the other, "
                                      "or ask Stuart to fix this.")

        logger.info(f"Fetching sparsevol for body {body}")
        ranges = fetch_sparsevol(*input_seg, body, format='ranges')
        logger.info("Sampling from sparsevol")

        if minimum_distance > 0:
            # Sample 4x extra so we still have enough after pruning.
            points = sample_points_from_ranges(ranges, 4*count, rng)
        else:
            points = sample_points_from_ranges(ranges, count, rng)

        points = pd.DataFrame(points, columns=[*'zyx'])

        if minimum_distance > 0:
            logger.info(f"Pruning close points from {len(points)} body points")
            prune_close_pairs(points, minimum_distance, rng)
            logger.info(f"After pruning, {len(points)} body points remain")

        points = points.iloc[:count]
        logger.info(f"Returning {len(points)} body points")
        return points

    elif roi:
        assert count
        logger.info(f"Fetching roi {roi}")
        roi_ranges = fetch_roi_roi(*input_seg[:2], roi, format='ranges')
        logger.info("Sampling from ranges")

        if minimum_distance > 0:
            # Sample 4x extra so we can prune some out if necessary.
            points_s5 = sample_points_from_ranges(roi_ranges, 4*count, rng)
        else:
            points_s5 = sample_points_from_ranges(roi_ranges, count, rng)

        corners_s0 = points_s5 * (2**5)
        points_s0 = rng.integers(corners_s0, corners_s0 + (2**5))
        points = pd.DataFrame(points_s0, columns=[*'zyx'])

        if minimum_distance > 0:
            logger.info(f"Pruning close points from {len(points)} roi points")
            prune_close_pairs(points, minimum_distance, rng)
            logger.info(f"After pruning, points from {len(points)} roi points remain")

        points = points.iloc[:count]
        logger.info(f"Returning {len(points)} roi points")
        return points
    else:
        # No body or roi specified, just sample from the whole non-zero segmentation area
        assert count
        logger.info("Sampling random points from entire input segmentation")
        logger.info("Fetching low-res input volume")
        box_s6 = round_box(fetch_volume_box(*input_seg), 2**6, 'out') // 2**6
        seg_s6 = fetch_labelmap_voxels(*input_seg, box_s6, scale=6)
        mask_s6 = seg_s6.astype(bool)
        logger.info("Encoding segmentation as ranges")
        seg_ranges = runlength_encode_mask_to_ranges(mask_s6, box_s6)

        logger.info("Sampling from ranges")

        if minimum_distance > 0:
            # Sample 4x extra so we can prune some out if necessary.
            points_s6 = sample_points_from_ranges(seg_ranges, 4*count, rng)
        else:
            points_s6 = sample_points_from_ranges(seg_ranges, count, rng)

        corners_s0 = points_s6 * (2**6)
        points_s0 = rng.integers(corners_s0, corners_s0 + (2**6))

        points = pd.DataFrame(points_s0, columns=[*'zyx'])

        if minimum_distance > 0:
            logger.info(f"Pruning close points from {len(points)} segmentation points")
            prune_close_pairs(points, minimum_distance, rng)
            logger.info(f"After pruning, points from {len(points)} segmentation points remain")

        points = points.iloc[:count]
        logger.info(f"Returning {len(points)} segmentation points")
        return points


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
        rng = default_rng()
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


def prune_close_pairs(points, radius, rng=None):
    """
    Given a DataFrame containing points in the xyz columns,
    find points that are too close to at least one other point
    (as defined by the given radius), and delete them one at a
    time in random order until no remaining points are too close
    to each other.
    """
    if radius == 0:
        return points

    if rng is None:
        rng = default_rng()

    # Contruct a graph of point IDs, with edges between points
    # that are too close to each other.
    kd = scipy.spatial.cKDTree(points[[*'zyx']].values)
    edges = kd.query_pairs(radius, output_type='ndarray')

    g = nx.Graph()
    g.add_nodes_from(range(len(points)))
    g.add_edges_from(edges)

    # In random order, delete points that still have any edges.
    nodes = list(g.nodes())
    rng.shuffle(nodes)
    for n in nodes:
        if g.degree(n) > 0:
            g.remove_node(n)

    # Remaining nodes indicate which points to return.
    return points.iloc[sorted(g.nodes())]


def write_assignment_file(seg_dst, points, path, config):
    server, uuid, mask_instance = seg_dst
    if not server.startswith('http'):
        server = f'https://{server}'

    src_server = config["input"]["server"]
    if not src_server.startswith('http'):
        src_server = f"http://{src_server}"

    seg_src = (src_server,
               config["input"]["uuid"],
               config["input"]["instance"])

    assignment = {
        "file version": 1,
        "grayscale source": config["grayscale-source"],
        "mito ROI source": f"dvid://{server}/{uuid}/neighborhood-masks",
        "DVID source": f"{server}/#/repo/{uuid}",
        "neuron segmentation": "dvid://{}/{}/{}".format(*seg_src),
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

            # Oops, the radius was omitted from the hemibrain mito tasks.
            # Next time it will be there.
            "neighborhood radius": config["radius"],
        }
        assignment["task list"].append(task)

    with open(path, 'w') as f:
        json.dump(assignment, f, indent=2)


def update_ng_settings(config):
    """
    Edit the layer sources in the config's neuroglancer-settings
    (which were probably copied and pasted from an old link)
    to make them consistent with the input/output sources at
    the top of the config.
    """
    input_server = config["input"]["server"]
    input_uuid = config["input"]["uuid"]
    input_instance = config["input"]["instance"]

    output_server = config["output"]["server"]
    output_uuid = config["output"]["uuid"]
    output_instance = config["output"]["instance"]

    if not input_server.startswith("http"):
        input_server = f"http://{input_server}"
    if not output_server.startswith("http"):
        output_server = f"http://{output_server}"

    for layer in config["neuroglancer"]["settings"]["layers"]:
        if layer["name"] in ("emdata", "grayscale"):
            layer["source"]["url"] = config["grayscale-source"]
        elif layer["name"] == "segmentation":
            layer["source"]["url"] = f"dvid://{input_server}/{input_uuid}/{input_instance}"
        elif layer["name"] == "neighborhood-masks":
            layer["source"]["url"] = f"dvid://{output_server}/{output_uuid}/{output_instance}"

        # Replace the server with the user's input server
        # Edit: Nah, force the user specify other layers themselves, in the config.
        # else:
        #     type, protocol, path = layer["source"]["url"].split("://")
        #     if type == "dvid":
        #         server, *endpoint = path.split("/")
        #         layer["source"]["url"] = f"{type}://{input_server}/" + "/".join(endpoint)


def add_link_col(points, config):
    """
    Add a column to points dataframe containing a
    neuroglancer links for each point.
    """
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
    update_ng_settings(config)
    input_seg = [*config["input"].values()]
    output_seg = [*config["output"].values()]
    radius = config["radius"]
    random_seed = config["random-seed"]

    if config["enforce-minimum-distance"]:
        minimum_distance = 2*radius
    else:
        minimum_distance = 0

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
        points = autogen_points(input_seg, args.count, args.roi, args.body, args.tbars, args.skeleton, random_seed, minimum_distance)

        uuid = input_seg[1]
        output_path = f'neighborhoods-from-{uuid[:6]}'

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

    kd = scipy.spatial.cKDTree(points[[*'zyx']].values)
    if len(kd.query_pairs(2*radius)) > 0:
        msg = ("Some of the chosen points are closer to each other than 2x the "
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
    write_assignment_file(output_seg, results_df, assignment_path, config)
    if not args.ng_links:
        del results_df['link']
    results_df.to_csv(csv_path, index=False, header=True, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        import os
        os.chdir('/tmp')
        # sys.argv += ['-g', '--ng-links', '-c=100', '/tmp/neighborhood-config.yaml']
        # sys.argv += ['-g', '--ng-links', '-c=100', '--roi=FB', '/tmp/neighborhood-config.yaml']
        # sys.argv += ['-g', '--ng-links', '-c=100', '--body=1071121755', '/tmp/neighborhood-config.yaml']
        # sys.argv += ['-g', '--ng-links', '-c=100', '--body=1071121755', '--tbars', '/tmp/neighborhood-config.yaml']
        # sys.argv += ['-g', '--ng-links', '-c=100', '--roi=FB', '--body=1071121755', '/tmp/neighborhood-config.yaml']
        # sys.argv += ['-g', '--ng-links', '-c=100', '--roi=FB', '--body=1071121755', '--skeleton', '/tmp/neighborhood-config.yaml']

        #sys.argv += ['-c=3', '--roi=FB', '--body=1113371822', '--skeleton', '/tmp/config.yaml']
        #sys.argv += ['-c=3', '--roi=FB', '--body=1113371822', '--tbars', '/tmp/config.yaml']
        sys.argv += ['-c=20', '--roi=FB', '--body=1113371822', '--tbars', '/tmp/config.yaml']

        #sys.argv += ['-c=10', '--roi=FB', '--body=1071121755', '--skeleton', '/tmp/cloud_config.yaml']

    main()
