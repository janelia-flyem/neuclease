#!/bin/env /python3
"""
For each body in a list of bodies, fetch the supervoxel meshes for the body,
combine them into a single mesh, decimate the mesh, and write the mesh to
an output file or DVID keyvalue instance (or both).
"""
import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd

from vol2mesh import Mesh

from neuclease import configure_default_logging
from neuclease.util import Timer, read_csv_header, read_csv_col, tqdm_proxy
from neuclease.dvid import fetch_tarfile, post_key

logger = logging.getLogger(__name__)


def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fraction', type=float,
                        help='Fraction of vertices to retain in the decimated mesh.  Between 0.0 and 1.0')
    parser.add_argument('--format',
                        help='Either obj or drc', required=True)
    parser.add_argument('--output-directory', '-d',
                        help='Directory to dump decimated meshes.')
    parser.add_argument('--output-url', '-u',
                        help='DVID keyvalue instance to write decimated mesh files to, '
                        'specified as a complete URL, e.g. http://emdata1:8000/api/node/123abc/my-meshes')
    parser.add_argument('server', help='dvid server, e.g. emdata3:8900')
    parser.add_argument('uuid', help='dvid node')
    parser.add_argument('tsv_instance', help='name of a tarsupervoxels instance, e.g. segmentation_sv_meshes')    
    parser.add_argument('bodies', nargs='+',
                        help='A list of body IDs OR a path to a CSV containing a column named "body", which will be read.\n'
                             'If no "body" column exists, the first column is used, regardless of the name.')

    args = parser.parse_args()

    if args.fraction is None:
        raise RuntimeError("Please specify a decimation fraction.")

    if args.format is None:
        raise RuntimeError("Please specify an output format (either 'drc' or 'obj' via --format")

    if args.output_directory:
        os.makedirs(args.output_directory, exist_ok=True)
            
    output_dvid = None    
    if args.output_url:
        if '/api/node' not in args.output_url:
            raise RuntimeError("Please specify the output instance as a complete URL, "
                               "e.g. http://emdata1:8000/api/node/123abc/my-meshes")
        
        # drop 'http://' (if present)
        url = args.output_url.split('://')[-1]
        parts = url.split('/')
        assert parts[1] == 'api'
        assert parts[2] == 'node'
        
        output_server = parts[0]
        output_uuid = parts[3]
        output_instance = parts[4]
        
        output_dvid = (output_server, output_uuid, output_instance)


    all_bodies = []
    for body in args.bodies:
        if body.endswith('.csv'):
            if 'body' in read_csv_header(body):
                bodies = pd.read_csv(body)['body'].drop_duplicates()
            else:
                # Just read the first column, no matter what it's named
                bodies = read_csv_col(body, 0, np.uint64).drop_duplicates()
        else:
            try:
                body = int(body)
            except ValueError:
                raise RuntimeError(f"Invalid body ID: '{body}'")
        
        all_bodies.extend(bodies)

    for body_id in tqdm_proxy(all_bodies):
        output_path = None
        if args.output_directory:
            output_path = f'{args.output_directory}/{body_id}.{args.format}'

        decimate_existing_mesh(args.server, args.uuid, args.tsv_instance, body_id, args.fraction, args.format, output_path, output_dvid)


def decimate_existing_mesh(server, uuid, instance, body_id, fraction, output_format=None, output_path=None, output_dvid=None, tar_bytes=None):
    """
    Fetch all supervoxel meshes for the given body, combine them into a
    single mesh, and then decimate that mesh at the specified fraction.
    The output will be written to a file, or to a dvid instance (or both).
    
    Args:
        tar_bytes:
            Optional. You can provide the tarfile contents (as bytes) directly,
            in which case the input server will not be used.
    """
    if output_path is not None:
        fmt = os.path.splitext(output_path)[1][1:]
        if output_format is not None and output_format != fmt:
            raise RuntimeError(f"Mismatch between output format '{output_format}'"
                               f" and output file extension in '{output_path}'")
        output_format = fmt
    
    if output_format is None:
        raise RuntimeError("You must specify an output format (or an output path with a file extension)")

    assert output_format in ('drc', 'obj'), \
        f"Unknown output format: {output_format}"

    assert output_path is not None or output_dvid is not None, \
        "No output location specified"

    if tar_bytes is None:
        with Timer(f"Body: {body_id} Fetching tarfile", logger):
            tar_bytes = fetch_tarfile(server, uuid, instance, body_id)
    
    with Timer(f"Body: {body_id}: Loading mesh for body {body_id}", logger):
        mesh = Mesh.from_tarfile(tar_bytes, keep_normals=False)

    mesh_mb = mesh.uncompressed_size() / 1e6
    logger.info(f"Body: {body_id}: Original mesh has {len(mesh.vertices_zyx)} vertices and {len(mesh.faces)} faces ({mesh_mb:.1f} MB)")
        
    with Timer(f"Body: {body_id}: Decimating at {fraction:.2f}", logger):
        mesh.simplify(fraction, in_memory=True)

    mesh_mb = mesh.uncompressed_size() / 1e6
    logger.info(f"Body: {body_id}: Final mesh has {len(mesh.vertices_zyx)} vertices and {len(mesh.faces)} faces ({mesh_mb:.1f} MB)")

    with Timer(f"Body: {body_id}: Serializing", logger):
        mesh_bytes = None
        if output_dvid is not None:
            assert len(output_dvid) == 3
            mesh_bytes = mesh.serialize(fmt=output_format)
            post_key(*output_dvid, f"{body_id}.{output_format}", mesh_bytes)
            
        if output_path:
            if mesh_bytes is None:
                mesh.serialize(output_path)
            else:
                with open(output_path, 'wb') as f:
                    f.write(mesh_bytes)
        

if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        sys.argv += "--fraction=0.05 --format=obj --output-directory=/tmp/decimated-eb-meshes emdata4:8900 137d segmentation_sv_meshes /tmp/some-eb-bodies.csv".split()

    main()
