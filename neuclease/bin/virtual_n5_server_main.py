#!/usr/bin/env python3
"""
Virtual N5 Server for DVID Labelmap

A Flask-based HTTP service that exposes DVID labelmap data as a virtual N5 volume
for neuroglancer visualization. Data is fetched on-demand from DVID and served
in N5 format without storing anything on disk.

Usage:
    dvid_virtual_n5_server emdata3:8900 abc123 segmentation --port 8000

Then open in neuroglancer with source:
    n5://http://localhost:8000        # body IDs (default)
    n5://http://localhost:8000/sv     # supervoxel IDs
"""
import argparse
import logging
from http import HTTPStatus

import numpy as np
import numcodecs
from flask import Flask, jsonify
from flask_cors import CORS
from zarr.n5 import N5ChunkWrapper

from neuclease.dvid.node import fetch_instance_info
from neuclease.dvid.labelmap import fetch_labelmap_voxels

logger = logging.getLogger(__name__)

# Global configuration, set during startup
DVID_CONFIG = None
CHUNK_ENCODER = None

BLOCK_SIZE = 64  # Fixed to match DVID's internal block size


def create_app(dvid_config):
    """
    Create and configure the Flask application.

    Args:
        dvid_config: dict with keys:
            - server: DVID server address
            - uuid: DVID UUID
            - instance: labelmap instance name
            - supervoxels: bool, whether to fetch supervoxel IDs
            - volume_box_xyz: np.array [[x0,y0,z0], [x1,y1,z1]]
            - voxel_size: list [x, y, z] resolution
            - voxel_units: str, e.g. 'nanometers'
            - max_scale: int, maximum downsampling level
    """
    global DVID_CONFIG, CHUNK_ENCODER
    DVID_CONFIG = dvid_config

    # Create N5 chunk encoder for uint64 labelmap data
    block_shape = np.array([BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE])
    CHUNK_ENCODER = N5ChunkWrapper(np.uint64, block_shape, compressor=numcodecs.GZip())

    app = Flask(__name__)
    CORS(app)

    def _get_top_level_attributes():
        """Return top-level N5 attributes describing the volume."""
        max_scale = DVID_CONFIG['max_scale']
        voxel_size = DVID_CONFIG['voxel_size']

        # N5 uses abbreviated units
        unit = _abbreviate_units(DVID_CONFIG['voxel_units'])

        scales = [[2**s, 2**s, 2**s] for s in range(max_scale + 1)]
        attr = {
            "pixelResolution": {
                "dimensions": voxel_size,
                "unit": unit
            },
            "ordering": "C",
            "scales": scales,
            "axes": ["x", "y", "z"],
            "units": [unit, unit, unit],
            "translate": [0, 0, 0]
        }
        return jsonify(attr), HTTPStatus.OK

    def _get_scale_attributes(scale):
        """Return attributes for a specific scale level."""
        if scale > DVID_CONFIG['max_scale']:
            return jsonify({"error": f"Scale {scale} exceeds max scale {DVID_CONFIG['max_scale']}"}), HTTPStatus.NOT_FOUND

        voxel_size = DVID_CONFIG['voxel_size']
        unit = _abbreviate_units(DVID_CONFIG['voxel_units'])

        # Compute dimensions at this scale (in X,Y,Z order)
        volume_box_xyz = DVID_CONFIG['volume_box_xyz']
        volume_shape_xyz = volume_box_xyz[1] - volume_box_xyz[0]
        scaled_shape = (volume_shape_xyz // (2 ** scale)).tolist()

        attr = {
            "transform": {
                "ordering": "C",
                "axes": ["x", "y", "z"],
                "scale": [2**scale, 2**scale, 2**scale],
                "units": [unit, unit, unit],
                "translate": [0.0, 0.0, 0.0]
            },
            "compression": {
                "type": "gzip",
                "useZlib": False,
                "level": -1
            },
            "blockSize": [BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
            "dataType": "uint64",
            "dimensions": scaled_shape
        }
        return jsonify(attr), HTTPStatus.OK

    def _serve_chunk(scale, chunk_x, chunk_y, chunk_z, supervoxels):
        """
        Serve a single chunk at the requested scale and location.

        The chunk is fetched from DVID and encoded in N5 format.
        """
        if scale > DVID_CONFIG['max_scale']:
            return jsonify({"error": f"Scale {scale} exceeds max scale"}), HTTPStatus.NOT_FOUND

        # Compute the bounding box for this chunk in X,Y,Z coordinates (at this scale)
        corner_xyz = np.array([chunk_x, chunk_y, chunk_z]) * BLOCK_SIZE
        box_xyz = np.array([corner_xyz, corner_xyz + BLOCK_SIZE])

        # Convert to DVID's Z,Y,X order
        box_zyx = box_xyz[:, ::-1]

        # Clip to volume bounds (at this scale)
        volume_box_xyz = DVID_CONFIG['volume_box_xyz']
        scaled_volume_box_xyz = volume_box_xyz // (2 ** scale)
        scaled_volume_box_zyx = scaled_volume_box_xyz[:, ::-1]

        # Check if chunk is completely outside volume
        if (box_zyx[0] >= scaled_volume_box_zyx[1]).any() or (box_zyx[1] <= scaled_volume_box_zyx[0]).any():
            # Return empty chunk
            empty_block = np.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype=np.uint64)
            return (
                CHUNK_ENCODER.encode(empty_block),
                HTTPStatus.OK,
                {'Content-Type': 'application/octet-stream'}
            )

        # Clip box to volume bounds
        clipped_box_zyx = np.array([
            np.maximum(box_zyx[0], scaled_volume_box_zyx[0]),
            np.minimum(box_zyx[1], scaled_volume_box_zyx[1])
        ])

        # Fetch data from DVID
        try:
            block_data_zyx = fetch_labelmap_voxels(
                DVID_CONFIG['server'],
                DVID_CONFIG['uuid'],
                DVID_CONFIG['instance'],
                clipped_box_zyx,
                scale=scale,
                supervoxels=supervoxels
            )
        except Exception as e:
            logger.error(f"Failed to fetch chunk s{scale}/{chunk_x}/{chunk_y}/{chunk_z}: {e}")
            return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

        # Create full-size block and place fetched data into it
        # (handles edge chunks that are partially outside the volume)
        block_vol_zyx = np.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype=np.uint64)

        # Compute where in the block to place the data
        rel_start = clipped_box_zyx[0] - box_zyx[0]
        rel_stop = rel_start + (clipped_box_zyx[1] - clipped_box_zyx[0])

        block_vol_zyx[
            rel_start[0]:rel_stop[0],
            rel_start[1]:rel_stop[1],
            rel_start[2]:rel_stop[2]
        ] = block_data_zyx

        # Encode to N5 chunk format (header + compressed data)
        # N5ChunkWrapper expects data in C-order with shape reversed from BLOCK_SHAPE.
        # BLOCK_SHAPE is [X, Y, Z], so encoder expects shape [Z, Y, X] which is
        # exactly what DVID gives us. No transpose needed.
        encoded = CHUNK_ENCODER.encode(block_vol_zyx)

        return (
            encoded,
            HTTPStatus.OK,
            {'Content-Type': 'application/octet-stream'}
        )

    # Routes for body IDs (default)
    @app.route('/attributes.json')
    def top_level_attributes():
        return _get_top_level_attributes()

    @app.route("/s<int:scale>/attributes.json")
    def scale_attributes(scale):
        return _get_scale_attributes(scale)

    @app.route("/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>")
    def serve_chunk(scale, chunk_x, chunk_y, chunk_z):
        return _serve_chunk(scale, chunk_x, chunk_y, chunk_z, supervoxels=False)

    # Routes for supervoxel IDs (via /sv prefix)
    @app.route('/sv/attributes.json')
    def top_level_attributes_sv():
        return _get_top_level_attributes()

    @app.route("/sv/s<int:scale>/attributes.json")
    def scale_attributes_sv(scale):
        return _get_scale_attributes(scale)

    @app.route("/sv/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>")
    def serve_chunk_sv(scale, chunk_x, chunk_y, chunk_z):
        return _serve_chunk(scale, chunk_x, chunk_y, chunk_z, supervoxels=True)

    return app


def _abbreviate_units(units):
    """Convert DVID unit names to abbreviated form for N5."""
    unit_map = {
        'nanometers': 'nm',
        'nanometer': 'nm',
        'micrometers': 'um',
        'micrometer': 'um',
        'microns': 'um',
        'micron': 'um',
        'millimeters': 'mm',
        'millimeter': 'mm',
    }
    # DVID may return units as a list (one per dimension) or a string
    if isinstance(units, list):
        units = units[0] if units else 'nm'
    return unit_map.get(units.lower(), units)


def fetch_dvid_metadata(server, uuid, instance):
    """
    Fetch instance metadata from DVID.

    Returns:
        dict with volume_box_xyz, voxel_size, voxel_units, max_scale
    """
    info = fetch_instance_info(server, uuid, instance)

    # Volume bounds (DVID returns in X,Y,Z order)
    min_point = info["Extended"].get("MinPoint")
    max_point = info["Extended"].get("MaxPoint")

    if min_point is None or max_point is None:
        raise ValueError(f"Instance {instance} has no data (MinPoint/MaxPoint is null)")

    # MaxPoint is inclusive in DVID, so add 1 to get exclusive bounds
    volume_box_xyz = np.array([min_point, [m + 1 for m in max_point]])

    # Voxel resolution
    voxel_size = info["Extended"].get("VoxelSize", [8.0, 8.0, 8.0])
    if isinstance(voxel_size, (int, float)):
        voxel_size = [voxel_size, voxel_size, voxel_size]

    voxel_units = info["Extended"].get("VoxelUnits", "nanometers")

    # Max scale (downsampling level)
    max_scale = int(info["Extended"].get("MaxDownresLevel", 0))

    return {
        'volume_box_xyz': volume_box_xyz,
        'voxel_size': list(voxel_size),
        'voxel_units': voxel_units,
        'max_scale': max_scale
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Virtual N5 server for DVID labelmap data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'dvid_server',
        help="DVID server address, e.g. 'emdata3:8900'"
    )
    parser.add_argument(
        'uuid',
        help="DVID UUID"
    )
    parser.add_argument(
        'instance',
        help="Labelmap instance name, e.g. 'segmentation'"
    )
    parser.add_argument(
        '-p', '--port', type=int, default=8000,
        help="Port to serve on (default: 8000)"
    )
    parser.add_argument(
        '--max-scale', type=int, default=None,
        help="Maximum scale level (default: auto-detect from DVID)"
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help="Run in debug mode"
    )
    return parser.parse_args()


def main(debug_mode=False):
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )

    # Fetch metadata from DVID
    logger.info(f"Fetching metadata from {args.dvid_server}/{args.uuid}/{args.instance}")
    metadata = fetch_dvid_metadata(args.dvid_server, args.uuid, args.instance)

    # Override max_scale if specified
    if args.max_scale is not None:
        metadata['max_scale'] = args.max_scale

    # Build config
    dvid_config = {
        'server': args.dvid_server,
        'uuid': args.uuid,
        'instance': args.instance,
        **metadata
    }

    logger.info(f"Volume bounds (XYZ): {dvid_config['volume_box_xyz'].tolist()}")
    logger.info(f"Voxel size: {dvid_config['voxel_size']} {dvid_config['voxel_units']}")
    logger.info(f"Max scale: {dvid_config['max_scale']}")

    # Create and run app
    app = create_app(dvid_config)

    logger.info(f"Starting server on port {args.port}")
    logger.info(f"Open in neuroglancer with source: n5://http://localhost:{args.port}")
    logger.info(f"  For supervoxel IDs, use: n5://http://localhost:{args.port}/sv")

    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug or debug_mode,
        threaded=not (args.debug or debug_mode),
        use_reloader=args.debug or debug_mode
    )


if __name__ == "__main__":
    main()
