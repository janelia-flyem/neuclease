#!/usr/bin/env python3
"""
Virtual N5 Server for Neuroglancer Precomputed Volumes

A Flask-based HTTP service that reads from a neuroglancer precomputed volume
on disk and serves it as a virtual N5 volume for neuroglancer visualization.

Usage:
    precomputed_virtual_n5_server /path/to/precomputed --port 8000

Then open in neuroglancer with source: n5://http://localhost:8000

With an optional mapping file (feather format):
    precomputed_virtual_n5_server /path/to/precomputed --mapping mapping.feather

The mapping file should have a 'sv' or 'body' column as the source IDs,
and one or more other columns as target IDs. Then use URLs like:
    n5://http://localhost:8000/<target_col>              # body -> target_col
    n5://http://localhost:8000/body/<target_col>         # body -> target_col (explicit)
    n5://http://localhost:8000/sv/<target_col>           # sv -> target_col
"""
import argparse
import json
import logging
from http import HTTPStatus
from pathlib import Path

import numpy as np
import pandas as pd
import numcodecs
import tensorstore as ts
from flask import Flask, jsonify
from flask_cors import CORS
from zarr.n5 import N5ChunkWrapper

from dvidutils import LabelMapper

logger = logging.getLogger(__name__)

# Global configuration, set during startup
PRECOMPUTED_CONFIG = None
CHUNK_ENCODER = None
MAPPERS = None  # Dict of {(source_col, target_col): LabelMapper}
STORES = None  # Dict of {scale: tensorstore}

BLOCK_SIZE = 64  # Fixed block size for N5 output


def create_app(precomputed_config, mappers=None):
    """
    Create and configure the Flask application.

    Args:
        precomputed_config: dict with keys:
            - path: path to precomputed volume
            - volume_box_xyz: np.array [[x0,y0,z0], [x1,y1,z1]]
            - voxel_size: list [x, y, z] resolution
            - voxel_units: str, e.g. 'nm'
            - max_scale: int, maximum downsampling level
            - dtype: numpy dtype of the volume
        mappers: optional dict of {(source_col, target_col): LabelMapper}
    """
    global PRECOMPUTED_CONFIG, CHUNK_ENCODER, MAPPERS, STORES
    PRECOMPUTED_CONFIG = precomputed_config
    MAPPERS = mappers

    # Open tensorstore for each scale
    STORES = {}
    for scale in range(precomputed_config['max_scale'] + 1):
        STORES[scale] = ts.open({
            "driver": "neuroglancer_precomputed",
            "kvstore": {
                "driver": "file",
                "path": precomputed_config['path'],
            },
            "scale_index": scale
        }).result()
        logger.info(f"Opened tensorstore for scale {scale}: shape={STORES[scale].shape}")

    # Create N5 chunk encoder for uint64 labelmap data
    block_shape = np.array([BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE])
    CHUNK_ENCODER = N5ChunkWrapper(precomputed_config['dtype'], block_shape, compressor=numcodecs.GZip())

    app = Flask(__name__)
    CORS(app)

    def _get_top_level_attributes():
        """Return top-level N5 attributes describing the volume."""
        scales_info = PRECOMPUTED_CONFIG['scales']
        base_resolution = scales_info[0]['resolution']
        unit = PRECOMPUTED_CONFIG['voxel_units']

        # Compute scale factors relative to base resolution
        scales = []
        for scale_info in scales_info:
            res = scale_info['resolution']
            scale_factor = [int(res[i] / base_resolution[i]) for i in range(3)]
            scales.append(scale_factor)

        attr = {
            "pixelResolution": {
                "dimensions": base_resolution,
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
        if scale > PRECOMPUTED_CONFIG['max_scale']:
            return jsonify({"error": f"Scale {scale} exceeds max scale {PRECOMPUTED_CONFIG['max_scale']}"}), HTTPStatus.NOT_FOUND

        scales_info = PRECOMPUTED_CONFIG['scales']
        scale_info = scales_info[scale]
        base_resolution = scales_info[0]['resolution']
        unit = PRECOMPUTED_CONFIG['voxel_units']

        # Get actual dimensions and resolution for this scale
        dimensions = scale_info['size']  # [x, y, z]
        resolution = scale_info['resolution']

        # Compute scale factor relative to base resolution
        scale_factor = [int(resolution[i] / base_resolution[i]) for i in range(3)]

        # Get voxel offset if present
        voxel_offset = scale_info.get('voxel_offset', [0, 0, 0])
        translate = [float(voxel_offset[i] * resolution[i]) for i in range(3)]

        # Determine N5 datatype string
        dtype = PRECOMPUTED_CONFIG['dtype']
        dtype_str = str(np.dtype(dtype))

        attr = {
            "transform": {
                "ordering": "C",
                "axes": ["x", "y", "z"],
                "scale": scale_factor,
                "units": [unit, unit, unit],
                "translate": translate
            },
            "compression": {
                "type": "gzip",
                "useZlib": False,
                "level": -1
            },
            "blockSize": [BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
            "dataType": dtype_str,
            "dimensions": dimensions
        }
        return jsonify(attr), HTTPStatus.OK

    def _serve_chunk(scale, chunk_x, chunk_y, chunk_z, supervoxels, mapping_col=None):
        """
        Serve a single chunk at the requested scale and location.

        Args:
            scale: downsampling scale level
            chunk_x, chunk_y, chunk_z: chunk coordinates
            supervoxels: if True, use 'sv' column for mapping source; otherwise use 'body'
            mapping_col: if provided, apply mapping to this target column
        """
        if scale > PRECOMPUTED_CONFIG['max_scale']:
            return jsonify({"error": f"Scale {scale} exceeds max scale"}), HTTPStatus.NOT_FOUND

        # Compute the bounding box for this chunk in X,Y,Z coordinates (at this scale)
        corner_xyz = np.array([chunk_x, chunk_y, chunk_z]) * BLOCK_SIZE
        box_xyz = np.array([corner_xyz, corner_xyz + BLOCK_SIZE])

        # Get volume bounds at this scale from the scale-specific metadata
        scale_info = PRECOMPUTED_CONFIG['scales'][scale]
        scale_size_xyz = np.array(scale_info['size'])
        scaled_volume_box_xyz = np.array([[0, 0, 0], scale_size_xyz])

        # Check if chunk is completely outside volume
        if (box_xyz[0] >= scaled_volume_box_xyz[1]).any() or (box_xyz[1] <= scaled_volume_box_xyz[0]).any():
            # Return empty chunk
            empty_block = np.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype=PRECOMPUTED_CONFIG['dtype'])
            return (
                CHUNK_ENCODER.encode(empty_block),
                HTTPStatus.OK,
                {'Content-Type': 'application/octet-stream'}
            )

        # Clip box to volume bounds
        clipped_box_xyz = np.array([
            np.maximum(box_xyz[0], scaled_volume_box_xyz[0]),
            np.minimum(box_xyz[1], scaled_volume_box_xyz[1])
        ])

        # Convert to Z,Y,X for tensorstore (it uses .T which gives ZYX indexing)
        clipped_box_zyx = clipped_box_xyz[:, ::-1]

        # Fetch data from tensorstore
        try:
            store = STORES[scale]
            block_data_zyx = store.T[
                clipped_box_zyx[0, 0]:clipped_box_zyx[1, 0],
                clipped_box_zyx[0, 1]:clipped_box_zyx[1, 1],
                clipped_box_zyx[0, 2]:clipped_box_zyx[1, 2]
            ].read().result()
            block_data_zyx = np.asarray(block_data_zyx)
        except Exception as e:
            logger.error(f"Failed to fetch chunk s{scale}/{chunk_x}/{chunk_y}/{chunk_z}: {e}")
            return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

        # Create full-size block and place fetched data into it
        # (handles edge chunks that are partially outside the volume)
        block_vol_zyx = np.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype=PRECOMPUTED_CONFIG['dtype'])

        # Compute where in the block to place the data
        box_zyx = box_xyz[:, ::-1]
        rel_start = clipped_box_zyx[0] - box_zyx[0]
        rel_stop = rel_start + (clipped_box_zyx[1] - clipped_box_zyx[0])

        block_vol_zyx[
            rel_start[0]:rel_stop[0],
            rel_start[1]:rel_stop[1],
            rel_start[2]:rel_stop[2]
        ] = block_data_zyx

        # Apply mapping if requested
        if mapping_col is not None:
            block_vol_zyx = _apply_mapping(block_vol_zyx, supervoxels, mapping_col)

        # Encode to N5 chunk format (header + compressed data)
        encoded = CHUNK_ENCODER.encode(block_vol_zyx)

        return (
            encoded,
            HTTPStatus.OK,
            {'Content-Type': 'application/octet-stream'}
        )

    # Routes for default IDs (body)
    @app.route('/attributes.json')
    def top_level_attributes():
        return _get_top_level_attributes()

    @app.route("/s<int:scale>/attributes.json")
    def scale_attributes(scale):
        return _get_scale_attributes(scale)

    @app.route("/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>")
    def serve_chunk(scale, chunk_x, chunk_y, chunk_z):
        return _serve_chunk(scale, chunk_x, chunk_y, chunk_z, supervoxels=False)

    # Routes for supervoxel source (via /sv prefix) - no mapping, just indicates source type
    @app.route('/sv/attributes.json')
    def top_level_attributes_sv():
        return _get_top_level_attributes()

    @app.route("/sv/s<int:scale>/attributes.json")
    def scale_attributes_sv(scale):
        return _get_scale_attributes(scale)

    @app.route("/sv/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>")
    def serve_chunk_sv(scale, chunk_x, chunk_y, chunk_z):
        return _serve_chunk(scale, chunk_x, chunk_y, chunk_z, supervoxels=True)

    # Routes for mapped IDs (requires --mapping file)
    # /<target_col>/... implies body -> target_col
    @app.route('/<target_col>/attributes.json')
    def top_level_attributes_mapped(target_col):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _get_top_level_attributes()

    @app.route("/<target_col>/s<int:scale>/attributes.json")
    def scale_attributes_mapped(target_col, scale):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _get_scale_attributes(scale)

    @app.route("/<target_col>/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>")
    def serve_chunk_mapped(target_col, scale, chunk_x, chunk_y, chunk_z):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _serve_chunk(scale, chunk_x, chunk_y, chunk_z, supervoxels=False, mapping_col=target_col)

    # /body/<target_col>/... explicit body -> target_col
    @app.route('/body/<target_col>/attributes.json')
    def top_level_attributes_body_mapped(target_col):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _get_top_level_attributes()

    @app.route("/body/<target_col>/s<int:scale>/attributes.json")
    def scale_attributes_body_mapped(target_col, scale):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _get_scale_attributes(scale)

    @app.route("/body/<target_col>/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>")
    def serve_chunk_body_mapped(target_col, scale, chunk_x, chunk_y, chunk_z):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _serve_chunk(scale, chunk_x, chunk_y, chunk_z, supervoxels=False, mapping_col=target_col)

    # /supervoxels/<target_col>/... sv -> target_col
    @app.route('/supervoxels/<target_col>/attributes.json')
    def top_level_attributes_sv_mapped(target_col):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _get_top_level_attributes()

    @app.route("/supervoxels/<target_col>/s<int:scale>/attributes.json")
    def scale_attributes_sv_mapped(target_col, scale):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _get_scale_attributes(scale)

    @app.route("/supervoxels/<target_col>/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>")
    def serve_chunk_sv_mapped(target_col, scale, chunk_x, chunk_y, chunk_z):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _serve_chunk(scale, chunk_x, chunk_y, chunk_z, supervoxels=True, mapping_col=target_col)

    # /sv/<target_col>/... alias for /supervoxels/<target_col>/...
    @app.route('/sv/<target_col>/attributes.json')
    def top_level_attributes_sv_mapped_short(target_col):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _get_top_level_attributes()

    @app.route("/sv/<target_col>/s<int:scale>/attributes.json")
    def scale_attributes_sv_mapped_short(target_col, scale):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _get_scale_attributes(scale)

    @app.route("/sv/<target_col>/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>")
    def serve_chunk_sv_mapped_short(target_col, scale, chunk_x, chunk_y, chunk_z):
        if not _validate_mapping_col(target_col):
            return jsonify({"error": f"Unknown mapping column: {target_col}"}), HTTPStatus.NOT_FOUND
        return _serve_chunk(scale, chunk_x, chunk_y, chunk_z, supervoxels=True, mapping_col=target_col)

    return app


def _validate_mapping_col(target_col):
    """Check if the target column exists in the mapping data."""
    if MAPPERS is None:
        return False
    # Check if any mapper has this target_col
    return any(t == target_col for (_, t) in MAPPERS.keys())


def _apply_mapping(data, supervoxels, target_col):
    """
    Apply ID mapping to the data array.

    Args:
        data: numpy array of IDs
        supervoxels: if True, use 'sv' column as source; otherwise use 'body'
        target_col: name of the target column in the mapping

    Returns:
        numpy array with mapped IDs (unmapped IDs pass through unchanged)
    """
    if MAPPERS is None:
        return data

    # Determine source column
    source_col = 'sv' if supervoxels else 'body'
    key = (source_col, target_col)

    if key not in MAPPERS:
        logger.warning(f"No mapper for {source_col} -> {target_col}, returning unmapped")
        return data

    mapper = MAPPERS[key]
    return mapper.apply(data, allow_unmapped=True)


def load_precomputed_metadata(path):
    """
    Load metadata from a neuroglancer precomputed volume.

    Args:
        path: path to the precomputed volume directory

    Returns:
        dict with path, scales (full info per scale), voxel_units, max_scale, dtype
    """
    info_path = Path(path) / "info"
    with open(info_path, 'r') as f:
        info = json.load(f)

    # Get scales info - store the full array for per-scale metadata
    scales = info['scales']
    max_scale = len(scales) - 1

    # Get data type
    data_type = info.get('data_type', 'uint64')
    dtype = np.dtype(data_type)

    # Units - precomputed typically uses nm
    voxel_units = 'nm'

    return {
        'path': str(path),
        'scales': scales,  # Full per-scale metadata
        'voxel_units': voxel_units,
        'max_scale': max_scale,
        'dtype': dtype
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Virtual N5 server for neuroglancer precomputed volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'path',
        help="Path to neuroglancer precomputed volume directory"
    )
    parser.add_argument(
        '-p', '--port', type=int, default=8000,
        help="Port to serve on (default: 8000)"
    )
    parser.add_argument(
        '--max-scale', type=int, default=None,
        help="Maximum scale level (default: auto-detect from info)"
    )
    parser.add_argument(
        '--mapping', type=str, default=None,
        help="Path to feather file with ID mapping. Should have 'sv' or 'body' column "
             "as source IDs and other columns as target IDs."
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

    # Load metadata from precomputed volume
    logger.info(f"Loading metadata from {args.path}")
    config = load_precomputed_metadata(args.path)

    # Override max_scale if specified
    if args.max_scale is not None:
        config['max_scale'] = args.max_scale

    # Log scale info
    base_scale = config['scales'][0]
    logger.info(f"Base resolution: {base_scale['resolution']} {config['voxel_units']}")
    logger.info(f"Base volume size (XYZ): {base_scale['size']}")
    logger.info(f"Number of scales: {config['max_scale'] + 1}")
    logger.info(f"Data type: {config['dtype']}")
    for i, scale_info in enumerate(config['scales']):
        logger.info(f"  Scale {i}: resolution={scale_info['resolution']}, size={scale_info['size']}")

    # Load mapping file if provided
    mappers = None
    if args.mapping:
        logger.info(f"Loading mapping from {args.mapping}")
        mapping_df = pd.read_feather(args.mapping)
        logger.info(f"Mapping columns: {list(mapping_df.columns)}")
        logger.info(f"Mapping rows: {len(mapping_df)}")

        # Validate that we have a source column
        source_cols = []
        if 'sv' in mapping_df.columns:
            source_cols.append('sv')
        if 'body' in mapping_df.columns:
            source_cols.append('body')

        if not source_cols:
            raise ValueError("Mapping file must have 'sv' or 'body' column as source IDs")

        # List available target columns
        target_cols = [c for c in mapping_df.columns if c not in ('sv', 'body')]
        logger.info(f"Source columns: {source_cols}")
        logger.info(f"Target columns: {target_cols}")

        # Build LabelMapper for each (source, target) combination
        mappers = {}
        for source_col in source_cols:
            source_ids = mapping_df[source_col].values.astype(np.uint64)
            for target_col in target_cols:
                target_ids = mapping_df[target_col].values.astype(np.uint64)
                mappers[(source_col, target_col)] = LabelMapper(source_ids, target_ids)
                logger.info(f"Built mapper: {source_col} -> {target_col}")

    # Create and run app
    app = create_app(config, mappers)

    logger.info(f"Starting server on port {args.port}")
    logger.info(f"Open in neuroglancer with source: n5://http://localhost:{args.port}")
    logger.info(f"  For supervoxel source, use: n5://http://localhost:{args.port}/sv")
    if mappers is not None:
        logger.info(f"  For mapped IDs, use: n5://http://localhost:{args.port}/<target_col>")
        logger.info(f"  Or: n5://http://localhost:{args.port}/body/<target_col>")
        logger.info(f"  Or: n5://http://localhost:{args.port}/sv/<target_col>")

    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug or debug_mode,
        threaded=not (args.debug or debug_mode),
        use_reloader=args.debug or debug_mode
    )


if __name__ == "__main__":
    main()
