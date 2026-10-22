"""
Multi-resolution body meshes (neuroglancer's 'neuroglancer_multilod_draco' format),
generated chunk by chunk.

Overview
--------
The body is divided into chunks (e.g. 512^3 full-resolution voxels), which are the
LOD-0 cells of the multi-resolution mesh.  Each chunk is meshed independently (with a
halo), and decimated to a fixed (quadric) error threshold per level of detail
(see MultiresMeshConfigSchema['lod-error-thresholds']).  After each decimation, the
chunk's mesh is trimmed to the chunk's own cell, yielding one *piece* per LOD.

Decimating each chunk to a fixed error (rather than decimating the whole body to
a target face count) allocates the faces across the body about as well, and it can be
done for all chunks in parallel (e.g. by separate workers; see generate_chunk_pieces()).

The body mesh is assembled from the pieces without any further decimation:

- LOD 0's fragments are the chunks' LOD-0 pieces, verbatim.
- An LOD-k fragment (whose cell spans 2^k chunks per axis) is the concatenation of
  its chunks' LOD-k pieces.  (Neuroglancer requires that no triangle in an LOD-k
  fragment crosses the fragment's 2x2x2 mid-planes.  Since each piece is trimmed to
  its own chunk, that holds automatically.)

Coarser LODs are made by decimating the chunk mesh further, not by meshing
downsampled segmentation (which would break up thin neurites).  The cut borders of
each piece are never decimated, at any LOD.

When a body changes, only its changed chunks are regenerated.  The other chunks'
pieces are recovered from the body's previous mesh, which therefore serves as the
chunk cache.  (No chunk meshes are stored separately.)  See stale_chunks() for how
changed chunks are detected.

Storage
-------
See MultiresMeshConfigSchema['storage'].  For each body, three files are stored:

    <body>          The mesh's fragment data
    <body>.index    The mesh's manifest
    <body>.chunks   Metadata about the chunks the mesh was made from (JSON)

The mesh's 'mesh_info' (see MESH_INFO_VERSION) is stored in DVID, in the
'<segmentation>_multires_mesh_info' keyvalue instance.  If the meshes are stored in
DVID, they're stored in the '<segmentation>_multires_meshes' instance.  Use
init_multires_mesh_instances() to create those instances.
"""
import json
import time
import zlib
import logging
import datetime
from functools import partial
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import skimage.morphology
from requests import HTTPError

from confiddler import validate, flow_style
from vol2mesh.mesh import Mesh
from vol2mesh import multires as v2m_multires

from neuclease import PrefixFilter
from neuclease.util import Timer, compute_parallel
from neuclease.util.segmentation import fill_holes_in_mask

from neuclease.dvid import set_default_dvid_session_timeout
from neuclease.dvid.repo import resolve_ref
from neuclease.dvid.keyvalue import fetch_key, post_key, delete_key
from neuclease.dvid.labelmap import fetch_lastmod, fetch_sparsevol
from neuclease.misc.meshstore import open_store, PreconditionFailed
from neuclease.misc.bodymesh import (
    DummyResourceMgr, mesh_instance_names, init_mesh_instances, body_chunk_surface_table, fetch_resolution_zyx)

logger = logging.getLogger(__name__)

MultiresMeshConfigSchema = {
    "description": "Settings for multi-resolution body meshes, generated chunk by chunk.\n",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "chunk-shape-s0": {
            "description":
                "The shape of the chunks (in full-resolution voxels, XYZ order) which are meshed\n"
                "independently.  The chunks are the cells of the mesh's finest level of detail.\n"
                "Must be a multiple of 64 (DVID's block size).\n",
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3,
            "default": flow_style([512, 512, 512])
        },
        "chunk-halo": {
            "description":
                "How much halo to fetch around each chunk before meshing it, in voxels at the source-scale.\n"
                "The halo is trimmed away after the chunk is smoothed and decimated, so that\n"
                "adjacent chunks' meshes line up closely.\n",
            "type": "integer",
            "minimum": 0,
            "default": 8
        },
        "source-scale": {
            "description": "Which scale of the segmentation to mesh.\n",
            "type": "integer",
            "minimum": 0,
            "default": 2
        },
        "morphological-closing-s0": {
            "description":
                "Apply morphological closing to each chunk's mask with the given radius,\n"
                "in full-resolution voxels.  (0 means no closing.)\n",
            "type": "integer",
            "minimum": 0,
            "default": 0
        },
        "fill-holes": {
            "description":
                "If true, fill holes in each chunk's mask before meshing it.\n"
                "(Holes which touch the chunk's boundary (including its halo) are not filled.)\n",
            "type": "boolean",
            "default": True
        },
        "smoothing": {
            "description": "How many iterations of Laplacian smoothing to apply to each chunk's mesh before decimating it.\n",
            "type": "integer",
            "minimum": 0,
            "default": 2
        },
        "lod-error-thresholds": {
            "description":
                "The decimation error threshold for each level of detail, from finest to coarsest.\n"
                "The number of LODs is the length of this list.\n"
                "Each chunk's mesh is decimated until no further edge can be collapsed without\n"
                "exceeding the threshold (see vol2mesh's Mesh.simplify_to_error()).  The error is\n"
                "the (quadric) sum of squared distances to the original faces' planes, in units of\n"
                "squared full-resolution voxels.  So, roughly speaking, sqrt(threshold) is the\n"
                "typical deviation from the undecimated mesh, in full-resolution voxels.\n"
                "Each coarser LOD is decimated from the previous one, so the thresholds must increase.\n"
                "Since each LOD's cells are 8x the volume of the previous LOD's, a threshold about\n"
                "10x the previous LOD's keeps roughly the same number of faces per cell.\n"
                "(With source-scale 2, chunk-halo 8, and smoothing 2, the thresholds [8, 75] yield\n"
                "about 16% and 4% of the undecimated faces at LOD 0 and LOD 1, respectively.)\n",
            "type": "array",
            "items": {"type": "number", "minimum": 0.0},
            "minItems": 1,
            "default": flow_style([8.0, 75.0])
        },
        "vertex-quantization-bits": {
            "description":
                "The number of bits used to quantize each vertex coordinate within a fragment\n"
                "(as required by the neuroglancer_multilod_draco format).  Must be 10 or 16.\n",
            "type": "integer",
            "enum": [10, 16],
            "default": 16
        },
        "base-uuid": {
            "description":
                "The UUID just BEFORE the 'surface_mutid' property was enabled in DVID.\n"
                "For blocks whose surface_mutid is unset (0), the body's lastmod in this UUID is\n"
                "used instead, to determine whether a stored mesh's chunks are still up-to-date.\n"
                "If left unset, then chunks containing any such blocks are never considered\n"
                "up-to-date, so they're regenerated every time the body's mesh is updated.\n",
            "type": "string",
            "default": ""
        },
        "storage": {
            "description":
                "Where to store the body meshes.  Either 'dvid' (the '<segmentation>_multires_meshes'\n"
                "keyvalue instance), or an object store URL such as 'gs://my-bucket/some/prefix'.\n"
                "In an object store, the mesh for the newest state of each body (on the 'live-ref'\n"
                "branch) is stored as '<body>' and '<body>.index' (along with an 'info' file),\n"
                "i.e. a standard neuroglancer precomputed mesh source which can be read directly\n"
                "from the bucket.  Meshes for other UUIDs are stored as '<body>-<uuid>' and\n"
                "'<body>-<uuid>.index'.  When the newest mesh for a body is replaced by one for\n"
                "a newer UUID, the old one is copied to '<body>-<old-uuid>' first, so each UUID's\n"
                "mesh remains available.  (Each mesh's '.chunks' file is stored alongside it.)\n"
                "Either way, the mesh_info is stored in DVID.  (See locate_multires_body_mesh().)\n"
                "Note: The objects' names don't include the segmentation instance name,\n"
                "so use a separate prefix for each segmentation.\n",
            "type": "string",
            "default": "dvid"
        },
        "live-ref": {
            "description":
                "For meshes in an object store: the DVID branch whose newest body meshes are stored\n"
                "under the standard names ('<body>' and '<body>.index').  See 'storage'.\n",
            "type": "string",
            "default": "master"
        },
    }
}

# The config settings which determine the meshes' geometry.
# Stored meshes (and their chunks) generated with different settings are not reused.
MESH_SETTINGS = [
    'chunk-shape-s0',
    'chunk-halo',
    'source-scale',
    'morphological-closing-s0',
    'fill-holes',
    'smoothing',
    'lod-error-thresholds',
    'vertex-quantization-bits',
]

# The version of the mesh_info format.
# The fields which clients (e.g. neuroglancer) may rely on are:
#
#   mesh-info-version:          This version number.
#   format:                     "neuroglancer_multilod_draco"
#   lastmod:                    The body's mutation id when the mesh was generated.
#   written-uuid:               The UUID in which the mesh was generated (and stored).
#   storage:                    "dvid", or the URL of the object store containing the mesh.
#   vertex-quantization-bits:   As in the neuroglancer 'info' file.
#
# All other fields are informational.  See locate_multires_body_mesh() for how to find
# the mesh files.  (The mesh_info is inherited by descendant UUIDs, like any DVID data.)
#
# (Version 1 was produced by the multires support in neuclease.misc.bodymesh,
# which assembled meshes from separately stored chunk meshes.)
MESH_INFO_VERSION = 2

# The mesh_info 'method' of the meshes produced by this module.
MESH_METHOD = 'multires-chunks'

# Body meshes in an object store may be overwritten, so browsers must not cache them.
BODY_MESH_CACHE_CONTROL = 'no-cache'

# The files which comprise a stored body mesh, by their suffix.
MESH_FILE_SUFFIXES = ('', '.index', '.chunks')


def validate_config(config):
    """Validate the config (injecting defaults), and check the things the schema can't express."""
    validate(config, MultiresMeshConfigSchema, inject_defaults=True)
    if (np.array(config['chunk-shape-s0']) % 64).any():
        raise ValueError(f"chunk-shape-s0 must be a multiple of 64, not {config['chunk-shape-s0']}")
    thresholds = config['lod-error-thresholds']
    if any(b < a for a, b in zip(thresholds[:-1], thresholds[1:])):
        raise ValueError(f"lod-error-thresholds must not decrease: {thresholds}")
    return config


def mesh_settings(config):
    """The config's geometry-related settings (see MESH_SETTINGS), as stored with each mesh."""
    # (Round-tripped through JSON, to match what we read back from storage.)
    return json.loads(json.dumps({k: config[k] for k in MESH_SETTINGS}))


def init_multires_mesh_instances(server, uuid, seg_instance):
    """Create the DVID keyvalue instances for the multires body meshes and their mesh_info, if necessary."""
    init_mesh_instances(server, uuid, seg_instance, body=True, chunks=False, sv=False, multires=True)


#
# Chunks
#

def generate_chunk_pieces(server, uuid, seg_instance, body, config, chunk_zyx, resource_mgr=None):
    """
    Mesh one chunk of a body, and return its piece for each level of detail.

    The chunk's mask is fetched (with a halo), meshed, smoothed, and then decimated
    to each LOD's error threshold in turn.  After each decimation, the mesh is
    trimmed to the chunk, and encoded as that LOD's piece: a Draco fragment
    whose position is the LOD-k cell which contains the chunk (with the
    vertex quantization of an LOD-k fragment).

    This is a pure function of the body's voxels within the chunk (and the config),
    so it can be performed by any worker.

    Args:
        config:
            A validated MultiresMeshConfigSchema config.
        chunk_zyx:
            The chunk's corner, in full-resolution voxels (ZYX).

    Returns:
        A list of Draco bytes, one per LOD (b'' if the chunk has no geometry at that LOD).
    """
    resource_mgr = resource_mgr or DummyResourceMgr()
    num_lods = len(config['lod-error-thresholds'])
    bits = config['vertex-quantization-bits']
    scale = config['source-scale']
    halo = config['chunk-halo']
    chunk_shape_xyz = np.array(config['chunk-shape-s0'])
    chunk_zyx = np.asarray(chunk_zyx, dtype=int)
    chunk_xyz = chunk_zyx[::-1]

    box = np.array([chunk_zyx, chunk_zyx + chunk_shape_xyz[::-1]]) // (2**scale)
    box += [[-halo] * 3, [halo] * 3]

    start = time.perf_counter()
    try:
        with resource_mgr.access_context(server, True, 1, 0):
            mask, mask_box = fetch_sparsevol(server, uuid, seg_instance, body, scale, mask_box=box, format='mask')
    except HTTPError as ex:
        # The chunk list comes from the body's scale-0 blocks, but a tiny piece of
        # the body may vanish at the (coarser) source scale, in which case DVID
        # returns 404 for this box.
        if ex.response is None or ex.response.status_code != 404:
            raise
        mask = None
    fetch_seconds = time.perf_counter() - start

    chunk_id = ','.join(map(str, chunk_xyz))
    if mask is None or not mask.any():
        logger.debug(f"Chunk {chunk_id}: no voxels at scale {scale}")
        return [b''] * num_lods

    start = time.perf_counter()
    if (closing_radius := config['morphological-closing-s0'] // 2**scale):
        # (It's okay to use out=mask since closing() operates on a temporary array.)
        footprints = skimage.morphology.ball(closing_radius, decomposition='sequence')
        skimage.morphology.closing(mask, footprints, out=mask)

    if config['fill-holes']:
        fill_holes_in_mask(mask, inplace=True)

    # The mesh is in full-resolution voxel coordinates (the "stored model" space of the info).
    mesh = Mesh.from_binary_vol(mask, mask_box * 2**scale, method='skimage')
    del mask
    mesh.laplacian_smooth(config['smoothing'])

    cell_xyz = chunk_xyz // chunk_shape_xyz
    pieces = []
    for lod, max_error in enumerate(config['lod-error-thresholds']):
        # Each LOD is decimated further from the previous one (with the halo intact).
        mesh.simplify_to_error(max_error)
        vertices_xyz, faces = v2m_multires.trim_mesh_to_box(
            mesh.vertices_zyx[:, ::-1], mesh.faces, chunk_xyz, chunk_xyz + chunk_shape_xyz)
        piece = v2m_multires.encode_fragment(
            (vertices_xyz, faces), cell_xyz >> lod, chunk_shape_xyz, [0, 0, 0],
            lod=lod, vertex_quantization_bits=bits)
        pieces.append(piece or b'')

    mesh_seconds = time.perf_counter() - start
    logger.debug(f"Chunk {chunk_id}: fetched mask in {fetch_seconds:.1f}s, "
                 f"meshed in {mesh_seconds:.1f}s ({[len(p) for p in pieces]} bytes)")
    return pieces


def encode_chunk_pieces(pieces):
    """
    Pack a chunk's per-LOD pieces (as returned by generate_chunk_pieces()) into a single
    bytestring (e.g. for an HTTP response).  See decode_chunk_pieces().
    """
    sizes = np.array([len(p) for p in pieces], dtype='<u4')
    return b''.join([np.uint32(len(pieces)).astype('<u4').tobytes(), sizes.tobytes(), *pieces])


def decode_chunk_pieces(buf):
    """The inverse of encode_chunk_pieces()."""
    n = int(np.frombuffer(buf, '<u4', 1, 0)[0])
    sizes = np.frombuffer(buf, '<u4', n, 4).astype(int)
    offsets = 4 * (n + 1) + np.concatenate(([0], np.cumsum(sizes)))
    if offsets[-1] != len(buf):
        raise ValueError(f"Invalid chunk pieces: expected {offsets[-1]} bytes, got {len(buf)}")
    return [bytes(buf[a:b]) for a, b in zip(offsets[:-1], offsets[1:])]


def generate_pieces_for_chunks(server, uuid, seg_instance, body, config, chunk_zyxs,
                               processes=0, resource_mgr=None, chunk_fn=None, chunk_fn_threads=8):
    """
    Generate the pieces for several chunks (see generate_chunk_pieces()).

    Args:
        chunk_zyxs:
            The chunks' corners (full-resolution, ZYX).
        processes:
            How many processes to use.  If 'dask-worker-client', it is assumed that you're
            calling this from within a dask worker, and the chunks are submitted to the dask cluster.
        chunk_fn:
            Optional.  A function to call instead of generate_chunk_pieces() for each chunk,
            e.g. to send the work to a remote service.  It's called as chunk_fn((z, y, x))
            and must return what generate_chunk_pieces() would.  If given, 'processes' is
            ignored, and chunk_fn is called from chunk_fn_threads threads.

    Returns:
        A list of results (in the same order as chunk_zyxs).
    """
    chunk_zyxs = [tuple(int(c) for c in zyx) for zyx in chunk_zyxs]
    if not chunk_zyxs:
        return []

    if chunk_fn is not None:
        with Timer(f"Generating {len(chunk_zyxs)} chunks via {chunk_fn_threads} threads", logger):
            return compute_parallel(chunk_fn, chunk_zyxs, threads=chunk_fn_threads)

    fn = partial(generate_chunk_pieces, server, uuid, seg_instance, body, config, resource_mgr=resource_mgr)
    if processes != 'dask-worker-client':
        with Timer(f"Generating {len(chunk_zyxs)} chunks using {processes} processes", logger):
            return compute_parallel(fn, chunk_zyxs, processes=processes)

    from dask.distributed import worker_client
    with (
        Timer(f"Generating {len(chunk_zyxs)} chunks using the dask cluster", logger),
        worker_client() as client
    ):
        task_names = [f'generate_chunk_pieces-body-{body}-chunk-{x},{y},{z}' for z, y, x in chunk_zyxs]
        futures = client.map(fn, chunk_zyxs, key=task_names, priority=10)
        return client.gather(futures)


#
# Assembly
#

def assemble_body_mesh(new_pieces, chunk_shape_s0_xyz, voxel_size_xyz, config, reused_cells=(), previous=None):
    """
    Assemble a multi-resolution body mesh from its chunks' pieces.  (No decimation is needed.)

    The pieces of unchanged chunks can be recovered from the body's previous mesh,
    rather than regenerated: LOD-0 pieces are simply copied (they are the previous mesh's
    LOD-0 fragments), and coarser pieces are extracted from the previous mesh's fragments
    (see vol2mesh.multires.partition_fragment_by_cells()).  If none of the chunks within a
    coarse fragment have changed, the previous fragment is copied as-is.

    Pure function (no I/O), so it can be unit-tested directly.

    Args:
        new_pieces:
            ``{chunk_cell_xyz: [lod0_bytes, lod1_bytes, ...]}``, as returned by
            generate_chunk_pieces(), keyed by each chunk's (LOD-0) grid cell index.
        chunk_shape_s0_xyz:
            The chunk shape (the LOD-0 cell size), in full-resolution voxels.
        voxel_size_xyz:
            The voxel size (e.g. in nm), for the 'info' transform.
        config:
            The validated config.
        reused_cells:
            The cells of the chunks whose pieces should be recovered from the previous mesh.
            (Disjoint from new_pieces.)
        previous:
            The previous mesh (see _load_previous_mesh()), required if reused_cells is non-empty.

    Returns:
        (data_bytes, index_bytes, info_json, stats)
    """
    bits = config['vertex-quantization-bits']
    num_lods = len(config['lod-error-thresholds'])
    chunk_shape_s0_xyz = np.asarray(chunk_shape_s0_xyz, dtype=float)
    grid_origin = [0, 0, 0]
    reused_cells = set(reused_cells)
    assert not (reused_cells & set(new_pieces)), "Reused chunks can't also be new"
    assert previous is not None or not reused_cells

    def previous_fragment(lod, position):
        """The previous mesh's fragment (bytes) at the given position, or None."""
        offset, size = previous.fragments[lod].get(position, (0, 0))
        return previous.data[offset:offset + size] if size else None

    fragments_by_lod = {lod: {} for lod in range(num_lods)}
    copied_per_lod = [0] * num_lods
    reencoded_per_lod = [0] * num_lods

    # LOD 0: The pieces are the fragments.
    for cell, pieces in new_pieces.items():
        if pieces[0]:
            fragments_by_lod[0][cell] = pieces[0]
    for cell in reused_cells:
        if (b := previous_fragment(0, cell)):
            fragments_by_lod[0][cell] = b
            copied_per_lod[0] += 1

    # Coarser LODs: Each fragment is the concatenation of its chunks' pieces.
    for lod in range(1, num_lods):
        new_by_parent = {}
        for cell, pieces in new_pieces.items():
            new_by_parent.setdefault(tuple(c >> lod for c in cell), []).append(pieces[lod])
        reused_by_parent = {}
        for cell in reused_cells:
            reused_by_parent.setdefault(tuple(c >> lod for c in cell), set()).add(cell)
        previous_by_parent = {}
        for cell in (previous.cells if previous else ()):
            previous_by_parent.setdefault(tuple(c >> lod for c in cell), set()).add(cell)

        for parent in new_by_parent.keys() | reused_by_parent.keys():
            new_bytes = [b for b in new_by_parent.get(parent, []) if b]
            reused = reused_by_parent.get(parent, set())

            if parent not in new_by_parent and reused == previous_by_parent.get(parent):
                # The fragment's chunks are exactly the same as before.
                if (b := previous_fragment(lod, parent)):
                    fragments_by_lod[lod][parent] = b
                    copied_per_lod[lod] += 1
                continue

            if len(new_bytes) == 1 and not reused:
                # The only piece is already encoded as this LOD's fragment.
                fragments_by_lod[lod][parent] = new_bytes[0]
                continue

            meshes = [
                v2m_multires.decode_fragment(
                    b, parent, chunk_shape_s0_xyz, grid_origin, lod=lod, vertex_quantization_bits=bits)
                for b in new_bytes
            ]
            if reused and (b := previous_fragment(lod, parent)):
                v, f = v2m_multires.decode_fragment(
                    b, parent, chunk_shape_s0_xyz, grid_origin, lod=lod, vertex_quantization_bits=bits)
                previous_pieces = v2m_multires.partition_fragment_by_cells(v, f, chunk_shape_s0_xyz)
                meshes += [previous_pieces[cell] for cell in reused if cell in previous_pieces]
            if meshes:
                fragments_by_lod[lod][parent] = _concatenate(meshes)
                reencoded_per_lod[lod] += 1

    data_bytes, index_bytes, num_fragments_per_lod = v2m_multires.encode_multilod_object(
        fragments_by_lod, chunk_shape_s0_xyz, grid_origin, vertex_quantization_bits=bits)
    info_json = info_json_for(config, voxel_size_xyz)

    stats = {
        'num_lods': num_lods,
        'num_fragments_per_lod': num_fragments_per_lod,
        'num_fragments': int(sum(num_fragments_per_lod)),
        'copied_fragments_per_lod': copied_per_lod,
        'reencoded_fragments_per_lod': reencoded_per_lod,
    }
    return data_bytes, index_bytes, info_json, stats


def _concatenate(meshes):
    """Concatenate (vertices, faces) pairs, without welding."""
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(v)
        faces.append(np.asarray(f, dtype=np.uint32) + np.uint32(offset))
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def info_json_for(config, voxel_size_xyz):
    """The 'info' file for a mesh source of meshes generated with the given config."""
    transform = [voxel_size_xyz[0], 0, 0, 0,
                 0, voxel_size_xyz[1], 0, 0,
                 0, 0, voxel_size_xyz[2], 0]
    return v2m_multires.build_info(vertex_quantization_bits=config['vertex-quantization-bits'], transform=transform)


#
# The previous mesh as a chunk cache
#
# A stored body mesh contains every piece of every chunk it was made from:
# the LOD-0 pieces are its LOD-0 fragments, and each coarser fragment is the
# concatenation of its chunks' pieces (each within its own chunk's cell).
# So when a body changes, the pieces of its unchanged chunks can be recovered
# from its previous mesh, and only the changed chunks need to be regenerated.
#
# To determine which chunks are unchanged, the mesh's '.chunks' file lists the
# chunks it was made from, with their block hashes (see body_chunk_surface_table()),
# and the body's lastmod when the mesh was generated.
#

CHUNKS_FILE_VERSION = 1


def _chunks_file(body, uuid, lastmod, config, chunk_df, data_bytes, index_bytes):
    """
    The contents of a body mesh's '.chunks' file (JSON):

        version:        CHUNKS_FILE_VERSION
        body, uuid:     The body, and the UUID in which the mesh was generated
        lastmod:        The body's mutation id when the mesh was generated.
                        (The mesh reflects the body's voxels as of then, or later.)
        mesh-settings:  See MESH_SETTINGS
        data-bytes:     The size of the mesh's data file
        index-crc32:    The CRC32 of the mesh's index file
        chunks:         [[x, y, z, block_hash], ...]  The chunks the mesh was made from
                        (their corners, in full-resolution voxels)

    The last two fields are used to verify that the '.chunks' file matches the mesh files.
    """
    chunks = [[int(x), int(y), int(z), str(h)] for x, y, z, h in chunk_df[[*'xyz', 'block_hash']].values]
    contents = {
        "version": CHUNKS_FILE_VERSION,
        "body": int(body),
        "uuid": uuid,
        "lastmod": int(lastmod),
        "mesh-settings": mesh_settings(config),
        "data-bytes": len(data_bytes),
        "index-crc32": zlib.crc32(index_bytes),
        "chunks": chunks,
    }
    return json.dumps(contents, separators=(',', ':')).encode()


def _load_previous_mesh(server, uuid, seg_instance, body, mesh_info, config, resource_mgr=None):
    """
    Load the body's stored mesh (if any), to reuse its chunks' pieces.

    Returns None if there's no stored mesh, or if it can't be reused (e.g. it was generated with
    different settings, or its files are inconsistent).  Otherwise, returns a SimpleNamespace with:

        lastmod:    The body's mutation id when the mesh was generated.
        chunks:     {chunk_cell_xyz: block_hash} of the chunks it was made from.
        cells:      The same chunk cells, as a set.
        data:       The mesh's data file (bytes).
        fragments:  [{position_xyz: (offset, size)}, ...] The location of each fragment in the data, per LOD.
    """
    resource_mgr = resource_mgr or DummyResourceMgr()
    if (mesh_info is None
            or mesh_info.get('method') != MESH_METHOD
            or mesh_info.get('mesh-info-version') != MESH_INFO_VERSION
            or mesh_info.get('mesh-settings') != mesh_settings(config)):
        return None

    location = locate_multires_body_mesh(server, uuid, seg_instance, body, mesh_info)
    if location is None:
        return None

    with (
        Timer("Fetching the previous mesh", logger),
        resource_mgr.access_context(server, True, 1, 0)
    ):
        files = _fetch_mesh_files(server, uuid, seg_instance, location)

    chunks_file, data, index = files['.chunks'], files[''], files['.index']
    if chunks_file is None or data is None or index is None:
        logger.warning("The previous mesh's files are missing; regenerating all chunks.")
        return None

    contents = json.loads(chunks_file)
    if (contents.get('version') != CHUNKS_FILE_VERSION
            or contents.get('lastmod') != mesh_info['lastmod']
            or contents.get('mesh-settings') != mesh_settings(config)
            or contents.get('data-bytes') != len(data)
            or contents.get('index-crc32') != zlib.crc32(index)):
        logger.warning("The previous mesh's files are inconsistent; regenerating all chunks.")
        return None

    chunk_shape_xyz = np.array(config['chunk-shape-s0'])
    chunks = {
        tuple(int(c) for c in np.array(xyz) // chunk_shape_xyz): block_hash
        for *xyz, block_hash in contents['chunks']
    }

    manifest = v2m_multires.parse_manifest(index)
    if (manifest['num_lods'] != len(config['lod-error-thresholds'])
            or (manifest['chunk_shape_xyz'] != chunk_shape_xyz).any()):
        logger.warning("The previous mesh's manifest doesn't match the config; regenerating all chunks.")
        return None

    fragments = [
        {tuple(int(c) for c in p): (int(o), int(s))
         for p, o, s in zip(lod['positions'], lod['offsets'], lod['sizes'])}
        for lod in manifest['lods']
    ]
    return SimpleNamespace(lastmod=contents['lastmod'], chunks=chunks, cells=set(chunks),
                           data=data, fragments=fragments)


def _fetch_mesh_files(server, uuid, seg_instance, location):
    """Fetch a stored mesh's files.  Returns {suffix: bytes or None}."""
    names = {'': location.data_name, '.index': location.index_name, '.chunks': location.chunks_name}
    if location.store:
        return {suffix: location.store.get(name) for suffix, name in names.items()}

    mesh_instance = mesh_instance_names(seg_instance, multires=True).body
    files = {}
    for suffix, name in names.items():
        try:
            files[suffix] = fetch_key(server, uuid, mesh_instance, name)
        except HTTPError as ex:
            if ex.response is None or ex.response.status_code != 404:
                raise
            files[suffix] = None
    return files


def stale_chunks(chunk_df, chunk_shape_s0_xyz, previous):
    """
    Determine which of a body's chunks must be regenerated, given its previous mesh (or None).

    A chunk's pieces can be reused from the previous mesh only if the chunk is listed in it with
    the same block hash, and its surface_mutid is no greater than the previous mesh's lastmod.
    (Merges don't update the surface_mutids of the merged blocks, so the block hash is needed to
    detect them.)  A chunk whose surface_mutid is unknown (0) is always stale.

    Args:
        chunk_df:
            As returned by body_chunk_surface_table()

    Returns:
        A boolean array (aligned with chunk_df's rows), which is True for the stale chunks.
    """
    if previous is None:
        return np.ones(len(chunk_df), dtype=bool)

    cells = chunk_df[[*'xyz']].values // np.asarray(chunk_shape_s0_xyz)
    previous_hashes = np.array([previous.chunks.get(tuple(int(c) for c in cell)) for cell in cells], dtype=object)
    return (
        (previous_hashes != chunk_df['block_hash'].values)
        | (chunk_df['surface_mutid'].values == 0)
        | (chunk_df['surface_mutid'].values > previous.lastmod)
    )


#
# Updating a body's mesh
#

@PrefixFilter.with_context("Body {body}")
def update_multires_body_mesh(
    server,
    uuid,
    seg_instance,
    body,
    config,
    *,
    force=False,
    processes=0,
    chunk_fn=None,
    chunk_fn_threads=8,
    expect_concurrent_updates=False,
    resource_mgr=None,
):
    """
    Bring a body's stored multires mesh up-to-date (if it isn't already).

    Args:
        config:
            A MultiresMeshConfigSchema config.
        force:
            If True, regenerate the mesh (from scratch) even if the stored one is up-to-date.
        processes, chunk_fn, chunk_fn_threads:
            How to generate the chunks.  See generate_pieces_for_chunks().
        expect_concurrent_updates:
            Set this if other processes (e.g. other instances of a mesh service) may be
            updating the same body at the same time.  Then the mesh isn't stored if another
            process has stored an up-to-date one in the meantime.

    Returns:
        The new mesh_info if a mesh was stored, otherwise None.
    """
    resource_mgr = resource_mgr or DummyResourceMgr()
    validate_config(config)
    uuid = resolve_ref(server, uuid, True)
    names = mesh_instance_names(seg_instance, multires=True)

    try:
        with resource_mgr.access_context(server, True, 1, 0):
            lastmod = fetch_lastmod(server, uuid, seg_instance, body)['mutation id']
    except HTTPError as ex:
        if ex.response is not None and ex.response.status_code == 404:
            logger.info(f"Body {body} doesn't exist in {uuid}; deleting its stored mesh (if any)")
            delete_multires_body_mesh(server, uuid, seg_instance, body, config)
            return None
        raise

    mesh_info = _fetch_mesh_info(server, uuid, names.info, body, resource_mgr)
    if not force and mesh_info is not None and _mesh_info_is_current(mesh_info, lastmod, config):
        logger.info(f"Stored mesh is up-to-date (lastmod {lastmod})")
        return None

    if mesh_info is None or mesh_info.get('method') == 'deleted':
        logger.info(f"No stored mesh yet; generating it (lastmod {lastmod})")
    elif force:
        logger.info("Regenerating mesh (forced)")
    else:
        logger.info(f"Stored mesh is stale (stored lastmod {mesh_info.get('lastmod')}, current lastmod {lastmod}, "
                    "or the mesh settings have changed); updating it")

    chunk_shape_s0_xyz = np.array(config['chunk-shape-s0'])
    chunk_df = body_chunk_surface_table(
        server, uuid, seg_instance, body, chunk_shape_s0_xyz, config['base-uuid'], resource_mgr)

    # Unchanged chunks can be recovered from the previous mesh.
    previous = None
    if not force:
        previous = _load_previous_mesh(server, uuid, seg_instance, body, mesh_info, config, resource_mgr)
    stale = stale_chunks(chunk_df, chunk_shape_s0_xyz, previous)
    logger.info(f"{len(chunk_df)} chunks: {stale.sum()} must be generated, "
                f"{(~stale).sum()} can be reused from the previous mesh")

    chunk_cells = [tuple(int(c) for c in xyz) for xyz in chunk_df[[*'xyz']].values // chunk_shape_s0_xyz]
    reused_cells = [cell for cell, s in zip(chunk_cells, stale) if not s]

    stale_zyxs = chunk_df.loc[stale, [*'zyx']].values
    results = generate_pieces_for_chunks(
        server, uuid, seg_instance, body, config, stale_zyxs, processes, resource_mgr, chunk_fn, chunk_fn_threads)
    new_pieces = {
        tuple(int(c) for c in zyx[::-1] // chunk_shape_s0_xyz): pieces
        for zyx, pieces in zip(stale_zyxs, results)
    }

    if expect_concurrent_updates:
        mesh_info = _fetch_mesh_info(server, uuid, names.info, body, resource_mgr)
        if mesh_info is not None and _mesh_info_is_current(mesh_info, lastmod, config):
            logger.info("Another process already stored an up-to-date body mesh")
            return None

    voxel_size_xyz = np.asarray(fetch_resolution_zyx(server, uuid, seg_instance))[::-1]
    with Timer(f"Assembling multires mesh from {len(new_pieces)} new and {len(reused_cells)} reused chunks", logger):
        data_bytes, index_bytes, info_json, stats = assemble_body_mesh(
            new_pieces, chunk_shape_s0_xyz, voxel_size_xyz, config, reused_cells, previous)
    chunks_file = _chunks_file(body, uuid, lastmod, config, chunk_df, data_bytes, index_bytes)

    mesh_info = {
        "mesh-info-version": MESH_INFO_VERSION,
        "body": int(body),
        "uuid": uuid,
        "written-uuid": uuid,
        "lastmod": lastmod,
        "method": MESH_METHOD,
        "format": "neuroglancer_multilod_draco",
        "storage": config['storage'],
        "mesh-settings": mesh_settings(config),
        "mesh-timestamp": str(datetime.datetime.now(ZoneInfo("US/Eastern"))),
        "data-bytes": len(data_bytes),
        "index-bytes": len(index_bytes),
        "chunk-count": len(chunk_df),
        "generated-chunk-count": len(new_pieces),
        "reused-chunk-count": len(reused_cells),
        "num-lods": stats['num_lods'],
        "fragment-count": stats['num_fragments'],
        "fragment-count-per-lod": stats['num_fragments_per_lod'],
        "vertex-quantization-bits": config['vertex-quantization-bits'],
    }
    files = {'': data_bytes, '.index': index_bytes, '.chunks': chunks_file}

    # FIXME... had to hard-code this because dask workers didn't have the configured dvid timeout.
    set_default_dvid_session_timeout(600.0, 600.0)

    storage = config['storage']
    with Timer(f"Storing multires body mesh in {storage} "
               f"({len(data_bytes) / 1e6:.1f} MB data, {len(index_bytes) / 1e3:.1f} kB index)", logger):
        if not _store_body_mesh(server, uuid, seg_instance, body, config, files, info_json, lastmod):
            return None
        # The mesh_info is posted last, since readers rely on it to find the mesh.
        post_key(server, uuid, names.info, body, json=mesh_info)
    return mesh_info


@PrefixFilter.with_context("Body {body}")
def delete_multires_body_mesh(server, uuid, seg_instance, body, config):
    """
    Record that the body has no mesh (in its mesh_info).
    If the meshes are stored in DVID, the mesh files are deleted, too.
    (In an object store, they're left in place.)
    """
    names = mesh_instance_names(seg_instance, multires=True)
    if config['storage'] == 'dvid':
        for suffix in MESH_FILE_SUFFIXES:
            key = f"{body}{suffix}"
            if fetch_key(server, uuid, names.body, key, check_head=True):
                logger.info(f"Deleting {key}")
                delete_key(server, uuid, names.body, key)

    mesh_info = _fetch_mesh_info(server, uuid, names.info, body)
    if mesh_info is not None and mesh_info.get('method') == 'deleted':
        return

    mesh_info = {
        "mesh-info-version": MESH_INFO_VERSION,
        "body": int(body),
        "uuid": uuid,
        "mesh-timestamp": str(datetime.datetime.now(ZoneInfo("US/Eastern"))),
        "method": "deleted",
    }
    post_key(server, uuid, names.info, body, json=mesh_info)


def fetch_multires_mesh_info(server, uuid, seg_instance, body):
    """Return the body's mesh_info, or None if it has none."""
    names = mesh_instance_names(seg_instance, multires=True)
    return _fetch_mesh_info(server, uuid, names.info, body)


def _fetch_mesh_info(server, uuid, info_instance, body, resource_mgr=None):
    resource_mgr = resource_mgr or DummyResourceMgr()
    try:
        with resource_mgr.access_context(server, True, 1, 0):
            return fetch_key(server, uuid, info_instance, body, as_json=True)
    except HTTPError as ex:
        if ex.response is not None and ex.response.status_code == 404:
            return None
        raise


def _mesh_info_is_current(mesh_info, lastmod, config):
    """
    Return True if a stored body mesh (according to its mesh_info) was generated by this module
    from the body's lastmod (or later), with the configured settings, in the configured storage.
    """
    return (
        mesh_info.get('method') == MESH_METHOD
        and mesh_info.get('mesh-info-version') == MESH_INFO_VERSION
        and mesh_info.get('lastmod', -1) >= lastmod
        and mesh_info.get('storage') == config['storage']
        and mesh_info.get('mesh-settings') == mesh_settings(config)
    )


#
# Storage
#

def locate_multires_body_mesh(server, uuid, seg_instance, body, mesh_info=None):
    """
    Find the stored multires mesh for the given body, as seen from the given UUID.

    Args:
        mesh_info:
            The body's mesh_info in the given UUID, if you've already fetched it.

    Returns:
        None if there's no stored mesh, or a SimpleNamespace with attributes:

        - store: None if the mesh is stored in DVID, otherwise the object store (see meshstore.py)
        - data_name, index_name, chunks_name: The names of the mesh's files, either
          DVID keys (in the '<segmentation>_multires_meshes' instance, in the given UUID)
          or object names (in the object store).
        - mesh_info: The body's mesh_info

    The rule for finding a body mesh in an object store is as follows:
    Let W be the mesh_info's 'written-uuid'.  If the object '<body>' was written in W
    (according to its custom metadata) from the same lastmod as the mesh_info indicates,
    then the mesh files are '<body>', '<body>.index' (and '<body>.chunks').
    Otherwise, they are '<body>-<W>', '<body>-<W>.index' (and '<body>-<W>.chunks').
    """
    if mesh_info is None:
        mesh_info = fetch_multires_mesh_info(server, uuid, seg_instance, body)
        if mesh_info is None:
            return None

    if mesh_info.get('method') == 'deleted' or 'lastmod' not in mesh_info:
        return None

    storage = mesh_info.get('storage', 'dvid')
    if storage == 'dvid':
        store = None
        name = f'{body}'
    else:
        store = open_store(storage)
        written_uuid = mesh_info['written-uuid']
        live = store.stat(f'{body}')
        if (live and live.metadata.get('uuid') == written_uuid
                and live.metadata.get('lastmod') == str(mesh_info['lastmod'])):
            name = f'{body}'
        else:
            name = f'{body}-{written_uuid}'

    return SimpleNamespace(store=store, data_name=name, index_name=f'{name}.index',
                           chunks_name=f'{name}.chunks', mesh_info=mesh_info)


def _store_body_mesh(server, uuid, seg_instance, body, config, files, info_json, lastmod):
    """
    Store a body mesh's files (``{suffix: bytes}``; see MESH_FILE_SUFFIXES)
    in DVID or the configured object store, along with the dataset's 'info' file.

    Returns True if the mesh was stored, or False if it wasn't, because another
    process has already stored a mesh for a newer state of the body in this UUID.
    """
    storage = config['storage']
    if storage != 'dvid':
        return _store_body_mesh_in_object_store(
            server, uuid, seg_instance, body, open_store(storage), config['live-ref'], files, info_json, lastmod)

    names = mesh_instance_names(seg_instance, multires=True)
    # The 'info' key is dataset-wide; posting it idempotently keeps
    # the instance self-describing for neuroglancer.
    post_key(server, uuid, names.body, "info", json=info_json)
    for suffix, data in files.items():
        post_key(server, uuid, names.body, f"{body}{suffix}", data)
    return True


def _store_body_mesh_in_object_store(server, uuid, seg_instance, body, store, live_ref, files, info_json, lastmod):
    """
    Store a multires body mesh's files (generated in the given UUID) in an object store.
    (See MultiresMeshConfigSchema['properties']['storage'] for the naming scheme.)

    If the UUID is the head of the live_ref branch, the mesh is stored as '<body>'
    (and '<body>.index', etc.), after preserving the existing mesh (if it was generated
    in a different UUID) as '<body>-<old-uuid>'.  Otherwise, it is stored as '<body>-<uuid>'.

    Returns True if the mesh was stored, or False if it wasn't, because another
    process has already stored a mesh for a newer state of the body in this UUID.
    """
    _ensure_store_info(store, info_json)
    metadata = {'uuid': uuid, 'lastmod': str(lastmod)}

    if not _is_branch_head(server, uuid, live_ref):
        for suffix, data in files.items():
            store.put(f'{body}-{uuid}{suffix}', data, metadata=metadata, cache_control=BODY_MESH_CACHE_CONTROL)
        return True

    # The live mesh might be modified by other processes at the same time,
    # so we only overwrite it if it hasn't changed since we inspected it.
    for _ in range(5):
        stats = {suffix: store.stat(f'{body}{suffix}') for suffix in MESH_FILE_SUFFIXES}
        data_stat = stats['']
        if data_stat and data_stat.metadata.get('uuid') == uuid and int(data_stat.metadata.get('lastmod', -1)) > lastmod:
            logger.info("A mesh for a newer state of this body has already been stored.")
            return False
        if data_stat and data_stat.metadata.get('uuid') != uuid:
            _preserve_superseded_body_mesh(server, seg_instance, body, store, stats)
        try:
            for suffix, data in files.items():
                stat = stats[suffix]
                store.put(f'{body}{suffix}', data, metadata=metadata, cache_control=BODY_MESH_CACHE_CONTROL,
                          if_generation_match=stat.generation if stat else 0)
            return True
        except PreconditionFailed:
            logger.info("Another process modified the stored body mesh at the same time; trying again.")
    raise RuntimeError(f"Body {body}: Could not store the body mesh in {store.url}, "
                       "due to repeated concurrent modifications by other processes.")


def _preserve_superseded_body_mesh(server, seg_instance, body, store, stats):
    """
    The live body mesh ('<body>') is about to be replaced with a mesh from a different UUID.
    Copy its files to '<body>-<old-uuid>' (etc.), so that its UUID (and that UUID's descendants)
    can still find it.  But only do so if the old UUID's mesh_info refers to it.
    (Otherwise, nothing refers to it anymore.)

    Args:
        stats:
            ``{suffix: ObjectInfo}`` of the live mesh's files.
    """
    data_stat = stats['']
    old_uuid = data_stat.metadata.get('uuid')
    old_lastmod = data_stat.metadata.get('lastmod')
    present = {suffix: stat for suffix, stat in stats.items() if stat is not None}
    if not old_uuid or any(stat.metadata != data_stat.metadata for stat in present.values()) or '.index' not in present:
        logger.warning(f"The stored mesh files for body {body} are inconsistent; not preserving them.")
        return

    old_mesh_info = fetch_multires_mesh_info(server, old_uuid, seg_instance, body)
    if old_mesh_info is None:
        return
    if old_mesh_info.get('written-uuid') != old_uuid or str(old_mesh_info.get('lastmod')) != old_lastmod:
        return

    with Timer(f"Preserving the mesh from {old_uuid[:6]} as {body}-{old_uuid}", logger):
        for suffix, stat in present.items():
            store.copy(f'{body}{suffix}', f'{body}-{old_uuid}{suffix}', source_generation=stat.generation)


def _is_branch_head(server, uuid, branch):
    """Return True if the given (full) UUID is the head of the given branch (in the same repo)."""
    head = resolve_ref(server, f'{uuid}:{branch}', expand=True)
    return uuid.startswith(head) or head.startswith(uuid)


# The (store URL, info) combinations which this process has already written or verified.
_verified_store_infos = set()


def _ensure_store_info(store, info_json):
    """
    Ensure that the object store's (dataset-wide) 'info' file matches the given info.

    We avoid writing it unless necessary: Cloud Storage permits only about one write
    per second to any single object, and many processes (e.g. cluster workers) may be
    storing body meshes at the same time.  So we only write it if it's missing or
    different, and each process checks it only once.
    """
    info_bytes = json.dumps(info_json).encode()
    if (store.url, info_bytes) in _verified_store_infos:
        return

    if store.get('info') != info_bytes:
        stat = store.stat('info')
        try:
            # Only one of the processes that try to create (or replace) it at the same time will succeed.
            store.put('info', info_bytes, cache_control=BODY_MESH_CACHE_CONTROL,
                      content_type='application/json',
                      if_generation_match=stat.generation if stat else 0)
        except PreconditionFailed:
            # Another process wrote it in the meantime.
            if store.get('info') != info_bytes:
                raise RuntimeError(f"The 'info' file in {store.url} doesn't match the current config, "
                                   "and another process is writing it concurrently.  "
                                   "Are processes with different configs using the same storage?")

    _verified_store_infos.add((store.url, info_bytes))
