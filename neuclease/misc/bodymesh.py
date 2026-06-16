import json
import logging
import datetime
from types import SimpleNamespace
from contextlib import contextmanager
from string import Formatter
from itertools import chain
from functools import cache, partial, wraps
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import skimage.morphology
from requests import HTTPError
from cityhash import CityHash64

from confiddler import validate, flow_style
from vol2mesh.mesh import Mesh
from vol2mesh import multires as v2m_multires

from neuclease import PrefixFilter
from neuclease.util import Timer, compute_parallel
from neuclease.util.segmentation import fill_holes_in_mask

from neuclease.dvid import set_default_dvid_session_timeout
from neuclease.dvid.repo import resolve_ref, create_instance, fetch_repo_instances
from neuclease.dvid.node import fetch_instance_info
from neuclease.dvid.keyvalue import fetch_keyrange, fetch_key, post_key, delete_key, fetch_keyvalues, fetch_keyrangevalues
from neuclease.dvid.tarsupervoxels import create_tarsupervoxel_instance
from neuclease.dvid.labelmap import fetch_lastmod, fetch_sparsevol, fetch_labelindex
from neuclease.dvid.tarsupervoxels import fetch_tarfile, fetch_missing, post_supervoxel
from neuclease.dvid.rle import blockwise_masks_from_ranges

logger = logging.getLogger(__name__)

SV_MESH_SCALE = 2
SV_MESH_GRID_S0 = 512
SV_MESH_DECIMATION_S0 = 0.004

def mesh_instance_names(seg_instance, multires=False):
    """
    The DVID keyvalue instance names used to store meshes for a segmentation.

    Single-resolution (ngmesh) meshes and multi-resolution
    (neuroglancer_multilod_draco) meshes are kept in entirely separate
    instances (distinguished by a '_multires' infix), so the two formats are
    never mixed in one instance.  The supervoxel instance is single-res only
    and is shared (no multires variant).

    Returns a SimpleNamespace with attributes: body, info, chunk, sv.
    """
    mr = "_multires" if multires else ""
    return SimpleNamespace(
        body=f"{seg_instance}{mr}_meshes",
        info=f"{seg_instance}{mr}_mesh_info",
        chunk=f"{seg_instance}{mr}_chunk_meshes",
        sv=f"{seg_instance}_sv_meshes",
    )

BodyMeshParametersSchema = {
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "description": "Settings for body mesh creation.\n",
    "properties": {
        "format": {
            "description":
                "Which mesh format to produce (and therefore which DVID instances to use).\n"
                "  - 'neuroglancer-single-res': the legacy single-resolution ngmesh format.\n"
                "  - 'neuroglancer-multi-res':  the neuroglancer multi-resolution\n"
                "    (neuroglancer_multilod_draco) format, stored in separate '_multires'\n"
                "    DVID instances.\n"
                "Note: init_mesh_instances() takes a separate 'multires' argument (it has no\n"
                "config); make sure the instances you initialize match this setting.\n",
            "type": "string",
            "enum": ["neuroglancer-single-res", "neuroglancer-multi-res"],
            "default": "neuroglancer-single-res"
        },
        "source-method": {
            "description":
                "Body meshes can be constructed by assembling supervoxel meshes or chunk meshes.\n"
                "This setting specifies which method to use for generating/retrieving the component meshes.\n"
                "To generate meshes by naively assembling supervoxel meshes, use 'concatenated-supervoxels'\n",
            "type": "string",
            "default": ""
        },
        "smoothing": {
            "description": "How many iterations of smoothing to apply to each mesh before decimation.",
            "type": "integer",
            "default": 0
        },
        "small-body-overall-decimation-s0": {
            "description": "Small-enough bodies will be decimated with this setting.",
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 1.0,
            "default": 0.01
        },
        "small-body-cutoff-vertices-s0": {
            "description":
                "Defines what counts as a 'small' body, according to the (approximate)\n"
                "vertex count bodies would have if they were meshed at scale 0.\n"
                "Bodies smaller than this will always be decimated at the least severe decimation setting,\n"
                "as specified via small-body-overall-decimation-s0, while bodies larger than this will\n"
                "be decimated more severely.",
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 20e6
        },
        "large-body-overall-decimation-s0": {
            "description":
                "Large bodies will be decimated with this setting.\n"
                "This is the most severe decimation we are willing to apply to any body,\n"
                "even though very large bodies could end up with heavy mesh files.\n",
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 1.0,
            "default": 0.0005
        },
        "large-body-cutoff-vertices-s0": {
            "description":
                "Defines what counts as a 'large' body, according to the (approximate)\n"
                "vertex count bodies would have if they were meshed at scale 0.\n"
                "Bodies smaller than this will be less severely decimated.\n"
                "Bodies larger than this will be decimated at the maximum allowed level,\n"
                "as configured in large-body-overall-decimation-s0\n",
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 200e6
        },
        "vertex-quantization-bits": {
            "description":
                "Multires meshes only: the number of bits used to quantize each vertex\n"
                "coordinate within a fragment, as required by the neuroglancer multi-resolution\n"
                "(neuroglancer_multilod_draco) format.  Must be 10 or 16.  Ignored for\n"
                "single-resolution (ngmesh) body meshes.\n",
            "type": "integer",
            "enum": [10, 16],
            "default": 16
        },
        "num-lods": {
            "description":
                "Multires meshes only: how many levels of detail to emit.  LOD 0 is the\n"
                "finest (assembled from the chunk fragments); each coarser LOD is derived\n"
                "by decimating the assembled mesh further and re-partitioning onto a grid\n"
                "with cells twice as large per level.  1 means a single-LOD multires mesh.\n",
            "type": "integer",
            "minimum": 1,
            "default": 1
        },
        "lod-decimation-factor": {
            "description":
                "Multires meshes only: the fraction of faces to keep at each successive\n"
                "(coarser) LOD relative to the previous one.  Each coarser LOD's cells are\n"
                "8x the volume, so ~0.25 keeps roughly constant on-screen vertex density.\n"
                "Only used when num-lods > 1.\n",
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 1.0,
            "default": 0.25
        },
    }
}

ChunkMeshParametersSchema = {
    "description": "Settings for chunk mesh creation, to be used if your body meshes are assembled from chunk meshes.\n",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "name": {
            "description": "A name to identify this quality level.\n",
            "type": "string",
            "default": ""
        },
        "source-scale": {
            "description": "Which scale of voxels to use for generating the mesh\n",
            "type": "integer",
            "minimum": 0,
            "default": 2,
        },
        "morphological-closing-s0": {
            "description": "Apply morphological closing to the chunk mask with the given radius, specified in units of scale-0 voxels.",
            "type": "integer",
            "default": 0
        },
        "fill-holes": {
            "description":
                "If True, erase holes (by filling them with ones) in the chunk mask before meshing.\n"
                "Objects which touch the chunk boundary will not be filled.\n",
            "type": "boolean",
            "default": True
        },
        "smoothing": {
            "description": "How many iterations of smoothing to apply to each mesh before decimation.",
            "type": "integer",
            "default": 0
        },
        "decimation-s0": {
            "description":
                "Mesh decimation aims to reduce the number of \n"
                "mesh vertices in the mesh to a fraction of the original mesh. \n"
                "To disable decimation, use 1.0.\n",
            "type": "number",
            "minimum": 0.0000001,
            "maximum": 1.0,  # 1.0 == disable

            # Ideal would be 1.0 (no decimation) at chunk level, followed by 0.005 at the body level
            # (where vertices can be distributed optimally across chunks according to their complexity).
            # But that would result in very heavy chunks and an expensive body-level decimation step.
            # We compromise by decimating part-way at the chunk level, leaving the final 16x decimation
            # (0.0625) in the body decimation step.
            "default": 0.08
        },
    }
}

MeshChunkConfigSchema = {
    "description": "Settings to define a single chunk 'quality'\n",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "config-name": {
            "description":
                "Arbitrary name for this set of chunk meshes.\n"
                "The name is prefixed to the chunk key, allowing us to experiment\n"
                "with different chunking schemes within the same keyvalue instance.\n"
                "Note: Must not contain any hyphen '-' characters.",
            "type": "string",
            "default": ""
        },
        "base-uuid": {
            "description":
                "The UUID just BEFORE the 'surface_mutid' property was enabled in DVID.\n"
                "When a chunk's surface_mutid is unset (0), it is assumed that any chunks\n"
                "in mesh-base-uuid are new enough.\n"
                "If left unset, then no stored chunks would be considered 'new enough',\n"
                "and chunks without a valid surface_mutid have to have fresh chunk meshes generated\n"
                "from scratch every time we want to construct the body mesh.\n",
            "type": "string",
            "default": ""
        },
        "chunk-shape-s0": {
            "description": "Size of the chunks at which meshes will be generated.",
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 3,
            "maxItems": 3,
            "default": flow_style([1024, 1024, 1024])
        },
        "chunk-halo": {
            "description":
                "How much halo to fetch for each chunk, specified at the\n"
                "resolution used to produce the initial mesh (not necessarily s0).\n",
            "type": "integer",
            "default": 2,
        },
        "quality-configs": {
            "description":
                "Multiple sets of chunk meshes can be produced, with different 'quality' settings.\n"
                "The idea here is that body meshes could be generated at different LOD depending on the set of chunks they start from.\n"
                "Ultimately, we could use this to quickly update multi-LOD body meshes for neuroglancer.\n"
                "For now, we typically list only one 'quality' configuration.\n",
            "type": "array",
            "items": ChunkMeshParametersSchema,
            "default": [{}]
        }
    }
}


def init_mesh_instances(server, uuid, seg_instance, body=True, chunks=True, sv=True, multires=False):
    """
    Create the DVID keyvalue instances used to store meshes.

    If multires=True, create the multi-resolution variants (with a
    '_multires' infix) instead of the single-resolution instances.  The
    supervoxel instance is single-resolution only and is never created in
    multires mode.

    Note: the instances that hold mesh *files* (body meshes and chunk meshes)
    are tagged with {'type': 'meshes'}, which tells DVID to store them in a
    separate internal database from the main voxel data.  The mesh_info
    instance holds JSON metadata (not mesh files) and is a plain keyvalue.
    """
    uuid = resolve_ref(server, uuid, True)
    seg = seg_instance
    names = mesh_instance_names(seg, multires)
    repo_instances = fetch_repo_instances(server, uuid)

    if body and names.body not in repo_instances:
        logger.info(f"Creating DVID instance: {names.body}")
        create_instance(server, uuid, names.body, 'keyvalue', {'type': 'meshes'})

    if body and names.info not in repo_instances:
        logger.info(f"Creating DVID instance: {names.info}")
        create_instance(server, uuid, names.info, 'keyvalue')

    if chunks and names.chunk not in repo_instances:
        logger.info(f"Creating DVID instance: {names.chunk}")
        create_instance(server, uuid, names.chunk, 'keyvalue', {'type': 'meshes'})

    if sv and not multires and names.sv not in repo_instances:
        logger.info(f"Creating DVID instance: {names.sv}")
        create_tarsupervoxel_instance(server, uuid, names.sv, seg, 'drc', {'type': 'meshes'})


class DummyResourceMgr:
    @contextmanager
    def access_context(self, *args, **kwargs):
        yield

    @classmethod
    def overwrite_none_kwarg(cls, f):
        """
        Decorator. If no resoruce_mgr was provided, supply a dummy object that
        can be used in a 'with' statement so the functions below don't have to
        check for "resource_mgr is None" all over the place.
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            if kwargs.get('resource_mgr', None) is None:
                kwargs['resource_mgr'] = DummyResourceMgr()
            return f(*args, **kwargs)
        return wrapper


@DummyResourceMgr.overwrite_none_kwarg
def update_body_mesh(
    server,
    uuid,
    seg_instance,
    body,
    body_mesh_config,
    chunk_config,
    force=False,
    processes=0,
    *,
    resource_mgr=None
):
    """
    FIXME: Too many parameters -- server, uuid, seg_instance, and resource_mgr should be rolled into one.

    Args:
        processes:
            If an int, then multiprocessing is used.
            If a string with value 'dask-worker-client', then it is assumed that you called
            this function from within a dask worker, and chunk mesh generation will be
            submitted to the dask cluster.

    Note:
        Whether to produce a single-resolution or multi-resolution mesh is determined
        by ``body_mesh_config['format']`` (see BodyMeshParametersSchema).  Multi-res
        output requires the chunk-based source-method (not supervoxels), and the
        corresponding '_multires' DVID instances must already exist (see
        init_mesh_instances(..., multires=True)).
    """
    seg = seg_instance
    uuid = resolve_ref(server, uuid, True)
    validate(body_mesh_config, BodyMeshParametersSchema, inject_defaults=True)
    multires = (body_mesh_config['format'] == 'neuroglancer-multi-res')
    names = mesh_instance_names(seg, multires)

    try:
        with resource_mgr.access_context(server, True, 1, 0):
            lastmod = fetch_lastmod(server, uuid, seg, body)['mutation id']
    except HTTPError as ex:
        # Note: If the instance name can't be found, DVID returns 400.
        # DVID only returns 404 if we are looking at a valid keyvalue
        # instance and the key isn't present.
        if ex.response.status_code == 404:
            delete_body_mesh(server, uuid, seg, body, multires=multires)
            return
        raise

    requested_method = body_mesh_config['source-method']
    try:
        mesh_info = fetch_key(server, uuid, names.info, body, as_json=True)
        needs_update = (lastmod > mesh_info['lastmod'] or mesh_info['method'] != requested_method)
    except HTTPError as ex:
        needs_update = True
        if ex.response.status_code != 404:
            raise

    if not (needs_update or force):
        return

    if requested_method == 'concatenated-supervoxels':
        if multires:
            raise RuntimeError("Multires body meshes can only be assembled from chunks, not supervoxels.")
        update_body_mesh_from_supervoxels(server, uuid, seg, body, body_mesh_config, resource_mgr=resource_mgr)
        return

    if '-' in chunk_config['config-name']:
        raise RuntimeError(f"Chunk config-name cannot contain a hyphen: {chunk_config['config-name']}")

    method_parts = requested_method.split('-')
    if len(method_parts) != 4 or method_parts[:2] != ['concatenated', 'chunks']:
        raise RuntimeError(
            f"Body mesh config requests invalid source-method: {requested_method}\n"
            "(source-method must start with 'concatenated-chunks-' and neither the config name nor 'quality' may contain hyphens)"
        )
    chunk_config_name, quality = method_parts[2:]
    if chunk_config_name != chunk_config['config-name']:
        raise RuntimeError(f"Body mesh config requests a chunk config which you didn't supply: {chunk_config_name}")
    available_qualities = {q['name'] for q in chunk_config['quality-configs']}
    if quality not in available_qualities:
        raise RuntimeError(f"Body mesh config requests a chunk quality which isn't listed in the chunk config: {quality}")

    update_body_mesh_from_chunks(server, uuid, seg, body, body_mesh_config, chunk_config, processes=processes, resource_mgr=resource_mgr)


@PrefixFilter.with_context("Body {body}")
def delete_body_mesh(server, uuid, seg_instance, body, multires=False):
    seg = seg_instance
    names = mesh_instance_names(seg, multires)

    if multires:
        # A multires body mesh is a data file ('{body}') plus its manifest ('{body}.index').
        for key in (f"{body}", f"{body}.index"):
            if fetch_key(server, uuid, names.body, key, check_head=True):
                logger.info(f"Deleting {key}")
                delete_key(server, uuid, names.body, key)
    else:
        if fetch_key(server, uuid, names.body, f"{body}.ngmesh", check_head=True):
            logger.info(f"Deleting {body}.ngmesh")
            delete_key(server, uuid, names.body, f"{body}.ngmesh")

    try:
        mesh_info = fetch_key(server, uuid, names.info, body, as_json=True)
        if mesh_info['method'] == 'deleted':
            return
    except HTTPError:
        pass

    mesh_info = {
        "body": int(body),
        "uuid": uuid,
        "mesh-timestamp": str(datetime.datetime.now(ZoneInfo("US/Eastern"))),
        "method": "deleted",
    }
    post_key(server, uuid, names.info, body, json=mesh_info)


@DummyResourceMgr.overwrite_none_kwarg
def create_and_upload_missing_supervoxel_meshes(server, uuid, seg_instance, body, *, resource_mgr=None):
    seg = seg_instance
    with resource_mgr.access_context(server, True, 1, 0):
        try:
            missing = fetch_missing(server, uuid, f"{seg}_sv_meshes", body)
        except HTTPError as ex:
            # Apparently it's possible for a supervoxel to be split into two children with 0 and N voxels, respectively.
            # The 0 voxel child will map to a body that -- if merged -- will not exist any more, but will still be in
            # the mapping for the 0-voxel child.
            # Long story short: Don't try to generate supervoxel meshes for bodies which don't have any supervoxels.
            if 'has no supervoxels' in ex.response.content.decode('utf-8'):
                return
            raise
    if len(missing) == 0:
        return

    logger.info(f"Creating supervoxel meshes for {len(missing)} missing supervoxel(s).")
    for sv in missing:
        mesh = create_supervoxel_mesh(server, uuid, seg, sv, decimation_s0=SV_MESH_DECIMATION_S0, resource_mgr=resource_mgr)
        if mesh is False:
            # The supervoxel doesn't exist so we don't store anything.
            return
        if mesh is None:
            # The supervoxel DOES exist, but it has no voxels at the requested scale.
            # By our convention, objects that were too small to
            # create meshes for are given an empty file in DVID.
            mesh_bytes = b''
        else:
            mesh_bytes = mesh.serialize(fmt='drc')

        post_supervoxel(server, uuid, f"{seg}_sv_meshes", sv, mesh_bytes)


def create_supervoxel_mesh(server, uuid, seg_instance, sv, smoothing=3, decimation_s0=0.005, resource_mgr=None):
    """
    Args:
        decimation_s0:
            The user should specify decimation as a fraction of the vertices
            the mesh WOULD have if we were generating the mesh from scale-0 voxels.
            We will then lighten the decimation accordingly based on the voxel
            resolution we are actually using (currently hard-coded).
            For example, since meshes generated from scale-1 voxels have 4x fewer
            vertices than those created from scale-0 voxels, then we need not
            decimate then as severely -- we increase the decimation fraction by 4x.
    """
    with resource_mgr.access_context(server, True, 1, 0):
        try:
            rng = fetch_sparsevol(server, uuid, seg_instance, sv, scale=SV_MESH_SCALE, supervoxels=True, format='ranges')
        except HTTPError as ex:
            # If the supervoxel has been split already, then we can't generate a mesh for it.
            if ex.response.status_code == 404:
                return False
            raise
    if len(rng) == 0:
        # Supervoxels that are VERY small might have no voxels at all at low scales.
        # We return None in that case.
        # In DVID, we'll store an empty file (0 bytes) instead of a drc file.
        return None

    blocks = (SV_MESH_GRID_S0 // 2**SV_MESH_SCALE)
    boxes, masks = blockwise_masks_from_ranges(rng, blocks, halo=1)

    mesh = Mesh.from_binary_blocks(masks, boxes * 2**SV_MESH_SCALE, stitch=True)
    mesh.laplacian_smooth(smoothing)

    decimation = min(decimation_s0 * (4**SV_MESH_SCALE), 1.0)
    mesh.simplify(decimation)
    return mesh


@PrefixFilter.with_context("Body {body}")
def update_body_mesh_from_supervoxels(server, uuid, seg_instance, body, body_mesh_config, resource_mgr):

    # FIXME... had to hard-code this because dask workers didn't have the configured dvid timeout.
    set_default_dvid_session_timeout(1200.0, 1200.0)

    seg = seg_instance
    uuid = resolve_ref(server, uuid, True)
    lastmod = fetch_lastmod(server, uuid, seg, body)['mutation id']
    create_and_upload_missing_supervoxel_meshes(server, uuid, seg, body, resource_mgr=resource_mgr)

    with (
        Timer("Fetching supervoxel tarfile", logger),
        resource_mgr.access_context(server, True, 1, 0)
    ):
        tar_bytes = fetch_tarfile(server, uuid, f"{seg}_sv_meshes", body)

    with Timer("Constructing Mesh", logger):
        mesh = Mesh.from_tarfile(tar_bytes, keep_normals=False)

    # neuroglancer meshes must be written in nanometer units.
    rescale = fetch_resolution_zyx(server, uuid, seg)
    mesh.vertices_zyx *= rescale

    orig_vertices = len(mesh.vertices_zyx)
    decimation_s0, decimation = select_body_decimation(
        SV_MESH_DECIMATION_S0,
        orig_vertices,
        body_mesh_config
    )

    mesh_mb = mesh.uncompressed_size() / 1e6
    logger.info(f"Original mesh has {orig_vertices} vertices and {len(mesh.faces)} faces ({mesh_mb:.1f} MB)")

    with Timer(f"Decimating at {decimation}", logger) as dec_timer:
        mesh.simplify(decimation)

    mesh_mb = mesh.uncompressed_size() / 1e6
    logger.info(f"Final mesh has {len(mesh.vertices_zyx)} vertices and {len(mesh.faces)} faces ({mesh_mb:.1f} MB)")

    with Timer("Serializing mesh", logger):
        mesh_bytes = mesh.serialize(fmt='ngmesh')

    with (
        Timer("Uploading mesh", logger),
        resource_mgr.access_context(server, False, 1, len(mesh_bytes))
    ):
        post_key(server, uuid, f"{seg}_meshes", f"{body}.ngmesh", mesh_bytes)

    mesh_info = {
        "body": int(body),
        "uuid": uuid,
        "lastmod": lastmod,
        "method": "concatenated-supervoxels",
        "mesh-timestamp": str(datetime.datetime.now(ZoneInfo("US/Eastern"))),
        "mesh-bytes": len(mesh_bytes),
        "supervoxel-vertex-total": orig_vertices,
        "target-body-decimation-s0": decimation_s0,
        "applied-body-decimation": decimation,
        "applied-body-decimation-seconds": dec_timer.seconds,
        "final-body-vertex-count": len(mesh.vertices_zyx)
    }
    post_key(server, uuid, f"{seg}_mesh_info", body, json=mesh_info)


def fetch_body_mesh_info(server, uuid, seg_instance, bodies, format='pandas', *, multires=False, session=None):
    assert format in ('json', 'pandas')
    names = mesh_instance_names(seg_instance, multires)
    if not hasattr(bodies, '__len__'):
        bodies = [bodies]
    bodies = [str(b) for b in bodies]
    kv = fetch_keyvalues(server, uuid, names.info, bodies, as_json=True, session=session)
    if format == 'json':
        return kv
    return pd.DataFrame(list(filter(None, kv.values()))).set_index('body')


@cache
def fetch_resolution_zyx(server, uuid, seg_instance):
    return fetch_instance_info(server, uuid, seg_instance)['Extended']['VoxelSize'][::-1]


def select_body_decimation(chunk_decimation_s0, chunk_total_vertices, body_mesh_config):
    s0_vertices = chunk_total_vertices / chunk_decimation_s0

    small_cutoff_vertices = body_mesh_config['small-body-cutoff-vertices-s0']
    small_decimation = body_mesh_config['small-body-overall-decimation-s0']
    large_cutoff_vertices = body_mesh_config['large-body-cutoff-vertices-s0']
    large_decimation = body_mesh_config['large-body-overall-decimation-s0']

    # Determine how much decimation to aim for by interpolating between small/large mesh decimation settings.
    target_decimation_s0 = np.interp(
        s0_vertices,
        [small_cutoff_vertices, large_cutoff_vertices],
        [small_decimation, large_decimation]
    )

    # We're aiming for that level of decimation overall, but some of that decimation
    # has already been achieved due to the chunks being fetched at low resolution
    # and then pre-decimated somewhat.  How much further decimation do we need to
    # apply to hit our target?
    remaining_needed_decimation = (target_decimation_s0 / chunk_decimation_s0)
    remaining_needed_decimation = min(1.0, remaining_needed_decimation)
    return target_decimation_s0, remaining_needed_decimation


CHUNK_KEY_FMT = "{config_name}-{body}-{chunk_id}-{mesh_mutid}-{mesh_block_hash}-{quality}"


@PrefixFilter.with_context("Body {body}")
def update_body_mesh_from_chunks(
    server,
    uuid,
    seg_instance,
    body,
    body_mesh_config,
    chunk_config,
    processes=0,
    resource_mgr=None
):
    validate(body_mesh_config, BodyMeshParametersSchema, inject_defaults=True)
    validate(chunk_config, MeshChunkConfigSchema, inject_defaults=True)
    if '-' in (name := chunk_config['config-name']) or name == "":
        raise RuntimeError(f"Invalid chunk config-name name: {name}")

    seg = seg_instance
    uuid = resolve_ref(server, uuid, True)
    multires = (body_mesh_config['format'] == 'neuroglancer-multi-res')
    names = mesh_instance_names(seg, multires)
    chunk_df = _chunk_table(server, uuid, seg_instance, body, chunk_config, resource_mgr, multires=multires)
    config_name = chunk_config['config-name']

    quality_names = {qc['name'] for qc in chunk_config['quality-configs']}
    for quality in quality_names:
        if body_mesh_config['source-method'] != f"concatenated-chunks-{config_name}-{quality}":
            # TODO: What, if anything, will I do with the other chunk qualities?
            continue

        if multires:
            data_bytes, index_bytes, info_json, mesh_info = _generate_multires_body_mesh_from_chunks(
                server, uuid, seg_instance,
                body, chunk_df, body_mesh_config, chunk_config, quality,
                processes, resource_mgr
            )

            # FIXME... had to hard-code this because dask workers didn't have the configured dvid timeout.
            set_default_dvid_session_timeout(600.0, 600.0)

            with Timer(f"Storing multires body mesh: {body}", logger):
                # The 'info' key is dataset-wide; posting it idempotently keeps
                # the instance self-describing for neuroglancer (no backend changes).
                post_key(server, uuid, names.body, "info", json=info_json)
                post_key(server, uuid, names.body, f"{body}", data_bytes)
                post_key(server, uuid, names.body, f"{body}.index", index_bytes)
                post_key(server, uuid, names.info, body, json=mesh_info)
        else:
            mesh_bytes, mesh_info = _generate_body_mesh_from_chunks(
                server, uuid, seg_instance,
                body,
                chunk_df,
                body_mesh_config,
                chunk_config,
                quality,
                processes,
                resource_mgr
            )

            # FIXME... had to hard-code this because dask workers didn't have the configured dvid timeout.
            set_default_dvid_session_timeout(600.0, 600.0)

            with Timer(f"Storing body mesh: {body}.ngmesh", logger):
                post_key(server, uuid, names.body, f"{body}.ngmesh", mesh_bytes)
                post_key(server, uuid, names.info, body, json=mesh_info)


def _chunk_table(server, uuid, seg_instance, body, chunk_config, resource_mgr, enforce_block_hash_match=True, multires=False):
    """
    Return a table with a row for each chunk in the given body.
    For chunks with a stored mesh in DVID, their properties are included as columns.
    (Only the most recent stored mesh for each chunk in the UUID listed, ignoring older ones.)
    For chunks in the body which lack a stored mesh, those property columns are NaN.

    Returns:
        DataFrame

    Example result:
                x      y      z  surface_mutid  block_hash  config_name   body           chunk_id  mesh_mutid quality                                                             key
        0   16384  38912  26624     1006321331  0x4f11fa6b    1k_04993d  90972  16384,38912,26624  1006321331  sc2_q1  1k_04993d-90972-16384,38912,26624-1006321331-0x4f11fa6b-sc2_q1
        1   16384  38912  27648     1006321331  0x854b11fa    1k_04993d  90972  16384,38912,27648  1006321331  sc2_q1  1k_04993d-90972-16384,38912,27648-1006321331-0x854b11fa-sc2_q1
        2   17408  36864  25600     1006321331  0xb7bcdfe0    1k_04993d  90972  17408,36864,25600  1006321331  sc2_q1  1k_04993d-90972-17408,36864,25600-1006321331-0xb7bcdfe0-sc2_q1
        3   17408  37888  23552     1006321331  0xaa12cc78    1k_04993d  90972  17408,37888,23552  1006321331  sc2_q1  1k_04993d-90972-17408,37888,23552-1006321331-0xaa12cc78-sc2_q1
        4   17408  37888  24576     1006321331  0x39915e1a    1k_04993d  90972  17408,37888,24576  1006321331  sc2_q1  1k_04993d-90972-17408,37888,24576-1006321331-0x39915e1a-sc2_q1
    """
    resource_mgr = resource_mgr or DummyResourceMgr()
    base_uuid = resolve_ref(server, chunk_config['base-uuid'], True)
    chunk_shape_s0 = np.array(chunk_config['chunk-shape-s0'])
    assert not (chunk_shape_s0 % 64).any(), "Chunk shape must be a multiple of 64"

    with resource_mgr.access_context(server, False, True, 0):
        pli = fetch_labelindex(server, uuid, seg_instance, body, format='pandas')

    # DVID blocks (64x64x64)
    block_df = pli.surface_mutids
    block_df[['cz', 'cy', 'cx']] = (block_df[[*'zyx']] // chunk_shape_s0) * chunk_shape_s0
    block_df = block_df.sort_values(['cz', 'cy', 'cx', *'zyx'], ignore_index=True)

    def compute_block_hash(df):
        """Compute a hash of the set of DVID block coordinates in a chunk"""
        return hex(CityHash64(df[[*'zyx']].values.copy('C')))

    block_hashes = (
        block_df
        .groupby(['cz', 'cy', 'cx'])
        .apply(compute_block_hash, include_groups=False)
        .rename('block_hash')
    )
    chunk_mutids = block_df.groupby(['cz', 'cy', 'cx'])['surface_mutid'].max()
    chunk_df = (
        pd.concat([chunk_mutids, block_hashes], axis=1)
        .reset_index()
        .rename(columns={'cx': 'x', 'cy': 'y', 'cz': 'z'})
    )

    if (chunk_df['surface_mutid'] == 0).any():
        # Any missing surface_mutids will be replaced with the lastmod in the BASE uuid.
        # It is assumed that any chunks that ARE present in the chunk store are at least
        # up-to-date with the base uuid.
        # And although we did not enable the surface_mutid feature until recently,
        # it was turned on immediately after locking the base_uuid, so all mutations
        # since then are captured by surface_mutid.
        with resource_mgr.access_context(server, False, True, 0):
            base_mutid = fetch_lastmod(server, base_uuid, seg_instance, body)['mutation id']
        chunk_df.loc[chunk_df['surface_mutid'] == 0, 'surface_mutid'] = base_mutid

    # Fetch all chunk mesh keys this body has
    key_df = fetch_stored_chunk_keys(server, uuid, seg_instance, chunk_config['config-name'], body, multires=multires)

    # Drop all but the most recent key for each chunk
    key_df = key_df.sort_values('mesh_mutid').drop_duplicates([*'xyz'], keep='last')

    # Append mesh_mutid and block_hash columns
    chunk_df = chunk_df.merge(key_df, 'left', on=[*'xyz'])
    chunk_df['mesh_mutid'] = chunk_df['mesh_mutid'].fillna(0).astype(int)

    if enforce_block_hash_match:
        # If the block_hash doesn't match, we treat it as out-of-date
        chunk_df.loc[chunk_df.eval('block_hash != mesh_block_hash'), 'mesh_mutid'] = 0
    return chunk_df


def fetch_stored_chunk_keys(server, uuid, seg_instance, config_name=None, body=None, multires=False):
    """
    Fetch the list of all mesh chunk keys in the database for the given segmentation,
    optionally limited to particular chunk configuration prefix, or further limited
    to a specific body under that prefix.

    The keys are parsed into their components (config_name, body, chunk_id, mesh_mutid, quality),
    and the chunk_id is parsed into x,y,z columns.

    Returns:
        DataFrame
    """
    seg = seg_instance
    for suffix in ('_multires_chunk_meshes', '_chunk_meshes'):
        if seg.endswith(suffix):
            seg = seg[:-len(suffix)]
            break
    chunk_instance = mesh_instance_names(seg, multires).chunk

    assert config_name or not body, \
        "Can't fetch keys for a specific body unless you also provide the config_name."

    prefix = ""
    if config_name:
        prefix = f"{config_name}"
    if body:
        prefix = f"{config_name}-{body}-"

    key_cols = [x[1] for x in Formatter().parse(CHUNK_KEY_FMT)]
    assert key_cols == ['config_name', 'body', 'chunk_id', 'mesh_mutid', 'mesh_block_hash', 'quality']
    keys = fetch_keyrange(server, uuid, chunk_instance, f"{prefix} ", f"{prefix}~")
    if len(keys) == 0:
        key_df = pd.DataFrame([], columns=key_cols)
    else:
        key_splits = pd.Series(keys, dtype=str).str.split('-')

        # Support old keys that didn't store the mesh_block_hash in the second-to-last position
        old_keys = (key_splits.map(len) < len(key_cols))
        key_splits.loc[old_keys] = key_splits.loc[old_keys].map(lambda x: [*x[:-1], '0x0', x[-1]])

        key_df = pd.DataFrame(key_splits.tolist(), columns=key_cols)
    key_df['key'] = keys
    key_df = key_df.sort_values('mesh_mutid')
    key_df = key_df.astype({'body': np.uint64, 'mesh_mutid': int})
    key_df[[*'zyx']] = parse_chunk_ids(key_df['chunk_id'])
    return key_df


def _generate_body_mesh_from_chunks(
    server,
    uuid,
    seg_instance,
    body,
    chunk_df,
    body_mesh_config,
    chunk_config,
    quality,
    processes,
    resource_mgr
):
    seg = seg_instance
    with resource_mgr.access_context(server, True, 1, 0):
        lastmod = fetch_lastmod(server, uuid, seg, body)['mutation id']

    quality_configs = {qc['name']: qc for qc in chunk_config['quality-configs']}
    quality_config = quality_configs[quality]

    # Select chunks which are out-of-date or not of the desired quality.
    missing = chunk_df.eval('mesh_mutid < surface_mutid or quality != @quality')
    missing_chunk_df = chunk_df.loc[missing]
    stored_chunk_df = chunk_df.loc[~missing]

    new_chunk_meshes = meshes_for_chunks(
        server, uuid, seg,
        body,
        chunk_config, quality, True,
        missing_chunk_df[[*'zyx', 'surface_mutid', 'block_hash']],
        processes, resource_mgr
    )

    with (
        Timer(f"Fetching {len(stored_chunk_df)} stored chunks", logger),
        resource_mgr.access_context(server, True, 1, 0)
    ):
        stored_chunk_mesh_bytes = fetch_keyvalues(server, uuid, f"{seg}_chunk_meshes", stored_chunk_df['key'].values, batch_size=10)

    with Timer("Combining chunks...", logger):
        stored_chunk_meshes = (Mesh.from_buffer(b, 'ngmesh') for b in stored_chunk_mesh_bytes.values())
        body_mesh = Mesh.concatenate_meshes(chain(new_chunk_meshes, stored_chunk_meshes), keep_normals=False)

    if (smoothing := body_mesh_config['smoothing']):
        with Timer(f"Smoothing body mesh with {smoothing} iterations", logger):
            body_mesh.laplacian_smooth(smoothing)

    # neuroglancer meshes must be written in nanometer units.
    rescale = fetch_resolution_zyx(server, uuid, seg)
    body_mesh.vertices_zyx *= rescale

    mesh_mb = body_mesh.uncompressed_size() / 1e6
    orig_vertices = len(body_mesh.vertices_zyx)
    logger.info(f"Original mesh has {orig_vertices} vertices and {len(body_mesh.faces)} faces ({mesh_mb:.1f} MB)")

    target_decimation_s0, decimation = select_body_decimation(
        quality_config['decimation-s0'],
        orig_vertices,
        body_mesh_config,
    )

    decimation_seconds = 0.0
    if decimation <= 1.0:
        with Timer(f"Decimating body mesh with {decimation}", logger) as dec_timer:
            body_mesh.simplify(decimation)
        decimation_seconds = dec_timer.seconds

    mesh_bytes = body_mesh.serialize(fmt='ngmesh')
    config_name = chunk_config['config-name']

    mesh_info = {
        "body": int(body),
        "uuid": uuid,
        "lastmod": lastmod,
        "method": f"concatenated-chunks-{config_name}-{quality}",
        "mesh-timestamp": str(datetime.datetime.now(ZoneInfo("US/Eastern"))),
        "mesh-bytes": len(mesh_bytes),
        "chunk-quality": quality,
        "chunk-count": len(chunk_df),
        "chunk-vertex-total": orig_vertices,
        "target-body-decimation-s0": target_decimation_s0,
        "applied-body-decimation": decimation,
        "applied-body-decimation-seconds": decimation_seconds,
        "final-body-vertex-count": len(body_mesh.vertices_zyx),
    }
    return mesh_bytes, mesh_info


def _assemble_multires_object(frag_bytes_by_cell, chunk_shape_s0_xyz, voxel_size_xyz,
                              chunk_decimation_s0, body_mesh_config):
    """
    Assemble a multi-resolution body mesh object from per-cell Draco fragments.

    Pure function (no DVID I/O), so it can be unit-tested directly.

    The fragments are already cell-quantized, cell-trimmed multires fragments
    (as produced by mesh_for_chunk in multires mode), expressed in
    full-resolution voxel "stored model" coordinates with a grid whose cell
    size is ``chunk_shape_s0_xyz``.

    LOD 0 is assembled from the (chunk) fragments.  If no body-level smoothing
    or decimation is needed, the cached LOD-0 fragment bytes are reused verbatim
    (fast path).  Coarser LODs (num-lods > 1) are derived by decimating the
    assembled mesh progressively and re-partitioning onto the LOD's 2x2x2 grid.
    Decimation/smoothing preserve cell boundaries so neighbors stay aligned and
    the mesh can be re-split.

    Args:
        frag_bytes_by_cell:
            ``{(x, y, z): draco_bytes}`` keyed by integer grid-cell index.
        chunk_shape_s0_xyz:
            (3,) LOD-0 cell extents in full-res voxels, XYZ order.
        voxel_size_xyz:
            (3,) voxel size in nm, XYZ order; written into the info transform.
        chunk_decimation_s0:
            The per-chunk ``decimation-s0`` that was already applied, used by
            select_body_decimation() to decide the remaining body decimation.
        body_mesh_config:
            The validated body mesh config (provides smoothing,
            small/large-body decimation thresholds, vertex-quantization-bits,
            num-lods, lod-decimation-factor).

    Returns:
        ``(data_bytes, index_bytes, info_json, stats)``.
    """
    bits = body_mesh_config['vertex-quantization-bits']
    num_lods = body_mesh_config['num-lods']
    lod_factor = body_mesh_config['lod-decimation-factor']
    chunk_shape_s0_xyz = np.asarray(chunk_shape_s0_xyz, dtype=float)
    grid_origin = [0, 0, 0]

    # Decode every fragment (needed to count vertices and, if requested, to
    # decimate as a single mesh).
    decoded = {}
    for cell, b in frag_bytes_by_cell.items():
        if not b:
            continue
        v_xyz, faces = v2m_multires.decode_fragment(
            b, cell, chunk_shape_s0_xyz, grid_origin, vertex_quantization_bits=bits)
        decoded[cell] = (v_xyz, faces)

    orig_vertices = sum(len(v) for v, _ in decoded.values())
    target_decimation_s0, decimation = select_body_decimation(
        chunk_decimation_s0, orig_vertices, body_mesh_config)
    smoothing = body_mesh_config['smoothing']
    body_ops = bool(smoothing) or (decimation < 1.0)

    fragments_by_lod = {}
    if num_lods == 1 and not body_ops:
        # Fast path: the cached LOD-0 fragments are already final -- reuse verbatim.
        fragments_by_lod[0] = {cell: b for cell, b in frag_bytes_by_cell.items() if b}
        applied_decimation = 1.0
        final_vertices = orig_vertices
    else:
        meshes = [Mesh(v[:, ::-1], faces) for (v, faces) in decoded.values()]
        body_mesh = Mesh.concatenate_meshes(meshes, keep_normals=False)
        if smoothing:
            body_mesh.laplacian_smooth(smoothing, preserve_border=True)
        if decimation < 1.0:
            body_mesh.simplify(decimation, preserve_border=True)
        applied_decimation = decimation
        final_vertices = len(body_mesh.vertices_zyx)

        # LOD 0: reuse cached bytes if we didn't modify the geometry, else split
        # the assembled mesh.
        if not body_ops:
            fragments_by_lod[0] = {cell: b for cell, b in frag_bytes_by_cell.items() if b}
        else:
            fragments_by_lod[0] = v2m_multires.split_mesh_for_lod(body_mesh, chunk_shape_s0_xyz, 0)

        # Coarser LODs: progressively decimate and re-partition.
        current = body_mesh
        for lod in range(1, num_lods):
            current = Mesh(current.vertices_zyx.copy(), current.faces.copy())
            current.simplify(lod_factor, preserve_border=True)
            fragments_by_lod[lod] = v2m_multires.split_mesh_for_lod(current, chunk_shape_s0_xyz, lod)

    data_bytes, index_bytes, num_fragments_per_lod = v2m_multires.encode_multilod_object(
        fragments_by_lod, chunk_shape_s0_xyz, grid_origin, vertex_quantization_bits=bits)

    transform = [voxel_size_xyz[0], 0, 0, 0,
                 0, voxel_size_xyz[1], 0, 0,
                 0, 0, voxel_size_xyz[2], 0]
    info_json = v2m_multires.build_info(vertex_quantization_bits=bits, transform=transform)

    stats = {
        'orig_vertices': orig_vertices,
        'target_decimation_s0': target_decimation_s0,
        'applied_decimation': applied_decimation,
        'final_vertices': final_vertices,
        'num_lods': num_lods,
        'num_fragments_per_lod': num_fragments_per_lod,
        'num_fragments': int(sum(num_fragments_per_lod)),
    }
    return data_bytes, index_bytes, info_json, stats


def _generate_multires_body_mesh_from_chunks(
    server, uuid, seg_instance, body, chunk_df, body_mesh_config,
    chunk_config, quality, processes, resource_mgr
):
    seg = seg_instance
    names = mesh_instance_names(seg, multires=True)
    bits = body_mesh_config['vertex-quantization-bits']

    with resource_mgr.access_context(server, True, 1, 0):
        lastmod = fetch_lastmod(server, uuid, seg, body)['mutation id']

    quality_config = {qc['name']: qc for qc in chunk_config['quality-configs']}[quality]

    # Select chunks which are out-of-date or not of the desired quality.
    missing = chunk_df.eval('mesh_mutid < surface_mutid or quality != @quality')
    missing_chunk_df = chunk_df.loc[missing]
    stored_chunk_df = chunk_df.loc[~missing]

    new_fragment_bytes = meshes_for_chunks(
        server, uuid, seg, body, chunk_config, quality, True,
        missing_chunk_df[[*'zyx', 'surface_mutid', 'block_hash']],
        processes, resource_mgr, multires=True, vertex_quantization_bits=bits)

    with (
        Timer(f"Fetching {len(stored_chunk_df)} stored chunks", logger),
        resource_mgr.access_context(server, True, 1, 0)
    ):
        stored_bytes = fetch_keyvalues(server, uuid, names.chunk, stored_chunk_df['key'].values, batch_size=10)

    chunk_shape_s0_xyz = np.asarray(chunk_config['chunk-shape-s0'], dtype=float)

    def cell_xyz(x, y, z):
        return (int(x // chunk_shape_s0_xyz[0]),
                int(y // chunk_shape_s0_xyz[1]),
                int(z // chunk_shape_s0_xyz[2]))

    # Both new and stored fragments are keyed by grid-cell index. (New and
    # stored use the same cell formula, so they remain mutually consistent.)
    frag_bytes_by_cell = {}
    for (z, y, x), b in zip(missing_chunk_df[[*'zyx']].values, new_fragment_bytes):
        if b:
            frag_bytes_by_cell[cell_xyz(x, y, z)] = b
    for (z, y, x), key in zip(stored_chunk_df[[*'zyx']].values, stored_chunk_df['key'].values):
        b = stored_bytes.get(key)
        if b:
            frag_bytes_by_cell[cell_xyz(x, y, z)] = b

    # stored-model space is full-res voxels, so the transform carries voxel->nm.
    voxel_size_xyz = np.asarray(fetch_resolution_zyx(server, uuid, seg))[::-1]

    data_bytes, index_bytes, info_json, stats = _assemble_multires_object(
        frag_bytes_by_cell, chunk_shape_s0_xyz, voxel_size_xyz,
        quality_config['decimation-s0'], body_mesh_config)

    config_name = chunk_config['config-name']
    mesh_info = {
        "body": int(body),
        "uuid": uuid,
        "lastmod": lastmod,
        "method": f"concatenated-chunks-{config_name}-{quality}",
        "format": "neuroglancer_multilod_draco",
        "mesh-timestamp": str(datetime.datetime.now(ZoneInfo("US/Eastern"))),
        "data-bytes": len(data_bytes),
        "index-bytes": len(index_bytes),
        "chunk-quality": quality,
        "chunk-count": len(chunk_df),
        "num-lods": stats['num_lods'],
        "fragment-count": stats['num_fragments'],
        "fragment-count-per-lod": stats['num_fragments_per_lod'],
        "chunk-vertex-total": stats['orig_vertices'],
        "target-body-decimation-s0": stats['target_decimation_s0'],
        "applied-body-decimation": stats['applied_decimation'],
        "final-body-vertex-count": stats['final_vertices'],
        "vertex-quantization-bits": bits,
    }
    return data_bytes, index_bytes, info_json, mesh_info


def meshes_for_chunks(
    server,
    uuid,
    seg_instance,
    body,
    chunk_config,
    quality,
    store,
    missing_chunk_df,
    processes,
    resource_mgr,
    multires=False,
    vertex_quantization_bits=16,
):
    fn = partial(mesh_for_chunk, server, uuid, seg_instance, body, chunk_config,
                 quality, store, resource_mgr, multires, vertex_quantization_bits)
    chunk_specs = missing_chunk_df[[*'zyx', 'surface_mutid', 'block_hash']].values.tolist()

    if processes != 'dask-worker-client':
        with Timer(f"Generating {len(chunk_specs)} missing chunks using {processes} processes", logger):
            return compute_parallel(fn, chunk_specs, processes=processes)

    from dask.distributed import worker_client
    with (
        Timer(f"Generating {len(missing_chunk_df)} missing chunks using the dask cluster", logger),
        worker_client() as client
    ):
        task_names = [
            f'mesh_for_chunk-body-{body}-chunk-{x},{y},{z}'
            for z,y,x in missing_chunk_df[[*'zyx']].values
        ]
        futures = client.map(fn, chunk_specs, key=task_names, priority=10)
        return client.gather(futures)


def mesh_for_chunk(
    server,
    uuid,
    seg_instance,
    body,
    chunk_config,
    quality,
    store,
    resource_mgr,
    multires,
    vertex_quantization_bits,
    chunk_spec
):
    """
    chunk_spec is a tuple of (*chunk_zyx, surface_mutid, block_hash)

    Returns the chunk mesh: a vol2mesh Mesh for single-resolution, or (for
    multires) the cell-quantized Draco fragment bytes that are also stored
    (an empty bytestring if the chunk produced no geometry).
    """
    seg = seg_instance
    names = mesh_instance_names(seg, multires)
    *chunk_zyx, surface_mutid, block_hash = chunk_spec
    chunk_zyx = np.asarray(chunk_zyx)
    _cfg = [
        cfg for cfg in chunk_config['quality-configs']
        if cfg['name'] == quality
    ]
    if not _cfg:
        raise RuntimeError(f"No quality config named '{quality}'")
    if len(_cfg) > 1:
        raise RuntimeError(f"More than one quality config named '{quality}'")
    quality_config = _cfg[0]

    chunk_shape_s0_zyx = chunk_config['chunk-shape-s0'][::-1]

    scale = quality_config['source-scale']
    chunk_box = np.array([chunk_zyx, chunk_zyx + chunk_shape_s0_zyx])
    chunk_box //= (2 ** scale)

    # Note: Halo is defined in units of the source scale.
    halo = chunk_config['chunk-halo']
    chunk_box.T[:] += [-halo, halo]

    # FIXME... had to hard-code this because dask workers didn't have the configured dvid timeout.
    set_default_dvid_session_timeout(600.0, 600.0)

    with resource_mgr.access_context(server, True, 1, 0):
        mask, mask_box = fetch_sparsevol(server, uuid, seg, body, scale, mask_box=chunk_box, format='mask')

    closing_radius = quality_config['morphological-closing-s0']
    closing_radius //= 2**scale
    if closing_radius:
        # Note that it's okay to use out=mask since the implementation
        # of binary_closing() operates on a temporary array.
        footprints = skimage.morphology.ball(closing_radius, decomposition='sequence')
        skimage.morphology.binary_closing(mask, footprints, out=mask)

    fill_holes = quality_config['fill-holes']
    if fill_holes:
        fill_holes_in_mask(mask, inplace=True)

    scaled_box = mask_box * 2**scale
    mesh = Mesh.from_binary_vol(mask, scaled_box, method='skimage')

    smoothing = quality_config['smoothing']
    decimation_s0 = quality_config['decimation-s0']

    # Reduce severity of decimation by 4x for each scale, since
    # low-res scales will start off with fewer vertices to begin with.
    decimation = min(decimation_s0 * 4**scale, 1.0)

    mesh.laplacian_smooth(smoothing)
    mesh.simplify(decimation)

    if multires:
        # Encode the chunk as its final, cell-quantized multires fragment:
        # trim to the grid cell and Draco-encode integer cell-relative
        # positions.  The cached value IS a final body-mesh fragment, so
        # assembly can reuse it verbatim when no further body decimation is
        # needed.  The stored-model space is full-resolution voxels (the
        # mesh's own coordinates here), so the cell grid is chunk-shape-s0.
        chunk_shape_s0_xyz = np.asarray(chunk_config['chunk-shape-s0'], dtype=float)
        fragment_position_xyz = tuple(int(c) for c in (chunk_zyx[::-1] // chunk_shape_s0_xyz))
        draco_bytes = v2m_multires.encode_fragment(
            mesh, fragment_position_xyz, chunk_shape_s0_xyz, [0, 0, 0],
            vertex_quantization_bits=vertex_quantization_bits, trim=True)
        chunk_value = draco_bytes if draco_bytes is not None else b''
    else:
        chunk_value = mesh.serialize(fmt='ngmesh')

    if store:
        key = CHUNK_KEY_FMT.format(
            config_name=chunk_config['config-name'],
            body=body,
            chunk_id=make_chunk_id(chunk_zyx),
            mesh_mutid=surface_mutid,
            mesh_block_hash=block_hash,
            quality=quality,
        )
        with resource_mgr.access_context(server, False, 1, 0):
            post_key(server, uuid, names.chunk, key, chunk_value)

    return chunk_value if multires else mesh


def make_chunk_ids(df):
    df = df[[*'xyz']].astype(str)
    return df['x'] + ',' + df['y'] + ',' + df['z']


def make_chunk_id(chunk_zyx):
    z, y, x = chunk_zyx
    return f"{x},{y},{z}"


def parse_chunk_ids(chunk_ids):
    df = chunk_ids.str.extract(r'(\d+),(\d+),(\d+)').astype(np.int32)
    df.columns = [*'xyz']
    return df[[*'zyx']]


def fetch_all_body_mesh_info(server, uuid, seg_instance, *, multires=False, session=None):
    names = mesh_instance_names(seg_instance, multires)
    kv = fetch_keyrangevalues(server, uuid, names.info, as_json=True, session=session)
    info_df = pd.DataFrame(kv.values())
    return info_df


def main_debug():
    from neuclease import configure_default_logging
    configure_default_logging()

    from neuclease.dvid import find_master

    # cns_test_server = 'http://emdata7.int.janelia.org:9000'
    # cns_test_uuid = find_master(cns_test_server)
    # cns_test = (cns_test_server, cns_test_uuid)
    # cns_test_seg = (*cns_test, 'segmentation')
    # init_mesh_instances(*cns_test_seg)

    cns_server = 'http://emdata6.int.janelia.org:9000'
    cns_uuid = find_master(cns_server)
    cns = (cns_server, cns_uuid)
    cns_seg = (*cns, 'segmentation')

    base_uuid = "04993d60dd594df8927c70f4f993b188"
    short_base_uuid = base_uuid[:6]

    body_mesh_config = {
        #"source-method": f"concatenated-chunks-1k_{short_uuid}-dec005_from_s2",
        "source-method": f"concatenated-chunks-1k_{short_base_uuid}-sc2_halo2_cl8_filled_sm3_dec005",
        "smoothing": 0,
        "small-body-overall-decimation-s0": 0.01,
        "large-body-overall-decimation-s0": 0.001,
        "small-body-cutoff-vertices-s0": 10e6,
        "large-body-cutoff-vertices-s0": 100e6,
    }

    chunk_config = {
        "config-name": f"1k_{short_base_uuid}",
        "base-uuid": base_uuid,
        "chunk-shape-s0": [1024, 1024, 1024],
        "chunk-halo": 2,
        "quality-configs": [
            {
                "name": "sc2_halo2_cl8_filled_sm3_dec005",
                "source-scale": 2,
                "morphological-closing-s0": 2 * (2**2),
                "fill-holes": True,
                "smoothing": 3,
                "decimation-s0": 0.005,
            }
        ]
    }

    # body = 17198
    # body = 805741
    # body = 11005  # Huge bilateral CB neuron
    # body = 10705
    # body = 11369
    # body = 807948  # Strange floating chunks???
    body = 90972

    # More test cases:
    # https://flyem-cns.slack.com/archives/C02QFC68HPX/p1733326471601639

    # update_body_mesh(*cns_test_seg, body, body_mesh_config, chunk_config, force=True, processes='dask-worker-client')
    update_body_mesh(*cns_seg, body, body_mesh_config, chunk_config, force=True, processes=8)

    from vol2mesh.mesh import Mesh
    buf = fetch_key(*cns, 'segmentation_meshes', f"{body}.ngmesh")
    mesh = Mesh.from_buffer(buf, 'ngmesh')
    mesh.serialize(f"/tmp/{body}.obj")

    # from dask.distributed import LocalCluster, Client
    # from contextlib import closing
    # with (
    #     closing(LocalCluster('foo', 8)) as cluster,
    #     closing(Client(cluster)) as client
    # ):
    #     configure_default_logging()
    #     fut = client.submit(update_body_mesh, *cns_test_seg, body, body_mesh_config, chunk_config, force=True, processes='dask-worker-client')
    #     fut.result()

    mesh_info = fetch_key(*cns, 'segmentation_mesh_info', body, as_json=True)
    logger.info(f"Mesh Info: {mesh_info}")


if __name__ == "__main__":
    main_debug()
