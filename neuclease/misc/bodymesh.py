import logging
import datetime
from string import Formatter
from itertools import chain
from functools import cache, partial
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from requests import HTTPError

from confiddler import validate, flow_style
from vol2mesh.mesh import Mesh

from neuclease import PrefixFilter
from neuclease.util import Timer, compute_parallel

from neuclease.dvid.repo import resolve_ref, create_instance, fetch_repo_instances
from neuclease.dvid.node import fetch_instance_info
from neuclease.dvid.keyvalue import fetch_keyrange, fetch_key, post_key, delete_key, fetch_keyvalues
from neuclease.dvid.tarsupervoxels import create_tarsupervoxel_instance
from neuclease.dvid.labelmap import fetch_lastmod, fetch_sparsevol, fetch_labelindex
from neuclease.dvid.tarsupervoxels import fetch_tarfile, fetch_missing, post_supervoxel
from neuclease.dvid.rle import blockwise_masks_from_ranges

logger = logging.getLogger(__name__)

SV_MESH_SCALE = 1
SV_MESH_GRID_S0 = 512

# from collections import namedtuple
# MeshInstances = namedtuple('MeshInstances', 'body_meshes sv_meshes chunk_meshes mesh_info ')

# def mesh_instances(seg_instance):
#     return MeshInstances(
#         f"{seg_instance}_meshes",
#         f"{seg_instance}_sv_meshes",
#         f"{seg_instance}_chunk_meshes"
#         f"{seg_instance}_mesh_info"
#     )

BodyMeshParametersSchema = {
    # TODO: skip-decimation-body-size
    # TODO: downsample-before-marching-cubes?
    "default": {},
    "additionalProperties": False,
    "properties": {
        "smoothing": {
            "description": "How many iterations of smoothing to apply to each mesh before decimation.",
            "type": "integer",
            "default": 0
        },
        "decimation": {
            "description": "Mesh decimation aims to reduce the number of \n"
                           "mesh vertices in the mesh to a fraction of the original mesh. \n"
                           "To disable decimation, use 1.0.\n",
            "type": "number",
            "minimum": 0.0000001,
            "maximum": 1.0,  # 1.0 == disable
            "default": 1.0
        },
        "max-vertices": {
            "description": "If necessary, decimate the mesh even further to avoid exceeding this maximum vertex count.\n",
            "type": "number",
            "minimum": 0,
            "default": 0  # no max
        },
        "compute-normals": {
            "description": "Compute vertex normals and include them in the uploaded results.",
            "type": "boolean",
            "default": False
        },
        "quality": {
            "description": "Which chunk quality to use as a starting point for the body mesh before further decimation.\n",
            "type": "string",
            "default": ""
        }
    }
}

ChunkMeshParametersSchema = {
    "default": {},
    "additionalProperties": False,
    "properties": {
        "name": {
            "description": "A name to identify this quality level.\n",
            "type": "string",
            # no default
        },
        "source-scale": {
            "description": "Which scale of voxels to use for generating the mesh\n",
            "type": "integer",
            "default": 2,
            "minimum": 0,
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
            "default": 1.0
        }
    }
}

MeshChunkConfigSchema = {
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
            # no default
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
            "type": "array",
            "items": ChunkMeshParametersSchema
        }
    }
}


def init_mesh_instances(server, uuid, seg_instance):
    uuid = resolve_ref(server, uuid, True)
    seg = seg_instance
    repo_instances = fetch_repo_instances(server, uuid)

    if f"{seg}_meshes" not in repo_instances:
        logger.info(f"Creating DVID instance: {seg}_meshes")
        create_instance(server, uuid, f"{seg}_meshes", 'keyvalue', {'type': 'meshes'})

    if f"{seg}_mesh_info" not in repo_instances:
        logger.info(f"Creating DVID instance: {seg}_mesh_info")
        create_instance(server, uuid, f"{seg}_mesh_info", 'keyvalue')

    if f"{seg}_sv_meshes" not in repo_instances:
        logger.info(f"Creating DVID instance: {seg}_sv_meshes")
        create_tarsupervoxel_instance(server, uuid, f"{seg}_sv_meshes", seg, 'drc', {'type': 'meshes'})

    if f"{seg}_chunk_meshes" not in repo_instances:
        logger.info(f"Creating DVID instance: {seg}_chunk_meshes")
        create_instance(server, uuid, f"{seg}_chunk_meshes", 'keyvalue', {'type': 'meshes'})


def update_body_mesh(server, uuid, seg_instance, body, mesh_params, force=False):
    seg = seg_instance
    uuid = resolve_ref(server, uuid, True)
    validate(mesh_params, BodyMeshParametersSchema, inject_defaults=True)

    try:
        lastmod = fetch_lastmod(server, uuid, seg, body)['mutation id']
    except HTTPError as ex:
        # Note: If the instance name can't be found, DVID returns 400.
        # DVID only returns 404 if we are looking at a valid keyvalue
        # instance and the key isn't present.
        if ex.response.status_code == 404:
            delete_body_mesh(server, uuid, seg, body)
            return
        raise

    try:
        mesh_info = fetch_key(server, uuid, f"{seg}_mesh_info", body, as_json=True)
        needs_update = (lastmod > mesh_info['lastmod'])
    except HTTPError as ex:
        needs_update = True
        if ex.response.status_code != 404:
            raise

    if needs_update or force:
        update_body_mesh_from_supervoxels(server, uuid, seg, body, mesh_params)


@PrefixFilter.with_context("Body {body}")
def delete_body_mesh(server, uuid, seg_instance, body):
    seg = seg_instance
    if fetch_key(server, uuid, f"{seg}_meshes", f"{body}.ngmesh", check_head=True):
        logger.info(f"Deleting {body}.ngmesh")
        delete_key(server, uuid, f"{seg}_meshes", f"{body}.ngmesh")

    try:
        mesh_info = fetch_key(server, uuid, f"{seg}_mesh_info", body)
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
    post_key(server, uuid, f"{seg}_mesh_info", body, json=mesh_info)


def create_and_upload_missing_supervoxel_meshes(server, uuid, seg_instance, body):
    seg = seg_instance
    missing = fetch_missing(server, uuid, f"{seg}_sv_meshes", body)
    if len(missing) == 0:
        return

    logger.info(f"Creating supervoxel meshes for {len(missing)} missing supervoxel(s).")
    for sv in missing:
        mesh = create_supervoxel_mesh(server, uuid, seg, sv)
        if mesh is None:
            # By our convention, objects that were too small to
            # create meshes for are given an empty file in DVID.
            mesh_bytes = b''
        else:
            mesh_bytes = mesh.serialize(fmt='drc')

        post_supervoxel(server, uuid, f"{seg}_sv_meshes", sv, mesh_bytes)


def create_supervoxel_mesh(server, uuid, seg_instance, sv, smoothing=3, decimation_s0=0.005):
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
    rng = fetch_sparsevol(server, uuid, seg_instance, sv, scale=SV_MESH_SCALE, format='ranges')
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
    mesh.simplify_openmesh(decimation)
    return mesh


@PrefixFilter.with_context("Body {body}")
def update_body_mesh_from_supervoxels(server, uuid, seg_instance, body, mesh_params):
    seg = seg_instance
    uuid = resolve_ref(server, uuid, True)
    lastmod = fetch_lastmod(server, uuid, seg, body)['mutation id']
    create_and_upload_missing_supervoxel_meshes(server, uuid, seg, body)

    with Timer("Fetching supervoxel tarfile", logger):
        tar_bytes = fetch_tarfile(server, uuid, f"{seg}_sv_meshes", body)

    with Timer("Constructing Mesh", logger):
        mesh = Mesh.from_tarfile(tar_bytes, keep_normals=False)

    # neuroglancer meshes must be written in nanometer units.
    rescale = fetch_resolution_zyx(server, uuid, seg)
    mesh.vertices_zyx *= rescale

    fraction = mesh_params['decimation']
    max_vertices = mesh_params['max-vertices']
    fraction = min(fraction, max_vertices / len(mesh.vertices_zyx))

    mesh_mb = mesh.uncompressed_size() / 1e6
    orig_vertices = len(mesh.vertices_zyx)
    logger.info(f"Original mesh has {orig_vertices} vertices and {len(mesh.faces)} faces ({mesh_mb:.1f} MB)")

    fraction = min(fraction, max_vertices / len(mesh.vertices_zyx))
    with Timer(f"Decimating at {fraction:.2f}", logger):
        mesh.simplify_openmesh(fraction)

    mesh_mb = mesh.uncompressed_size() / 1e6
    logger.info(f"Final mesh has {len(mesh.vertices_zyx)} vertices and {len(mesh.faces)} faces ({mesh_mb:.1f} MB)")

    with Timer("Uploading mesh", logger):
        post_key(server, uuid, f"{seg}_meshes", f"{body}.ngmesh", mesh.serialize(fmt='ngmesh'))

    mesh_info = {
        "body": int(body),
        "uuid": uuid,
        "lastmod": lastmod,
        "mesh-timestamp": str(datetime.datetime.now(ZoneInfo("US/Eastern"))),
        "method": "concatenated-supervoxels",
        "extra-decimation": fraction,
        "vertex-count": len(mesh.vertices_zyx)
    }
    post_key(server, uuid, f"{seg}_mesh_info", body, json=mesh_info)


@cache
def fetch_resolution_zyx(server, uuid, seg_instance):
    return fetch_instance_info(server, uuid, seg_instance)['Extended']['VoxelSize'][::-1]


CHUNK_KEY_FMT = "{config_name}-{body}-{chunk_id}-{mesh_mutid}-{quality}"


def update_body_mesh_from_chunks(server, uuid, seg_instance, body, chunk_config, body_mesh_config, processes=0):
    validate(chunk_config, MeshChunkConfigSchema, inject_defaults=True)
    if '-' in (cs := chunk_config['chunk-set']) or cs == "":
        raise RuntimeError(f"Invalid chunk-set name: {cs}")

    chunk_shape_s0 = chunk_config['chunk-shape-s0']
    base_uuid = resolve_ref(server, chunk_config['base-uuid'], True)

    seg = seg_instance
    uuid = resolve_ref(server, uuid, True)
    lastmod = fetch_lastmod(server, uuid, seg, body)['mutation id']

    pli = fetch_labelindex(server, uuid, seg, body, format='pandas')
    chunk_df = pli.surface_mutids.copy()
    chunk_df[[*'zyx']] //= chunk_shape_s0
    chunk_df[[*'zyx']] *= chunk_shape_s0
    chunk_mutids = chunk_df.groupby([*'zyx'])['surface_mutid'].max()
    if (chunk_mutids == 0).any():
        # Any missing surface_mutids will be replaced with the lastmod in the BASE uuid.
        # It is assumed that any chunks that ARE present in the chunk store are at least
        # up-to-date with the base uuid.
        # And although we did not enable the surface_mutid feature until recently,
        # it was turned on immediately after locking the base_uuid, so all mutations
        # since then are captured by surface_mutid.
        base_mutid = fetch_lastmod(server, base_uuid, seg, body)['mutation id']
        chunk_mutids[chunk_mutids == 0] = base_mutid

    # Fetch all chunk meshes this body has
    config_name = chunk_config['config-name']
    prefix = f"{config_name}-{body}-"
    key_cols = [x[1] for x in Formatter().parse(CHUNK_KEY_FMT)]
    assert key_cols == ['config_name', 'body', 'chunk_id', 'mesh_mutid', 'quality']
    body_keys = fetch_keyrange(server, uuid, f"{seg}_chunk_meshes", f"{prefix} ", f"{prefix}~")
    key_splits = pd.Series(body_keys).str.split('-').values
    key_df = pd.DataFrame(key_splits, columns=key_cols)
    key_df['key'] = body_keys
    key_df = key_df.sort_values('mutid')
    key_df[[*'zyx']] = parse_chunk_ids(key_df['chunk_id'])
    key_df = key_df.merge(chunk_mutids, 'right', on=[*'xyz'])
    key_df['mesh_mutid'] = key_df['mesh_mutid'].fillna(0).astype(int)

    quality_names = [qc['name'] for qc in chunk_config['quality-configs']]
    for quality in quality_names:
        # TODO: What, if anything, will I do with the other chunk qualities?
        if body_mesh_config['quality'] != quality:
            continue

        # Select chunks which are out-of-date or not of the desired quality.
        missing_chunk_df = key_df.query('mesh_mutid < surface_mutid or quality != @quality')
        fn = partial(mesh_for_chunk, server, uuid, seg, body, lastmod, chunk_config, quality, True)
        with Timer(f"Generating {len(missing_chunk_df)} missing chunks", logger):
            new_chunk_meshes = compute_parallel(fn, missing_chunk_df[[*'zyx']].values, processes=processes)

        # Download chunks that were already stored.
        stored_chunk_df = key_df.query('mesh_mutid >= surface_mutid and quality != @quality')
        stored_chunk_df.drop_duplicates('chunk_id', inplace=True)
        with Timer(f"Fetching {len(stored_chunk_df)} stored chunks", logger):
            stored_chunk_mesh_bytes = fetch_keyvalues(server, uuid, f"{seg}_mesh_chunks", stored_chunk_df['key'].values, batch_size=10)

        with Timer(f"Combining chunks...", logger):
            stored_chunk_meshes = (Mesh.from_buffer(b, 'ngmesh') for b in stored_chunk_mesh_bytes.values())
            body_mesh = Mesh.concatenate_meshes(chain(new_chunk_meshes, stored_chunk_meshes))

        if (smoothing := body_mesh_config['smoothing']):
            with Timer(f"Smoothing body mesh with {smoothing} iterations", logger):
                body_mesh.laplacian_smooth(smoothing)

        fraction = body_mesh_config['decimation']
        max_vertices = body_mesh_config['max-vertices']
        fraction = min(fraction, max_vertices / len(body_mesh.vertices_zyx))
        if fraction <= 1.0:
            with Timer(f"Decimating body mesh with {fraction}", logger):
                body_mesh.simplify_openmesh(smoothing)

        with Timer(f"Storing body mesh: {body}.ngmesh", logger):
            post_key(server, uuid, f"{seg}_meshes", f"{body}.ngmesh", body_mesh.serialize(fmt='ngmesh'))


def mesh_for_chunk(server, uuid, seg_instance, body, lastmod, chunk_config, quality, store, chunk_zyx):
    seg = seg_instance
    _cfg = [
        cfg for cfg in chunk_config['quality-configs']
        if cfg['name'] == quality
    ]
    if not _cfg:
        raise RuntimeError(f"No quality config named '{quality}'")
    if len(_cfg) > 1:
        raise RuntimeError(f"More than one quality config named '{quality}'")
    quality_config = _cfg[0]

    chunk_zyx = np.asarray(chunk_zyx)
    chunk_shape_s0_zyx = chunk_config['chunk-shape-s0'][::-1]

    scale = quality_config['source-scale']
    chunk_box = np.array([chunk_zyx, chunk_zyx + chunk_shape_s0_zyx])
    chunk_box //= (2 ** scale)

    # Note: Halo is defined in units of the source scale.
    halo = chunk_config['chunk-halo']
    chunk_box.T[:] += [-halo, halo]

    mask, mask_box = fetch_sparsevol(server, uuid, seg, body, scale, mask_box=chunk_box, format='mask')
    mesh = Mesh.from_binary_vol(mask, mask_box * 2**scale, method='skimage')

    smoothing = quality_config['smoothing']
    decimation_s0 = quality_config['decimation-s0']

    # Reduce severity of decimation by 4x for each scale, since
    # low-res scales will start off with fewer vertices to begin with.
    decimation = min(decimation_s0 * 4**scale, 1.0)

    mesh.laplacian_smooth(smoothing)
    mesh.simplify_openmesh(decimation)
    if store:
        key = CHUNK_KEY_FMT.format(
            config_name=chunk_config['config-name'],
            body=body,
            chunk_id=make_chunk_id(chunk_zyx),
            mesh_mutid=lastmod,
            quality=quality,
        )
        post_key(server, uuid, f"{seg}_mesh_chunks", key, mesh.serialize(fmt='ngmesh'))
    return mesh


def make_chunk_ids(df):
    df = df[[*'xyz']].astype(str)
    return df['x'] + ',' + df['y'] + ',' + df['z']


def make_chunk_id(chunk_zyx):
    z, y, x = chunk_zyx
    return f"{z},{y},{x}"


def parse_chunk_ids(chunk_ids):
    df = pd.Series(chunk_ids).str.extract(r'(\d+),(\d+),(\d+)').astype(int)
    df.columns = [*'xyz']
    return df[[*'zyx']]
