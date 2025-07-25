"""
Based on recent mutations in a DVID labelmap instance, update
segmentation-derived data such as meshes, skeletons, and annotations.
"""
import os
import re
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import json
from requests import HTTPError
import pandas as pd
from confiddler import load_config, dump_default_config

from neuclease import PrefixFilter
from neuclease.util import switch_cwd, tqdm_proxy_config, Timer, tqdm_proxy
from neuclease.dvid import (
    set_default_dvid_session_timeout, is_locked,
    fetch_branch_nodes, resolve_ref, fetch_repo_instances, find_repo_root,
    create_instance, fetch_key, fetch_keys, post_key, delete_key, fetch_keyrange, fetch_keyrangevalues,
    fetch_mapping,fetch_mutations, compute_affected_bodies, fetch_skeleton, fetch_lastmod, fetch_query
)
from neuclease.misc.bodymesh import update_body_mesh, BodyMeshParametersSchema, MeshChunkConfigSchema, create_and_upload_missing_supervoxel_meshes

logger = logging.getLogger(__name__)

SegmentationDvidInstanceSchema = {
    "type": "object",
    "required": ["server", "uuid", "segmentation-name"],

    "default": {},
    "additionalProperties": False,
    "properties": {
        "server": {
            "description": "location of DVID server to READ.",
            "type": "string",
            "default": ""
        },
        "ignore-mutations-before-uuid": {
            "description":
                "Optional. If given, mutations earlier than this uuid will be ignored, "
                "even if this is the first time you've run the update script.\n"
                "By default, all mutations are considered the first time the "
                "update script is run, starting at the root repo uuid.\n"
                "Use this only if you know what you're doing! Once you use this setting, earlier "
                "UUIDs will never be processed, even in subsequent calls to the update script.\n",
            "default": "",
            "type": ["string", "null"]
        },
        "uuid": {
            "description": "version node from dvid for which the derived data will be brought in sync with the segmentation",
            "type": "string",
            "default": ""
        },
        "segmentation-name": {
            "description": "Name of the instance to create",
            "type": "string",
            "default": ""
        },
        "timeout": {
            "description": "",
            "type": "number",
            "default": 600.0
        }
    }
}

SkeletonConfigSchema = {
    "description": "Settings for skeleton generation.",
    "type": "object",
    "required": ["scale"],
    "default": {},
    "additionalProperties": False,
    "properties": {
        "neutu-executable": {
            "description": "Full path to a neutu executable which will be used for generating skeletons from DVID bodies.\n",
            # NO DEFAULT!
            "type": "string",
        },
        "scale": {
            "description":
                "Which scale NeuTu should use when fetching the sparsevol to skeletonize.\n"
                "Either an integer 0-5 or the special string 'coarse'.\n",
            "default": "coarse",
            "oneOf": [
                {
                    "type": "string",
                    "enum": ["coarse"]
                },
                {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5
                }
            ]
        }
    }
}

ConfigSchema = {
    "type": "object",
    "additionalProperties": False,
    "default": {},
    "properties": {
        "logfile": {
            "type": "string",
            "default": "derived-updates"
        },
        "dvid": SegmentationDvidInstanceSchema,
        "update-derived-types": {
            "description": "Which types of derived data to update.\n",
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["meshes", "sv-meshes", "skeletons", "annotations"]
            },
            "default": ["meshes", "sv-meshes", "skeletons", "annotations"]
        },
        "body-meshes": BodyMeshParametersSchema,
        "chunk-meshes": MeshChunkConfigSchema,
        "skeletons": SkeletonConfigSchema,
        "force-update": {
            "description":
                "Ignore body meshes on the server and regenerate them from the component chunks.\n",
            "type": "boolean",
            "default": False
        }
    }
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-c', '--config',
                        help='The input configuration yaml file')
    parser.add_argument('-y', '--dump-default-yaml', action='store_true',
                        help='Print out the default config file')
    parser.add_argument('-Y', '--dump-verbose-yaml', action='store_true',
                        help='Print out the default config file, with verbose comments above each setting.')
    parser.add_argument('-p', '--processes', type=int, default=0,
                        help='How many processes to use when generating meshes')
    parser.add_argument('-t', '--tee-log', action='store_true',
                        help='Log to stdout in addition to the logfile.')
    args = parser.parse_args()

    if args.dump_default_yaml:
        dump_default_config(ConfigSchema, sys.stdout, 'yaml')
        sys.exit(0)

    if args.dump_verbose_yaml:
        dump_default_config(ConfigSchema, sys.stdout, 'yaml-with-comments')
        sys.exit(0)

    cfg = load_config(args.config, ConfigSchema, inject_defaults=True)
    init_logging(args.config, cfg['logfile'], args.tee_log)

    timeout = cfg['dvid']['timeout']
    set_default_dvid_session_timeout(timeout, timeout)

    # We must resolve the UUID before starting in case the UUID becomes
    # locked before we store our update receipts, which would result in
    # inconsistencies between the update receipt's stored UUID and the
    # stored mutation ID.
    dvid_seg = (
        cfg['dvid']['server'],
        resolve_ref(cfg['dvid']['server'], cfg['dvid']['uuid']),
        cfg['dvid']['segmentation-name'],
    )

    if 'meshes' in cfg['update-derived-types']:
        update_body_meshes(
            *dvid_seg,
            cfg['body-meshes'],
            cfg['chunk-meshes'],
            cfg['dvid']['ignore-mutations-before-uuid'],
            cfg['force-update'],
            args.processes
        )

    if 'sv-meshes' in cfg['update-derived-types']:
        update_sv_meshes(*dvid_seg, cfg['dvid']['ignore-mutations-before-uuid'])

    if 'skeletons' in cfg['update-derived-types']:
        update_skeletons(
            *dvid_seg,
            cfg['skeletons']['neutu-executable'],
            cfg['force-update'],
            cfg['skeletons']['scale'],
            cfg['dvid']['ignore-mutations-before-uuid']
        )

    if 'annotations' in cfg['update-derived-types']:
        update_annotations(*dvid_seg)

    logger.info("DONE.")


def init_logging(config_path, logfile, stdout_logging=False):
    logfile = Path(logfile)
    if not logfile.is_absolute():
        with switch_cwd(Path(config_path).parent):
            logfile = logfile.resolve()

    os.makedirs(logfile.parent, exist_ok=True)

    # I want logged progress bars, no matter what.
    tqdm_proxy_config['output_file'] = 'logger'

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s')

    handler = logging.handlers.RotatingFileHandler(logfile, maxBytes=int(10e6), backupCount=1000)
    handler.setFormatter(formatter)
    handler.addFilter(PrefixFilter())

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    rootLogger.addHandler(handler)

    if stdout_logging:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)

        # I don't understand why this isn't necessary, but it isn't.
        # Adding it results in double prefixes.
        # stdout_handler.addFilter(PrefixFilter())

        rootLogger.addHandler(stdout_handler)


def mutated_bodies_since_previous_update(dvid_server, uuid, seg_instance, derived_type, ignore_before_uuid=None):
    """
    Read the most recent update receipt to determine when the previous update
    occurred for the given 'derived_type' (e.g. meshes, sv-meshes, skeletons, annotations).

    Then read the mutation log for mutations since that update occurred,
    and return the affected bodies as a named tuple (as returned by compute_affected_bodies()).

    Also return the previous update receipt and the new mutation ID that corresponds to the newly affected bodies.

    Returns:
        prev_update, affected, last_mutid
        where prev_update is a dict with keys 'uuid' and 'mutid'
        and affected is a named tuple with attributes 'changed_bodies', 'removed_bodies', 'new_bodies'
        and last_mutid is the mutation ID of the most recent mutation in the mutation log (up to and including the given uuid).
    """
    assert uuid == resolve_ref(dvid_server, uuid, expand=True), \
        "The given UUID should be a valid, pre-resolved UUID."

    keys = []
    if "derived-data-checkpoints" in fetch_repo_instances(dvid_server, uuid):
        keys = fetch_keyrange(
            dvid_server,
            uuid,
            "derived-data-checkpoints",
            f"{seg_instance}-{derived_type}-",
            f"{seg_instance}-{derived_type}-a"
        )

    if keys:
        prev_update = fetch_key(dvid_server, uuid, "derived-data-checkpoints", max(keys), as_json=True)
    else:
        prev_update = {
            'uuid': find_repo_root(dvid_server, uuid),
            'mutid': 0
        }

    updated_uuid = prev_update['uuid']
    updated_mutid = prev_update['mutid']  # noqa

    if ignore_before_uuid:
        ignore_before_uuid = resolve_ref(dvid_server, ignore_before_uuid, expand=True)
        updated_uuid = resolve_ref(dvid_server, updated_uuid, expand=True)

        # Sadly, there's no more elegant way to do this, even with CategoricalDtype
        all_uuids = list(fetch_branch_nodes(dvid_server, uuid, uuid))
        uuid_index = all_uuids.index(uuid)
        updated_uuid_index = all_uuids.index(updated_uuid)
        ignore_before_index = all_uuids.index(ignore_before_uuid)

        if uuid_index < ignore_before_index:
            raise RuntimeError(
                f"Can't process for UUID {uuid} since it's earlier than "
                f"your 'ignore before' UUID {ignore_before_uuid}"
            )

        if updated_uuid_index < ignore_before_index:
            updated_uuid = ignore_before_uuid

    recent_muts = fetch_mutations(dvid_server, f"[{updated_uuid}, {uuid}]", seg_instance)
    recent_muts = recent_muts.query('mutid > @updated_mutid')
    if len(recent_muts) == 0:
        last_mutid = int(updated_mutid)
    else:
        last_mutid = int(recent_muts['mutid'].iloc[-1])

    affected = compute_affected_bodies(recent_muts)
    return prev_update, affected, last_mutid


def fetch_all_update_receipts(dvid_server, uuid, seg_instance, derived_type):
    if "derived-data-checkpoints" not in fetch_repo_instances(dvid_server, uuid):
        return []

    kv = fetch_keyrangevalues(
        dvid_server,
        uuid,
        "derived-data-checkpoints",
        f"{seg_instance}-{derived_type}-",
        f"{seg_instance}-{derived_type}-a",
        as_json=True
    )
    return pd.DataFrame(kv.values())


def store_update_receipt(dvid_server, uuid, seg_instance, derived_type, mutid):
    """
    TODO: Store the list of updated bodies.
    """
    assert uuid == resolve_ref(dvid_server, uuid, expand=True), \
        "The given UUID should be a valid, pre-resolved UUID."
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    update_value = {
        "uuid": uuid,
        "mutid": mutid,
        "timestamp": now
    }

    if "derived-data-checkpoints" not in fetch_repo_instances(dvid_server, uuid):
        if not os.environ.get('DVID_ADMIN_TOKEN'):
            raise RuntimeError(
                "The instance 'derived-data-checkpoints' does not yet exist, "
                "but it can't be created because DVID_ADMIN_TOKEN is not defined"
            )
        logger.info("Creating instance 'derived-data-checkpoints' in the ROOT repo UUID.")
        root_uuid = find_repo_root(dvid_server, uuid)
        create_instance(dvid_server, root_uuid, "derived-data-checkpoints", 'keyvalue')

    post_key(
        dvid_server,
        uuid,
        "derived-data-checkpoints",
        f"{seg_instance}-{derived_type}-{mutid:020d}",
        json=update_value
    )
    return update_value


def update_body_meshes(dvid_server, uuid, seg_instance, body_mesh_config, chunk_config, ignore_before_uuid=None, force=False, processes=0):
    dvid_seg = (dvid_server, uuid, seg_instance)
    prev_update, affected, last_mutid = mutated_bodies_since_previous_update(*dvid_seg, "meshes", ignore_before_uuid)
    prev_uuid = prev_update['uuid']
    prev_mutid = prev_update['mutid']
    prev_timestamp = prev_update.get('timestamp', '<unknown time>')
    bodies = [*affected.changed_bodies, *affected.removed_bodies, *affected.new_bodies]
    logger.info(
        f"Detected {len(bodies)} mutated bodies since the previous mesh update "
        f"in uuid {prev_uuid} (mutid: {prev_mutid}) at {prev_timestamp}"
    )
    for i, body in enumerate(bodies, start=1):
        with PrefixFilter.context(f"({i}/{len(bodies)})"), Timer(f"Body {body} Updating chunks/body", logger):
            update_body_mesh(*dvid_seg, body, body_mesh_config, chunk_config, force, processes)

    store_update_receipt(*dvid_seg, "meshes", last_mutid)


def update_sv_meshes(dvid_server, uuid, seg_instance, ignore_before_uuid=None):
    dvid_seg = (dvid_server, uuid, seg_instance)
    prev_update, affected, last_mutid = mutated_bodies_since_previous_update(*dvid_seg, "sv-meshes", ignore_before_uuid)
    prev_uuid = prev_update['uuid']
    prev_mutid = prev_update['mutid']
    prev_timestamp = prev_update.get('timestamp', '<unknown time>')

    # Due to race conditions, one of our 'new_svs' might already be invalid
    # by the time we fetch its mapping, so we filter out body 0 just in case.
    new_svs = set(affected.new_svs) - set(affected.deleted_svs)
    bodies = set(fetch_mapping(*dvid_seg, new_svs)) - {0}
    logger.info(
        f"Detected {len(bodies)} bodies with new supervoxels since the previous "
        f"sv-mesh update in uuid {prev_uuid} (mutid: {prev_mutid}) at {prev_timestamp}"
    )
    for i, body in enumerate(bodies, start=1):
        with PrefixFilter.context(f"({i}/{len(bodies)})"), Timer(f"Body {body} Updating new supervoxel meshes", logger):
            create_and_upload_missing_supervoxel_meshes(*dvid_seg, body)

    store_update_receipt(*dvid_seg, "sv-meshes", last_mutid)


def update_skeletons(dvid_server, uuid, seg_instance, neutu_executable, force, scale=5, ignore_before_uuid=None):
    if is_locked(dvid_server, uuid) and not os.environ.get('DVID_ADMIN_TOKEN'):
        # Without this error, NeuTu would silently just switch to the HEAD uuid without telling us!
        raise RuntimeError(
            "The UUID is locked, but DVID_ADMIN_TOKEN is not defined. "
            "You must define DVID_ADMIN_TOKEN to update skeletons."
        )

    dvid_seg = (dvid_server, uuid, seg_instance)
    prev_update, affected, last_mutid = mutated_bodies_since_previous_update(*dvid_seg, "skeletons", ignore_before_uuid)

    logger.info(f"Found {len(affected.removed_bodies)} removed bodies since last update.")
    logger.info(f"Found {len(affected.new_bodies)} new bodies since last update.")
    logger.info(f"Found {len(affected.changed_bodies)} changed bodies since last update.")

    keys_to_delete = {f"{body}_swc" for body in affected.removed_bodies}
    if len(keys_to_delete) >= 10_000:
        logger.info("Reading existing skeleton keys")
        stored_keys = fetch_keys(dvid_server, uuid, f'{seg_instance}_skeletons')
        keys_to_delete &= set(stored_keys)

    logger.info(f"Deleting {len(keys_to_delete)} skeleton keys.")
    for key in keys_to_delete:
        # Note: DVID doesn't complain if the key doesn't exist.
        delete_key(dvid_server, uuid, f"{seg_instance}_skeletons", key)

    logger.info(f"Updating skeletons for {len(affected.changed_bodies)} changed bodies and {len(affected.new_bodies)} new bodies.")
    failed_bodies = []
    for body in tqdm_proxy([*affected.changed_bodies, *affected.new_bodies]):
        try:
            lastmod = fetch_lastmod(*dvid_seg, body)
            mutid = lastmod["mutation id"]
        except HTTPError:
            logger.info(f"Failed to fetch lastmod for body {body}")
            failed_bodies.append(body)
        else:
            update_skeleton(*dvid_seg, body, mutid, neutu_executable, force, scale)

    if not failed_bodies:
        store_update_receipt(*dvid_seg, "skeletons", last_mutid)

    logger.info("Done updating skeletons.")


def update_skeleton(dvid_server, uuid, seg_instance, body, mutid, neutu_executable, force=False, scale=5):
    if not dvid_server.startswith('http'):
        dvid_server = f'http://{dvid_server}'

    if not force:
        try:
            swc = fetch_skeleton(dvid_server, uuid, f"{seg_instance}_skeletons", body, format='swc')
        except HTTPError:
            pass
        else:
            if m := re.search(r'{"mutation id": (\d+)}', swc):
                swc_mutid = int(m.groups()[0])
                if swc_mutid >= mutid:
                    return

    # We use --force here because we have already decided to regenerate the skeleton,
    # so we don't want NeuTu to second-guess our decision.
    target = f"{dvid_server}?uuid={uuid}&segmentation={seg_instance}&label_zoom={scale}"
    if admintoken := os.environ.get('DVID_ADMIN_TOKEN'):
        target = f"{target}&admintoken={admintoken}"

    cmd = f'{neutu_executable} --command --skeletonize --force --bodyid {body} "{target}"'
    logger.info(cmd)
    subprocess.run(cmd, shell=True, check=True)


def update_annotations(dvid_server, uuid, seg_instance, ignore_before_uuid=None):
    """
    TODO: For 'changed' bodies, we could make sure their
          annotated 'location' (if any) still lives in the body.
    """
    dvid_seg = (dvid_server, uuid, seg_instance)
    prev_update, affected, last_mutid = mutated_bodies_since_previous_update(*dvid_seg, "annotations", ignore_before_uuid)

    keys = fetch_keys(dvid_server, uuid, f"{seg_instance}_annotations")
    keys_to_delete = set(keys) & set(map(str, affected.removed_bodies))
    if not keys_to_delete:
        logger.info("No annotations to delete.")
        return

    q = {'bodyid': [*list(map(int, keys_to_delete))]}
    ann_to_delete = fetch_query(dvid_server, uuid, f"{seg_instance}_annotations", q, format='json')

    try:
        logger.info(f"Deleting {len(keys_to_delete)} annotations for removed (merged) bodies.")
        for key in keys_to_delete:
            # There's no harm in 'deleting' keys that don't actually exist.
            # (DVID doesn't complain.)
            delete_key(dvid_server, uuid, f"{seg_instance}_annotations", key)
        update_value = store_update_receipt(*dvid_seg, "annotations", last_mutid)
        os.makedirs("deleted-annotations", exist_ok=True)
        path = f"deleted-annotations/deleted-{seg_instance}-annotations-{update_value['mutid']:020d}-{update_value['timestamp']}.json"
        with open(path, 'w') as f:
            json.dump(ann_to_delete, f)
    except Exception:
        logger.error("Failed to delete annotations. Logging annotations to failed-annotation-deletions.json")
        with open("failed-annotation-deletions.json", 'w') as f:
            json.dump(ann_to_delete, f)
        raise


if __name__ == "__main__":
    main()
