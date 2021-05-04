import os
import json
import logging
import subprocess

import numpy as np

from neuclease.util import tqdm_proxy as tqdm, dump_json

logger = logging.getLogger()


def create_precomputed_roi_vol(roi_vol, bucket_name, bucket_path, max_scale=3):
    """
    Upload the given ROI volume (which must be a scale-5 volume, i.e. 256nm resolution)
    as a neuroglancer precomputed volume.

    An example of such a volume can be found here:

        gs://flyem-vnc-roi-d5f392696f7a48e27f49fa1a9db5ee3b/roi

    Requires tensorstore.

    TODO:
        - This doesn't upload the volume in "sharded" format.
        - This doesn't upload metadata for the ROI segment names.
        - This doesn't upload meshes.
    """
    import tensorstore as ts

    if bucket_name.startswith('gs://'):
        bucket_name = bucket_name[len('gs://'):]

    for scale in tqdm(range(max_scale)):
        store = ts.open({
            'driver': 'neuroglancer_precomputed',
            'kvstore': {
                'driver': 'gcs',
                'bucket': bucket_name,
            },
            'path': bucket_path,
            'create': True,
            "multiscale_metadata": {
                "type": "segmentation",
                "data_type": "uint64",
                "num_channels": 1
            },
            "scale_metadata": {
                "size": list(np.array(roi_vol.shape[::-1]) // 2**scale),
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8, 8, 8],
                "chunk_size": [64, 64, 64],
                "resolution": [256*2**scale, 256*2**scale, 256*2**scale]
            }
        }).result()
        if scale == 0:
            store[:] = roi_vol.transpose()[..., None]
        else:
            store[:] = roi_vol.transpose()[:-2**scale+1:2**scale, :-2**scale+1:2**scale, :-2**scale+1:2**scale, None]


def create_precomputed_ngmeshes(vol, vol_fullres_box, names, bucket_name, bucket_path, localdir=None):
    from vol2mesh import Mesh
    if not bucket_name.startswith('gs://'):
        bucket_name = 'gs://' + bucket_name

    if localdir is None:
        localdir = bucket_path.split('/')[-1]

    os.makedirs(f"{localdir}/mesh", exist_ok=True)
    dump_json({"@type": "neuroglancer_legacy_mesh"}, f"{localdir}/mesh/info")

    logger.info("Generating meshes")
    meshes = Mesh.from_label_volume(vol, vol_fullres_box, smoothing_rounds=2)

    logger.info("Simplifying meshes")
    for mesh in meshes.values():
        mesh.simplify(0.05)

    logger.info("Serializing meshes")
    for label, mesh in meshes.items():
        name = names.get(label, str(label))
        mesh.serialize(f"{localdir}/mesh/{name}.ngmesh")
        dump_json({"fragments": [f"{name}.ngmesh"]}, f"{localdir}/mesh/{label}:0")

    subprocess.run(f"gsutil cp {bucket_name}/{bucket_path}/info {localdir}/info", shell=True)
    with open(f"{localdir}/info", 'r') as f:
        info = json.load(f)

    info["mesh"] = "mesh"
    dump_json(info, f"{localdir}/info", unsplit_int_lists=True)

    logger.info("Uploading")
    subprocess.run(f"gsutil cp {localdir}/info {bucket_name}/{bucket_path}/info", shell=True)
    subprocess.run(f"gsutil cp -R {localdir}/mesh {bucket_name}/{bucket_path}/mesh", shell=True)


def create_precomputed_segment_properties(names, bucket_name, bucket_path, localdir=None):
    if not bucket_name.startswith('gs://'):
        bucket_name = 'gs://' + bucket_name

    if localdir is None:
        localdir = bucket_path.split('/')[-1]

    os.makedirs(f"{localdir}/segment_properties", exist_ok=True)

    props = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [],
            "properties": [
                {
                    "id": "source",
                    "type": "label",
                    "values": []
                }
            ]
        }
    }

    for label, name in names.items():
        props["inline"]["ids"].append(str(label))
        props["inline"]["properties"][0]["values"].append(name)

    dump_json(props, f"{localdir}/segment_properties/info", unsplit_int_lists=True)

    subprocess.run(f"gsutil cp {bucket_name}/{bucket_path}/info {localdir}/info", shell=True)
    with open(f"{localdir}/info", 'r') as f:
        info = json.load(f)

    info["segment_properties"] = "segment_properties"
    dump_json(info, f"{localdir}/info", unsplit_int_lists=True)

    subprocess.run(f"gsutil cp {localdir}/info {bucket_name}/{bucket_path}/info", shell=True)
    subprocess.run(f"gsutil cp -R {localdir}/segment_properties {bucket_name}/{bucket_path}/segment_properties", shell=True)


