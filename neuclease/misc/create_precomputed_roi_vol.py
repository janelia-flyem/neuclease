import os
import json
import copy
import logging
import tempfile
import subprocess

import numpy as np

from neuclease.util import tqdm_proxy as tqdm, dump_json
from neuclease.dvid import fetch_combined_roi_volume

logger = logging.getLogger()


# Useful if you need to load a volume without tensorstore,
# e.g. if you're just uploading meshes and segment_properties
DEFAULT_VOLUME_INFO = {
    "@type": "neuroglancer_multiscale_volume",
    "scales": [{}],
    "data_type": "uint64",
    "num_channels": 1,
    "mesh": "mesh",
    "type": "segmentation",
}


def construct_ng_precomputed_layer_from_rois(server, uuid, rois, bucket_name, bucket_path, scale_0_res=8, decimation=0.01,
                                             localdir=None, steps={'voxels', 'meshes', 'properties'}):
    """
    Given a list of ROIs, generate a neuroglancer precomputed layer for them.

    The process is as follows:

    1. Download the ROI data from dvid (RLE format), and load them into a single label volume.

    2. Upload the label volume to a google bucket in neuroglancer precomputed format, using tensorstore.

    3. Generate a mesh for each ROI in the label volume, and upload it in neuroglancer's "legacy" (single resolution) format.
        - Upload to a directory named .../mesh
        - Also edit json files:
            .../info
            .../mesh/info

    4. Update the neuroglancer "segment properties" metadata file for the layer.
        - Edit json files:
            .../info
            .../segment_properties/info
    """
    invalid_steps = set(steps) - {'voxels', 'meshes', 'properties'}
    assert not invalid_steps, "Invalid steps: {steps}"

    if not localdir:
        localdir = tempfile.mkdtemp()

    # First, verify that we have permission to edit the bucket.
    with open(f"{localdir}/test-file.txt", 'w') as f:
        f.write("Just testing my bucket access...\n")
    subprocess.run(f"gsutil cp {localdir}/test-file.txt {bucket_name}/{bucket_path}/test-file.txt", shell=True, check=True)
    subprocess.run(f"gsutil rm {bucket_name}/{bucket_path}/test-file.txt", shell=True, check=True)

    if sorted(rois) != rois:
        logger.warning("Your ROIs aren't sorted")
    roi_names = dict(enumerate(rois, start=1))

    logger.info("Consructing segmentation volume from ROI RLEs")
    roi_vol, roi_box, overlaps = fetch_combined_roi_volume(server, uuid, rois, box_zyx=[(0,0,0), None])
    if len(overlaps):
        raise RuntimeError("The ROIs you specified overlap:\n{overlaps}")

    if 'voxels' in steps:
        logger.info("Uploading segmentation volume")
        create_precomputed_roi_vol(roi_vol, bucket_name, bucket_path)

    if 'meshes' in steps:
        roi_res = scale_0_res * (2**5)

        logger.info("Preparing legacy neuroglancer meshes")
        # pad volume to ensure mesh faces on all sides
        roi_vol = np.pad(roi_vol, 1)
        roi_box += [[-1, -1, -1], [1, 1, 1]]
        create_precomputed_ngmeshes(roi_vol, roi_res * roi_box, roi_names, bucket_name, bucket_path, localdir, decimation)

    if 'properties' in steps:
        logger.info("Adding segment properties (ROI names)")
        create_precomputed_segment_properties(roi_names, bucket_name, bucket_path, localdir)

    logger.info(f"Done creating layer in {bucket_name}/{bucket_path}")


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
            # Subsample
            store[:] = roi_vol.transpose()[:-2**scale+1:2**scale, :-2**scale+1:2**scale, :-2**scale+1:2**scale, None]


def create_precomputed_ngmeshes(vol, vol_fullres_box, names, bucket_name, bucket_path, localdir=None, decimation=0.01, volume_info=None):
    """
    Create meshes for the given labelvolume and upload them to a google bucket in
    neuroglancer legacy mesh format (i.e. what flyem calls "ngmesh" format).
    """
    from vol2mesh import Mesh
    logger.info("Generating meshes")
    meshes = Mesh.from_label_volume(vol, vol_fullres_box, smoothing_rounds=2)

    logger.info("Simplifying meshes")
    for mesh in meshes.values():
        mesh.simplify(decimation)

    upload_precompted_ngmeshes(meshes, names, bucket_name, bucket_path, localdir, volume_info)


def upload_precompted_ngmeshes(meshes, names, bucket_name, bucket_path, localdir=None, volume_info=None):
    if not bucket_name.startswith('gs://'):
        bucket_name = 'gs://' + bucket_name

    if localdir is None:
        localdir = bucket_path.split('/')[-1]

    os.makedirs(f"{localdir}/mesh", exist_ok=True)
    dump_json({"@type": "neuroglancer_legacy_mesh"}, f"{localdir}/mesh/info")

    logger.info("Serializing meshes")
    for label, mesh in meshes.items():
        name = names.get(label, str(label))
        mesh.serialize(f"{localdir}/mesh/{name}.ngmesh")
        dump_json({"fragments": [f"{name}.ngmesh"]}, f"{localdir}/mesh/{label}:0")

    if volume_info:
        volume_info = copy.deepcopy(volume_info)
    else:
        subprocess.run(f"gsutil -h 'Cache-Control:public, no-store' cp {bucket_name}/{bucket_path}/info {localdir}/info", shell=True)
        with open(f"{localdir}/info", 'r') as f:
            volume_info = json.load(f)

    volume_info["mesh"] = "mesh"
    dump_json(volume_info, f"{localdir}/info", unsplit_int_lists=True)

    logger.info("Uploading")
    subprocess.run(f"gsutil -h 'Cache-Control:public, no-store' cp {localdir}/info {bucket_name}/{bucket_path}/info", shell=True)
    subprocess.run(f"gsutil cp -R {localdir}/mesh {bucket_name}/{bucket_path}/mesh", shell=True)


def create_precomputed_segment_properties(names, bucket_name, bucket_path, localdir=None, volume_info=None):
    """
    Write the "segment properties" for a neuroglancer precomputed volume,
    i.e. the segment names.
    """
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

    if volume_info:
        volume_info = copy.deepcopy(volume_info)
    else:
        subprocess.run(f"gsutil cp {bucket_name}/{bucket_path}/info {localdir}/info", shell=True)
        with open(f"{localdir}/info", 'r') as f:
            volume_info = json.load(f)

    volume_info["segment_properties"] = "segment_properties"
    dump_json(volume_info, f"{localdir}/info", unsplit_int_lists=True)

    subprocess.run(f"gsutil -h 'Cache-Control:public, no-store' cp {localdir}/info {bucket_name}/{bucket_path}/info", shell=True)
    subprocess.run(f"gsutil -h 'Cache-Control:public, no-store' cp -R {localdir}/segment_properties {bucket_name}/{bucket_path}/segment_properties", shell=True)
