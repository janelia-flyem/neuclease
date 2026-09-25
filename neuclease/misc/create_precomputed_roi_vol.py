import os
import re
import glob
import json
import copy
import logging
import tempfile
from functools import partial
from collections.abc import Mapping, Collection

import numpy as np
import pandas as pd

from vol2mesh import Mesh
from neuclease.util import tqdm_proxy as tqdm, dump_json, compute_parallel, region_boxes, box_to_slicing, compute_nonzero_box
from neuclease.util import gcs
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
                                             localdir=None, steps={'voxels', 'meshes', 'properties'}, permit_overlaps=False,
                                             processes=0):
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
    assert not invalid_steps, f"Invalid steps: {steps}"

    if not localdir:
        localdir = tempfile.mkdtemp()

    os.makedirs(localdir, exist_ok=True)

    # First, verify that we have permission to edit the bucket.
    gcs.check_bucket_access(f"{bucket_name}/{bucket_path}")

    if isinstance(rois, pd.Series):
        roi_names = dict(rois.items())
        rois = {name: label for label, name in roi_names.items()}
    else:
        roi_names = dict(enumerate(rois, start=1))
        if sorted(rois) != rois:
            logger.warning("Your ROIs aren't sorted")

    logger.info("Consructing segmentation volume from ROI RLEs")
    roi_vol, roi_box, overlaps = fetch_combined_roi_volume(server, uuid, rois, box_zyx=[(0,0,0), None])
    if len(overlaps) and not permit_overlaps:
        raise RuntimeError(f"The ROIs you specified overlap:\n{overlaps}")

    if 'voxels' in steps:
        logger.info("Uploading segmentation volume")
        create_precomputed_roi_vol(roi_vol, bucket_name, bucket_path)

    if 'meshes' in steps:
        roi_res = scale_0_res * (2**5)

        logger.info("Preparing legacy neuroglancer meshes")
        # pad volume to ensure mesh faces on all sides
        roi_vol = np.pad(roi_vol, 1)
        roi_box += [[-1, -1, -1], [1, 1, 1]]
        create_precomputed_ngmeshes(roi_vol, roi_res * roi_box, roi_names, bucket_name, bucket_path, localdir, decimation, processes=processes)

    if 'properties' in steps:
        logger.info("Adding segment properties (ROI names)")
        create_precomputed_segment_properties(roi_names, bucket_name, bucket_path, localdir)

    logger.info(f"Done creating layer in {bucket_name}/{bucket_path}")


def construct_ng_precomputed_layer_from_roi_seg(roi_vol, roi_names, bucket_name, bucket_path, scale_0_res=8, decimation=0.01,
                                                localdir=None, steps={'voxels', 'meshes', 'properties'}, processes=0):
    """
    Similar to above, but when you have an ROI volume from elsewhere (not ROIs in DVID).

    1. Upload the label volume to a google bucket in neuroglancer precomputed format, using tensorstore.

    2. Generate a mesh for each ROI in the label volume, and upload it in neuroglancer's "legacy" (single resolution) format.
        - Upload to a directory named .../mesh
        - Also edit json files:
            .../info
            .../mesh/info

    3. Update the neuroglancer "segment properties" metadata file for the layer.
        - Edit json files:
            .../info
            .../segment_properties/info

    Args:
        roi_vol:
            Must have scale-5 resolution, and must start at (0,0,0).

        roi_names:
            Should be a dict of {id: name}
    """
    invalid_steps = set(steps) - {'voxels', 'meshes', 'properties'}
    assert not invalid_steps, f"Invalid steps: {steps}"

    if set(steps) & {'meshes', 'properties'}:
        assert isinstance(roi_names, Mapping), \
            "roi_names should be dict of {id: name}"
        assert all(np.issubdtype(type(k), np.integer) for k in roi_names.keys()), \
            "roi_names should be dict of {id: name}"
        assert all(isinstance(v, str) for v in roi_names.values()), \
            "roi_names should be dict of {id: name}"

    if not localdir:
        localdir = tempfile.mkdtemp()
    os.makedirs(localdir, exist_ok=True)

    scale_0_res = np.asarray(scale_0_res)

    # First, verify that we have permission to edit the bucket.
    gcs.check_bucket_access(f"{bucket_name}/{bucket_path}")

    if 'voxels' in steps:
        logger.info("Uploading segmentation volume")
        create_precomputed_roi_vol(roi_vol, bucket_name, bucket_path, resolution_nm=scale_0_res*(2**5))

    if 'meshes' in steps:
        logger.info("Preparing legacy neuroglancer meshes")
        roi_res = scale_0_res * (2**5)
        roi_box = np.array([(0,0,0), roi_vol.shape])

        # pad volume to ensure mesh faces on all sides
        roi_vol = np.pad(roi_vol, 1)
        roi_box += [[-1, -1, -1], [1, 1, 1]]
        create_precomputed_ngmeshes(roi_vol, roi_res * roi_box, roi_names, bucket_name, bucket_path, localdir, decimation, processes=processes)

    if 'properties' in steps:
        logger.info("Adding segment properties (ROI names)")
        create_precomputed_segment_properties(roi_names, bucket_name, bucket_path, localdir)

    logger.info(f"Done creating layer in {bucket_name}/{bucket_path}")


def create_precomputed_roi_vol(roi_vol, bucket_name, bucket_path, max_scale=3, resolution_nm=8*(2**5)):
    """
    Upload the given ROI volume (which shoud usually be a scale-5 volume, i.e. 256nm resolution)
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

    if not isinstance(resolution_nm, Collection):
        resolution_nm = 3 * (resolution_nm,)

    resolution_nm = np.asarray(resolution_nm)[::-1]
    
    for scale in tqdm(range(1 + max_scale)):
        res = resolution_nm * 2**scale
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
                "resolution": res.tolist()
            }
        }).result()
        if scale == 0:
            v = roi_vol.transpose()[..., None]
            nzbox = compute_nonzero_box(v)
            store[box_to_slicing(*nzbox)] = v[box_to_slicing(*nzbox)]
        else:
            # Subsample
            v = roi_vol.transpose()[:-2**scale+1:2**scale, :-2**scale+1:2**scale, :-2**scale+1:2**scale, None]
            nzbox = compute_nonzero_box(v)
            store[box_to_slicing(*nzbox)] = v[box_to_slicing(*nzbox)]


def create_precomputed_ngmeshes(vol, vol_fullres_box, names, bucket_name, bucket_path, localdir=None, decimation=0.01, volume_info=None, processes=0):
    """
    Create meshes for the given label volume and upload them to a google bucket in
    neuroglancer legacy mesh format (i.e. what flyem calls "ngmesh" format).

    Args:
        vol_fullres_box:
            Full resolution box, in NANOMETERS
    """
    logger.info("Generating meshes")
    num_labels = len(set(pd.unique(vol.reshape(-1))) - {0})

    boxes = region_boxes(vol)
    valid = (boxes[:, 0, :] < boxes[:, 1, :]).all(axis=1)
    valid[0] = False  # skip label 0
    boxes = pd.Series(boxes.tolist()).iloc[valid]

    def _gen_masks():
        for label, box in boxes.items():
            subvol = vol[box_to_slicing(*box)]
            mask = (subvol == label)
            res = (vol_fullres_box[1] - vol_fullres_box[0]) / vol.shape
            yield label, box * res, mask

    fn = partial(_gen_mesh, 2, decimation)
    meshes = compute_parallel(fn, _gen_masks(), starmap=True, processes=processes, total=num_labels)
    meshes = dict(meshes)

    upload_precomputed_ngmeshes(meshes, names, bucket_name, bucket_path, localdir, volume_info)


def _gen_mesh(smoothing_rounds, decimation, label, fullres_box, mask):
    # Apparently the 'ilastik' method isn't process-safe anymore??
    # mesh = Mesh.from_binary_vol(mask, fullres_box, method='ilastik', ensure_halo=True, smoothing_rounds=smoothing_rounds)
    mesh = Mesh.from_binary_vol(mask, fullres_box, method='skimage', ensure_halo=True)
    mesh.laplacian_smooth(smoothing_rounds)
    mesh.simplify(decimation)
    return label, mesh


def upload_precomputed_ngmeshes(meshes, names, bucket_name, bucket_path, localdir=None, volume_info=None, fix_case_insensitive_clashes=True):
    """
    fix_case_insensitive_clashes:
        Although Linux and Google Cloud Storage are case-sensitive,
        anyone downloading the mesh files onto a Mac will be bitten if
        two file names differ only by their case.
        It just so happens that drosophila datasets have compartments named:

            - AL(L)/AL(R) (Antennal Lobe)
            - aL(L)/aL(R) (alpha lobe)
        
        On a Mac, AL(L).ngmesh and aL(L).ngmesh are treated as the SAME FILE,
        and gsutil does not warn you about this when downloading them.
        To make matters worse, gsutil will download both files simultaneously,
        intermingling their contents on disk, resulting in a corrupted file!

        If fix_case_insensitive_clashes is True, we will fix name clashes by appending
        one or more underscores (before the file .ngmesh file extension) to each of the
        "duplicate" names.
    """
    if not bucket_name.startswith('gs://'):
        bucket_name = 'gs://' + bucket_name

    if localdir is None:
        localdir = bucket_path.split('/')[-1]

    os.makedirs(f"{localdir}/mesh", exist_ok=True)
    dump_json({"@type": "neuroglancer_legacy_mesh"}, f"{localdir}/mesh/info")

    if fix_case_insensitive_clashes:
        names = pd.Series(names).sort_values()
        inames = names.str.lower()
        clashes = inames.duplicated()
        while clashes.any():
            names[clashes] = names[clashes] + "_"
            inames = names.str.lower()
            clashes = inames.duplicated()
        names = names.to_dict()

    logger.info("Serializing meshes")
    for label, mesh in meshes.items():
        name = names.get(label, str(label))
        mesh.serialize(f"{localdir}/mesh/{name}.ngmesh")
        dump_json({"fragments": [f"{name}.ngmesh"]}, f"{localdir}/mesh/{label}:0")

    if volume_info:
        volume_info = copy.deepcopy(volume_info)
    else:
        if not gcs.download_to_file(bucket_name, f"{bucket_path}/info", f"{localdir}/info"):
            raise FileNotFoundError(f"No existing 'info' file found at {bucket_name}/{bucket_path}/info")
        with open(f"{localdir}/info", 'r') as f:
            volume_info = json.load(f)

    volume_info["mesh"] = "mesh"
    dump_json(volume_info, f"{localdir}/info", unsplit_int_lists=True)

    logger.info("Uploading")
    gcs.upload_file(bucket_name, f"{bucket_path}/info", f"{localdir}/info", disable_cache=True)
    gcs.upload_directory(bucket_name, f"{bucket_path}/mesh", f"{localdir}/mesh", disable_cache=True)


def upload_precomputed_ngmesh_files(mesh_dir, names, bucket_name, bucket_path):
    if not bucket_name.startswith('gs://'):
        bucket_name = 'gs://' + bucket_name

    dump_json({"@type": "neuroglancer_legacy_mesh"}, f"{mesh_dir}/info")

    logger.info("Writing fragment index files")
    names = names or {}
    revnames = {v:k for k,v in names.items()}
    for p in tqdm_proxy(sorted(glob.glob(f"{mesh_dir}/*.ngmesh"))):
        if (m := re.match(mesh_dir + r"/(\d+).ngmesh", p)):
            label = m.groups()[0]
            name = names.get(label, str(label))
        elif (m := re.match(mesh_dir + r"/(.+).ngmesh", p)):
            name = m.groups()[0]
            label = names.get(name, None) or revnames.get(name, None)
            if not label:
                raise RuntimeError(f"Couldn't determine label for mesh '{name}'")

        dump_json({"fragments": [f"{name}.ngmesh"]}, f"{mesh_dir}/{label}:0")

    logger.info("Uploading")
    # Note: gsutil's `cp -R <dir> <dest>` nests the source dir (by basename) under <dest>,
    # so we replicate that here rather than uploading directly into bucket_path.
    dest_prefix = f"{bucket_path}/{os.path.basename(mesh_dir.rstrip('/'))}"
    gcs.upload_directory(bucket_name, dest_prefix, mesh_dir)


def create_precomputed_segment_properties(names, bucket_name, bucket_path, localdir=None, volume_info=None):
    """
    Write the "segment properties" for a neuroglancer precomputed volume,
    i.e. the segment names.

    Args:
        names:
            dict {label: name}
        bucket_name:
            destination bucket
        bucket_path:
            Location within bucket
        localdir:
            Where to store the files locally
        volume_info:
            A copy of the original segmentation's /info file.
            If not given, it will be fetched.
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

    if volume_info is not None:
        volume_info = copy.deepcopy(volume_info)
    else:
        if not gcs.download_to_file(bucket_name, f"{bucket_path}/info", f"{localdir}/info"):
            raise FileNotFoundError(f"No existing 'info' file found at {bucket_name}/{bucket_path}/info")
        with open(f"{localdir}/info", 'r') as f:
            volume_info = json.load(f)

    volume_info["segment_properties"] = "segment_properties"
    dump_json(volume_info, f"{localdir}/info", unsplit_int_lists=True)

    gcs.upload_file(bucket_name, f"{bucket_path}/info", f"{localdir}/info", disable_cache=True)
    gcs.upload_directory(bucket_name, f"{bucket_path}/segment_properties", f"{localdir}/segment_properties", disable_cache=True)


def create_legacy_mesh_info(mesh_dir, names=None):
    """
    Given a (local) directory of neuroglancer 'legacy'
    mesh files (we usually call them .ngmesh files),
    add the appropriate metadata files so that neuroglancer
    can fetch the correct mesh for each segment ID.

    Args:
        mesh_dir:
            Path to a local directory containing mesh files
        names:
            Optional.  A dict of {name: label}, which will be used to
            determine which mesh file corresponds to which label ID.
            The name should not include the .ngmesh file extension.
            If not provided, the mesh files must be named like '123.ngmesh'
    """
    paths = sorted(glob.glob(f'{mesh_dir}/*.ngmesh'))

    if names is None:
        names = [
            p.split('/')[-1][:-len('.ngmesh')]
            for p in paths
        ]
        names = {
            name: int(name)
            for name in names
        }
    else:
        names = {
            os.path.splitext(name)[0].split('/')[-1]: int(label)
            for name, label in names.items()
        }

    dump_json({"@type": "neuroglancer_legacy_mesh"}, f"{mesh_dir}/info")

    for path in tqdm(sorted(paths)):
        name = os.path.splitext(path)[0].split('/')[-1]
        dump_json(
            {"fragments": [f"{name}.ngmesh"]},
            f"{mesh_dir}/{names[name]}:0"
        )
