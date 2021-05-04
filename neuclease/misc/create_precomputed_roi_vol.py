import os
import json
import subprocess

import numpy as np

from neuclease.util import tqdm_proxy as tqdm, dump_json


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
