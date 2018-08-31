import requests
from tqdm import tqdm

import numpy as np
import pandas as pd

from neuclease.dvid import fetch_missing, fetch_sizes

def check_tarsupervoxels_status(server, uuid, tsv_instance, seg_instance, bodies):
    """
    For the given bodies, query the given tarsupervoxels instance and return a
    DataFrame indicating which supervoxels are 'missing' from the instance,
    along with their sizes.
    
    Bodies that no longer exist in the segmentation instance are ignored.
    """
    body_sv_sizes = []
    for body in tqdm(bodies):
        try:
            missing_svs = fetch_missing(server, uuid, tsv_instance, body)
        except requests.RequestException as ex:
            if 'has no supervoxels' in ex.args[0]:
                continue
            else:
                raise
            
        if len(missing_svs) == 0:
            continue

        sizes = fetch_sizes(server, uuid, seg_instance, missing_svs, supervoxels=True)
        body_sv_sizes += [(body, sv, size) for (sv, size) in zip(missing_svs, sizes)]

    df = pd.DataFrame(body_sv_sizes, columns=['body', 'sv', 'size'], dtype=np.uint64)
    df.set_index('sv', inplace=True)
    return df
