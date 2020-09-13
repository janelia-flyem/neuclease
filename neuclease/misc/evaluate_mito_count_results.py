"""
Compares the proofreaders' mito-count results with automated mito counts from a mito segmentation instance.
This script requires the proofreaders' mito-count results in table form,
as exported via export_mito_count_results.py
"""
import sys
import copy
import json
import pickle
import urllib
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label

from neuprint import Client, fetch_neurons
from neuclease.util import round_box, mask_for_labels, compute_nonzero_box, extract_subvol, contingency_table, binary_edge_mask, iter_batches, compute_parallel
from neuclease.dvid import fetch_info, fetch_labelmap_voxels, fetch_sizes, fetch_labels_batched, determine_point_rois

EXAMPLE_NEUROGLANCER_LINK = """\
http://hemibrain-dvid2.janelia.org:8000/neuroglancer/#!%7B%22dimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B9999.525390625%2C19841.533203125%2C13040%5D%2C%22crossSectionScale%22:0.406569659740599%2C%22projectionOrientation%22:%5B-0.30359750986099243%2C0.21658426523208618%2C-0.08182431757450104%2C0.9242427349090576%5D%2C%22projectionScale%22:804.6077478874928%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg%22%2C%22tab%22:%22source%22%2C%22name%22:%22jpeg%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.1/segmentation%22%2C%22tab%22:%22source%22%2C%22name%22:%22segmentation%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22dvid://http://hemibrain-dvid2.janelia.org:8000/a7e1303c0c294ed99b66be7ab679cbb5/neighborhood-masks%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%2256007023659456%22%5D%2C%22name%22:%22neighborhood-masks%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22dvid://http://emdata3:8900/62f6394a18d4490c93892fbd9f1416b5/masked-mito-cc%22%2C%22tab%22:%22source%22%2C%22name%22:%22masked-mito-cc%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22masked-mito-cc%22%2C%22visible%22:true%7D%2C%22layout%22:%224panel%22%7D
"""

NEUROGLANCER_ENDPOINT = "http://hemibrain-dvid2.janelia.org:8000/neuroglancer"
NEUROGLANCER_SETTINGS = {
    "dimensions": {
        "x": [8e-09, "m"], "y": [8e-09, "m"], "z": [8e-09, "m"]
    },
    "position": [10314, 14942, 11732],
    "crossSectionScale": 0.5,
    "projectionScale": 36.0,
    "layers": [
        {
            "type": "image",
            "source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
            "tab": "source",
            "name": "em-clahe"
        },
        {
            "type": "segmentation",
            "source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.1/segmentation",
            "tab": "source",
            "name": "segmentation",
            "visible": False
        },
        {
            "type": "segmentation",
            "source": "dvid://http://hemibrain-dvid2.janelia.org:8000/a7e1303c0c294ed99b66be7ab679cbb5/neighborhood-masks",
            "tab": "segments",
            "name": "neighborhood-masks"
        },
        {
            "type": "segmentation",
            "source": "dvid://http://emdata3:8900/62f6394a18d4490c93892fbd9f1416b5/masked-mito-cc",
            "tab": "segments",
            "name": "mito-svs"
        }
    ],
    "showSlices": False,
    "layout": "4panel"
}

# Unfortunately, this is not stored in the task!
RADIUS = 250
MITO_SVS = ('http://emdata3.int.janelia.org:8900', '62f6394a18d4490c93892fbd9f1416b5', 'masked-mito-cc') # for visualization
MITO_SEG = ('http://emdata3.int.janelia.org:8900', 'd31b64ac81444923a0319961736a6c31', 'masked-mito-cc')

# Ideally, we would have listed this in the assignments, indicating
# which segmentation was used to generate the neighborhoods.
NEURON_SEG = ('http://emdata4.int.janelia.org:8900', '20631f94c3f446d7864bc55bf515706e', 'segmentation')

NEUPRINT_CLIENT = Client('neuprint.janelia.org', 'hemibrain:v1.1')

# Mito size threshold (in terms of 8nm voxels)
# Mito segments smaller than this weren't assigned
# to any body and weren't loaded into neuprint,
# so mitos smaller than this should not be included in our counts here.
MIN_MITO_SIZE = 10_000

# Some of our mito segmentations are stored at 8nm resolution, some at 16nm,
# but we assume the neighborhood resolution is full-scale (8nm)
NEIGHBORHOOD_RES = 8

# It doesn't make sense to do this analysis at scale 0,
# since the mitos were generated at scale-1 anyway.
ANALYSIS_SCALE = 1

# Note about analysis scale and neighborhood erosion step
# -------------------------------------------------------
# Our mito segmentation was generated at scale 1, bur our neighborhoods were written at scale 0.
# Downsampling introduces discrepancies between the mito borders and neighborhood borders.
# Those discrepancies are not an issue for our published mito statistics, since our mito objects
# were generated from the scale-1 mask overlaid with the scale-1 neuron segmentation,
# and all of our subsequent analyses used scale-1 for both, too.
# (For that reason, perhaps we should have generated these neighborhood objects using scale-1, too. Oh well.)
# We can't simply use scale-1 version of the neighborhood, since it was not necessarily
# downsampled via the exact same algorithm that was used to produce the scale-1 neuron segmentation
# (and thus it won't necessarily conform to the scale-1 mito boundaries).
# As a workaround, we will simply erode the neighborhood slightly to make sure we aren't
# catching mitos from nearby neurons in this analysis.


def main():
    RESULTS_PKL_PATH = sys.argv[1]
    if len(sys.argv) == 3:
        PROCESSES = sys.arvc[2]
    else:
        PROCESSES = 4

    # Calculate the difference in resolution between the stored mito segmentation and neuron segmenation.
    # If they differ, it must be by a power of 2.
    mito_res = fetch_info(*MITO_SEG)["Extended"]["VoxelSize"][0]
    assert mito_res % NEIGHBORHOOD_RES == 0
    assert np.log2(mito_res / NEIGHBORHOOD_RES) == int(np.log2(mito_res / NEIGHBORHOOD_RES)), \
        "This script assumes that the mito resolution and neighborhood resolution differ by a power of 2."
    mito_res_scale_diff = int(np.log2(mito_res // NEIGHBORHOOD_RES))

    with open(RESULTS_PKL_PATH, 'rb') as f:
        mc_df = pickle.load(f)

    new_names = {col: col.replace(' ', '_') for col in mc_df.columns}
    new_names['result'] = 'proofreader_count'
    mc_df = mc_df.rename(columns=new_names)

    print("Evaluating mito count results")
    results = compute_parallel(
        partial(_task_results, mito_res_scale_diff),
        iter_batches(mc_df.drop_duplicates('neighborhood_id'), 1),
        total=len(mc_df),
        processes=PROCESSES,
        leave_progress=True,
        ordered=False
    )

    cols = ['neighborhood_id',
            'neighborhood_origin',
            'proofreader_count',
            'mito_id_count',
            'mito_ids',
            'mito_sizes',
            'num_ccs',
            'mito_cc_ids',
            'mito_cc_sizes',
            'ng_link']

    df = pd.DataFrame(results, columns=cols)

    # Add columns for cell type (from neuprint)
    print("Fetching neuron cell types")
    origins_df = pd.DataFrame(df['neighborhood_origin'].tolist(), columns=[*'xyz'])
    df['body'] = fetch_labels_batched(*NEURON_SEG, origins_df[[*'zyx']].values, processes=8)
    neurons_df, _ = fetch_neurons(df['body'].unique())
    neurons_df = neurons_df.rename(columns={'bodyId': 'body', 'type': 'body_type', 'instance': 'body_instance'})
    df = df.merge(neurons_df[['body', 'body_type', 'body_instance']], 'left', on='body')
    df['body_type'].fillna("", inplace=True)
    df['body_instance'].fillna("", inplace=True)

    # Append roi column
    print("Determining ROIs")
    determine_point_rois(*NEURON_SEG[:2], NEUPRINT_CLIENT.primary_rois, origins_df)
    df['roi'] = origins_df['roi']

    # Results only
    path = 'mito-seg-counts.pkl'
    print(f"Writing {path}")
    with open(path, 'wb') as f:
        pickle.dump(df, f)

    path = 'mito-seg-counts.tab-delimited.csv'
    print(f"Writing {path}")
    df.to_csv(path, sep='\t', header=True, index=False)

    # Full results (with task info columns)
    df = df.merge(mc_df.drop(columns=['neighborhood_origin', 'proofreader_count']), 'left', on='neighborhood_id')

    path = 'full-results-with-mito-seg-counts.pkl'
    print(f"Writing {path}")
    with open(path, 'wb') as f:
        pickle.dump(df, f)

    path = 'full-results-with-mito-seg-counts.tab-delimited.csv'
    print(f"Writing {path}")
    df.to_csv(path, sep='\t', header=True, index=False)

    print("DONE")


def _task_results(mito_res_scale_diff, task_df):
    """
    Process the given input row and return a tuple to be used as a row of output results.
    """
    assert len(task_df) == 1
    row = next(task_df.itertuples())
    mito_table = mitos_in_neighborhood( row.mito_ROI_source,
                                        row.neighborhood_origin,
                                        row.neighborhood_id,
                                        mito_res_scale_diff )

    mito_ids = mito_table.index.drop_duplicates().values
    mito_sizes = mito_table.reset_index().drop_duplicates('mito')['size']
    mito_ccs = mito_table.groupby(mito_table.index, sort=False).agg({'cc': tuple, 'cc_size': tuple})

    ng_settings = copy.deepcopy(NEUROGLANCER_SETTINGS)
    ng_settings['position'] = list(row.neighborhood_origin)

    def select_layer(name):
        return [l for l in ng_settings["layers"] if l['name'] == name][0]

    select_layer("neighborhood-masks")["source"] = row.mito_ROI_source
    select_layer("mito-svs")["source"] = "dvid://{}/{}/{}".format(*MITO_SVS)
    select_layer("mito-svs")["segments"] = [str(mito) for mito in mito_ids]

    ng_link = NEUROGLANCER_ENDPOINT + '/#!' + urllib.parse.quote(json.dumps(ng_settings))

    r = ( row.neighborhood_id,
          row.neighborhood_origin,
          row.proofreader_count,
          mito_table.index.nunique(),
          tuple(mito_ids),
          tuple(mito_sizes),
          mito_table['cc'].nunique(),
          tuple(mito_ccs['cc']),
          tuple(mito_ccs['cc_size']),
          ng_link )

    return r


def mitos_in_neighborhood(mito_roi_source, neighborhood_origin_xyz, neighborhood_id, mito_res_scale_diff):
    """
    Determine how many non-trivial mito objects overlap with the given "neighborhood object",
    and return a table of their IDs and sizes.

    1. Download the neighborhood mask for the given neighborhood_id.
    2. Erode the neighborhood mask by 1 px (see note in the comment above).
    3. Fetch the mito segmentation for the voxels within the neighborhood.
    4. Fetch (from dvid) the sizes of each mito object.
    5. Filter out the mitos that are smaller than the minimum size that is
       actually used in our published mito analyses.
    6. Just for additional info, determine how many connected components
       are formed by the mito objects.
    7. Return the mito IDs, sizses, and CC info as a DataFrame.
    """
    # The neighborhood segmentation source
    protocol, url = mito_roi_source.split('://')[-2:]
    server, uuid, instance = url.split('/')
    server = f'{protocol}://{server}'

    origin_zyx = np.array(neighborhood_origin_xyz[::-1])
    box = [origin_zyx - RADIUS, 1 + origin_zyx + RADIUS]

    # Align box to the analysis scale before scaling it.
    box = round_box(box, (2**ANALYSIS_SCALE))

    # Scale box
    box //= (2**ANALYSIS_SCALE)

    neighborhood_seg = fetch_labelmap_voxels(server, uuid, instance, box, scale=ANALYSIS_SCALE)
    neighborhood_mask = (neighborhood_seg == neighborhood_id)

    # This is equivalent to a 1-px erosion
    # See note above for why we do this.
    neighborhood_mask ^= binary_edge_mask(neighborhood_mask, 'inner')

    mito_seg = fetch_labelmap_voxels(*MITO_SEG, box, supervoxels=True, scale=ANALYSIS_SCALE-mito_res_scale_diff)
    assert neighborhood_mask.shape == mito_seg.shape
    mito_seg = np.where(neighborhood_mask, mito_seg, 0)

    # The mito segmentation includes little scraps and slivers
    # that were filtered out of the "real" mito set.
    # Filter those scraps out of our results here.
    mito_ids = set(pd.unique(mito_seg.ravel())) - {0}
    mito_sizes = fetch_sizes(*MITO_SEG, [*mito_ids], supervoxels=True)
    mito_sizes = mito_sizes.rename_axis('mito')
    mito_sizes *= (2**mito_res_scale_diff)**3

    # This is our main result: mito IDs (and their sizes)
    mito_sizes = mito_sizes.loc[mito_sizes >= MIN_MITO_SIZE]

    # Just for extra info, group the mitos we found into connected components.
    mito_mask = mask_for_labels(mito_seg, mito_sizes.index)
    mito_box = compute_nonzero_box(mito_mask)
    mito_mask = extract_subvol(mito_mask, mito_box)
    mito_seg = extract_subvol(mito_seg, mito_box)
    mito_cc = label(mito_mask)
    ct = contingency_table(mito_seg, mito_cc).reset_index()
    ct = ct.rename(columns={'left': 'mito', 'right': 'cc', 'voxel_count': 'cc_size'})
    ct = ct.set_index('mito')
    mito_sizes = pd.DataFrame(mito_sizes).merge(ct, 'left', left_index=True, right_index=True)
    return mito_sizes


if __name__ == "__main__":
    #import os
    #os.chdir('/Users/bergs/Documents/FlyEM/mito-project/proofreading/mito-count')
    #sys.argv.append('mito-count-results.pkl')
    main()
