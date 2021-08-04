from functools import partial
import numpy as np
import pandas as pd
from requests.exceptions import HTTPError

from neuclease.dvid import generate_sample_coordinate
from neuclease.clio.api import fetch_json_annotations_all
from neuclease.misc.vnc_statuses import fetch_vnc_statuses
from neuclease.util import compute_parallel, Timer


def determine_clio_positions(server, uuid, clio_ann=None, dvid_ann=None):
    """
    A one-off function for updating the 'position' field in all
    Clio body annotations which currently lack a position.

    Final position is chosen as follows:

    - Use the user-supplied point if one already exists in clio
    - Otherwise, use the soma position (from DVID) if one exists
    - Otherwise, use the neck position (from DVID) if one exists
    - Otherwise, auto-select a point using generate_sample_coordinate()

    Returns:
        DataFrame with the original Clio annotations, with 'position'
        overwritten as well as some extra columns.

        A 'position_type' column is added to the results to indicate which
        of the above cases applies to each row.
    """
    if clio_ann is None:
        with Timer("Fetching Clio annotations"):
            clio_ann = fetch_json_annotations_all('VNC')

    if dvid_ann is None:
        with Timer("Fetching DVID annotations"):
            dvid_ann = fetch_vnc_statuses(server, uuid)

    # deduplicate to eliminate double-ascending entries
    dvid_ann = dvid_ann.loc[~dvid_ann.index.duplicated()].copy()
    dvid_ann['soma_position'] = dvid_ann[['soma_x', 'soma_y', 'soma_z']].values.tolist()
    dvid_ann['neck_position'] = dvid_ann[['neck_x', 'neck_y', 'neck_z']].values.tolist()

    # Append DVID columns to the main Clio set.
    dvid_ann = dvid_ann[['has_soma', 'is_cervical', 'soma_position', 'neck_position']]
    clio_ann = clio_ann.merge(dvid_ann, 'left', left_on='bodyid', right_index=True)

    # Find an auto-selected position for every body in the list.
    gen_auto_pos = partial(_auto_position, server, uuid)
    auto_positions = compute_parallel(gen_auto_pos, clio_ann['bodyid'], processes=16, ordered=False)
    auto_positions = pd.DataFrame(auto_positions, columns=['bodyid', 'auto_position'])
    clio_ann = clio_ann.merge(auto_positions, on='bodyid')

    # Cache the old position
    clio_ann['user_position'] = clio_ann['position']

    # For empty soma/neck positions, use NaN instead of [0,0,0]
    empty_somapos = clio_ann['soma_position'].map(lambda p: p == [0,0,0] or (isinstance(p, float) and np.isnan(p)))
    clio_ann.loc[clio_ann.index[empty_somapos], 'soma_position'] = np.nan

    empty_neckpos = clio_ann['neck_position'].map(lambda p: p == [0,0,0] or (isinstance(p, float) and np.isnan(p)))
    clio_ann.loc[clio_ann.index[empty_neckpos], 'neck_position'] = np.nan

    # Start with a completely empty position column
    clio_ann['position'] = np.nan
    clio_ann['position_type'] = np.nan

    # Overwrite empty rows in prioritized order: user > soma > neck > auto
    idx = clio_ann.query('position.isnull()').index
    clio_ann.loc[idx, 'position'] = clio_ann.loc[idx, 'user_position']
    clio_ann.loc[idx, 'position_type'] = 'user'

    idx = clio_ann.query('position.isnull()').index
    clio_ann.loc[idx, 'position'] = clio_ann.loc[idx, 'soma_position']
    clio_ann.loc[idx, 'position_type'] = 'soma'

    idx = clio_ann.query('position.isnull()').index
    clio_ann.loc[idx, 'position'] = clio_ann.loc[idx, 'neck_position']
    clio_ann.loc[idx, 'position_type'] = 'neck'

    idx = clio_ann.query('position.isnull()').index
    clio_ann.loc[idx, 'position'] = clio_ann.loc[idx, 'auto_position']
    clio_ann.loc[idx, 'position_type'] = 'auto'

    return clio_ann


def _auto_position(server, uuid, body):
    try:
        z,y,x = generate_sample_coordinate(server, uuid, 'segmentation', body)
        return (body, [x,y,z])
    except HTTPError:
        return (body, np.nan)
