"""
neuroglancer-related utility functions
"""

import copy
import numpy as np

LOCAL_ANNOTATION_JSON = {
    "name": "annotations",
    "type": "annotation",
    "source": {
        "url": "local://annotations",
        "transform": {
            "outputDimensions": {
                "x": [
                    8e-09,
                    "m"
                ],
                "y": [
                    8e-09,
                    "m"
                ],
                "z": [
                    8e-09,
                    "m"
                ]
            }
        }
    },
    "tool": "annotatePoint",
    "panels": [
        {
            "row": 1,
            "flex": 1.22,
            "tab": "annotations"
        }
    ],
    "annotations": [
        # {
        #     "point": [23367, 35249, 68171],
        #     "type": "point",
        #     "id": "149909688276769607",
        #     "description": "soma"
        # },
    ]
}


def point_annotation_layer_json(points_df):
    """
    Construct the JSON data for a neuroglancer local point annotations layer.
    This does not result in a complete neuroglancer link; it results in something
    that can be added to the layers list in the neuroglancer viewer JSON state.
    """
    assert {*'xyz'} <= set(points_df.columns), 'x,y,z are required columns'
    points_df = points_df.copy()
    points_df = points_df.astype({c: np.int64 for c in 'xyz'})

    if 'id' not in points_df.columns:
        ids = (points_df['z'].values << 42) | (points_df['y'].values << 21) | (points_df['x'].values)
        points_df['id'] = [*map(str, ids)]

    data = copy.deepcopy(LOCAL_ANNOTATION_JSON)
    data['annotations'].clear()

    for row in points_df.itertuples():
        entry = {}
        entry['point'] = [row.x, row.y, row.z]
        entry['id'] = row.id
        if 'description' in points_df.columns:
            entry['description'] = row.description
        data['annotations'].append(entry)

    return data
