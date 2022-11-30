"""
neuroglancer-related utility functions
"""
import os
import copy
import json
import urllib
import numpy as np
import pandas as pd
from textwrap import dedent


def parse_nglink(link):
    url_base, pseudo_json = link.split('#!')
    pseudo_json = urllib.parse.unquote(pseudo_json)
    data = json.loads(pseudo_json)
    return data


def format_nglink(ng_server, link_json_settings):
    return ng_server + '/#!' + urllib.parse.quote(json.dumps(link_json_settings))


def extract_annotations(link, link_index=None, user=None):
    if isinstance(link, str):
        link = parse_nglink(link)
    annotation_layers = [layer for layer in link['layers'] if layer['type'] == "annotation"]

    data = []
    for layer in annotation_layers:
        for a in layer['annotations']:
            data.append((layer['name'], *a['point'], a.get('description', '')))

    df = pd.DataFrame(data, columns=['layer', *'xyz', 'description'])

    cols = []
    if link_index is not None:
        df['link_index'] = link_index
        cols += ['link_index']
    if user is not None:
        df['user'] = user
        cols += ['user']

    df = df.astype({k: np.int64 for k in 'xyz'})
    cols += ['layer', *'xyz', 'description']
    return df[cols]


SHADER_FMT = dedent("""\
    void main() {{
        setColor(defaultColor());
        setPointMarkerSize({size:.1f});
    }}
""")

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
    "shader": "\nvoid main() {\n  setColor(defaultColor());\n  setPointMarkerSize(8.0);\n}\n",
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


def point_annotation_layer_json(points_df, name="annotations", color="#ffff00", size=8.0, linkedSegmentationLayer=None):
    """
    Construct the JSON data for a neuroglancer local point annotations layer.
    This does not result in a complete neuroglancer link; it results in something
    that can be added to the layers list in the neuroglancer viewer JSON state.

    Args:
        points_df:
            DataFrame with columns ['x', 'y', 'z'] and optionally 'id' and 'description'.
    Returns:
        dict (JSON data)
    """
    assert {*'xyz'} <= set(points_df.columns), 'x,y,z are required columns'
    points_df = points_df.copy()
    points_df = points_df.astype({c: np.int64 for c in 'xyz'})

    if 'id' not in points_df.columns:
        ids = (points_df['z'].values << 42) | (points_df['y'].values << 21) | (points_df['x'].values)
        points_df['id'] = [*map(str, ids)]

    data = copy.deepcopy(LOCAL_ANNOTATION_JSON)
    data['name'] = name
    data['annotationColor'] = color
    data['shader'] = SHADER_FMT.format(size=size)
    data['annotations'].clear()
    if linkedSegmentationLayer:
        data['linkedSegmentationLayer'] = linkedSegmentationLayer
        data['filterBySegmentation'] = ['segments']

    for row in points_df.itertuples():
        entry = {}
        entry['type'] = "point"
        entry['point'] = [row.x, row.y, row.z]
        entry['id'] = row.id
        if 'description' in points_df.columns:
            entry['description'] = row.description

        if linkedSegmentationLayer and 'segments' in points_df.columns:
            segments = row.segments
            if not hasattr(segments, '__len__'):
                segments = [segments]
            segments = [str(s) for s in segments]
            entry['segments'] = segments

        data['annotations'].append(entry)

    return data


def upload_ngstate(bucket_path, state):
    """
    Upload the given JSON state to a gbucket location.
    """
    assert bucket_path.startswith('gs://')
    bucket_path = bucket_path[len('gs://'):]

    bucket = bucket_path.split('/')[0]
    filename = bucket_path[1 + len(bucket):]

    state_string = json.dumps(state, indent=2)
    return _upload_to_bucket(bucket, filename, state_string)


def _upload_to_bucket(bucket_name, blob_name, blob_contents):
    """
    Upload a blob of data to the specified google storage bucket.
    """
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.cache_control = 'public, no-store'
    blob.upload_from_string(blob_contents, content_type='application/json')
    return blob.public_url
