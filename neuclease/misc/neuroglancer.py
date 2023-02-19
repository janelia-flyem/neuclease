"""
neuroglancer-related utility functions

See also: neuclease/notebooks/hemibrain-neuroglancer-video-script.txt
"""
import copy
import json
import urllib
import logging
import numpy as np
import pandas as pd
from textwrap import dedent

logger = logging.getLogger(__name__)

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
            If you are providing a linkedSegmentationLayer, your dataframe should contain
            a 'semgents' column to indicate which segments are associated with each annotation.
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


VALID_PROP_TYPES = ['label', 'description', 'tags', 'string', 'number']


def serialize_segment_properties_info(df, prop_types={}, output_path=None):
    """
    Construct segment properties JSON info file according to the neuroglancer spec:
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/segment_properties.md

    Note:
        This function doesn't yet support 'tags'.

    Args:
        df:
            DataFrame or Series.  Index must be named 'body'.
            Every column will be interpreted as a segment property.

        prop_types:
            Dict to specify the neuroglancer property type of each column, e.g. {'instance': 'label'}.
            For columns not listed in the dict, the property type is inferred from the name of the column
            (if the name is 'label' or 'description') or the dtype of the column (string vs. number).

        output_path:
            If provided, export the JSON to a file.

    Returns:
        JSON data (as a dict)
    """
    assert df.index.name == 'body'
    if isinstance(df, pd.Series):
        df = df.to_frame()
    invalid_prop_types = set(prop_types.values()) - set(VALID_PROP_TYPES)
    assert not invalid_prop_types, \
        f"Invalid property types: {invalid_prop_types}"

    assert 'tags' not in prop_types.values(), \
        "Sorry, 'tags' properties aren't yet supported by this function."

    info = {
        '@type': 'neuroglancer_segment_properties',
        'inline': {
            'ids': [*map(str, df.index)],
            'properties': []
        }
    }

    # If there's only one column, assume it's the 'label' property
    if not prop_types and len(df.columns) == 1:
        prop_types = {df.columns[0]: 'label'}

    default_prop_types = {
        'label': 'label',
        'description': 'description'
    }
    prop_types = default_prop_types | prop_types

    for col in df.columns:
        prop = {}
        prop['id'] = col

        if np.issubdtype(df[col].dtype, np.number):
            assert not df[col].dtype in (np.int64, np.uint64), \
                "Neuroglancer doesn't support 64-bit integer properties.  Use int32 or float64"
            prop['type'] = 'number'
            prop['data_type'] = df[col].dtype.name
            assert not df[col].isnull().any(), \
                (f"Column {col} contans NaN entries. "
                 "I'm not sure what to do with NaN values in numeric properties.")
            prop['values'] = df[col].tolist()
        else:
            prop['type'] = prop_types.get(col, 'string')
            prop['values'] = df[col].fillna("").astype(str).tolist()

        info['inline']['properties'].append(prop)

    _validate_property_type_counts(info)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
    return info


def _validate_property_type_counts(info):
    type_counts = (
        pd.Series([prop['type'] for prop in info['inline']['properties']])
        .value_counts()
        .reindex(VALID_PROP_TYPES)
        .fillna(0)
        .astype(int)
    )
    for t in ['label', 'description', 'tags']:
        assert type_counts.loc[t] <= 1, \
            f"Can't have more than one property with type '{t}'"

    if type_counts.loc['label'] == 0 and type_counts.loc['string'] > 0:
        logger.warning("None of your segment properties are of type 'label', "
                       "so none will be displayed in the neuroglancer UI.")
