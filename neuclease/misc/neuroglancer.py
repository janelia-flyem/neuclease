"""
neuroglancer-related utility functions

See also: neuclease/notebooks/hemibrain-neuroglancer-video-script.txt
"""
import re
import sys
import copy
import json
import urllib
import logging
import tempfile
import subprocess
from itertools import chain
from collections.abc import Mapping, Collection
import numpy as np
import pandas as pd
from textwrap import indent, dedent

logger = logging.getLogger(__name__)


def parse_nglink(link):
    _, pseudo_json = link.split('#!')
    if pseudo_json.endswith('.json'):
        return download_ngstate(pseudo_json)
    pseudo_json = urllib.parse.unquote(pseudo_json)
    data = json.loads(pseudo_json)
    return data


def format_nglink(ng_server, link_json_settings):
    return ng_server + '/#!' + urllib.parse.quote(json.dumps(link_json_settings))


def download_ngstate(link):
    import requests
    if link.startswith('gs://'):
        url = f'https://storage.googleapis.com/{link[len("gs://"):]}'
        return requests.get(url, timeout=10).json()

    if not link.startswith('http'):
        raise ValueError(f"Don't understand state link: {link}")

    if link.count('://') == 1:
        return requests.get(link, timeout=10).json()

    if link.count('://') == 2:
        url = f'https://storage.googleapis.com/{link.split("://")[2]}'
        return requests.get(url, timeout=10).json()

    raise ValueError(f"Don't understand state link: {link}")


def layer_dict(state):
    return {layer['name']: layer for layer in state['layers']}


def layer_state(state, name):
    matches = []
    for layer in state['layers']:
        if re.match(name, layer['name']):
            matches.append(layer)
    if len(matches) > 1:
        raise RuntimeError(f"Found more than one layer matching to the regex '{name}'")
    return layer


def extract_annotations(link, *, link_index=None, user=None, visible_only=False):
    """
    Extract local://annotations data from a neuroglancer link.
    The annotation coordinates (point, pointA, pointB, radii) are extracted
    into separate columns, named x,y,z/xa,ya,za/xb,yb,zb/rx,ry,rz
    (consitent with annotation_layer_json() in this file).

    Args:
        link:
            Either a neuroglancer JSON state (dict), or a neuroglancer link with embedded state,
            or a neuroglancer link that references a JSON state file, such as
            https://neuroglancer-demo.appspot.com/#!gs://flyem-views/hemibrain/v1.2/base.json

        link_index:
            Deprecated.
            Adds a column named 'link_index' populated with the provided value in all rows.

        user:
            Deprecated.
            Adds a column named 'user' populated with the provided value in all rows.

        visible_only:
            If True, do not extract annotations from layers that are not currently
            visible in the given link state.

    Returns:
        DataFrame
    """
    if isinstance(link, str):
        link = parse_nglink(link)
    annotation_layers = [layer for layer in link['layers'] if layer['type'] == "annotation"]

    dfs = []
    for layer in annotation_layers:
        if visible_only and (layer.get('archived', False) or not layer.get('visible', True)):
            continue

        _df = pd.DataFrame(layer['annotations'])
        _df['layer'] = layer['name']
        dfs.append(_df)

    df = pd.concat(dfs, ignore_index=True)
    if 'point' in df.columns:
        rows = df['point'].notnull()
        df.loc[rows, [*'xyz']] = df.loc[rows, 'point'].tolist()
    if 'pointA' in df.columns:
        rows = df['pointA'].notnull()
        df.loc[rows, ['xa', 'ya', 'za']] = df.loc[rows, 'pointA'].tolist()
    if 'pointB' in df.columns:
        rows = df['pointB'].notnull()
        df.loc[rows, ['xb', 'yb', 'zb']] = df.loc[rows, 'pointB'].tolist()
    if 'radii' in df.columns:
        rows = df['radii'].notnull()
        df.loc[rows, ['rx', 'ry', 'rz']] = df.loc[rows, 'radii'].tolist()

    # Convert to int if possible.
    for col in [*'xyz', 'xa', 'ya', 'za', 'xb', 'yb', 'zb', 'rx', 'ry', 'rz', 'radii']:
        if col in df and df[col].notnull().all() and not (df[col] % 1).any():
            df[col] = df[col].astype(np.int64)

    if link_index is not None:
        df['link_index'] = link_index
    if user is not None:
        df['user'] = user

    df = df.drop(columns=['point', 'pointA', 'pointB', 'radii'], errors='ignore')
    cols = ['link_index', 'user', 'layer', 'type', *'xyz', 'rx', 'ry', 'rz', 'xa', 'ya', 'za', 'xb', 'yb', 'zb', 'id', 'description']
    df = df[[c for c in cols if c in df.columns]]
    return df


# Tip: Here's a nice repo with lots of colormaps implemented in GLSL.
# https://github.com/kbinani/colormap-shaders
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


def annotation_layer_json(df, name="annotations", color="#ffff00", size=8.0, linkedSegmentationLayer=None, show_panel=False, properties=[], shader=None, res_nm_xyz=(8,8,8)):
    """
    Construct the JSON data for a neuroglancer local annotations layer.
    This does not result in a complete neuroglancer link; it results in something
    that can be added to the layers list in the neuroglancer viewer JSON state.


    Args:
        df:
            DataFrame containing the annotation data.
            Which columns you must provide depends on which annotation type(s) you want to display.

            - For point annotations, provide ['x', 'y', 'z']
            - For line annotations or axis_aligned_bounding_box annotations,
              provide ['xa', 'ya', 'za', 'xb', 'yb', 'zb']
            - For ellipsoid annotations, provide ['x', 'y', 'z', 'rx', 'ry', 'rz']
              for the center point and radii.

            You may also provide a column 'type' to explicitly set the annotation type.
            In some cases, 'type' isn't needed since annotation type can be inferred from the
            columns you provided. But in the case of line and box annotations, the input
            columns are the same, so you must provide a 'type' column.

            You may also provide additional columns to use as annotation properties,
            in which case they should be listed in the 'properties' argument. (See below.)

        name:
            The name of the annotation layer

        color:
            The default color for annotations, which can be overridden by the annotation shader.

        size:
            The annotation size to hard-code into the default annotation shader used by this function.
            (Only used for points and line endpoints.)

        linkedSegmentationLayer:
            If the annotations should be associated with another layer in the view,
            this specifies the name of that layer.
            This function sets the 'filterBySegmentation' key to hide annotations from non-selected segments.
            If you are providing a linkedSegmentationLayer, your dataframe should contain
            a 'segments' column to indicate which segments are associated with each annotation.

        show_panel:
            If True, the selection panel will be visible in the side bar by default.

        properties:
            The list column names to use as annotation properties.
            Properties are visible in the selection panel when an annotation is selected,
            and they can also be used in annotation shaders via special functions neuroglancer
            defines for each property.  For example, for a property named 'confidence',
            you could write setPointMarkerSize(prop_confidence()) in your annotation shader.

            This function supports annotation color proprties via strings (e.g. '#ffffff') and
            also annotation 'enum' properties if you pass them via pandas categorical columns.

            By default, the annotation IDs are the same as the column names and the annotation types are inferred.
            You can override the property 'spec' by supplying a dict-of-dicts here instead of a list of columns:

                properties={
                    "my_column": {
                        "id": "my property",
                        "description": "This is my annotation property.",
                        "type": "float32",
                        "enum_values": [0.0, 0.5, 1.0],
                        "enum_labels": ["nothing", "something", "everything"],
                    },
                    "another_column: {...}
                }

    Returns:
        dict (JSON data)
    """
    df = _standardize_annotation_dataframe(df)
    if isinstance(properties, str):
        properties = [properties]

    data = copy.deepcopy(LOCAL_ANNOTATION_JSON)
    res_m = (np.array(res_nm_xyz) * 1e-9).tolist()
    output_dim = {k: [r, 'm'] for k,r in zip('xyz', res_m)}
    data['source']['transform']['outputDimensions'] = output_dim
    data['name'] = name
    data['annotationColor'] = color

    if shader:
        data['shader'] = shader
    else:
        data['shader'] = _default_shader(df['type'].unique(), size)

    if linkedSegmentationLayer:
        data['linkedSegmentationLayer'] = linkedSegmentationLayer
        data['filterBySegmentation'] = ['segments']

    if not show_panel:
        del data['panels']

    prop_specs = _annotation_property_specs(df, properties)
    if prop_specs:
        data['annotationProperties'] = prop_specs

    data['annotations'].clear()
    data['annotations'] = _annotation_list_json(
        df, linkedSegmentationLayer, properties
    )
    return data


def _annotation_list_json(df, linkedSegmentationLayer, properties):
    """
    Helper for annotation_layer_json().

    Generate the list of annotations for an annotation layer,
    assuming the input dataframe has already been pre-conditioned.
    """
    # Replace categoricals with their integer codes.
    # The corresponding enum_labels are already stored in the property specs
    for col in properties:
        if df[col].dtype == "category":
            df[col] = df[col].cat.codes

    annotations = []
    for row in df.itertuples():
        entry = {}
        entry['type'] = row.type
        entry['id'] = row.id
        if 'description' in df.columns:
            entry['description'] = row.description

        if row.type == 'point':
            entry['point'] = [row.x, row.y, row.z]
        elif row.type in ('line', 'axis_aligned_bounding_box'):
            entry['pointA'] = [row.xa, row.ya, row.za]
            entry['pointB'] = [row.xb, row.yb, row.zb]
        elif row.type == 'ellipsoid':
            entry['point'] = [row.x, row.y, row.z]
            entry['radii'] = [row.rx, row.ry, row.rz]
        else:
            raise RuntimeError(f'Invalid annotation type: {row.type}')

        if linkedSegmentationLayer and 'segments' in df.columns:
            segments = row.segments
            if not hasattr(segments, '__len__'):
                segments = [segments]
            segments = [str(s) for s in segments]
            entry['segments'] = segments

        if properties:
            entry['props'] = [getattr(row, prop) for prop in properties]

        annotations.append(entry)
    return annotations


def _standardize_annotation_dataframe(df):
    """
    Helper for annotation_layer_json().
    Add empty columns as needed until the dataframe
    has all possible annotation columns.

    Also populate the 'type' column with inferred
    annotation types based on the columns the user DID provide.
    """
    df = df.copy()
    id_cols = [*'xyz', 'xa', 'ya', 'za', 'xb', 'yb', 'zb', 'rx', 'ry', 'rz', 'type']
    for col in id_cols:
        if col not in df.columns:
            df[col] = np.nan

    if 'id' not in df.columns:
        ids = [str(hex(abs(hash(tuple(x))))) for x in df[id_cols].values.tolist()]
        df['id'] = ids

    assert (df['x'].isnull() ^ df['xa'].isnull()).all(), \
        "You must supply either x,y,z or xa,ya,za,xb,yb,zb for every row."

    df['type'] = df['type'].fillna(
        df['rx'].isnull().map({
            True: None,
            False: 'ellipsoid'
        })
    )

    # We have no way of choosing between 'line' and 'axis_aligned_bounding_box'
    # unless the user provides the 'type' explicitly.
    # We default to 'axis_aligned_bounding_box'.
    df['type'] = df['type'].fillna(
        df['x'].isnull().map({
            True: 'axis_aligned_bounding_box',
            False: 'point'
        })
    )
    return df


def _default_shader(annotation_types, default_size):
    """
    Create a default annotation shader that is pre-populated with
    the annotation API functions so you don't have to look them up.
    """
    shader_body = ""
    if 'point' in annotation_types:
        # Note:
        #   In older versions of neuroglancer,
        #   setPointMarkerColor(vec3) doesn't exist yet,
        #   so we must use vec4.
        #   https://github.com/google/neuroglancer/pull/475
        shader_body += dedent(f"""\
            //
            // Point Marker API
            //
            setPointMarkerSize({default_size});
            setPointMarkerColor(vec4(defaultColor(), 1.0));
            setPointMarkerBorderWidth(1.0);
            setPointMarkerBorderColor(defaultColor());
        """)

    if 'line' in annotation_types:
        shader_body += dedent(f"""\
            //
            // Line API
            //
            setLineColor(defaultColor());
            setEndpointMarkerSize({default_size}, {default_size});
            setEndpointMarkerColor(defaultColor(), defaultColor());
            setEndpointMarkerBorderWidth(1.0, 1.0);
            setEndpointMarkerBorderColor(defaultColor(), defaultColor());
        """)

    if 'axis_aligned_bounding_box' in annotation_types:
        shader_body += dedent("""\
            //
            // Bounding Box API
            //
            setBoundingBoxBorderWidth(1.0);
            setBoundingBoxBorderColor(defaultColor());
            setBoundingBoxFillColor(vec4(defaultColor(), 0.5));
        """)

    if 'ellipsoid' in annotation_types:
        shader_body += dedent("""\
            //
            // Ellipsoid API
            //
            setEllipsoidFillColor(defaultColor());
        """)

    shader_main = dedent(f"""\
        void main() {{
            {indent(shader_body, ' '*12)[12:]}\
        }}
    """)

    return shader_main


def _annotation_property_specs(df, properties):
    """
    Helper for annotation_layer_json().

    Given an input dataframe for annotations and a list of columns
    from which to generate annotation properties, generate a
    JSON property type specification for inserting into the layer
    JSON state in the 'annotationProperties' section.

    The annotation property type is inferred from each column's dtype.
    Categorical pandas columns result in neuroglancer enum annotation properties.

    Args:
        df:
            DataFrame.  The property columns will be inspected to infer
            the ultimate property types (numeric vs enum vs color).
        properties:
            list of column names from which to generate properties,
            or a dict-of-dicts containing pre-formulated property specs
            as descrbed in the docstring for annotation_layer_json().
    Returns:
        JSON dict
    """
    def proptype(col):
        dtype = df[col].dtype
        if dtype in (np.float64, np.int64, np.uint64):
            raise RuntimeError('neuroglancer doesnt support 64-bit property types.')
        if dtype in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32):
            return str(dtype)

        if dtype == 'category':
            num_cats = len(dtype.categories)
            for utype in (np.uint8, np.uint16, np.uint32):
                if num_cats <= 1 + np.iinfo(utype).max:
                    return str(np.dtype(utype))
            raise RuntimeError(f"Column {col} has too many categories")

        if df[col].dtype != object:
            raise RuntimeError(f"Unsupported property dtype: {dtype} for column {col}")

        is_str = df[col].map(lambda x: isinstance(x, str)).all()
        is_color = is_str and df[col].str.startswith('#').all()
        if not is_color:
            msg = (
                f"Column {col}: I don't know what to do with object dtype that isn't rbg or rgba.\n"
                "If you want to create an enum property, then supply a pandas Categorical column."
            )
            raise RuntimeError(msg)
        if (df[col].map(len) == len("#rrggbb")).all():
            return 'rgb'
        if (df[col].map(len) == len("#rrggbbaa")).all():
            return 'rgba'
        raise RuntimeError("Not valid RGB or RGBA colors")

    if isinstance(properties, Mapping):
        property_specs = properties
    else:
        assert isinstance(properties, Collection)
        property_specs = {col: {} for col in properties}

    default_property_specs = {
        col: {
            'id': col,
            'type': proptype(col),
        }
        for col in property_specs
    }

    for col in default_property_specs.keys():
        if df[col].dtype == "category":
            cats = df[col].cat.categories.tolist()
            default_property_specs[col]['enum_values'] = [*range(len(cats))]
            default_property_specs[col]['enum_labels'] = cats

    property_specs = [
        {**default_property_specs[col], **property_specs[col]}
        for col in property_specs
    ]

    return property_specs


# Deprecated name (now supports more than just points)
point_annotation_layer_json = annotation_layer_json


def upload_ngstates(bucket_dir, states, threads=0, processes=0):
    """
    Use multithreading or multiprocessing to upload many files in parallel,
    similar to `gsutil -m cp []...]`, except that in this case you must choose
    between multithreading or multiprocessing (not a combination of the two).
    """
    from neuclease.util import upload_to_bucket

    assert bucket_dir.startswith('gs://')
    bucket_dir = bucket_dir[len('gs://'):]
    bucket = bucket_dir.split('/')[0]
    dirpath = bucket_dir[1 + len(bucket):]

    blob_names = [dirpath + '/' + name for name in states.keys()]
    blobs = map(json.dumps, states.values())
    args = [(bucket, blobname, blob) for blobname, blob in zip(blob_names, blobs)]

    from neuclease.util import compute_parallel
    urls = compute_parallel(upload_to_bucket, args, starmap=True, threads=threads, processes=processes)
    return urls


def upload_ngstate(bucket_path, state):
    """
    Upload the given JSON state to a gbucket location,
    such as 'gs://flyem-user-links/short/foobar.json'
    """
    from neuclease.util import upload_to_bucket

    assert bucket_path.startswith('gs://')
    bucket_path = bucket_path[len('gs://'):]

    bucket = bucket_path.split('/')[0]
    filename = bucket_path[1 + len(bucket):]

    state_string = json.dumps(state, indent=2)
    return upload_to_bucket(bucket, filename, state_string)


def upload_to_bucket(bucket, blob_name, blob_contents):
    """
    Upload a blob of data to the specified google storage bucket.
    """
    if isinstance(bucket, str):
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket)

    blob = bucket.blob(blob_name)
    blob.cache_control = 'public, no-store'
    blob.upload_from_string(blob_contents, content_type='application/json')
    return blob.public_url


def make_bucket_public(bucket=None):
    if bucket is None:
        bucket = sys.argv[1]
    if bucket.startswith('gs://'):
        bucket = bucket[len('gs://'):]
    subprocess.run(f'gsutil iam ch allUsers:objectViewer gs://{bucket}', shell=True, check=True)

    with tempfile.NamedTemporaryFile('w') as f:
        cors_settings = [{
            "maxAgeSeconds": 3600,
            "method": ["GET"],
            "origin": ["*"],
            "responseHeader": ["Content-Type", "Range"]
        }]
        json.dump(cors_settings, f)
        f.flush()
        subprocess.run(f'gsutil cors set {f.name} gs://{bucket}', shell=True, check=True)

    print(f"Configured bucket for public neuroglancer access: gs://{bucket}")


def segment_properties_json(
    df,
    label_col=None,
    *,
    description_col=None,
    string_cols=[],
    number_cols=[],
    tag_cols=[],
    prefix_tags=False,
    output_path=None,
):
    """
    Construct segment properties JSON info file according to the neuroglancer spec:
    https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/segment_properties.md

    Args:
        df:
            DataFrame whose index contains segment IDs.
            To avoid confusion, we require that the Index be named 'body' or 'segment'.
            All columns will be converted to segment properties. If a column is not
            explicitly listed in the following arguments, its property type will be inferred
            from the column name and dtype.

        label_col:
            Which column to use for the 'label' property which is shown in neuroglancer by default.

        description_col:
            Which column to use as the 'description' property.

        string_cols:
            Columns to represent as 'string' properties.

        number_cols:
            Columns to represent as 'number' properties.

        tag_cols:
            Columns which should be used to generate the (combined) 'tags' property.

        prefix_tags:
            If True, all tags will be prefixed with the name of the column they came from,
            e.g. 'status:Anchor'

        output_path:
            If provided, export the JSON to a file.

    Returns:
        JSON data (as a dict) which should be written into an 'info' file to
        host neuroglancer precomputed segment properties.
    """
    assert df.index.name in ('body', 'segment')
    if isinstance(df, pd.Series):
        df = df.to_frame()

    scalar_types = _scalar_property_types(
        df, label_col, description_col, string_cols, number_cols, tag_cols
    )

    json_props = []
    for col, prop_type in scalar_types.items():
        j = _scalar_property_json(df[col], prop_type)
        json_props.append(j)

    if tag_cols:
        j = _tags_property_json(df, tag_cols, prefix_tags)
        json_props.append(j)

    info = {
        '@type': 'neuroglancer_segment_properties',
        'inline': {
            'ids': [*map(str, df.index)],
            'properties': json_props
        }
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(info, f)

    return info


def _scalar_property_types(df, label_col, description_col, string_cols, number_cols, tag_cols):
    prop_types = {}
    if label_col:
        prop_types[label_col] = 'label'
    if description_col:
        prop_types[description_col] = 'description'

    prop_types |= {c: 'string' for c in string_cols}
    prop_types |= {c: 'number' for c in number_cols}
    prop_types |= {c: 'tags' for c in tag_cols}

    if unknown_cols := set(prop_types.keys()) - set(df.columns):
        raise RuntimeError(f"Columns not found: {unknown_cols}")

    # If there's only one column and it wasn't explicitly listed,
    # we assume its intended as the 'label' column.
    if not prop_types and len(df.columns) == 1:
        prop_types = {df.columns[0]: 'label'}

    # Infer the types of unlisted columns from either the name or dtype
    for name, dtype in df.dtypes.items():
        if name in prop_types:
            continue
        elif name == 'label':
            prop_types['label'] = 'label'
        elif name == 'description':
            prop_types['description'] = 'description'
        elif np.issubdtype(dtype, np.number):
            prop_types[name] = 'number'
        else:
            prop_types[name] = 'string'

    # drop tag columns
    return {k:v for k,v in prop_types.items() if v != 'tags'}


def _scalar_property_json(s, prop_type):
    """
    Helper for serialize_segment_properties_info().

    Constructs the JSON for a segment property, other than the 'tags'
    property, which is implemented in _tags_property_json()
    """
    if prop_type == 'number':
        assert s.dtype not in (np.int64, np.uint64), \
            "Neuroglancer doesn't support 64-bit integer properties.  Use int32 or float64"

        assert not s.isnull().any(), \
            (f"Column {s.name} contans NaN entries. "
             "I'm not sure what to do with NaN values in numeric properties.")

        return {
            'id': s.name,
            'type': 'number',
            'data_type': s.dtype.name,
            'values': s.tolist()
        }

    return {
        'id': s.name,
        'type': prop_type,
        'values': s.fillna("").astype(str).tolist()
    }


def _tags_property_json(df, tags_columns, prepend_colnames):
    """
    Helper for serialize_segment_properties_info().
    Constructs the JSON for the 'tags' segment property.
    """
    df = df[[*tags_columns]].copy()

    # Clean and convert each column to categorical
    # before we combine categories below.
    for c in df.columns:
        df[c] = (
            df[c]
            .astype('string')
            .str.replace(' ', '_')  # spaces are forbidden in tags
            .replace('', None)      # discard empty strings
            .astype('category')
        )
        if prepend_colnames:
            prefix = c.replace(' ', '_')
            prefixed_categories = [
                f'{prefix}:{cat}'
                for cat in df[c].dtype.categories
            ]
            df[c] = df[c].cat.rename_categories(prefixed_categories)

    # Convert to a single big categorical dtype
    all_tags = sorted({*chain(*(df[col].dtype.categories for col in df.columns))})
    df = df.astype(pd.CategoricalDtype(categories=all_tags))

    # Tags are written as a list-of-lists of sorted codes
    codes_df = pd.DataFrame({c: df[c].cat.codes for c in df.columns})
    sorted_codes = np.sort(codes_df.values, axis=1)
    codes_lists = [
        [x for x in row if x != -1]  # Drop nulls
        for row in sorted_codes.tolist()
    ]

    return {
        'id': 'tags',
        'type': 'tags',
        'tags': all_tags,
        'values': codes_lists,
    }
