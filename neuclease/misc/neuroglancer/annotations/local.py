import copy
from textwrap import indent, dedent

import numpy as np
import pandas as pd

from ..util import parse_nglink
from .util import annotation_property_specs

def extract_local_annotations(link, *, link_index=None, user=None, visible_only=False):
    """
    Extract local://annotations data from a neuroglancer link.
    The annotation coordinates (point, pointA, pointB, radii) are extracted
    into separate columns, named x,y,z/xa,ya,za/xb,yb,zb/rx,ry,rz
    (consistent with local_annotation_json() in this file).

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

        try:
            _df = pd.DataFrame(layer['annotations'])
        except KeyError as e:
            continue
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


# legacy name
extract_annotations = extract_local_annotations

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


def local_annotation_json(df, name="annotations", color="#ffff00", size=8.0, linkedSegmentationLayer=None,
                          show_panel=False, properties=[], shader=None, res_nm_xyz=(8,8,8)):
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

    prop_specs = annotation_property_specs(df, properties)
    if prop_specs:
        data['annotationProperties'] = prop_specs
        properties = [p['id'] for p in prop_specs]
    
    data['annotations'].clear()
    data['annotations'] = _annotation_list_json(
        df, linkedSegmentationLayer, properties
    )
    return data


# Deprecated name (now supports more than just points)
point_annotation_layer_json = local_annotation_json

# Deprecated name
annotation_layer_json = local_annotation_json


def _annotation_list_json(df, linkedSegmentationLayer, properties):
    """
    Helper for local_annotation_json().

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
            segments = [str(int(s)) for s in segments]
            entry['segments'] = segments

        if properties:
            entry['props'] = [getattr(row, prop) for prop in properties]

        annotations.append(entry)
    return annotations


def _standardize_annotation_dataframe(df):
    """
    Helper for local_annotation_json().
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

    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)
    else:
        df['id'] = [
            str(hex(abs(hash(tuple(x)))))
            for x in df[id_cols].values.tolist()
        ]

    is_point_or_ellipsoid = df[[*'xyz']].notnull().all(axis=1)
    is_line_or_box = df[['xa', 'ya', 'za', 'xb', 'yb', 'zb']].notnull().all(axis=1)
    assert (is_point_or_ellipsoid ^ is_line_or_box).all(), \
        "You must supply either [x,y,z] or [xa,ya,za,xb,yb,zb] for every row (and not both)."

    df['type'] = df['type'].fillna(
        df['rx'].isnull().map({
            True: None,
            False: 'ellipsoid'
        })
    )

    # We have no way of choosing between 'line' and 'axis_aligned_bounding_box'
    # unless the user provides the 'type' explicitly.
    # We default to 'axis_aligned_bounding_box' because it's harder to type than 'line' :-)
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
