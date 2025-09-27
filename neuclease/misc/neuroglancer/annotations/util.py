import re
from collections.abc import Mapping, Collection

import numpy as np


# Copied from neuroglancer/write_annotations.py
_PROPERTY_DTYPES: dict[str, tuple[tuple[str] | tuple[str, tuple[int, ...]], int]] = {
    "uint8": (("|u1",), 1),
    "uint16": (("<u2",), 2),
    "uint32": (("<u4",), 3),
    "int8": (("|i1",), 1),
    "int16": (("<i2",), 2),
    "int32": (("<i4",), 4),
    "float32": (("<f4",), 4),
    "rgb": (("|u1", (3,)), 1),
    "rgba": (("|u1", (4,)), 1),
}


def annotation_property_specs(df, properties):
    """
    Given an input dataframe for annotations and a list of columns
    from which to generate annotation properties, generate a
    JSON property type specification.

    The annotation property type is inferred from each column's dtype.
    Categorical pandas columns result in neuroglancer enum annotation properties.

    In addition to constructing the property specs, this function sorts the properties
    according to the rules required for precomputed annotations (for padding efficiency).

    For local annotation layers, the property specs should be written
    as the 'annotationProperties' part of the layer JSON state.
   
    For precomputed annotations, the property specs should be written as
    the "properties" section of the info JSON.

    Args:
        df:
            DataFrame.  The property columns will be inspected to infer
            the ultimate property types (numeric vs enum vs color).
        properties:
            list of column names from which to generate properties,
            or a dict-of-dicts containing pre-formulated property specs
            as descrbed in the docstring for local_annotation_json().
    Returns:
        JSON dict
        The order of the properties may not be the same as what you passed in.
        They are sorted according to the layout rules of neuroglancer's
        precomputed annotation format.
    """
    try:
        from neuroglancer.viewer_state import AnnotationPropertySpec
    except ImportError:
        class AnnotationPropertySpec:
            pass

    if isinstance(properties, Mapping):
        property_specs = {}
        for key, spec in properties.items():
            if isinstance(spec, AnnotationPropertySpec):
                assert spec.id == key
                property_specs[key] = spec.to_json()
            else:
                property_specs[key] = spec
        property_specs = properties
    else:
        assert isinstance(properties, Collection)
        property_specs = {}
        for spec in properties:
            if isinstance(spec, AnnotationPropertySpec):
                property_specs[spec.id] = spec.to_json()
            elif isinstance(spec, dict):
                property_specs[spec['id']] = spec
            elif isinstance(spec, str):
                property_specs[spec] = {}
            else:
                raise ValueError(f"Invalid property spec: {spec}")

    default_property_specs = {}
    for propname in property_specs:
        if propname in df.columns:
            default_property_specs[propname] = {
                'id': propname,
                'type': _proptype(df[propname]),
            }
        elif {f'{propname}_{channel}' for channel in 'rgba'} <= set(df.columns):
            default_property_specs[propname] = {
                'id': propname,
                'type': 'rgba',
            }
        elif {f'{propname}_{channel}' for channel in 'rgb'} <= set(df.columns):
            default_property_specs[propname] = {
                'id': propname,
                'type': 'rgb',
            }
        else:
            raise ValueError(f"Property '{propname}' not found in dataframe")

    for col in default_property_specs.keys():
        if col in df and df[col].dtype == "category":
            if df[col].isnull().any():
                raise ValueError(f"Column {col} can't be used for an enum property because it has null values")

            cats = df[col].cat.categories.tolist()
            default_property_specs[col]['enum_values'] = [*range(len(cats))]
            default_property_specs[col]['enum_labels'] = cats

    property_specs = [
        {**default_property_specs[col], **property_specs[col]}
        for col in property_specs
    ]

    # For precomputed annotations, the property specs
    # must be sorted in the following order:
    # 32-bit properties, 16-bit properties, 8-bit-properties (including rgb and rgba)
    property_specs.sort(key=lambda x: -_PROPERTY_DTYPES[x['type']][1])

    # Validate the property names according to the neuroglancer spec.
    for prop in property_specs:
        if not re.match(r'^[a-z][a-zA-Z0-9_]*$', prop['id']):
            raise ValueError(f"Invalid property name: {prop}")

    return property_specs


def _proptype(s):
    """
    Helper for _annotation_property_specs().
    Given a Series, determine the corresponding neuroglancer property type.

    Returns: str
        Either a numeric type (e.g. 'uint16') or a color type ('rgb' or 'rgba').
    """
    if s.dtype in (np.float64, np.int64, np.uint64):
        raise RuntimeError("neuroglancer doesn't support 64-bit property types.")
    if s.dtype in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32):
        return str(s.dtype)

    if s.dtype == 'category':
        num_cats = len(s.dtype.categories)
        if s.cat.codes.dtype == np.int8:
            # Old versions of neuroglancer had a bug for int8 property types,
            # so we store them as uint8.
            # https://github.com/google/neuroglancer/pull/830
            return 'uint8'

        # Pandas already stores categorical codes with a minimal-width dtype
        return str(np.dtype(s.cat.codes.dtype))

    if s.dtype != object:
        raise RuntimeError(f"Unsupported property dtype: {s.dtype} for column {s.name}")

    is_str = s.map(lambda x: isinstance(x, str)).all()
    is_color = is_str and s.str.startswith('#').all()
    if not is_color:
        msg = (
            f"Column {s.name}: I don't know what to do with object dtype that isn't rbg or rgba.\n"
            "If you want to create an enum property, then supply a pandas Categorical column."
        )
        raise RuntimeError(msg)
    if (s.map(len) == len("#rrggbb")).all():
        return 'rgb'
    if (s.map(len) == len("#rrggbbaa")).all():
        return 'rgba'
    raise RuntimeError("Not valid RGB or RGBA colors")
