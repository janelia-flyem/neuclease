from dataclasses import dataclass
from typing import Literal
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

    default_property_specs = {
        col: {
            'id': col,
            'type': _proptype(df[col]),
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

    # For precomputed annotations, the property specs
    # must be sorted in the following order:
    # 32-bit properties, 16-bit properties, 8-bit-properties (including rgb and rgba)
    property_specs.sort(key=lambda x: -_PROPERTY_DTYPES[x['type']][1])

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
        for utype in (np.uint8, np.uint16, np.uint32):
            if num_cats <= 1 + np.iinfo(utype).max:
                return str(np.dtype(utype))
        raise RuntimeError(f"Column {s.name} has too many categories")

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


@dataclass
class ShardSpec:
    type: str
    hash: Literal["murmurhash3_x86_128", "identity_hash"]
    preshift_bits: int
    shard_bits: int
    minishard_bits: int
    data_encoding: Literal["raw", "gzip"]
    minishard_index_encoding: Literal["raw", "gzip"]

    def to_json(self):
        return {
            "@type": self.type,
            "hash": self.hash,
            "preshift_bits": self.preshift_bits,
            "shard_bits": self.shard_bits,
            "minishard_bits": self.minishard_bits,
            "data_encoding": str(self.data_encoding),
            "minishard_index_encoding": str(self.minishard_index_encoding),
        }


def choose_output_spec(
    total_count,
    total_bytes,
    hashtype: Literal["murmurhash3_x86_128", "identity_hash"] = "murmurhash3_x86_128",
    gzip_compress=True,
):
    """
    Copied from Forrest Collman's PR:
    https://github.com/google/neuroglancer/pull/522
    """
    import tensorstore as ts
    MINISHARD_TARGET_COUNT = 1000
    SHARD_TARGET_SIZE = 50000000

    if total_count == 1:
        return None
    if ts is None:
        return None

    # test if hashtype is valid
    if hashtype not in ["murmurhash3_x86_128", "identity_hash"]:
        raise ValueError(
            f"Invalid hashtype {hashtype}."
            "Must be one of 'murmurhash3_x86_128' "
            "or 'identity_hash'"
        )

    total_minishard_bits = 0
    while (total_count >> total_minishard_bits) > MINISHARD_TARGET_COUNT:
        total_minishard_bits += 1

    shard_bits = 0
    while (total_bytes >> shard_bits) > SHARD_TARGET_SIZE:
        shard_bits += 1

    preshift_bits = 0
    while MINISHARD_TARGET_COUNT >> preshift_bits:
        preshift_bits += 1

    minishard_bits = total_minishard_bits - min(total_minishard_bits, shard_bits)
    data_encoding: Literal["raw", "gzip"] = "raw"
    minishard_index_encoding: Literal["raw", "gzip"] = "raw"

    if gzip_compress:
        data_encoding = "gzip"
        minishard_index_encoding = "gzip"

    return ShardSpec(
        type="neuroglancer_uint64_sharded_v1",
        hash=hashtype,
        preshift_bits=preshift_bits,
        shard_bits=shard_bits,
        minishard_bits=minishard_bits,
        data_encoding=data_encoding,
        minishard_index_encoding=minishard_index_encoding,
    )
