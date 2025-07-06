import os
import json
import pandas as pd
import numpy as np
from itertools import chain

from tqdm import tqdm
import tensorstore as ts
from neuroglancer.coordinate_space import CoordinateSpace
from neuroglancer.viewer_state import AnnotationPropertySpec

from .util import annotation_property_specs


# TODO:
# - support more than just points
# - Use tensorstore for writing
# - accept a tensorstore KVStore or somehow let the user specify what they want.
# - Refactor to make it easier to obtain the data for a single annotation/segment in isolation (for a web service)

def write_annotations(
    df: pd.DataFrame,
    coord_space: CoordinateSpace,
    annotation_type: str | None = None,
    properties: list[str] | list[AnnotationPropertySpec] | dict[str, AnnotationPropertySpec] | list[dict] = (),
    relationships: list[str] = (),
    output_dir: str = 'annotations',
    sharding_spec=None
):
    if annotation_type is None:
        annotation_type = _infer_annotation_type(df)
    
        # spec = {
        #     "driver": "neuroglancer_uint64_sharded",
        #     "metadata": shard_spec.to_json(),
        #     "base": f"file://{path}",
        # }
        # dataset = ts.KvStore.open(spec).result()
        # txn = ts.Transaction()

    properties = annotation_property_specs(df, properties)
    _write_info(df, coord_space, annotation_type, properties, relationships, output_dir)

    id_bufs, ann_bufs, rel_bufs = _encode_annotations(df, coord_space, annotation_type, properties, relationships, output_dir)
    df['id_buf'] = id_bufs
    df['ann_buf'] = ann_bufs
    if rel_bufs is not None:
        df['rel_buf'] = rel_bufs

    _write_annotations_by_id(df, output_dir)
    _write_annotations_by_relationship(df, relationships, output_dir)


def _write_info(df, coord_space, annotation_type, properties, relationships, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    geometry_cols = _geometry_cols(coord_space.names, annotation_type)

    lower_bound = [np.inf] * len(coord_space.names)
    upper_bound = [-np.inf] * len(coord_space.names)
    for cols in geometry_cols:
        lower_bound = np.minimum(lower_bound, df[cols].min().to_numpy())
        upper_bound = np.maximum(upper_bound, df[cols].max().to_numpy())

    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": coord_space.to_json(),
        "lower_bound": lower_bound.tolist(),
        "upper_bound": upper_bound.tolist(),
        "annotation_type": annotation_type,
        "properties": properties,
        "relationships": [
            {
                "id": relationship,
                "key": f"by_rel_{relationship}"
            }
            for relationship in relationships
        ],
        "by_id": {
            "key": "by_id"
        },

        # TODO
        "spatial": []
    }

    with open(f"{output_dir}/info", 'w') as f:
        json.dump(info, f)


def _geometry_cols(coord_names, annotation_type):
    """
    Determine the list of column groups that express
    the geometry of annotations of the given type.
    Point annotations have only one group,
    but other annotation types have two.
    
    Examples:
    
        >>> _geometry_cols([*'xyz'], 'point')
        [['x', 'y', 'z']]

        >>> _geometry_cols([*'xyz'], 'ellipsoid')
        [['x', 'y', 'z'], ['rx', 'ry', 'rz']]

        >>> _geometry_cols([*'xyz'], 'line')
        [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']]

        >>> _geometry_cols([*'xyz'], 'axis_aligned_bounding_box')
        [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']]
    """
    if annotation_type == 'point':
        return [[c for c in coord_names]]

    if annotation_type == 'ellipsoid':
        return [
            [c for c in coord_names],
            [f'r{c}' for c in coord_names]
        ]

    if annotation_type in ('line', 'axis_aligned_bounding_box'):
        return [
            [f'{c}a' for c in coord_names],
            [f'{c}b' for c in coord_names]
        ]

    raise ValueError(f"Annotation type {annotation_type} not supported")


def _encode_annotations(df, coord_space, annotation_type, properties, relationships, output_dir):
    id_bufs = _encode_uint64_series(df.index)
    ann_bufs = _encode_geometries_and_properties(df, coord_space, annotation_type, properties)
    rel_bufs = _encode_relationships(df, relationships)
    return id_bufs, ann_bufs, rel_bufs


def _encode_uint64_series(s):
    id_buf = s.to_numpy(np.uint64).tobytes()
    id_bufs = [
        id_buf[offset:(offset+8)]
        for offset in range(0, len(id_buf), 8)
    ]
    return np.array(id_bufs, dtype=object)


def _encode_geometries_and_properties(df, coord_space, annotation_type, properties):
    geometry_cols = _geometry_cols(coord_space.names, annotation_type)
    prop_cols = [p['id'] for p in properties]
    geometry_prop_df = df[[*chain(*geometry_cols), *prop_cols]].copy(deep=False)

    property_widths = [
        {'rgb': 3, 'rgba': 4}.get(p['type'], np.dtype(p['type']).itemsize)
        for p in properties
    ]
    property_padding = (4 - (sum(property_widths) % 4)) % 4
    for i in range(property_padding):
        geometry_prop_df[f'__padding_{i}__'] = np.uint8(0)

    dtypes = {c: np.float32 for c in chain(*geometry_cols)}
    for p in properties:
        if df[p['id']].dtype == 'category':
            geometry_prop_df[p['id']] = geometry_prop_df[p['id']].cat.codes
            dtypes[p['id']] = p['type']

    # vectorized serialization
    records = geometry_prop_df.to_records(index=False, column_dtypes=dtypes)
    buf = records.tobytes()

    # extract bytes from the appropriate slice for each record
    recsize = records.dtype.itemsize
    encoded_annotations = [buf[i*recsize:(i+1)*recsize] for i in range(len(records))]
    ann_bufs = pd.Series(encoded_annotations, index=df.index)
    return ann_bufs


def _encode_relationships(df, relationships):
    if not relationships:
        return None

    encoded_relationships = {}
    for rel_col in relationships:
        encoded_relationships[rel_col] = _encode_relationship(df[rel_col])

    # concatenate buffers on each row
    rel_bufs = pd.DataFrame(encoded_relationships, index=df.index).sum(axis=1)
    return rel_bufs


def _encode_relationship(related_ids):
    """
    Given a Series containing lists of IDs, encode each list of IDs
    in the format neuroglancer expects for each relationship in an
    annotation.

    Each item in related_ids is a list, such as:

        [7, 8, 9]

    which gets encoded into buffer as <count><id_1><id_2><id_3>,
    where <count> is uint32 and <id_1><id_2><id_3> are each uint64:

        (
            b'\x00\x00\x00\x03' +
            b'\x00\x00\x00\x00\x00\x00\x00\x07' +
            b'\x00\x00\x00\x00\x00\x00\x00\x08' +
            b'\x00\x00\x00\x00\x00\x00\x00\x09'
        )

    Args:
        related_ids:
            A Series of length N and dtype=object, containing lists of IDs.
            As a special convenience in the case where every row contains
            exactly one ID, you may pass a series with dtype=np.uint64,
            which will be interpreted as if each entry were a list of length 1.
            (In this case, the implementation is slightly faster than in the general case.)

    Returns:
        A numpy array with N entries, where each entry is a buffer as shown above.
    """
    # Special case if the relationship contains only a single ID.
    if np.issubdtype(related_ids.dtype, np.integer):
        buf = (
            pd.DataFrame({'count': np.uint32(1), 'id': related_ids})
            .astype({'count': np.uint32, 'id': np.uint64}, copy=False)
            .to_records(index=False)
            .tobytes()
        )

        encoded_ids = [
            buf[i*12:(i+1)*12]
            for i in range(len(related_ids))
        ]
        return np.array(encoded_ids, dtype=object)

    # Otherwise, the relationship contains lists.
    assert related_ids.dtype == object
    counts = related_ids.map(len).to_numpy(np.uint32)
    offsets = 8 * np.cumulative_sum(counts, include_initial=True)

    ids_buf = np.concatenate(related_ids, dtype=np.uint64).tobytes()
    counts_buf = counts.tobytes()

    encoded_ids = [
        counts_buf[i*4:(i+1)*4] + ids_buf[start:end]
        for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:]))
    ]
    return np.array(encoded_ids, dtype=object)


def _write_annotations_by_id(df, output_dir):
    os.makedirs(f"{output_dir}/by_id", exist_ok=True)

    if 'rel_buf' in df.columns:
        ann_bufs = df['ann_buf'] + df['rel_buf']
    else:
        ann_bufs = df['ann_buf']

    for ann_id, ann_buf in tqdm(ann_bufs.items()):
        with open(f"{output_dir}/by_id/{ann_id}", 'wb') as f:
            f.write(ann_buf)


def _write_annotations_by_relationship(df, relationships, output_dir):
    for relationship in relationships:
        rel_df = df[['id_buf', 'ann_buf', relationship]]
        bufs_by_segment = (
            rel_df
            [['id_buf', 'ann_buf', relationship]]
            .dropna(subset=relationship)
            .explode(relationship)
            .groupby(relationship)
            .agg({'id_buf': ['count', 'sum'], 'ann_buf': 'sum'})
        )
        bufs_by_segment.columns = ['count', 'id_buf', 'ann_buf']
        bufs_by_segment['count_buf'] = _encode_uint64_series(bufs_by_segment['count'])
        bufs_by_segment['combined_buf'] = bufs_by_segment[['count_buf', 'ann_buf', 'id_buf']].sum(axis=1)

        for segment, buf in tqdm(bufs_by_segment['combined_buf'].items()):
            os.makedirs(f"{output_dir}/by_rel_{relationship}", exist_ok=True)
            with open(f"{output_dir}/by_rel_{relationship}/{segment}", 'wb') as f:
                f.write(buf)


def _infer_annotation_type(df):
    if 'rx' in df.columns:
        return 'ellipsoid'
    if 'xa' in df.columns:
        return 'axis_aligned_bounding_box'
    if 'xa' in df.columns:
        return 'line'
    if 'x' in df.columns:
        return 'point'
    raise ValueError("Annotation type could not be inferred from the columns in the dataframe")


def test():
    """
    pixi init
    pixi add nodejs
    pixi run npm install -g serve
    pixi run serve --cors -p 9999 /tmp/

    # http://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/stuart-annotation-test-view.json
    """
    #%load_ext autoreload
    #%autoreload 2

    from neuroglancer.coordinate_space import CoordinateSpace
    from neuclease.misc.neuroglancer.annotations.precomputed import write_annotations
    from neuclease.misc.neuroglancer.annotations.precomputed import (
        write_annotations, _encode_annotations, _encode_geometries_and_properties, _encode_relationships, 
        _encode_relationship, _write_annotations_by_relationship, _geometry_cols
    )

    body = 11064
    _df, df = fetch_label(*brain_master, 'synapses', body, relationships=True, format='pandas')
    df = (
        df
        .reset_index()
        .assign(body_a=body)
        .astype({'rel': 'category'})
        .rename(columns={'rel': 'kind'})
    )

    df['body_b'] = fetch_labels_batched(*brain_seg, df[['to_z', 'to_y', 'to_x']].values, processes=4)

    df = df.rename(columns={**{k: f'{k}a' for k in 'xyz'}, **{f'to_{k}': f'{k}b' for k in 'xyz'}})
    swap_df_cols(df, None, df['kind'] == "PostSynTo", ['a', 'b'])

    df = df.rename(columns={'body_a': 'pre_synaptic_body', 'body_b': 'post_synaptic_body'})

    cs = CoordinateSpace(
        names=[*'zyx'],
        scales=[8,8,8],
        units=['nm', 'nm', 'nm']
    )

    #%load_ext line_profiler
    #%lprun -f write_annotations -f _encode_annotations -f _encode_geometries_and_properties -f _encode_relationships -f _encode_relationship -f _write_annotations_by_relationship 
    write_annotations(df, cs, 'line', ['kind'], ['pre_synaptic_body', 'post_synaptic_body'], '/tmp/test-annotations')
