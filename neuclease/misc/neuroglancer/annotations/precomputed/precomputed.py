import os
import json
import logging
from itertools import chain
from typing import Literal

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorstore as ts

try:
    from neuroglancer.coordinate_space import CoordinateSpace
    from neuroglancer.viewer_state import AnnotationPropertySpec
except ImportError:
    # For now, we support importing this module without neuroglancer,
    # even though these functions require it.
    # We define these names here so the type hints below don't fail.
    class CoordinateSpace:
        pass
    class AnnotationPropertySpec:
        pass

from ..util import annotation_property_specs, choose_output_spec

logger = logging.getLogger(__name__)


def write_precomputed_annotations(
    df: pd.DataFrame,
    coord_space: CoordinateSpace,
    annotation_type: Literal['point', 'line', 'ellipsoid', 'axis_aligned_bounding_box'],
    properties: list[str] | list[AnnotationPropertySpec] | dict[str, AnnotationPropertySpec] | list[dict] = (),
    relationships: list[str] = (),
    output_dir: str = 'annotations',
    write_sharded: bool = True,
    *,
    write_by_id: bool = True,
    write_by_relationship: bool = True,
    write_single_spatial_level: bool = False,
):
    """
    Export the data from a pandas DataFrame into neuroglancer's precomputed annotations format
    as described in https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md

    A progress bar is shown when writing each portion of the export (annotation ID index, related ID indexes),
    but there may be a significant amount of preprocessing time that occurs before the actual writing begins.

    Limitations:

        - This implementation only supports a single spatial grid level, which means neuroglancer
          will download all annotations at once if they are viewed without respect to a relationship.
          This means that spatial indexing should not be used for large datasets (more than 1M annotations or so.)
        - We do not yet support rgb or rgba properties.
          In a future version, we might support them via strings like '#ff0000' or via MultiIndex columns.

    Args:
        df:
            DataFrame.
            The required columns depend on the annotation_type and the coordinate space.
            For example, assuming ``coord_space.names == ['x', 'y', 'z']``,
            then provide the following columns:

            - For point annotations, provide ['x', 'y', 'z']
            - For line annotations or axis_aligned_bounding_box annotations,
              provide ['xa', 'ya', 'za', 'xb', 'yb', 'zb']
            - For ellipsoid annotations, provide ['x', 'y', 'z', 'rx', 'ry', 'rz']
              for the center point and radii.

            You may also provide additional columns to use as annotation properties,
            in which case they should be listed in the 'properties' argument. (See below.)

            The index of the DataFrame is used as the annotation ID.

        coord_space:
            CoordinateSpace.
            The coordinate space of the annotations.
            This is used to determine the geometry of the annotations.

        annotation_type:
            Literal['point', 'line', 'ellipsoid', 'axis_aligned_bounding_box']
            The type of annotation to export. Note that the columns you provide in
            the DataFrame depend on the annotation type.

        properties:
            If your dataframe contains columns that you want to use as annotation properties,
            list the names of those columns here.
            The full property spec for each property will be inferred from the column dtype,
            but if you want to specify the property spec yourself, you can pass a list of
            AnnotationPropertySpec objects here.

        relationships:
            list[str]
            If your annotations have related segment IDs, such relationships can be provided
            in the columns of your DataFrame. Each relationship should be listed in a single column,
            whose values are lists of segment IDs.  In the special case where each annotation has
            exactly one related segment, the column may have dtype=np.uint64 instead of containing lists.

        output_dir:
            str
            The directory into which the exported annotations will be written.
            Subdirectories will be created for the "annotation ID index" and each
            "related object id index" as needed.

        write_sharded:
            bool
            Whether to write the output as sharded files.
            The sharded format is preferable for most use cases.
            Without sharding, every annotation results in a separate file in the annotation ID index.
            Similarly, every related ID results in a separate file in the related ID index.

        write_by_id:
            bool
            Whether to write the annotations to the "Annotation ID Index".
            If False, skip writing.

        write_relationships:
            bool
            Whether to write the relationships to the "Related Object ID Index".
            If False, skip writing.

        write_single_spatial_level:
            bool
            If True, write the spatial index as a single grid level. With a spatial index
            all annotations can be viewed at once, independent of any relationships.
            However, as the argument name suggests, we only support a single spatial grid level
            (containing all annotations), which is not suitable for millions of annotations.
            If False, no spatial index will be written at all.
    """
    # Verify that the neuroglancer package is available.
    from neuroglancer.coordinate_space import CoordinateSpace
    from neuroglancer.viewer_state import AnnotationPropertySpec

    os.makedirs(output_dir, exist_ok=True)
    annotation_type = annotation_type.lower()
    property_specs = annotation_property_specs(df, properties)
    lower_bound, upper_bound = _get_bounds(df, coord_space, annotation_type)

    # Construct a buffer for each annotation and additional buffers
    # for each annotation's relationships, stored in extra columns of df.
    df = _encode_annotations(
        df,
        coord_space,
        annotation_type,
        property_specs,
        relationships
    )

    by_id_metadata = {}
    if write_by_id:
        by_id_metadata = _write_annotations_by_id(
            df,
            output_dir,
            write_sharded
        )

    by_rel_metadata = []
    if write_by_relationship:    
        by_rel_metadata = _write_annotations_by_relationships(
            df,
            relationships,
            output_dir,
            write_sharded
        )

    spatial_metadata = []
    if write_single_spatial_level:
        spatial_metadata = _write_annotations_spatial(
            df,
            lower_bound,
            upper_bound,
            output_dir,
            write_sharded
        )
    
    # Write the top-level 'info' file for the annotation output directory.
    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": coord_space.to_json(),
        "lower_bound": lower_bound.tolist(),
        "upper_bound": upper_bound.tolist(),
        "annotation_type": annotation_type,
        "properties": property_specs,
        "by_id": by_id_metadata,
        "relationships": by_rel_metadata,
        "spatial": spatial_metadata,
    }

    with open(f"{output_dir}/info", 'w') as f:
        json.dump(info, f)


def _get_bounds(df, coord_space, annotation_type):
    """
    Inspect the geometry columns of the given dataframe to
    determine the overall upper and lower bounds of the annotations.

    Returns:
        lower_bound, upper_bound
        (both numpy arrays of length 3)
    """
    geometry_cols = _geometry_cols(coord_space.names, annotation_type)
    if (gc := set(chain(*geometry_cols))) > set(df.columns):
        raise ValueError(
            "Dataframe does not have all required geometry columns for the given coordinate space.\n"
            f"Required columns: {gc}"
        )

    if annotation_type == 'point':
        points = df[geometry_cols[0]]
        return (
            points.min(axis=0).to_numpy(),
            points.max(axis=0).to_numpy()
        )

    if annotation_type in ('line', 'axis_aligned_bounding_box'):
        points_a = df[geometry_cols[0]]
        points_b = df[geometry_cols[1]]
        return (
            np.minimum(points_a.min().to_numpy(), points_b.min().to_numpy()),
            np.maximum(points_a.max().to_numpy(), points_b.max().to_numpy())
        )

    if annotation_type == 'ellipsoid':
        center = df[geometry_cols[0]]
        radii = df[geometry_cols[1]]
        return (
            (center - radii).min().to_numpy(),
            (center + radii).max().to_numpy()
        )

    raise ValueError(f"Annotation type {annotation_type} not supported")


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


def _encode_annotations(df, coord_space, annotation_type, property_specs, relationships):
    """
    Returns a (shallow) copy of the dataframe with additional columns containing
    buffers for each encoded annotation and its encoded id and encoded relationships.
    """
    df = df.copy(deep=False)

    logger.info("Encoding annotation IDs")
    df['id_buf'] = _encode_uint64_series(df.index)

    logger.info("Encoding annotation geometries and properties")
    df['ann_buf'] = _encode_geometries_and_properties(
        df, coord_space, annotation_type, property_specs
    )

    logger.info("Encoding relationships")
    rel_bufs = _encode_relationships(df, relationships)
    if rel_bufs is not None:
        df['rel_buf'] = rel_bufs

    return df


def _encode_uint64_series(s, dtype='<u8'):
    """
    Encode a pandas Series (or Index) of N values
    into a numpy array of N buffers (bytes objects).
    """
    id_buf = s.to_numpy(dtype).tobytes()
    id_bufs = [
        id_buf[offset:(offset+8)]
        for offset in range(0, len(id_buf), 8)
    ]
    return np.array(id_bufs, dtype=object)


def _encode_geometries_and_properties(df, coord_space, annotation_type, property_specs):
    """
    For each annotation in the given dataframe, encode its geometry columns (e.g. x,y,z)
    and property columns into a buffer, plus any padding that was necessary to align the
    buffer to a 4-byte boundary, per the neuroglancer spec.

    (In the precomputed format, geometry and properties always appear together,
    regardless of whether they're being written to the "Annotation ID Index",
    the "Related Object ID Index" or the "Spatial Index".)

    Returns:
        pd.Series of dtype=object, containing one buffer for each annotation.
    """
    geometry_cols = _geometry_cols(coord_space.names, annotation_type)
    prop_cols = [p['id'] for p in property_specs]
    geometry_prop_df = df[[*chain(*geometry_cols), *prop_cols]].copy(deep=False)

    property_widths = [
        {'rgb': 3, 'rgba': 4}.get(p['type'], np.dtype(p['type']).itemsize)
        for p in property_specs
    ]
    property_padding = (4 - (sum(property_widths) % 4)) % 4
    for i in range(property_padding):
        geometry_prop_df[f'__padding_{i}__'] = np.uint8(0)

    dtypes = {c: np.float32 for c in chain(*geometry_cols)}
    dtypes.update({
        p['id']: p['type']
        for p in property_specs
        if df[p['id']].dtype != 'category' and p['type'] not in ('rgb', 'rgba')
    })

    # Convert category columns to their integer equivalents
    for p in property_specs:
        if df[p['id']].dtype == 'category':
            geometry_prop_df[p['id']] = geometry_prop_df[p['id']].cat.codes
            dtypes[p['id']] = p['type']

    # Vectorized serialization
    records = geometry_prop_df.to_records(index=False, column_dtypes=dtypes)
    recsize = records.dtype.itemsize
    buf = records.tobytes()

    # Reduce RAM usage
    del records

    # extract bytes from the appropriate slice for each record
    encoded_annotations = [buf[i*recsize:(i+1)*recsize] for i in range(len(df))]
    ann_bufs = pd.Series(encoded_annotations, index=df.index)
    return ann_bufs


def _encode_relationships(df, relationships):
    """
    For each annotation in the given dataframe, encode the related IDs
    for all relationships into a buffer according to the neuroglancer spec.

    Returns:
        pd.Series of dtype=object, containing one buffer for each annotation.
    """
    if not relationships:
        return None

    encoded_relationships = {}
    for rel_col in relationships:
        encoded_relationships[rel_col] = _encode_related_ids(df[rel_col])

    # Concatenate buffers on each row.
    # Note:
    #   Using sum() is O(N^2) in the number of relationships, but we generally
    #   expect few relationships, so this is faster than df.apply(b''.join, axis=1)
    rel_bufs = pd.DataFrame(encoded_relationships, index=df.index).sum(axis=1)
    return rel_bufs


def _encode_related_ids(related_ids):
    """
    Given a Series containing lists of IDs, encode each list of IDs
    in the format neuroglancer expects for each relationship in an
    annotation.

    Each item in related_ids is a list, which gets encoded as
    <count><id_1><id_2><id_3>..., where <count> is uint32 and
    <id_1><id_2><id_3>... are each uint64.

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
    else:
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


def _write_annotations_by_id(df, output_dir, write_sharded):
    """
    Write the annotations to the "Annotation ID Index", a subdirectory of output_dir.

    Returns:
        JSON metadata to be written under the 'by_id' key in the top-level 'info' file.
        Currently, this is always {"key": "by_id"}
    """
    if 'rel_buf' in df.columns:
        ann_bufs = df['ann_buf'] + df['rel_buf']
    else:
        ann_bufs = df['ann_buf']

    logger.info("Writing annotations to 'by_id' index")
    metadata = _write_buffers(ann_bufs, output_dir, "by_id", write_sharded)
    return metadata


def _write_annotations_by_relationships(df, relationships, output_dir, write_sharded):
    """
    Write the annotations to a "Related Object ID Index" for each relationship.
    Each relationship is written to a separate subdirectory of output_dir.

    Returns:
        JSON metadata to be written under the 'relationships' key in the top-level 'info' file,
        consisting of a list of JSON objects (one for each relationship).
    """
    by_rel_metadata = []
    for relationship in relationships:
        metadata = _write_annotations_by_relationship(
            df,
            relationship,
            output_dir,
            write_sharded
        )
        by_rel_metadata.append(metadata)

    return by_rel_metadata


def _write_annotations_by_relationship(df, relationship, output_dir, write_sharded):
    """
    Write the annotations to a "Related Object ID Index" for a single relationship.

    Returns:
        JSON metadata for the relationship, including the key and sharding spec if applicable.
    """
    logger.info(f"Grouping annotations by relationship {relationship}")
    bufs_by_segment = (
        df[['id_buf', 'ann_buf', relationship]]
        .dropna(subset=relationship)
        .explode(relationship)
        .groupby(relationship, sort=False)
        # Use b''.join() instead of 'sum' to avoid O(N^2) performance for large groups.
        .agg({'id_buf': ['count', b''.join], 'ann_buf': b''.join})
    )
    logger.info(f"Combining annotation and ID buffers for relationship '{relationship}'")
    bufs_by_segment.columns = ['count', 'id_buf', 'ann_buf']
    bufs_by_segment['count_buf'] = _encode_uint64_series(bufs_by_segment['count'])
    bufs_by_segment['combined_buf'] = bufs_by_segment[['count_buf', 'ann_buf', 'id_buf']].sum(axis=1)

    logger.info(f"Writing annotations to 'by_rel_{relationship}' index")
    metadata = _write_buffers(
        bufs_by_segment['combined_buf'],
        output_dir,
        f"by_rel_{relationship}",
        write_sharded
    )
    metadata['id'] = relationship
    return metadata


def _write_annotations_spatial(df, lower_bound, upper_bound, output_dir, write_sharded):
    """
    Write the annotations to the spatial index.
    Currently, we only support a single spatial grid level,
    resulting in a single annotation list.
    """
    # According to the spec[1]:
    #   "For the spatial index, the annotations should be ordered randomly."
    #
    # This probably doesn't matter here since we're using a limit of 1,
    # but let's go ahead and follow the spec.
    #
    # [1]: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md
    logger.info("Shuffling annotations for spatial index")
    df = df.sample(frac=1)

    logger.info("Concatenating all annotation buffers for the spatial index")
    count_buf = np.uint64(len(df)).tobytes()
    all_annotations_buf = b''.join(df['ann_buf'])
    all_ids_buf = b''.join(df['id_buf'])
    combined_buf = b''.join([count_buf, all_annotations_buf, all_ids_buf])

    # For now, just one big buffer.
    logger.info(f"Writing annotations to spatial index")
    key = '_'.join('0' for _ in lower_bound)
    metadata_0 = _write_buffers(
        pd.Series([combined_buf], index=[key]),
        output_dir,
        "spatial0",
        write_sharded
    )

    metadata = [
        {
            **metadata_0,
            "grid_shape": [1] * len(lower_bound),
            "chunk_size": np.maximum(upper_bound - lower_bound, 1).tolist(),

            # According to jbms:
            #   Neuroglancer "subsamples" by showing only a prefix of the list of
            #   annotations according to the spacing setting.  If you set "limit" to 1 in
            #   the info file, you won't get subsampling by default.  If you want the
            #   subsampling to do something reasonable, then you can randomly shuffle the
            #   order in which you write the annotations.
            # Source:
            #   https://github.com/google/neuroglancer/issues/227#issuecomment-651944575
            "limit": 1,
        },
    ]
    return metadata


def _write_buffers(buf_series, output_dir, subdir, write_sharded):
    """
    Write the buffers to the appropriate subdirectory of output_dir,
    in sharded or unsharded format.

    Args:
        buf_series:
            pd.Series of dtype=object, whose values are buffers (bytes objects).
            The index of the series provides the keys under which each item is stored.

        output_dir:
            str
            The directory into which the exported annotations will be written.

        subdir:
            str
            The subdirectory into which the buffers will be written.

        write_sharded:
            bool
            If True, write the buffers in sharded format.
            If False, write the buffers in unsharded format, i.e. one file per item.

    Returns:
        JSON metadata for the written data, including the key (subdir)
        and sharding spec if applicable.
    """
    if write_sharded:
        return _write_buffers_sharded(buf_series, output_dir, subdir)
    else:
        return _write_buffers_unsharded(buf_series, output_dir, subdir)


def _write_buffers_unsharded(buf_series, output_dir, subdir):
    """
    Write the buffers to the appropriate subdirectory of output_dir,
    in unsharded format, i.e. one file per item.

    The index of buf_series is used as the key for each item, after being
    converted to a string (as decimal values in the case of integer keys).

    Returns:
        JSON metadata, always {"key": subdir}
    """
    # In the unsharded format, the keys are just strings (e.g. decimal IDs).
    string_keys = buf_series.index.astype(str)
    buf_series = buf_series.set_axis(string_keys)

    # Since we're writing unsharded files, we could have just used
    # standard Python open() and write() here for each key.
    # Using tensorstore here is mostly just a matter of taste, but it will
    # become useful if we ever support alternative storage backends such as gcs.
    kvstore = ts.KvStore.open(f"file://{output_dir}/{subdir}/").result()

    # Using a transaction here is not necessary, at least for plain files.
    # I'm not sure if it helps or hurts, but it probably doesn't matter much
    # for small datasets, which is presumably what we're dealing with if the
    # user has chosen the unsharded format.
    with ts.Transaction() as txn:
        for segment_key, buf in tqdm(buf_series.items(), total=len(buf_series)):
            kvstore.with_transaction(txn)[segment_key] = buf

    metadata = {"key": subdir}
    return metadata


def _write_buffers_sharded(buf_series, output_dir, subdir):
    """
    Write the buffers to the appropriate subdirectory of output_dir,
    in sharded format.

    The index of buf_series is used as the key for each item,
    after being encoded as a bigendian uint64.

    Returns:
        JSON metadata, including the output "key" (subdir) and sharding spec.
    """
    # When writing sharded data, we must use encoded bigendian uint64 as the key.
    # https://github.com/google/neuroglancer/pull/522#issuecomment-1923137085
    bigendian_keys = _encode_uint64_series(buf_series.index, '>u8')
    buf_series = buf_series.set_axis(bigendian_keys)

    shard_spec = choose_output_spec(
        total_count=len(buf_series),
        total_bytes=buf_series.map(len).sum(),  # fixme, might be slow
        hashtype='murmurhash3_x86_128',
        gzip_compress=True
    )
    spec = {
        "driver": "neuroglancer_uint64_sharded",
        "metadata": shard_spec.to_json(),
        "base": f"file://{output_dir}/{subdir}",
    }
    kvstore = ts.KvStore.open(spec).result()

    # Note:
    #   At the time of this writing, tensorstore uses a
    #   surprising amount of RAM to perform the writes.
    with ts.Transaction() as txn:
        for segment_key, buf in tqdm(buf_series.items(), total=len(buf_series)):
            kvstore.with_transaction(txn)[segment_key] = buf

    metadata = {
        "key": subdir,
        "sharding": shard_spec.to_json()
    }
    return metadata


def _assign_spatial_grid(df, coord_space, annotation_type, lower_bound, upper_bound, max_grid_depth):
    """
    Assign each annotation to a spatial grid cell.
    """
    grid_shape = np.ceil((upper_bound - lower_bound) / 2**max_grid_depth)
    