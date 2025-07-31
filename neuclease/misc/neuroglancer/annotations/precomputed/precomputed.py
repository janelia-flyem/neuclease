import os
import json
import logging
from itertools import chain
from typing import Literal

import pandas as pd
import numpy as np

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

from ..util import annotation_property_specs

from ._util import _encode_uint64_series, _geometry_cols
from ._id import _write_annotations_by_id
from ._relationships import _write_annotations_by_relationships, _encode_relationships
from ._spatial import _write_annotations_by_spatial_chunk

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
    write_by_spatial_chunk: bool = True,
    num_spatial_levels: int = 7,
    target_chunk_limit: int = 10_000,
    description: str = "",
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
            Categorical columns will be automatically converted to integers with associated
            enum labels.
            The full property spec for each property will be inferred from the column dtype,
            but if you want to explicitly override any property specs yourself, you can pass
            a list of AnnotationPropertySpec objects here.

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

        write_by_spatial_chunk:
            bool
            Whether to write the spatial index.

        num_spatial_levels:
            int
            The maximum number of spatial index levels to write.
            If not all levels are needed (because all annotations fit within the first N levels),
            then the actual number of levels written will be less than this value.

        target_chunk_limit:
            int
            For the spatial index, how many annotations we aim to place in each chunk (regardless of the level).
        
        description:
            str
            A description of the annotation collection.
    """
    # Verify that the neuroglancer package is available.
    from neuroglancer.coordinate_space import CoordinateSpace
    from neuroglancer.viewer_state import AnnotationPropertySpec

    os.makedirs(output_dir, exist_ok=True)
    annotation_type = annotation_type.lower()
    property_specs = annotation_property_specs(df, properties)
    bounds = _get_bounds(df, coord_space, annotation_type)

    # Construct a buffer for each annotation and additional buffers
    # for each annotation's relationships, stored in new columns of df.
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
    if write_by_spatial_chunk:
        spatial_metadata = _write_annotations_by_spatial_chunk(
            df,
            coord_space,
            annotation_type,
            bounds,
            num_spatial_levels,
            target_chunk_limit,
            output_dir,
            write_sharded
        )
    
    # Write the top-level 'info' file for the annotation output directory.
    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": coord_space.to_json(),
        "lower_bound": bounds[0].tolist(),
        "upper_bound": bounds[1].tolist(),
        "annotation_type": annotation_type,
        "properties": property_specs,
        "by_id": by_id_metadata,
        "relationships": by_rel_metadata,
        "spatial": spatial_metadata,
    }

    if description:
        info['description'] = description

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
        center = df[geometry_cols[0]].to_numpy()
        radii = df[geometry_cols[1]].to_numpy()
        return np.asarray([
            (center - radii).min(axis=0),
            (center + radii).max(axis=0)
        ])

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
