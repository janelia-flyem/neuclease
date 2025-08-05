import logging
import numpy as np
import pandas as pd

from ._util import _encode_uint64_series, TableHandle
from ._write_buffers import _write_buffers

logger = logging.getLogger(__name__)


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
    #   Using sum() is O(R^2) in the number of relationships R, but we generally
    #   expect few relationships, so this is faster than df.apply(b''.join, axis=1),
    #   since .sum() uses a single C call whereas using b''.join() would
    #   use many Python calls.
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
            exactly one ID, you may pass a series with dtype=auint64,
            which will be interpreted as if each entry were a list of length 1.
            (In this case, the implementation is slightly faster than in the general case.)

    Returns:
        A numpy array with N entries, where each entry is a buffer as shown above.
    """
    # Special case if the relationship contains only a single ID for each annotation.
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


def _write_annotations_by_relationships(df_handle: TableHandle, relationships, output_dir, write_sharded):
    """
    Write the annotations to a "Related Object ID Index" for each relationship.
    Each relationship is written to a separate subdirectory of output_dir.

    Args:
        df_handle:
            TableHandle holding a DataFrame with columns ['id_buf', 'ann_buf', *relationships].
            The handle's reference will be unset before this function returns.

        relationships:
            List of relationship column names.

        output_dir:
            Directory to write the annotations to.
            Each relationship is written to a separate subdirectory of output_dir.

        write_sharded:
            Whether to write the annotations in sharded format.

    Returns:
        JSON metadata to be written under the 'relationships' key in the top-level 'info' file,
        consisting of a list of JSON objects (one for each relationship).
    """
    handles = {
        r: TableHandle(df_handle.df) for r in relationships
    }
    df_handle.df = None

    by_rel_metadata = []
    for relationship, df_handle in handles.items():
        metadata = _write_annotations_by_relationship(
            df_handle,
            relationship,
            output_dir,
            write_sharded
        )
        by_rel_metadata.append(metadata)

    return by_rel_metadata


def _write_annotations_by_relationship(df_handle: TableHandle, relationship, output_dir, write_sharded):
    """
    Write the annotations to a "Related Object ID Index" for a single relationship.

    Returns:
        JSON metadata for the relationship, including the key and sharding spec if applicable.
    """
    logger.info(f"Grouping annotations by relationship {relationship}")
    bufs_by_segment = (
        df_handle.df[['id_buf', 'ann_buf', relationship]]
        .dropna(subset=relationship)
        .explode(relationship)
        .groupby(relationship, sort=False)
        # Use b''.join() instead of 'sum' to avoid O(N^2) performance for large groups.
        .agg({'id_buf': ['count', b''.join], 'ann_buf': b''.join})
    )
    df_handle.df = None

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
