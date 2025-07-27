import logging
import numpy as np
import pandas as pd

from ._write_buffers import _write_buffers

logger = logging.getLogger(__name__)


def _write_annotations_spatial(df, bounds, output_dir, write_sharded):
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
    key = '_'.join('0' for _ in bounds[0])
    metadata_0 = _write_buffers(
        pd.Series([combined_buf], index=[key]),
        output_dir,
        "spatial0",
        write_sharded
    )

    metadata = [
        {
            **metadata_0,
            "grid_shape": [1] * len(bounds[0]),
            "chunk_size": np.maximum(bounds[1] - bounds[0], 1).tolist(),

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
