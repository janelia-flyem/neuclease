import logging
from ._write_buffers import _write_buffers

logger = logging.getLogger(__name__)


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

