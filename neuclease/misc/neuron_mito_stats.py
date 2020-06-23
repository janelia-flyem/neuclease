import sys
import logging
import argparse

logger = logging.getLogger(__name__)

try:
    from confiddler import load_config, dump_default_config
except ImportError:
    # https://confiddler.readthedocs.io/en/latest/quickstart.html
    msg = ("\nError: Missing dependency.\n"
           "Please install the 'confiddler' module:\n\n"
           "  conda install -c flyem-flows confiddler\n")
    sys.exit(msg)

# In the mito mask, empty space is given label 4 instead of label 0
EMPTY_MITO = 4

LabelmapSchema = {
    "description": "dvid labelmap location",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "required": ["server", "uuid", "instance"],
    "properties": {
        "server": {
            "description": "DVID server",
            "type": "string"
        },
        "uuid": {
            "description": "DVID node",
            "type": "string"
        },
        "instance": {
            "description": "Name of the labelmap instance",
            "type": "string"
        }
    }
}

ConfigSchema = {
    "description": "Volume source config",
    "default": {},
    "required": ["segmentation", "mito-objects", "mito-masks"],
    "additionalProperties": False,
    "properties": {
        "scale": {
            "description": "Which scale to use when downloading segmentation/mask data and analyzing it.",
            "type": "integer",
            "minValue": 0,
            "default": 0
        },
        "min-size": {
            "description": "Exclude mitochondria below this size (in voxels) from the results.",
            "type": "number",
            "default": 0
        },
        "centroid-adjustment-radius": {
            "description": "Since mitochondria are not always convex, their centroids can\n"
                           "sometimes lie outside of their boundaries.  In those cases, we\n"
                           "search for the closest in-bounds point to the true centroid, \n"
                           "but we only search within a given radius of the true centroid.\n"
                           "This setting specifies the search radius (in voxels).\n",
            "type": "integer",
            "default": 100
        },
        "segmentation": {**LabelmapSchema, "description": "Neuron segmentation location"},
        "mito-objects": {**LabelmapSchema, "description": "Mitochondria connected component location"},
        "mito-masks": {**LabelmapSchema, "description": "Mitochondria classification mask location"},
    }
}


def neuron_mito_stats(seg_src, mito_cc_src, mito_class_src, body_id, scale=0, min_size=0, search_radius=50, processes=1):
    from functools import partial
    import numpy as np
    import pandas as pd

    from neuclease.util import compute_parallel
    from neuclease.dvid import fetch_sparsevol_coarse, resolve_ref, fetch_labels, fetch_labelmap_voxels

    seg_src[1] = resolve_ref(*seg_src[:2])
    mito_cc_src[1] = resolve_ref(*mito_cc_src[:2])
    mito_class_src[1] = resolve_ref(*mito_class_src[:2])

    # Fetch block coords; re-scale for the analysis scale
    block_coords = (2**6) * fetch_sparsevol_coarse(*seg_src, body_id)
    bc_df = pd.DataFrame(block_coords, columns=[*'zyx'])
    bc_df[[*'zyx']] //= 2**scale
    block_coords = bc_df.drop_duplicates().values

    #
    # Blockwise stats
    #
    block_fn = partial(_process_block, seg_src, mito_cc_src, mito_class_src, body_id, scale)
    block_tables = compute_parallel(block_fn, block_coords, processes=processes)
    block_tables = [*filter(lambda t: t is not None, block_tables)]

    #
    # Combine stats
    #
    full_table = pd.concat(block_tables).fillna(0)
    class_cols = [*filter(lambda c: c.startswith('class'), full_table.columns)]
    full_table = full_table.astype({c: np.int32 for c in class_cols})

    # Weight each block centroid by the block's voxel count before taking the mean
    full_table[[*'zyx']] *= full_table[['total_size']].values
    stats_df = full_table.groupby('mito_id').sum()
    stats_df[[*'zyx']] /= stats_df[['total_size']].values

    # Drop tiny mitos
    stats_df = stats_df.query("total_size >= @min_size").copy()

    # Assume all centroids are 'exact' by default (overwritten below if necessary)
    stats_df['centroid_type'] = 'exact'

    # Include a column for 'body' even thought its the same on every row,
    # just as a convenience for concatenating these results with the results
    # from other bodies if desired.
    stats_df['body'] = body_id

    stats_df = stats_df.astype({a: np.int32 for a in 'zyx'})
    stats_df = stats_df[['body', *'xyz', 'total_size', *class_cols, 'centroid_type']]

    #
    # Check for centroids that fall outside of the mito,
    # and adjust them if necessary.
    #
    centroid_mitos = fetch_labels(*mito_cc_src, stats_df[[*'zyx']].values, scale=scale)
    mismatches = stats_df.index[(stats_df.index != centroid_mitos)]

    if len(mismatches) == 0:
        return stats_df

    logger.warning("Some mitochondria centroids do not lie within the mitochondria itself. "
                   "Searching for pseudo-centroids.")

    # construct field of distances from the central voxel
    sr = search_radius
    cz, cy, cx = np.ogrid[-sr:sr+1, -sr:sr+1, -sr:sr+1]
    distances = np.sqrt(cz**2 + cy**2 + cx**2)

    pseudo_centroids = []
    error_mito_ids = []
    for row in stats_df.loc[mismatches].itertuples():
        mito_id = row.Index
        centroid = np.array((row.z, row.y, row.x))
        box = (centroid - sr, 1 + centroid + sr)
        mito_mask = (mito_id == fetch_labelmap_voxels(*mito_cc_src, box, scale))

        if not mito_mask.any():
            pseudo_centroids.append((row.z, row.y, row.x))
            error_mito_ids.append(mito_id)
            continue

        # Find minimum distance
        masked_distances = np.where(mito_mask, distances, np.inf)
        new_centroid = np.unravel_index(np.argmin(masked_distances), masked_distances.shape)
        new_centroid = np.array(new_centroid) + centroid - sr
        pseudo_centroids.append(new_centroid)

    stats_df.loc[mismatches, ['z', 'y', 'x']] = np.array(pseudo_centroids, dtype=np.int32)
    stats_df.loc[mismatches, 'centroid_type'] = 'adjusted'
    stats_df.loc[error_mito_ids, 'centroid_type'] = 'error'

    if error_mito_ids:
        logger.warning("Some mitochondria pseudo-centroids could not be found.")

    stats_df = stats_df.astype({a: np.int32 for a in 'zyx'})
    return stats_df


def _process_block(seg_src, mito_cc_src, mito_class_src, body_id, scale, block_coord):
    import numpy as np
    import pandas as pd

    from neuclease.util import ndindex_array
    from neuclease.dvid import fetch_labelmap_voxels

    block_box = np.array((block_coord, block_coord+64))
    block_seg = fetch_labelmap_voxels(*seg_src, block_box, scale)
    mito_labels = fetch_labelmap_voxels(*mito_cc_src, block_box, scale)
    mito_classes = fetch_labelmap_voxels(*mito_class_src, block_box, scale)

    body_mask = (block_seg == body_id)
    mito_mask = (mito_labels != 0) & (mito_classes != EMPTY_MITO)
    mask = (body_mask & mito_mask)
    if not mask.any():
        # No mito voxels of interest in this block
        return None

    unraveled_df = pd.DataFrame({'mito_id': mito_labels.reshape(-1),
                                 'mito_class': mito_classes.reshape(-1)})

    # pivot_table() doesn't work without a data column to aggregate
    unraveled_df['voxels'] = 1

    # Add coordinate columns to compute centroids
    raster_coords = ndindex_array(*(64, 64, 64), dtype=np.int32)
    raster_coords += block_coord
    unraveled_df['z'] = np.int8(0)
    unraveled_df['y'] = np.int8(0)
    unraveled_df['x'] = np.int8(0)
    unraveled_df[['z', 'y', 'x']] = raster_coords

    # Drop non-body voxels and non-mito-voxels
    unraveled_df = unraveled_df.iloc[mask.reshape(-1)]

    block_table = (unraveled_df[['mito_id', 'mito_class', 'voxels']]
                    .pivot_table(index='mito_id',  # noqa
                                 columns='mito_class',
                                 values='voxels',
                                 aggfunc='sum',
                                 fill_value=0))

    block_table.columns = [f"class_{c}" for c in block_table.columns]
    block_table['total_size'] = block_table.sum(axis=1).astype(np.int32)

    # Compute block centroid for each mito
    mito_points = unraveled_df.groupby('mito_id')[['z', 'y', 'x']].mean().astype(np.float32)
    block_table = block_table.merge(mito_points, 'left', left_index=True, right_index=True)
    return block_table


def main():
    # Early exit if we're dumping the config
    # (Parse it ourselves to allow omission of otherwise required parameters.)
    if ({'--dump-config-template', '-d'} & {*sys.argv}):
        dump_default_config(ConfigSchema, sys.stdout, "yaml-with-comments")
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-config-template', '-d', action='store_true',
                        help='Dump out a template yaml config file and exit.')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help="Size of the process pool to use")
    parser.add_argument('config')
    parser.add_argument('body_id', type=int)
    args = parser.parse_args()

    import numpy as np
    from neuclease import configure_default_logging
    configure_default_logging()

    config = load_config(args.config, ConfigSchema)
    seg_src = [*config["segmentation"].values()]
    mito_cc_src = [*config["mito-objects"].values()]
    mito_class_src = [*config["mito-masks"].values()]

    stats_df = neuron_mito_stats(seg_src, mito_cc_src, mito_class_src, args.body_id,
                                 config["scale"], config["min-size"], config["centroid-adjustment-radius"],
                                 args.processes)

    csv_path = f"mito-stats-{args.body_id}-scale-{config['scale']}.csv"
    logger.info(f"Writing {csv_path}")
    stats_df.to_csv(csv_path, index=True, header=True)

    npy_path = f"mito-stats-{args.body_id}-scale-{config['scale']}.npy"
    logger.info(f"Writing {npy_path}")
    np.save(npy_path, stats_df.to_records(index=True))

    logger.info("DONE")


if __name__ == "__main__":
    main()
