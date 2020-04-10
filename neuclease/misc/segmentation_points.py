import pandas as pd
from tqdm import tqdm

from neuclease.util import boxes_from_grid, ndindex_array
from neuclease.dvid import fetch_volume_box, fetch_labelmap_voxels


def one_point_per_body(server, uuid, seg_instance, box_zyx=None):
    """
    Given a labelmap instance, return a list of all its bodies and
    a single coordinate within each body.  The point will be at
    an arbitrary location within the body.

    Downloads the segmentation in blocks, keeping track of which bodies
    have been seen and an example coordinate within each.

    Args:
        server, uuid, seg_instance:
            A dvid segmentation instance

        box_zyx:
            Box [[z0, y0, x0], [z1, y1, x1]]
            Optional. If provided, only the given subvolume will be processed.

    Returns:
        DataFrame with columns [z, y, x, body]
    """
    if box_zyx is None:
        box_zyx = fetch_volume_box(server, uuid, seg_instance)

    BLOCK_SHAPE = (512, 64, 64)

    block_dfs = []
    boxes = boxes_from_grid(box_zyx, BLOCK_SHAPE, clipped=False)
    for block_box in tqdm(boxes):
        # Download the block segmentation
        block_vol = fetch_labelmap_voxels(server, uuid, seg_instance, block_box)

        # Construct a DataFrame containing every voxel coordinate and label
        block_coords = ndindex_array(*BLOCK_SHAPE) + block_box[0]
        block_df = pd.DataFrame(block_coords, columns=[*'zyx'])
        block_df['body'] = block_vol.reshape(-1)

        # We only need one row for each body
        block_df = block_df.drop_duplicates('body')
        block_dfs.append(block_df)

    # Combine results for all blocks
    full_df = pd.concat(block_dfs, ignore_index=True)
    full_df.drop_duplicates('body', inplace=True)
    return full_df


if __name__ == "__main__":
    #
    # EXAMPLES for VNC hotknife ground-truth volumes.
    #
    print("Finding hk1 body points")
    hk1_points_df = one_point_per_body('emdata2:8500', '8293', 'segmentation-hk1', [[16576, 13888, 16896], [16832, 16000, 19008]])
    hk1_points_df[[*'xyz', 'body']].to_csv('hk1-body-points.csv', header=True, index=False)

    print("Finding hk2 body points")
    hk2_points_df = one_point_per_body('emdata2:8500', '8293', 'segmentation-hk2')
    hk2_points_df[[*'xyz', 'body']].to_csv('hk2-body-points.csv', header=True, index=False)

    print("Finding hk3 body points")
    hk3_points_df = one_point_per_body('emdata2:8500', '8293', 'segmentation-hk3')
    hk3_points_df[[*'xyz', 'body']].to_csv('hk3-body-points.csv', header=True, index=False)

    print("DONE")
