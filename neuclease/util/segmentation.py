import vigra
import numpy as np
import pandas as pd
from collections import OrderedDict

from . import Grid, boxes_from_grid, box_intersection, box_to_slicing

BLOCK_STATS_DTYPES = OrderedDict([ ('segment_id', np.uint64),
                                   ('z', np.int32),
                                   ('y', np.int32),
                                   ('x', np.int32),
                                   ('count', np.uint32) ])


def block_stats_for_volume(block_shape, volume, physical_box):
    """
    Get the count of voxels for each segment (excluding segment 0)
    in each block within the given volume, returned as a DataFrame.
    
    Returns a DataFrame with the following columns:
        ['segment_id', 'z', 'y', 'x', 'count']
        where z,y,z are the starting coordinates of each block.
    """
    block_grid = Grid(block_shape)
    
    block_dfs = []
    block_boxes = boxes_from_grid(physical_box, block_grid)
    for box in block_boxes:
        clipped_box = box_intersection(box, physical_box) - physical_box[0]
        block_vol = volume[box_to_slicing(*clipped_box)]
        counts = pd.Series(block_vol.reshape(-1)).value_counts(sort=False)
        segment_ids = counts.index.values
        counts = counts.values.astype(np.uint32)

        box = box.astype(np.int32)

        block_df = pd.DataFrame( { 'segment_id': segment_ids,
                                   'count': counts,
                                   'z': box[0][0],
                                   'y': box[0][1],
                                   'x': box[0][2] } )

        # Exclude segment 0 from output
        block_df = block_df[block_df['segment_id'] != 0]

        block_dfs.append(block_df)

    brick_df = pd.concat(block_dfs, ignore_index=True)
    brick_df = brick_df[['segment_id', 'z', 'y', 'x', 'count']]
    assert list(brick_df.columns) == list(BLOCK_STATS_DTYPES.keys())
    return brick_df


def contingency_table(left_vol, right_vol):
    """
    Overlay left_vol and right_vol and compute the table of
    overlapping label pairs, along with the size of each overlapping
    region.
    
    Args:
        left_vol, right_vol:
            np.ndarrays of equal shape
    
    Returns:
        pd.Series of sizes with a multi-level index (left,right)
    """
    assert left_vol.shape == right_vol.shape
    df = pd.DataFrame( {"left": left_vol.reshape(-1),
                        "right": right_vol.reshape(-1)},
                       dtype=left_vol.dtype )
    sizes = df.groupby(['left', 'right']).size()
    sizes.name = 'voxel_count'
    return sizes


def compute_cc(img, min_component=1):
    """
    Compute the connected components of the given label image,
    and return a pd.Series that maps from the CC label back to the original label.
    
    Pixels of value 0 are treated as background and not labeled.
    
    Args:
        img:
            ND label image, either np.uint8, np.uint32, or np.uint64
        
        min_component:
            Output components will be indexed starting with this value
            (but 0 is not affected)
        
    Returns:
        img_cc, cc_mapping, where:
            - img_cc is the connected components image (as np.uint32)
            - cc_mapping is pd.Series, indexed by CC labels, data is original labels.
    """
    assert min_component > 0
    if img.dtype in (np.uint8, np.uint32):
        img_cc = vigra.analysis.labelMultiArrayWithBackground(img)
    elif img.dtype == np.uint64:
        # Vigra's labelMultiArray() can't handle np.uint64,
        # so we must convert it to np.uint32 first.
        # We can't simply truncate the label values,
        # so we "consecutivize" them.
        img32 = np.zeros_like(img, dtype=np.uint32, order='C')
        _, _, _ = vigra.analysis.relabelConsecutive(img, out=img32)
        img_cc = vigra.analysis.labelMultiArrayWithBackground(img32)
    else:
        raise AssertionError(f"Invalid label dtype: {img.dtype}")    
    
    cc_mapping_df = pd.DataFrame( { 'orig': img.flat, 'cc': img_cc.flat } )
    cc_mapping_df.drop_duplicates(inplace=True)
    
    if min_component > 1:
        img_cc[img_cc != 0] += np.uint32(min_component-1)
        cc_mapping_df.loc[cc_mapping_df['cc'] != 0, 'cc'] += np.uint32(min_component-1)

    cc_mapping = cc_mapping_df.set_index('cc')['orig']
    return img_cc, cc_mapping
    

