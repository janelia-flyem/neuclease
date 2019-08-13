from collections import OrderedDict

import numpy as np
import pandas as pd
from numba import njit

from dvidutils import LabelMapper

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


def mask_for_labels(volume, label_ids):
    """
    Given a label volume and a subset of labels to keep,
    return a boolean mask indicating which voxels fall on the given label_ids.
    """
    if volume.flags.c_contiguous:
        flatvol = volume.reshape(-1)
    else:
        flatvol = volume.copy('C').reshape(-1)

    if not isinstance(label_ids, (set, pd.Index)):
        label_ids = set(label_ids)

    valid_positions = pd.DataFrame(flatvol, columns=['label']).eval('label in @label_ids')
    return valid_positions.values.reshape(volume.shape)


def apply_mask_for_labels(volume, label_ids, inplace=False):
    """
    Given a label volume and a subset of labels to keep,
    mask out all voxels that do not fall on the given label_ids
    (i.e. set them to 0).
    """
    if inplace:
        assert volume.flags.c_contiguous
        flatvol = volume.reshape(-1)
    else:
        flatvol = volume.copy('C').reshape(-1)

    if not isinstance(label_ids, (set, pd.Index)):
        label_ids = set(label_ids)

    erase_positions = pd.DataFrame(flatvol, columns=['label']).eval('label not in @label_ids')

    flatvol[erase_positions.values] = 0
    return flatvol.reshape(volume.shape)


def box_from_coords(coords):
    """
    Determine the bounding box of the given list of coordinates.
    """
    start = coords.min(axis=0)
    stop = 1+coords.max(axis=0)
    box = np.array((start, stop))
    return box


def mask_from_coords(coords):
    """
    Given a list of coordinates, create a dense mask array,
    whose shape is determined by the bounding box of
    the coordinates.
    
    Returns:
        (mask, box)
        where mask.shape == box[1] - box[0],
    """
    coords = np.asarray(coords)
    box = box_from_coords(coords)
    mask = np.zeros(box[1] - box[0], bool)
    mask[(*(coords - box[0]).transpose(),)] = True
    return mask, box


def compute_nonzero_box(mask, save_ram=False):
    """
    Given a mask image, return the bounding box of the nonzero voxels in the mask.
    
    Equivalent to:
    
        coords = np.transpose(np.nonzero(mask))
        if len(coords) == 0:
            return np.zeros((2, mask.ndim))
        box = np.array([coords.min(axis=0), 1+coords.max(axis=0)])
        return box
    
    but if save_ram=True, this function avoids allocating
    the coords array, but performs slightly worse than the simple
    implementation unless your array is so very large and dense
    that allocating the coordinate list is a problem.
    """
    if save_ram:
        box = _compute_nonzero_box(mask)
    
        # If the volume is completely empty,
        # the helper returns an invalid box.
        # In that case, return zeros
        if (box[1] <= box[0]).any():
            return np.zeros_like(box)

        return box

    else:
        coords = np.transpose(np.nonzero(mask))
        if len(coords) == 0:
            return np.zeros((2, mask.ndim))
        box = np.array([coords.min(axis=0), 1+coords.max(axis=0)])
        return box


@njit
def _compute_nonzero_box(mask):
    """
    Helper for compute_nonzero_box().

    Note:
        If the mask has no nonzero voxels, an "invalid" box is returned,
        i.e. the start is above the stop.
    """
    box = np.zeros((2, mask.ndim), np.int32)
    box[0, :] = mask.shape

    c = np.zeros((mask.ndim,), np.int32)
    for i, val in np.ndenumerate(mask):
        if val != 0:
            c[:] = i
            box[0] = np.minimum(c, box[0])
            box[1] = np.maximum(c, box[1])
    box[1] += 1
    return box


def contingency_table(left_vol, right_vol):
    """
    Overlay left_vol and right_vol and compute the table of
    overlapping label pairs, along with the size of each overlapping
    region.
    
    Args:
        left_vol, right_vol:
            np.ndarrays of equal shape
    
    Returns:
        pd.Series of sizes with a multi-level index (left,right),
        named 'voxel_count'.
    """
    assert left_vol.shape == right_vol.shape
    df = pd.DataFrame( {"left": left_vol.reshape(-1),
                        "right": right_vol.reshape(-1)},
                       dtype=left_vol.dtype )
    sizes = df.groupby(['left', 'right']).size()
    sizes.name = 'voxel_count'
    return sizes


def split_disconnected_bodies(labels_orig):
    """
    Produces 3D volume split into connected components.

    This function identifies bodies that are the same label
    but are not connected.  It splits these bodies and
    produces a dict that maps these newly split bodies to
    the original body label.

    Special exception: Segments with label 0 are not relabeled.
    
    Note:
        Requires scikit-image (which, currently, is not otherwise
        listed as a dependency of neuclease's conda-recipe).

    Args:
        labels_orig (numpy.array): 3D array of labels

    Returns:
        (labels_new, new_to_orig)

        labels_new:
            The partially relabeled array.
            Segments that were not split will keep their original IDs.
            Among split segments, the largest 'child' of a split segment retains the original ID.
            The smaller segments are assigned new labels in the range (N+1)..(N+1+S) where N is
            highest original label and S is the number of new segments after splitting.
        
        new_to_orig:
            A pseudo-minimal (but not quite minimal) mapping of labels
            (N+1)..(N+1+S) -> some subset of (1..N),
            which maps new segment IDs to the segments they came from.
            Segments that were not split at all are not mentioned in this mapping,
            for split segments, every mapping pair for the split is returned, including the k->k (identity) pair.
        
        new_unique_labels:
            An array of all label IDs in the newly relabeled volume.
            The original label set can be selected via:
            
                new_unique_labels[new_unique_labels < min(new_to_orig.keys())]
        
    """
    import skimage.measure as skm
    # Compute connected components and cast back to original dtype
    labels_cc = skm.label(labels_orig, background=0, connectivity=1)
    assert labels_cc.dtype == np.int64
    if labels_orig.dtype == np.uint64:
        labels_cc = labels_cc.view(np.uint64)
    else:
        labels_cc = labels_cc.astype(labels_orig.dtype, copy=False)

    # Find overlapping segments between orig and CC volumes
    overlap_table_df = contingency_table(labels_orig, labels_cc).reset_index()
    assert overlap_table_df.columns.tolist() == ['left', 'right', 'voxel_count']
    overlap_table_df.columns = ['orig', 'cc', 'voxels']
    overlap_table_df.sort_values('voxels', ascending=False, inplace=True)
    
    # If a label in 'orig' is duplicated, it has multiple components in labels_cc.
    # The largest component gets to keep the original ID;
    # the other components must take on new values.
    # (The new values must not conflict with any of the IDs in the original, so start at orig_max+1)
    new_cc_pos = overlap_table_df['orig'].duplicated()
    orig_max = overlap_table_df['orig'].max()
    new_cc_values = np.arange(orig_max+1, orig_max+1+new_cc_pos.sum(), dtype=labels_orig.dtype)

    overlap_table_df['final_cc'] = overlap_table_df['orig'].copy()
    overlap_table_df.loc[new_cc_pos, 'final_cc'] = new_cc_values
    
    # Relabel the CC volume to use the 'final_cc' labels
    mapper = LabelMapper(overlap_table_df['cc'].values, overlap_table_df['final_cc'].values)
    mapper.apply_inplace(labels_cc)

    # Generate the mapping that could (if desired) convert the new
    # volume into the original one, as described in the docstring above.
    emitted_mapping_rows = overlap_table_df['orig'].duplicated(keep=False)
    emitted_mapping_pairs = overlap_table_df.loc[emitted_mapping_rows, ['final_cc', 'orig']].values

    new_to_orig = dict(emitted_mapping_pairs)

    new_unique_labels = pd.unique(overlap_table_df['final_cc'].values)
    new_unique_labels = new_unique_labels.astype(overlap_table_df['final_cc'].dtype)
    new_unique_labels.sort()
    
    return labels_cc, new_to_orig, new_unique_labels

