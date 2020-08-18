import logging
from collections import OrderedDict

import vigra
import numpy as np
import pandas as pd
from numba import njit

from dvidutils import LabelMapper

from . import Timer, Grid, boxes_from_grid, box_intersection, box_to_slicing, extract_subvol

logger = logging.getLogger(__name__)

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


def compute_nonzero_box(mask, save_ram=True):
    """
    Given a mask image, return the bounding box
    of the nonzero voxels in the mask.
    
    Equivalent to:
    
        coords = np.transpose(np.nonzero(mask))
        if len(coords) == 0:
            return np.zeros((2, mask.ndim))
        box = np.array([coords.min(axis=0), 1+coords.max(axis=0)])
        return box
    
    ...but faster.
    
    Args:
        mask:
            A binary image
    
        save_ram:
            Deprecated.  Now ignored.
    
    Returns:
        box, e.g. [(1,2,3), (10, 20,30)]
        If the mask is completely empty, zeros are returned,
        e.g. [(0,0,0), (0,0,0)]
    """
    box = _compute_nonzero_box_numpy(mask)

    # If the volume is completely empty,
    # the helper returns an invalid box.
    # In that case, return zeros
    if (box[1] <= box[0]).any():
        return np.zeros_like(box)

    return box


def _compute_nonzero_box_numpy(mask):
    """
    Helper for compute_nonzero_box().
 
    Note:
        If the mask has no nonzero voxels, an "invalid" box is returned,
        i.e. the start is above the stop.
    """
    # start with an invalid box
    box = np.zeros((2, mask.ndim), np.int32)
    box[0, :] = mask.shape
    
    # For each axis, reduce along the other axes
    axes = [*range(mask.ndim)]
    for axis in axes:
        other_axes = tuple({*axes} - {axis})
        pos = np.logical_or.reduce(mask, axis=other_axes).nonzero()[0]
        if len(pos):
            box[0, axis] = pos[0]
            box[1, axis] = pos[-1]+1

    return box
    

@njit
def _compute_nonzero_box_numba(mask):
    """
    Altenative helper for compute_nonzero_box().
    
    This turns out to be slower than the pure-numpy version above,
    despite the fact that this version makes only one pass over the
    data and the numpy version makes a pass for each axis.
    
    It would be interesting to see if the performance of this
    version improves with future version of numpy. 
 
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


def edge_mask(label_img, mode='before', mark_volume_edges=False):
    """
    Find all boundaries between labels in the given ND volume,
    and return a boolean mask that selects all voxels that lie
    just "before" the inter-voxel boundary,
    or just "after" the inter-voxel boudnary,
    or both.

    If mark_volume_edge=True, also mask the first/last voxel
    along all sides of the volume.

    Example:

        >>> labels = [[1,1,1,1,2,2],
        ...           [1,1,1,1,2,2],
        ...           [1,1,2,2,2,2],
        ...           [1,1,2,2,2,2],
        ...           [1,1,2,2,2,2],
        ...           [1,2,2,2,2,2]]

        >>> mask = edge_mask(labels, 'before')
        >>> print(mask.astype(int))
        [[0 0 0 1 0 0]
         [0 0 1 1 0 0]
         [0 1 0 0 0 0]
         [0 1 0 0 0 0]
         [0 1 0 0 0 0]
         [1 0 0 0 0 0]]

        >>> mask = edge_mask(labels, 'after')
        >>> print(mask.astype(int))
        [[0 0 0 0 1 0]
         [0 0 0 0 1 0]
         [0 0 1 1 0 0]
         [0 0 1 0 0 0]
         [0 0 1 0 0 0]
         [0 1 0 0 0 0]]

        >>> mask = edge_mask(labels, 'both')
        >>> print(mask.astype(int))
        [[0 0 0 1 1 0]
         [0 0 1 1 1 0]
         [0 1 1 1 0 0]
         [0 1 1 0 0 0]
         [0 1 1 0 0 0]
         [1 1 0 0 0 0]]
    """
    label_img = np.asarray(label_img)
    mask = np.zeros(label_img.shape, bool)

    for axis in range(label_img.ndim):
        left_slicing = ((slice(None),) * axis) + (np.s_[:-1],)
        right_slicing = ((slice(None),) * axis) + (np.s_[1:],)

        m = edge_mask_for_axis(label_img, axis)
        if mode in ('before', 'both'):
            mask[left_slicing] |= m

        if mode in ('after', 'both'):
            mask[right_slicing] |= m

    if mark_volume_edges:
        for axis in range(mask.ndim):
            left_slicing = ((slice(None),) * axis) + (0,)
            right_slicing = ((slice(None),) * axis) + (-1,)

            mask[left_slicing] = 1
            mask[right_slicing] = 1

    return mask


def edge_mask_for_axis( label_img, axis ):
    """
    Find all supervoxel edges along the given axis and return
    a 'left-hand' mask indicating where the edges are located
    (i.e. a boolean array indicating voxels that are just to the left of an edge).
    Note that this mask is less wide (by 1 pixel) than ``label_img`` along the chosen axis.
    """
    if axis < 0:
        axis += label_img.ndim
    assert label_img.ndim > axis
    
    if label_img.shape[axis] == 1:
        return np.zeros_like(label_img)

    left_slicing = ((slice(None),) * axis) + (np.s_[:-1],)
    right_slicing = ((slice(None),) * axis) + (np.s_[1:],)

    edge_mask = (label_img[left_slicing] != label_img[right_slicing])
    return edge_mask


def compute_adjacencies(label_vol, max_dilation=0, include_zero=False, return_dilated=False, disable_quantization=False):
    """
    Compute the size of the borders between label segments in a label volume.
    Returns a pd.Series of edge sizes, indexed by adjacent label pairs.
    
    If you're interested in labels that are close, but perhaps not exactly adjacent,
    the max_dilation parameter can be used to pre-process the data by simultaneously
    dilating all labels within the empty regions between labels, using a distance transform.
    
    Args:
        label_vol:
            ND label volume
        
        max_dilation:
            Before computing adjacencies, grow the labels by the given
            radius within the empty (zero) regions of the volume as
            described above.
            Note:
                The dilation is performed from all labels simultaneously,
                so if max_dilation=1, gaps of up to 2 voxels wide will be closed.
                (With this method, there is no way to close gaps of 1 voxel
                wide without also closing gaps that are 2 voxels wide.)
            
            Note:
                This feature is optimized if you choose a value that is divisible by 0.25,
                and <= 63.  In that case, the distance transform is quantized and converted to uint8.
                Some precision is lost in the quantization, but the watershed step is much faster.
                This optimization can be disabled via disable_quantization=True.
        
        include_zero:
            If True, include adjacencies to label 0 in the output.
        
        return_dilated:
            If True, also return the pre-processed label volume.
            If max_dilation=0, then the input volume is returned unchanged.

            Note:
                To avoid unnecessary computation and save RAM, the label_vol
                is cropped (if possible) to included only the non-zero voxels.
                Therefore, the 'dilated' output may not have the same shape
                as the original input.
        
        disable_quantization:
            If True, do not enable the optimization described above.

    Returns:
        pd.Series of edgea area values, indexed by label pair
        If return_dilated=True, then a tuple is returned:
        (pd.Series, np.ndarray)

    Example:
    
            >>> labels = [[1,1,1,1,2,2],
            ...           [1,1,1,0,2,2],
            ...           [1,0,0,0,2,2],
            ...           [1,0,0,0,2,2],
            ...           [1,0,0,0,2,2],
            ...           [1,0,2,2,2,2]]
    

        >>> compute_adjacencies(labels)
        label_a  label_b
        1        2          1
        Name: edge_area, dtype: int64

        >>> adj, dil = compute_adjacencies(labels, 1, return_dilated=True)
        
        >>> print(adj)
        label_a  label_b
        1        2          6
        Name: edge_area, dtype: int64
        
        >>> print(dil)
        [[1 1 1 1 2 2]
         [1 1 1 1 2 2]
         [1 1 1 2 2 2]
         [1 1 0 2 2 2]
         [1 1 2 2 2 2]
         [1 1 2 2 2 2]]
    """
    label_vol = np.asarray(label_vol)
    if max_dilation > 0:
        with Timer("Computing distance transform", logger):
            nonzero = (label_vol != 0).astype(np.uint32)
            nz_box = compute_nonzero_box(nonzero)
            nonzero = extract_subvol(nonzero, nz_box)
            label_vol = extract_subvol(label_vol, nz_box)
            dt = vigra.filters.distanceTransform(nonzero)

            # If we can safely convert the distance transform to uint8,
            # that will allow a much faster watershed ("turbo mode").
            # We convert to uint8 if max_dilation is divisible by 0.25.
            if not disable_quantization and max_dilation <= 63 and int(4*max_dilation) == 4*max_dilation:
                logger.info("Quantizing distance transform as uint8 to enable 'turbo' watershed")
                max_dilation *= 4
                dt[dt > 63] = 63
                dt *= 4
                dt = dt.astype(np.uint8)
        
        with Timer("Computing watershed to fill gaps", logger):
            label_vol = label_vol.astype(np.uint32, copy=True)
            vigra.analysis.watersheds(dt, seeds=label_vol, out=label_vol)
            label_vol[dt > max_dilation] = 0

    with Timer("Computing adjacencies", logger):
        adj = _compute_adjacencies(label_vol, include_zero)

    if return_dilated:
        return adj, label_vol
    else:
        return adj


def _compute_adjacencies(label_vol, include_zero=False):
    all_label_pairs = []
    for axis in range(label_vol.ndim):
        left_slicing = ((slice(None),) * axis) + (np.s_[:-1],)
        right_slicing = ((slice(None),) * axis) + (np.s_[1:],)

        edge_mask = (label_vol[left_slicing] != label_vol[right_slicing])
        left_labels = label_vol[left_slicing][edge_mask]
        right_labels = label_vol[right_slicing][edge_mask]
        
        label_pairs = np.array([left_labels, right_labels]).transpose()
        label_pairs.sort(axis=1)
        
        if not include_zero:
            keep_rows = (label_pairs[:,0] != 0) & (label_pairs[:,1] != 0)
            label_pairs = label_pairs[keep_rows]
        
        all_label_pairs.append(label_pairs)
    
    all_label_pairs = np.concatenate(all_label_pairs, axis=0)
    df = pd.DataFrame(all_label_pairs, columns=['label_a', 'label_b'])
    areas = df.groupby(['label_a', 'label_b']).size().rename('edge_area')
    return areas
    


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
    sizes = df.groupby(['left', 'right'], sort=False).size()
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

