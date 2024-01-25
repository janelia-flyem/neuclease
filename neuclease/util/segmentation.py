import logging
from collections import OrderedDict

import scipy
import vigra
import numpy as np
import pandas as pd
from numba import njit
from skimage.util import view_as_blocks

from dvidutils import LabelMapper

from .util import Timer, tqdm_proxy, downsample_mask, upsample
from .grid import Grid, boxes_from_grid, box_intersection
from .box import extract_subvol, box_to_slicing

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
        try:
            counts = pd.Series(block_vol.reshape(-1)).value_counts(sort=False)
        except ValueError:
            # Bizarrely, I've encountered this error emerging from
            # pandas._libs.hashtable.value_count_uint64():
            # "ValueError: buffer source array is read-only"
            # ...so if this fails, just try again, with a clean copy...
            counts = pd.Series(block_vol.ravel().copy()).value_counts(sort=False)
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

    # Fast path for the single label case
    if len(label_ids) == 1:
        label = next(iter(label_ids))
        return (volume == label)

    valid_positions = pd.Series(flatvol).isin(label_ids).values
    return valid_positions.reshape(volume.shape)


def apply_mask_for_labels(volume, label_ids, inplace=False):
    """
    Given a label volume and a subset of labels to keep,
    mask out all voxels that do not fall on the given label_ids
    (i.e. set them to 0).
    """
    label_ids = np.fromiter(label_ids, dtype=volume.dtype)

    # Fast path for the single label case
    if len(label_ids) == 1:
        if inplace:
            ret = volume
        else:
            ret = np.empty_like(volume)
        label = label_ids[0]
        ret[:] = np.where(volume != label, 0, label)
        return ret

    if inplace:
        assert volume.flags.c_contiguous
        flatvol = volume.reshape(-1)
    else:
        flatvol = volume.copy('C').reshape(-1)

    keep = pd.Series(flatvol).isin(label_ids).values
    flatvol[~keep] = 0
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
            # Edit: I now know that calling np.minimum and np.maximum is SLOW in numba.
            #       Would be good to rewrite this without calling those functions.
            box[0] = np.minimum(c, box[0])
            box[1] = np.maximum(c, box[1])
    box[1] += 1
    return box


def has_nonzero_exterior(mask):
    """
    Return True if the mask has any non-zero voxels on the 'exterior'
    of the volume, i.e. in the first or last slice along any axis.
    Otherwise, return False
    """
    for axis in range(mask.ndim):
        v = np.moveaxis(mask, axis, 0)
        if v[0].any() or v[-1].any():
            return True
    return False


def volume_face_ids(vol, thickness=1):
    """
    Determine the set of segment IDs (including ID 0)
    that lie on any of the faces of a volume,
    e.g. the 6 faces of a 3D volume, or the 4 edges
    of a 2D rectangle, etc.
    """
    thickness = np.broadcast_to(thickness, (2, vol.ndim))
    ids = []
    for axis in range(vol.ndim):
        v = np.rollaxis(vol, axis, 0)
        t0, t1 = thickness[:, axis]
        u0 = pd.unique(v[:t0].reshape(-1))
        u1 = pd.unique(v[-t1:].reshape(-1))
        ids += [u0, u1]
    ids = np.concatenate(ids)
    return np.sort(pd.unique(ids))


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


def binary_edge_mask(mask, mode='inner', mark_volume_edges=False):
    """
    For a binary image (mask), select the pixels that lie
    immediately inside or outside of the masked region, or both.

    """
    if mask.dtype != bool:
        mask = (mask != 0)

    if mode in ('before', 'after', 'both'):
        return edge_mask(mask, mode, mark_volume_edges)

    em = edge_mask(mask, 'both')
    if mode == 'inner':
        em &= mask
    if mode == 'outer':
        em &= ~mask

    if mark_volume_edges:
        for axis in range(mask.ndim):
            left_slicing = ((slice(None),) * axis) + (0,)
            right_slicing = ((slice(None),) * axis) + (-1,)

            em[left_slicing] = 1
            em[right_slicing] = 1

    return em

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
                        "right": right_vol.reshape(-1)})
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


def fill_hull(image):
    """
    Compute the convex hull of the given binary image and
    return a mask of the filled hull.

    Adapted from:
    https://stackoverflow.com/a/46314485/162094

    This version is slightly (~40%) faster for 3D volumes,
    by being a little more stingy with RAM.
    """
    # (The variable names below assume 3D input,
    # but this would still work in 4D, etc.)

    assert (np.array(image.shape) <= np.iinfo(np.int16).max).all(), \
        f"This function assumes your image is smaller than {2**15} in each dimension"

    mask = np.zeros_like(image, dtype=bool)
    points = np.argwhere(image).astype(np.int16)

    # Restrict analysis to the bounding-box
    box = (points.min(axis=0), 1+points.max(axis=0))
    mask_view = mask[box_to_slicing(*box)]
    points -= box[0]

    _fill_hull(points, mask_view)
    return mask


def _fill_hull(points, mask):
    """
    Compute the convex hull of the given points
    set and write it into the given mask.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except scipy.spatial.qhull.QhullError:
        # If there aren't enough points, or the points are coplanar,
        # we might see an error like this:
        #   QhullError: QH6214 qhull input error: not enough points(1) to construct initial simplex (need 4)
        # In that case, just write the points themselves into the result.
        mask[tuple(points.transpose())] = True
        return

    deln = scipy.spatial.Delaunay(points[hull.vertices])

    # Instead of allocating a giant array for all indices in the volume,
    # just iterate over the slices one at a time.
    idx_2d = np.indices(mask.shape[1:], np.int16)
    idx_2d = np.moveaxis(idx_2d, 0, -1)

    idx_3d = np.zeros((*mask.shape[1:], mask.ndim), np.int16)
    idx_3d[:, :, 1:] = idx_2d

    for z in range(len(mask)):
        idx_3d[:,:,0] = z

        # find_simplex() returns -1 for points that don't fall within any simplex.
        s = deln.find_simplex(idx_3d)
        mask[z, (s != -1)] = 1


def fill_hull_for_segment(label_img, segment_id=1):
    """
    Compute the hull for the given segment and generate a mask.
    Faster than bare fill_hull() because it filters out non-boundary
    points from the mask first.
    """
    all_edges = edge_mask(label_img, 'both', True)
    seg_edges = np.where(label_img == segment_id, all_edges, False)
    return fill_hull(seg_edges)


def approximate_hull_for_segment(label_img, segment_id, downsample_factor=2):
    mask = downsample_mask(label_img == segment_id, downsample_factor)
    hull = fill_hull_for_segment(mask)
    return upsample(hull, downsample_factor)


def approximate_hulls_for_segments(label_img, downsample_factor=1, as_masks=False, overlap_rule='erase', progress=False):
    """
    Compute the hulls for all of the non-zero segments in the label image.
    Return them either as a set of masks, or as a single label image.

    Two output formats are supported.  See description of ``as_masks``.

    Args:
        label_img:
            A volume with integer labels.  All non-zero segments will be processed.

        downsample_factor:
            For speed, if an approximate hull will suffice, the image can be downsampled
            before the hull is computed. The result is upsampled to the original resolution.

        as_masks:
            Specifies the return format. See return info.

        overlap_rule:
            How to resolve conflicts between overlapping hulls when as_masks=False,
            as described above.  Currently 'erase' is the only valid option.

    Returns:
        If as_masks=False, all hulls are written to a single label image.
        Since hulls can overlap with each other, the overlap_rule specifies how to resolve conflicts.
        Currently, the only available overlap_rule is 'erase', meaning any voxels within
        intersecting hulls are given a label of 0.  In the future, other overlap_rules will be implemented,
        such as 'keep-largest' or 'keep-max', etc.
        The hull label image and the bounding box of each hull within it are returned.
        (The bounding box refers to the hull's original size, before applying the overlap_rule.)
        The bounding boxes are returned as a pd.Series, where each value is a box.
        Returns (hull_label_img, segment_boxes)

        If as_masks=True, then a mask for each hull is generated, and all masks are returned.
        The returned mask volumes are only large enough to contain each hull's bounding box.
        The bounding box that each mask corresponds to is also returned.
        The output is a dictionary: {label: (box, mask)}
    """
    assert overlap_rule in ['erase']

    # Drop interior points for efficiency's sake
    all_edges = edge_mask(label_img, 'both', True)
    label_img = np.where(all_edges, label_img, 0)

    assert downsample_factor > 0, \
        "downsample_factor should be a factor >= 1, not a scale >= 0."
    if downsample_factor > 1:
        d = downsample_factor
        downsampled = view_as_blocks(label_img, (d,d,d)).max(axis=(3,4,5))
    else:
        downsampled = label_img

    boxes = region_features(downsampled, None, ['Box'])['Box']
    boxes = boxes.drop(0, errors='ignore')
    segment_ids = boxes.index.values

    def _hull_mask(s):
        """
        Compute the hull mask and its box
        within the downsampled volume.
        """
        box = boxes.loc[s]
        vol = extract_subvol(downsampled, box)
        mask = (vol == s)
        points = np.argwhere(mask)
        _fill_hull(points, mask)
        return box, mask

    if as_masks:
        hull_masks = {}
        for s in tqdm_proxy(segment_ids, disable=not progress):
            box, mask = _hull_mask(s)
            mask = upsample(mask, downsample_factor)
            box *= downsample_factor
            hull_masks[s] = (box, mask)
        return hull_masks
    else:
        hull_label_img = label_img.copy()
        OVERLAP_MARKER = max(segment_ids) + 1
        for s in tqdm_proxy(segment_ids, disable=not progress):
            box, hull_mask = _hull_mask(s)
            hull_mask = upsample(hull_mask, downsample_factor)
            box *= downsample_factor
            out_view = extract_subvol(hull_label_img, box)

            if overlap_rule == 'erase':
                # Mark voxels which overlap more than one hull,
                # so they can be deleted below.
                overlap_mask = hull_mask & (out_view != 0) & (out_view != s)
                out_view[:] = np.where(hull_mask, s, out_view)
                out_view[:] = np.where(overlap_mask, OVERLAP_MARKER, out_view)
            else:
                raise AssertionError("Invalid overlap rule")

        if overlap_rule == 'erase':
            hull_label_img[hull_label_img == OVERLAP_MARKER] = 0

        return hull_label_img, boxes


def fill_triangle(image, verts):
    """
    Paint a triangle into the given image
    by successively calling line_nd() from every
    corner to all points on the opposite edge.
    (If you only do it from one corner, you can
    end up with holes.)

    Note: This requires a version of line_nd that can
    handle a list of point pairs and process all of them.
    """
    def _fill(image, verts):
        a,b,c = verts
        ab = np.transpose(line_nd(a,b,endpoint=True))
        z, y, x = line_nd(ab,c,endpoint=True)
        z = z.reshape(-1)
        y = y.reshape(-1)
        x = x.reshape(-1)
        image[(z, y, x)] = True

    a,b,c = verts
    _fill(image, (a,b,c))
    _fill(image, (b,c,a))
    _fill(image, (c,a,b))


def line_nd(start, stop, *, endpoint=False, integer=True):
    """
    This is a copy of ``skimage.morphology.line_nd()``,
    but slightly modified so that it supports lists of start/stop
    coordinates instead of only one line at a time.

    Draw a single-pixel thick line in n dimensions.
    The line produced will be ndim-connected. That is, two subsequent
    pixels in the line will be either direct or diagonal neighbours in
    n dimensions.
    Parameters
    ----------
    start : array-like, shape (D,) or (N,D)
        The start coordinates of the line.
    stop : array-like, shape (D,) or (N,D)
        The end coordinates of the line.
    endpoint : bool, optional
        Whether to include the endpoint in the returned line. Defaults
        to False, which allows for easy drawing of multi-point paths.
    integer : bool, optional
        Whether to round the coordinates to integer. If True (default),
        the returned coordinates can be used to directly index into an
        array. `False` could be used for e.g. vector drawing.
    Returns
    -------
    coords : tuple of arrays
        The coordinates of points on the line.
    Examples
    --------
    >>> lin = line_nd((1, 1), (5, 2.5), endpoint=False)
    >>> lin
    (array([1, 2, 3, 4]), array([1, 1, 2, 2]))
    >>> im = np.zeros((6, 5), dtype=int)
    >>> im[lin] = 1
    >>> im
    array([[0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> line_nd([2, 1, 1], [5, 5, 2.5], endpoint=True)
    (array([2, 3, 4, 4, 5]), array([1, 2, 3, 4, 5]), array([1, 1, 2, 2, 2]))

    >>> line_nd([0,0,0], [10,10,10])
    >>> line_nd([[10,10,10], [-10, -10, -10]], [[0,0,0], [1,1,1]])
    >>> line_nd([0,0,0], [[10,10,10], [-10, -10, -10]])
    >>> line_nd([[10,10,10], [-10, -10, -10]], [0,0,0])
    """
    def _round_safe(coords):
        # Round coords while ensuring successive values are less than 1 apart.
        # When rounding coordinates for `line_nd`, we want coordinates that are less
        # than 1 apart (always the case, by design) to remain less than one apart.
        # However, NumPy rounds values to the nearest *even* integer, so:
        # >>> np.round([0.5, 1.5, 2.5, 3.5, 4.5])
        # array([0., 2., 2., 4., 4.])
        # As a workaround, just nudge all .5-values down so they always round down.
        coords[(coords % 1 == 0.5) & (coords[...,1:2] - coords[...,0:1] == 1)] -= 0.01
        return np.round(coords).astype(int)

    start = np.asarray(start)
    stop = np.asarray(stop)
    npoints = int(np.ceil(np.max(np.abs(stop - start))))
    if endpoint:
        npoints += 1
    coords = []
    for dim in range(start.shape[-1]):
        dimcoords = np.linspace(start[..., dim], stop[..., dim], npoints, endpoint)
        if integer:
            dimcoords = _round_safe(dimcoords).astype(int)
        coords.append(dimcoords)
    return tuple(coords)


def normalize_image_range(img, dtype):
    """
    Change the range of the given image to be in
    the range 0.0-1.0 for float dtypes, or 0-(MAX-1)
    for integer dtypes.
    """
    img[:] -= img.min()
    img[:] /= img.max()
    assert img.min() == 0.0
    assert img.max() == 1.0

    if np.issubdtype(dtype, np.integer):
        # Scale up to max-1, leaving 1 value as an 'infinity',
        # in case caller wants to place "no-go" areas in the image.
        img[:] *= np.iinfo(dtype).max-1

    img = img.astype(dtype)
    return img


def distance_transform(mask, background=False, smoothing=0.0, negate=False, pad=True):
    """
    Compute the distance transform of voxels inside (or outside) the given mask,
    with smoothing and negation post-processing options as a convenience.

    Args:
        mask:
            A boolean mask
        background:
            If True, calculate distances within the background (zero voxels) to the nearest foreground voxel.
            If False, calculate distances within the foreground (non-zero voxels) to the nearest background voxel.
            Note: Our default setting is the OPPOSITE of vigra's default!
        smoothing:
            How must gaussian blur to apply before computing the max distance.
        negate:
            Negate the distance transform result before returning.
            Useful if you plan to run a watershed segmentation on the results.
        pad:
            If True, pad the mask with a halo of zeros before computing the distance tranform,
            in case the foreground occupies the edge of the volume and results in strange results.
            But strip off the padded voxels before returning the result.
            The result will always have the same shape as the input.
    Returns:
        Same shape as mask, but float32.
    """
    mask = mask.astype(bool, copy=False)
    if pad and has_nonzero_exterior(mask):
        mask = np.pad(mask, 1)
    else:
        pad = False

    mask = vigra.taggedView(mask, 'zyx').astype(np.uint32)
    dt = vigra.filters.distanceTransform(mask, background=background)
    del mask

    if smoothing > 0.0:
        vigra.filters.gaussianSmoothing(dt, smoothing, dt, window_size=2.0)

    if negate:
        dt[:] *= -1

    if pad:
        sl = (slice(1, -1),) * dt.ndim
        return dt[sl]

    return dt


def thickest_point_in_mask(mask):
    dt = distance_transform(mask, pad=True)
    maxpoint = np.unravel_index(np.argmax(dt), dt.shape)
    return maxpoint, dt[maxpoint]


def distance_transform_watershed(mask, smoothing=0.0, seed_mask=None, seed_labels=None, flood_from='interior'):
    """
    Compute a watershed over the distance transform within a mask.
    You can either compute the watershed from inside-to-outside or outside-to-inside.
    For the former, the watershed is seeded from the most interior points,
    and the distance transform is inverted so the watershed can proceed from low to high as usual.
    For the latter, the distance transform is seeded from the voxels immediately outside the mask,
    using labels as found in the seed_labels volume. In this mode, the results effectively tell
    you which exterior segment (in the seed volume) is closest to any given point within the
    interior of the mask.

    Or you can provide your own seeds if you think you know what you're doing.

    Args:
        mask:
            Only the masked area will be processed
        smoothing:
            If non-zero, run gaussian smoothing on the distance transform with the
            given sigma before defining seed points or running the watershed.

        seed_mask:
        seed_labels:
        flood_from:

    Returns:
        dt, labeled_seeds, ws

    Notes:
        This function provides a subset of the options that can be found in other
        libraries, such as:
            - https://github.com/ilastik/wsdt/blob/3709b27/wsdt/wsDtSegmentation.py#L26
            - https://github.com/constantinpape/elf/blob/34f5e76/elf/segmentation/watershed.py#L69

        This uses vigra's efficient true euclidean distance transform, which is superior to
        distance transform approximations, e.g. as found in https://imagej.net/Distance_Transform_Watershed

        The imglib2 distance transform uses the same algorithm as vigra:
        https://github.com/imglib/imglib2-algorithm/tree/master/src/main/java/net/imglib2/algorithm/morphology/distance
        http://www.theoryofcomputing.org/articles/v008a019/
    """
    mask = mask.astype(bool, copy=False)
    mask = vigra.taggedView(mask, 'zyx').astype(np.uint32)

    # Widen seeds dtype if necessary
    # (The 64-bit case is handled below, with a mapping.)
    if seed_labels is not None and seed_labels.dtype in (np.uint8, np.uint16, np.int8, np.int16, np.int32):
        seed_labels = seed_labels.astype(np.uint32)

    imask = np.logical_not(mask)
    outer_edge_mask = binary_edge_mask(mask, 'outer')

    assert flood_from in ('interior', 'exterior')
    if flood_from == 'interior':
        # Negate the distance transform result,
        # since watershed must start at minima, not maxima.
        # Convert to uint8 to benefit from 'turbo' watershed mode
        # (uses a bucket queue).
        dt = distance_transform(mask, False, smoothing, negate=True)

        if seed_mask is None:
            # requires float32 input for some reason
            if dt.ndim == 2:
                minima = vigra.analysis.localMinima(dt, marker=np.nan, neighborhood=8, allowAtBorder=True, allowPlateaus=False)
            else:
                minima = vigra.analysis.localMinima3D(dt, marker=np.nan, neighborhood=26, allowAtBorder=True, allowPlateaus=False)
            seed_mask = np.isnan(minima)
            del minima

        dt = normalize_image_range(dt, np.uint8)
    else:
        if seed_labels is None and seed_mask is None:
            logger.warning("Without providing your own seed mask and/or seed labels, "
                           "the watershed operation will simply be the same as a "
                           "connected components operation.  Is that what you meant?")

        if seed_mask is None:
            seed_mask = outer_edge_mask.copy()

        # Dilate the mask once more.
        outer_edge_mask[:] |=  binary_edge_mask(outer_edge_mask | mask, 'outer')

        dt = distance_transform(mask, False, smoothing, negate=False)
        dt = normalize_image_range(dt, np.uint8)

    if seed_labels is None:
        seed_mask = vigra.taggedView(seed_mask, 'zyx')
        labeled_seeds = vigra.analysis.labelMultiArrayWithBackground(seed_mask.view('uint8'))
    else:
        labeled_seeds = np.where(seed_mask, seed_labels, 0)

    # Make sure seed_mask matches labeled_seeds,
    # Even if some seed_labels were zero-valued
    seed_mask = (labeled_seeds != 0)

    # Must remap to uint32 before calling vigra's watershed.
    seed_mapper = None
    seed_values = None
    if labeled_seeds.dtype in (np.uint64, np.int64):
        labeled_seeds = labeled_seeds.astype(np.uint64)
        seed_values = np.sort(pd.unique(labeled_seeds.reshape(-1)))
        if seed_values[0] != 0:
            seed_values = np.array([0] + list(seed_values), np.uint64)

        assert seed_values.dtype == np.uint64
        assert labeled_seeds.dtype == np.uint64

        ws_seed_values = np.arange(len(seed_values), dtype=np.uint32)
        seed_mapper = LabelMapper(seed_values, ws_seed_values)
        ws_seeds = seed_mapper.apply(labeled_seeds)
        assert ws_seeds.dtype == np.uint32
    else:
        ws_seeds = labeled_seeds

    # Fill the non-masked area with one big seed,
    # except for a thin border around the mask.
    # This saves time in the watershed step,
    # since these voxels now don't need to be
    # consumed in the watershed.
    dummy_seed = ws_seeds.max()+np.uint32(1)
    ws_seeds[np.logical_not(mask | outer_edge_mask)] = dummy_seed
    ws_seeds[outer_edge_mask & ~seed_mask] = 0

    dt[outer_edge_mask] = 255
    dt[seed_mask] = 0

    dt = vigra.taggedView(dt, 'zyx')
    ws_seeds = vigra.taggedView(ws_seeds, 'zyx')
    ws, max_id = vigra.analysis.watershedsNew(dt, seeds=ws_seeds, method='Turbo')

    # Areas that were unreachable without crossing over the border
    # could end up with the dummy seed.
    # We treat such areas as if they are outside of the mask.
    ws[ws == dummy_seed] = 0
    ws_seeds[imask] = 0

    # If we converted from uint64 to uint32 to perform the watershed,
    # convert back before returning.
    if seed_mapper is not None:
        ws = seed_values[ws]
    return dt, labeled_seeds, ws


SEGMENTATION_FEATURE_NAMES = [
    'Box', # This is a special add-on to the vigra names, equivalent to [Coord<Min>, 1+Coord<Max>]
    'Box0', # This is a special add-on to the vigra names, equivalent to Coord<Min>
    'Box1', # This is a special add-on to the vigra names, equivalent to 1+Coord<Max>
    'Count', # size
    'Coord<Maximum>',
    'Coord<Minimum>',
    'Coord<DivideByCount<Principal<PowerSum<2>>>>',
    'Coord<PowerSum<1>>',
    'Coord<Principal<Kurtosis>>',
    'Coord<Principal<PowerSum<2>>>',
    'Coord<Principal<PowerSum<3>>>',
    'Coord<Principal<PowerSum<4>>>',
    'Coord<Principal<Skewness>>',
    'RegionAxes',
    'RegionCenter',
    'RegionRadii',
]

# These features require both a segmentation image and a paired grayscale image
GRAYSCALE_FEATURE_NAMES = [
    'Central<PowerSum<2>>',
    'Central<PowerSum<3>>',
    'Central<PowerSum<4>>',
    'Coord<ArgMaxWeight>',
    'Coord<ArgMinWeight>',
    'Global<Maximum>',
    'Global<Minimum>',
    'Histogram',
    'Kurtosis',
    'Maximum',
    'Mean',
    'Minimum',
    'Quantiles',
    'Skewness',
    'Sum',
    'Variance',
    'Weighted<Coord<DivideByCount<Principal<PowerSum<2>>>>>',
    'Weighted<Coord<PowerSum<1>>>',
    'Weighted<Coord<Principal<Kurtosis>>>',
    'Weighted<Coord<Principal<PowerSum<2>>>>',
    'Weighted<Coord<Principal<PowerSum<3>>>>',
    'Weighted<Coord<Principal<PowerSum<4>>>>',
    'Weighted<Coord<Principal<Skewness>>>',
    'Weighted<PowerSum<0>>',
    'Weighted<RegionAxes>',
    'Weighted<RegionCenter>',
    'Weighted<RegionRadii>'
]


@njit
def region_boxes(vol):
    """
    Determine the bounding boxes of all label regions (segments) in a label volume.

    Note:
        Since the result is indexed by segment ID, this function is only
        suitable for volumes in which the maximum label ID is relatively low.
        For instance, if the volume contains labels [1,2,3, int(1e9)],
        then the result will have length 1e9.

    Args:
        vol:
            ndarray, integer dtype and arbitrary dimensionality D
    Returns:
        ndarray, shape (N, 2, D)
        where N is the number of unique label values in the array (including label 0).
        The min box coordinate for label i is given in entry [i, 0, :] and the max in entry [i, 1, :].

        Note:
            If the input array contains non-consecutive label IDs,
            (i.e., some labels are not present), then the results in those
            entries will be intentionally nonsensical: the 'min' will
            be GREATER than the 'max'.

    See Also:
        ``neuclease.util.segmentation.region_features()``,
        which computes more than just bounding boxes and handles arbitrarily large label IDs.
    """
    N = vol.max()

    # Initialize box min (and max) coords with extreme max (and min)
    # values so that any encountered coordinate overrides the initial value.
    boxes = np.empty((N+1, 2, vol.ndim), np.int32)
    boxes[:, 0, :] = np.array(vol.shape)
    boxes[:, 1, :] = 0

    for idx in np.ndindex(*vol.shape):
        label = vol[idx]
        for axis, i in enumerate(idx):
            boxes[label, 0, axis] = min(i, boxes[label, 0, axis])
            boxes[label, 1, axis] = max(i, boxes[label, 1, axis])
    boxes[:, 1, :] += 1
    return boxes


def region_boxes_numpy(vol):
    """
    Same as region_boxes(), but 100x slower.
    However, this version requries no JIT.
    """
    Z, Y, X = vol.shape
    grid = np.ogrid[:Z, :Y, :X]

    boxes = np.zeros((vol.max()+1, 2, 3), dtype=int)
    boxes[:, 0, :] = vol.shape

    for axis in (0,1,2):
        np.minimum.at(boxes[:, 0, axis], vol, grid[axis])
        np.maximum.at(boxes[:, 1, axis], vol, grid[axis])

    boxes[:, 1, :] += 1
    return boxes


def region_boxes_numpy_mgrid(vol):
    """
    Alternative implementation of region_boxes_numpy().
    Slightly slower, so, still much slower than region_boxes().
    This version is here only for study.
    It passes over the volume 2x instead of 6x, but it requires
    a "fleshed out" and transposed mgrid instead of the tiny ogrid.
    Apparently the tradeoff isn't worth it.
    """
    Z, Y, X = vol.shape
    grid = np.empty((*vol.shape, 3), int)
    grid[..., 0], grid[..., 1], grid[..., 2] = np.ogrid[:Z, :Y, :X]

    boxes = np.zeros((vol.max()+1, 2, 3), dtype=int)
    boxes[:, 0, :] = vol.shape

    np.minimum.at(boxes[:, 0], vol, grid)
    np.maximum.at(boxes[:, 1], vol, grid)

    boxes[:, 1, :] += 1
    return boxes


@njit
def region_boxes_dict(vol):
    """
    FIXME:
        This has much worse performance than region_boxes(),
        for unknown reasons.

    Determine the bounding boxes of all label regions (segments) in a label volume.
    Unlike region_boxes() above, this function works for segmentations with
    non-consective, arbitrarily high-valued segment IDs.

    Args:
        vol:
            ndarray, integer dtype and arbitrary dimensionality D
    Returns:
        dict of {segment_id: box}

    See Also:
        - ``neuclease.util.segmentation.region_boxes()``
        - ``neuclease.util.segmentation.region_features()``
    """
    boxes = dict()

    for idx in np.ndindex(*vol.shape):
        label = vol[idx]
        if label not in boxes:
            box = np.empty((2, vol.ndim), np.int32)
            boxes[label] = box
            # Initialize box min (and max) coords with extreme max (and min)
            # values so that any encountered coordinate overrides the initial value.
            box[0, :] = np.array(vol.shape)
            box[1, :] = 0

        box = boxes[label]
        for axis, i in enumerate(idx):
            box[0, axis] = min(i, box[0, axis])
            box[1, axis] = max(i, box[1, axis])

    for box in boxes.values():
        box[1, :] += 1
    return boxes


def region_features(label_img, grayscale_img=None, features=['Box', 'Count'], ignore_label=0):
    """
    Wrapper around vigra.analysis.extractRegionFeatures() that supports uint64 and
    returns each feature as a pandas Series or DataFrame, indexed by object ID.

    For simple features such as 'Box' and 'Count', most of the time is spent remapping the
    input array from uint64 to uint32, which is the only label image type supported by vigra.

    See vigra docs regarding the supported features:
        - http://ukoethe.github.io/vigra/doc-release/vigranumpy/index.html#vigra.analysis.extractRegionFeatures
        - http://ukoethe.github.io/vigra/doc-release/vigra/group__FeatureAccumulators.html


    See Also:
        region_boxes(), which only computes bounding boxes on consecutively labeled regions,
        but is much faster for that one use case than this function.

    Args:
        label_img:
            An integer-valued label image, containing no negative values

        grayscle_img:
            Optional.  If provided, then weighted features are available.
            See GRAYSCALE_FEATURE_NAMES, above.

        features:
            List of strings.  If no grayscale image was provided, you can only
            ask for the features in ``SEGMENTATION_FEATURE_NAMES``, above.

        ignore_label:
            A background label to ignore. If you don't want to ignore any thing, pass ``None``.

    Returns:
        dict {name: feature}, where each feature value is indexed by label ID.
        For keys where the feature is scalar-valued for each label, the returned value is a Series.
        For keys where the feature is a 1D array for each label, a DataFrame is returned, with columns 'zyx'.
        For keys where the feature is a 2D array (e.g. Box, RegionAxes, etc.), a Series is returned whose
        dtype=object, and each item in the series is a 2D array.
        TODO: This might be a good place to use Xarray
    """
    assert label_img.ndim in (2,3)
    axes = 'zyx'[-label_img.ndim:]

    if isinstance(features, str):
        features = [features]

    vfeatures = {*features}

    valid_names = {*SEGMENTATION_FEATURE_NAMES, *GRAYSCALE_FEATURE_NAMES}
    invalid_names = vfeatures - valid_names
    assert not invalid_names, \
        f"Invalid feature names: {invalid_names}"

    if 'Box' in features:
        vfeatures -= {'Box'}
        vfeatures |= {'Coord<Minimum>', 'Coord<Maximum>'}

    if 'Box0' in features:
        vfeatures -= {'Box0'}
        vfeatures |= {'Coord<Minimum>'}

    if 'Box1' in features:
        vfeatures -= {'Box1'}
        vfeatures |= {'Coord<Maximum>'}

    assert np.issubdtype(label_img.dtype, np.integer)

    # LabelMapper requires unsigned dtype
    if label_img.dtype == np.int8:
        label_img = label_img.view(np.uint8)
    if label_img.dtype == np.int16:
        label_img = label_img.view(np.uint16)
    if label_img.dtype == np.int32:
        label_img = label_img.view(np.uint32)
    if label_img.dtype == np.int64:
        label_img = label_img.view(np.uint64)

    # Oops, labelmapper doesn't support uint8 -> u32 nor uint16 -> u32
    if label_img.dtype in (np.uint8, np.uint16):
        label_img = label_img.astype(np.uint32)

    # Map from nonconsecutive[u64 or u32] -> consecutive uint32
    label_ids = np.sort(pd.unique(label_img.reshape(-1)))
    label_ids_32 = np.arange(len(label_ids), dtype=np.uint32)
    mapper = LabelMapper(label_ids, label_ids_32)
    label_img32 = mapper.apply(label_img)
    if ignore_label is not None and ignore_label in label_ids:
        ignore_label = mapper.apply(np.array([ignore_label], np.uint64))[0]

    assert label_img32.dtype == np.uint32

    if grayscale_img is None:
        invalid_names = vfeatures - {*SEGMENTATION_FEATURE_NAMES}
        assert not invalid_names, \
            f"Invalid segmentation feature names: {invalid_names}"
        grayscale_img = label_img32.view(np.float32)
    else:
        assert grayscale_img.dtype == np.float32, \
            "Grayscale image must be float32"

    grayscale_img = vigra.taggedView(grayscale_img, axes)
    label_img32 = vigra.taggedView(label_img32, axes)

    # TODO: provide histogramRange options
    acc = vigra.analysis.extractRegionFeatures(grayscale_img, label_img32, [*vfeatures], ignoreLabel=ignore_label)

    results = {}
    if 'Box0' in features:
        v = acc['Coord<Minimum >'].astype(np.int32)
        results['Box0'] = pd.DataFrame(v, columns=[*axes])
    if 'Box1' in features:
        v = 1+acc['Coord<Maximum >'].astype(np.int32)
        results['Box1'] = pd.DataFrame(v, columns=[*axes])
    if 'Box' in features:
        box0 = acc['Coord<Minimum >'].astype(np.int32)
        box1 = (1+acc['Coord<Maximum >']).astype(np.int32)
        boxes = np.stack((box0, box1), axis=1)
        obj_boxes = np.zeros(len(boxes), object)
        obj_boxes[:] = list(boxes)
        results['Box'] = pd.Series(obj_boxes, name='Box')

    for k, v in acc.items():
        k = k.replace(' ', '')

        # Only return the features the user explicitly requested.
        if k not in features:
            continue

        if v.ndim == 1:
            results[k] = pd.Series(v, name=k)
        elif v.ndim == 2:
            results[k] = pd.DataFrame(v, columns=[*axes])
        else:
            # If the data doesn't neatly fit into a 1-d Series
            # or a 2-d DataFrame, then construct a Series with dtype=object
            # and make each row a separate ndarray object.
            obj_v = np.zeros(len(v), dtype=object)
            obj_v[:] = list(v)
            results[k] = pd.Series(obj_v, name=k)

    # Set index to the original uint64 values
    for v in results.values():
        v.index = label_ids

    # vigra didn't process the ignore_label,
    # but it still appears in the results (with uninitialized values)
    # Remove it from our results.
    for k in [*results.keys()]:
        v = results[k]
        v = v[v.index != ignore_label].copy()
        results[k] = v

    return results


def meshes_from_volume(vol, fullres_box_zyx=None, subset_labels=None, *,
                       cuffs=False, capped=False,
                       min_voxels=0, max_voxels=None,
                       smoothing=0, constrain_exterior=False,
                       decimation=1.0, minimum_decimation_vertices=20,
                       keep_normals=False,
                       progress=True):
    """
    Generate meshes for all (or some subset) of the segments in a label image.
    To do this efficiently, care is taken to pre-calculate the extents of each
    object so that the minimal mask can be extracted for marching cubes to use.
    Otherwise, it's easy to accidentally degrade to quadratic behavior.

    Args:
        vol:
            3D label image
        fullres_box_zyx:
            The spatial extents of the volume, in spatial (mesh) coordinates, not voxel coordinates.
        subset_labels:
            A list of labels to process.  By default, all labels are processed except for label 0.
        cuffs:
            If True, pad the volume by duplicating each of the six volume edges.
            For objects which lie on the volume edge, this will add an extra "cuff"
            to the edge of the object.
            This can be useful for artificially closing gaps between segments in neighboring volumes,
            assuming you're computing a large set of meshes in block-wise fashion without "properly"
            stitching the objects in neighboring blocks together.
        capped:
            If True, pad the volume with an empty layer of voxels around each of the six volume edges.
            This will ensure that volumes on the edge of the volume are "capped" rather then left open.
        min_voxels:
            Only process objects with at least this many voxels in the volume.
            Skip the others and don't include them in the results.
        smoothing:
            How many rounds of "laplacian smoothing" to apply to the resulting mesh, before decimation.
        constrain_exterior:
            If True, don't allow the smoothing operation to adjust the coordinates of mesh vertices at
            the edge of the volume.  Leave those vertices "unsmoothed" to avoid mesh shrinkage.
            Has no effect unless cuffs=True.
        decimation:
            Decimate the mesh with the given target fraction.
            Examples:
                - 1.0: No decimation is performed.
                - 0.25: Remove vertices until only 25% of the original vertices remain.
        minimum_decimation_vertices:
            Don't even bother decimating meshes of objects that have fewer vertices than this count.
        keep_normals:
            If True, keep the normal vectors in the mesh before returning.
            Otherwise, discard them.  (If you aren't planning to use them anyway,
            you can reduce the RAM footprint of the mesh.)
        progress:
            If True, show a progress bar while meshes are being generated.

    Returns:
        df, mesh_gen
        Where df is a DataFrame indexed by the labels that will
        be processed along with their sizes and (local) extents,
        and mesh_gen is a generator which will produce a single mesh per
        iteration until all requested meshes have been produced.
    """
    from vol2mesh import Mesh

    if fullres_box_zyx is None:
        fullres_box_zyx = np.array([(0,0,0), vol.shape])
    else:
        fullres_box_zyx = np.asarray(fullres_box_zyx)

    # Infer the resolution of the downsampled volume
    fullres_shape = fullres_box_zyx[1] - fullres_box_zyx[0]
    resolution = (fullres_box_zyx[1] - fullres_box_zyx[0]) // vol.shape

    # The fullres start/end do not need to be even multiples of the resolution,
    # but the *width* of each dimension must divide cleanly.
    assert not (fullres_shape % vol.shape).any(), \
        "Mask volume dimensions must divide cleanly into full-res dimensions."

    fullres_padded_box = fullres_box_zyx.copy()
    if cuffs:
        vol = np.pad(vol, 1, 'edge')
        fullres_padded_box += resolution * np.array([[-1, -1, -1], [1, 1, 1]])

    if capped:
        vol = np.pad(vol, 1, 'constant')
        fullres_padded_box += resolution * np.array([[-1, -1, -1], [1, 1, 1]])

    max_voxels = max_voxels or np.prod(vol.shape)

    feat = region_features(vol, ignore_label=0)
    feat_df = feat['Box'].to_frame()
    feat_df['Count'] = feat['Count']
    feat_df = feat_df.rename_axis('label')
    feat_df = feat_df.query('label != 0 and Count >= @min_voxels and Count <= @max_voxels')

    if subset_labels is not None and len(subset_labels):
        feat_df = feat_df.query('label in @subset_labels')

    # Now convert (if possible)
    if vol.dtype in (np.uint64, np.int64) and vol.max() < np.iinfo(np.uint32).max:
        vol = vol.astype(np.uint32)

    def _meshes_for_volume():
        for label, segment_box in tqdm_proxy(feat_df['Box'].items(), total=len(feat_df), disable=not progress):
            mask_box = segment_box.copy()
            mask_box[0] = np.maximum(0, segment_box[0] - 1)
            mask_box[1] = np.minimum(vol.shape, segment_box[1] + 1)
            mask = vol[box_to_slicing(*mask_box)] == label
            m = Mesh.from_binary_vol(mask, mask_box * resolution + fullres_padded_box[0])

            m.box = segment_box * resolution + fullres_padded_box[0]

            if constrain_exterior:
                m.laplacian_smooth(smoothing, constrain_exterior=fullres_box_zyx)
            else:
                m.laplacian_smooth(smoothing)

            if len(m.vertices_zyx) > minimum_decimation_vertices:
                m.simplify_openmesh(decimation)

            if not keep_normals:
                m.drop_normals()

            yield label, m

    return feat_df.copy(), _meshes_for_volume()
