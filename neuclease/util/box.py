from itertools import starmap
import numpy as np


def extract_subvol(array, box):
    """
    Extract a subarray according to the given box.
    """
    assert all(b >= 0 for b in box[0])
    assert all(b <= s for b,s in zip(box[1], array.shape))
    return array[box_to_slicing(*box)]


def overwrite_subvol(array, box, subarray):
    """
    Overwrite a portion of the given array.
    """
    assert all(b >= 0 for b in box[0])
    assert all(b <= s for b,s in zip(box[1], array.shape))
    try:
        array[box_to_slicing(*box)] = subarray
    except:
        assert (subarray.shape == box[1] - box[0]).all(), \
            f"subarray is the wrong shape {subarray.shape} for the given box {box}"
        raise


def box_to_slicing(start, stop):
    """
    For the given bounding box (start, stop),
    return the corresponding slicing tuple.

    Example:
    
        >>> assert bb_to_slicing([1,2,3], [4,5,6]) == np.s_[1:4, 2:5, 3:6]
    """
    return tuple( starmap( slice, zip(start, stop) ) )


def box_as_tuple(box):
    if isinstance(box, np.ndarray):
        box = box.tolist()
    return (tuple(box[0]), tuple(box[1]))


def box_shape(box):
    box = np.asarray(box)
    return box[1] - box[0]


def round_coord(coord, grid_spacing, how):
    """
    Round the given coordinate up or down to the nearest grid position.
    (If how='closest' each axis is rounded up/down independently,
    so some might go up and some might go down.)
    """
    coord = np.asarray(coord)
    assert how in ('down', 'up', 'closest')
    if how == 'down':
        return (coord // grid_spacing) * grid_spacing
    if how == 'up':
        return ((coord + grid_spacing - 1) // grid_spacing) * grid_spacing
    if how == 'closest':
        down = (coord // grid_spacing) * grid_spacing
        up = ((coord + grid_spacing - 1) // grid_spacing) * grid_spacing
        both = np.array((down, up))
        both_diffs = np.abs(both - coord)
        return both[np.argmin(both_diffs, axis=0), range(len(coord))]


def round_box(box, grid_spacing, how='out'):
    # FIXME: Better name would be align_box()
    """
    Expand/shrink the given box (or boxes) out/in to align it to a grid.

    box: ND array with shape (..., 2, D)
    grid_spacing: int or shape
    how: One of ['out', 'in', 'down', 'up', 'closest'].
         Determines which direction the box corners are moved.

    See also: pad_for_grid
    """
    directions = { 'out':  ('down', 'up'),
                   'in':   ('up', 'down'),
                   'down': ('down', 'down'),
                   'up':   ('up', 'up'),
                   'closest': ('closest', 'closest') }

    box = np.asarray(box)
    assert how in directions.keys()

    box0 = round_coord(box[..., 0, :], grid_spacing, directions[how][0])[..., None, :]
    box1 = round_coord(box[..., 1, :], grid_spacing, directions[how][1])[..., None, :]
    return np.concatenate( (box0, box1), axis=-2 )


def pad_for_grid(a, grid_spacing, box_zyx=None):
    """
    For an array which currently occupies the given box in space,
    pad the array such that its edges align to the given grid.
    """
    if box_zyx is None:
        box_zyx = [(0,)*a.ndim, a.shape]

    box_zyx = np.asarray(box_zyx)
    assert ((box_zyx[1] - box_zyx[0]) == a.shape).all()
    rounded_box = round_box(box_zyx, grid_spacing, 'out')
    box_padding = np.array([box_zyx[0] - rounded_box[0],
                            rounded_box[1] - box_zyx[1]])
    padded = np.pad(a, box_padding.T,)
    return padded, rounded_box


def choose_pyramid_depth(bounding_box, top_level_max_dim=512):
    """
    If a 3D volume pyramid were generated to encompass the given bounding box,
    determine how many pyramid levels you would need such that the top
    level of the pyramid is no wider than `top_level_max_dim` in any dimension.
    """
    from numpy import ceil, log2
    bounding_box = np.asarray(bounding_box)
    global_shape = bounding_box[1] - bounding_box[0]

    full_res_max_dim = float(global_shape.max())
    assert full_res_max_dim > 0.0, "Subvolumes encompass no volume!"
    
    depth = int(ceil(log2(full_res_max_dim / top_level_max_dim)))
    return max(depth, 0)


def box_intersection(box_A, box_B):
    """
    Compute the intersection of the two given boxes.
    If the two boxes do not intersect at all, then the returned box will have non-positive shape:

    >>> intersection = box_intersection(box_A, box_B)
    >>> assert (intersection[1] - intersection[0] > 0).all(), "Boxes do not intersect."

    You may pass multiple boxes in either argument, in which case broadcasting rules apply.
    """
    box_A = np.asarray(box_A)
    box_B = np.asarray(box_B)

    assert box_A.shape[-2:] == box_B.shape[-2:], \
        f"Incompatible shapes: {box_A.shape} and {box_B.shape}"

    intersection = np.empty(np.broadcast(box_A, box_B).shape, np.int32)
    intersection[..., 0, :] = np.maximum( box_A[..., 0, :], box_B[..., 0, :] )
    intersection[..., 1, :] = np.minimum( box_A[..., 1, :], box_B[..., 1, :] )
    return intersection


def box_union(*boxes):
    """
    Compute the bounding box of the given boxes,
    i.e. the smallest box that still encompasses
    all of the given boxes.
    """
    boxes = np.asarray(boxes)
    union = boxes[0].copy()
    union[0] = boxes[:, 0].min(axis=0)
    union[1] = boxes[:, 1].max(axis=0)
    return union


def is_box_coverage_complete(boxes, full_box):
    """
    When overlaid with each other, do the given
    boxes fully cover the given full_box?
    
    Args:
        boxes:
            Array, shape (N,2,D)

        full_box:
            Array, shape (2,D)
        
    Returns:
        bool
    
    Note:
        Boxes that extend beyond the extents of
        full_box are permitted; they'll be cropped.
    """
    boxes = np.array(boxes, dtype=int)
    full_box = np.array(full_box, dtype=int)

    assert boxes.ndim == 3
    assert full_box.ndim == 2
    assert boxes.shape[1:] == full_box.shape

    # Offset so full_box starts at (0,0,...)
    boxes = boxes - full_box[0]
    full_box = full_box - full_box[0]

    # Clip
    boxes[:,0,:] = np.maximum(boxes[:,0,:], full_box[0])
    boxes[:,1,:] = np.minimum(boxes[:,1,:], full_box[1])

    # Save RAM by dividing boxes by their greatest common factor along each dimension,
    # allowing for a smaller mask.
    all_boxes = np.concatenate((boxes, full_box[None]), axis=0)
    factors = np.gcd.reduce(all_boxes, axis=(0,1))

    assert not (boxes % factors).any()
    assert not (full_box % factors).any()

    boxes = boxes // factors
    full_box = full_box // factors

    # Now simply create a binary mask and fill each box within it.
    mask = np.zeros(full_box[1], dtype=bool)
    for box in boxes:
        overwrite_subvol(mask, box, True)

    return mask.all()


def boxlist_to_json( bounds_list, indent=0 ):
    # The 'json' module doesn't have nice pretty-printing options for our purposes,
    # so we'll do this ourselves.
    from io import StringIO

    buf = StringIO()
    buf.write('    [\n')
    
    bounds_list, last_item = bounds_list[:-1], bounds_list[-1:]
    
    for bounds_zyx in bounds_list:
        start_str = '[{}, {}, {}]'.format(*bounds_zyx[0])
        stop_str  = '[{}, {}, {}]'.format(*bounds_zyx[1])
        buf.write(' '*indent + '[ ' + start_str + ', ' + stop_str + ' ],\n')

    # Write last entry
    if last_item:
        last_item = last_item[0]
        start_str = '[{}, {}, {}]'.format(*last_item[0])
        stop_str  = '[{}, {}, {}]'.format(*last_item[1])
        buf.write(' '*indent + '[ ' + start_str + ', ' + stop_str + ' ]')

    buf.write('\n')
    buf.write(' '*indent + ']')

    return str(buf.getvalue())
