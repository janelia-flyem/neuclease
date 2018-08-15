from itertools import starmap
import numpy as np


def extract_subvol(array, box):
    """
    Extract a subarray according to the given box.
    """
    return array[box_to_slicing(*box)]


def overwrite_subvol(array, box, subarray):
    """
    Overwrite a portion of the given array.
    """
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


def round_coord(coord, grid_spacing, how):
    """
    Round the given coordinate up or down to the nearest grid position.
    """
    assert how in ('down', 'up')
    if how == 'down':
        return (coord // grid_spacing) * grid_spacing
    if how == 'up':
        return ((coord + grid_spacing - 1) // grid_spacing) * grid_spacing


def round_box(box, grid_spacing, how='out'):
    # FIXME: Better name would be align_box()
    """
    Expand/shrink the given box out/in to align it to a grid.

    box: (start, stop)
    grid_spacing: int or shape
    how: One of ['out', 'in', 'down', 'up'].
         Determines which direction the box corners are moved.
    """
    directions = { 'out':  ('down', 'up'),
                   'in':   ('up', 'down'),
                   'down': ('down', 'down'),
                   'up':   ('up', 'up') }

    box = np.asarray(box)
    assert how in directions.keys()
    return np.array( [ round_coord(box[0], grid_spacing, directions[how][0]),
                       round_coord(box[1], grid_spacing, directions[how][1]) ] )


def box_intersection(box_A, box_B):
    """
    Compute the intersection of the two given boxes.
    If the two boxes do not intersect at all, then the returned box will have non-positive shape:

    >>> intersection = box_intersection(box_A, box_B)
    >>> assert (intersection[1] - intersection[0] > 0).all(), "Boxes do not intersect."
    """
    intersection = np.empty_like(box_A)
    intersection[0] = np.maximum( box_A[0], box_B[0] )
    intersection[1] = np.minimum( box_A[1], box_B[1] )
    return intersection


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
