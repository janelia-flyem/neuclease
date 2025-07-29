import numpy as np
from numba import njit


def _encode_uint64_series(s, dtype='<u8'):
    """
    Encode a pandas Series (or Index) of N values
    into a numpy array of N buffers (bytes objects).
    """
    id_buf = s.to_numpy(dtype).tobytes()
    id_bufs = [
        id_buf[offset:(offset+8)]
        for offset in range(0, len(id_buf), 8)
    ]
    return np.array(id_bufs, dtype=object)


def _geometry_cols(coord_names, annotation_type):
    """
    Determine the list of column groups that express
    the geometry of annotations of the given type.
    Point annotations have only one group,
    but other annotation types have two.
    
    Examples:
    
        >>> _geometry_cols([*'xyz'], 'point')
        [['x', 'y', 'z']]

        >>> _geometry_cols([*'xyz'], 'ellipsoid')
        [['x', 'y', 'z'], ['rx', 'ry', 'rz']]

        >>> _geometry_cols([*'xyz'], 'line')
        [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']]

        >>> _geometry_cols([*'xyz'], 'axis_aligned_bounding_box')
        [['xa', 'ya', 'za'], ['xb', 'yb', 'zb']]
    """
    if annotation_type == 'point':
        return [[c for c in coord_names]]

    if annotation_type == 'ellipsoid':
        return [
            [c for c in coord_names],
            [f'r{c}' for c in coord_names]
        ]

    if annotation_type in ('line', 'axis_aligned_bounding_box'):
        return [
            [f'{c}a' for c in coord_names],
            [f'{c}b' for c in coord_names]
        ]

    raise ValueError(f"Annotation type {annotation_type} not supported")


def ndindex_array(shape):
    """
    Like np.ndindex, but returns an array.
    
    Example:
    
        >>>  _ndindex_array([3,4])
        array([[0, 0],
               [0, 1],
               [0, 2],
               [0, 3],
               [1, 0],
               [1, 1],
               [1, 2],
               [1, 3],
               [2, 0],
               [2, 1],
               [2, 2],
               [2, 3]])
    """
    return _ndindex_array(np.asarray(shape))

@njit
def _ndindex_array(shape):
    total_indices = np.prod(shape)
    ndim = len(shape)
    
    indices = np.zeros((total_indices, ndim), dtype=np.uint64)
    
    for linear_idx in range(total_indices):
        temp_idx = linear_idx

        # Work backwards through dimensions
        for dim in range(ndim - 1, -1, -1):
            indices[linear_idx, dim] = temp_idx % shape[dim]
            temp_idx = temp_idx // shape[dim]
    
    return indices
