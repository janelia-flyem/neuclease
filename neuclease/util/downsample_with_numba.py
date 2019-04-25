import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def downsample_box( box, block_shape ):
    """
    Given a box (i.e. start and stop coordinates) and a
    block_shape (downsampling factor), return the corresponding box
    in downsampled coordinates.
    """
    assert block_shape.shape[0] == box.shape[1]
    downsampled_box = np.zeros_like(box)
    downsampled_box[0] = box[0] // block_shape
    downsampled_box[1] = (box[1] + block_shape - 1) // block_shape
    return downsampled_box

def make_blockwise_reducer_3d(reducer_func, nopython=True):
    """
    Returns a function that can reduce an array of shape (Z*Bz, Y*By, X*Bx)
    into an array of shape (Z,Y,X), by dividing the array into shapes of blocks (Bz,By,Bx)
    and calling the given 'reducer' function on each block.

    The reducer function must return a scalar.
    Ideally, the reducer should be jit-compileable with numba. If not, set nopython=False.
    
    See reduce_blockwise(), below, for details regarding the returned function.
    
    Implemented according to guidelines in numba FAQ:
    http://numba.pydata.org/numba-doc/dev/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function
    """
    @jit(nopython=nopython)
    def _reduce_blockwise_compiled(data, block_shape, data_box, reduced_box):
        _output_shape = reduced_box[1] - reduced_box[0]
        output_shape = (_output_shape[0], _output_shape[1], _output_shape[2])
        output = np.zeros(output_shape, data.dtype)

        for block_index in np.ndindex(*output_shape):
            # block_bounds = block_shape * ( (block_index, 1+block_index) + reduced_box[0] )
            block_bounds = np.zeros( (2,3), dtype=np.int32 )
            block_bounds[0] = block_index
            block_bounds[1] = 1 + block_bounds[0]
            block_bounds[:] += reduced_box[0]
            block_bounds[:] *= block_shape
            
            block_bounds[0] = np.maximum(block_bounds[0], data_box[0])
            block_bounds[1] = np.minimum(block_bounds[1], data_box[1])
    
            z0, y0, x0 = block_bounds[0] - data_box[0]
            z1, y1, x1 = block_bounds[1] - data_box[0]
            
            block_data = data[z0:z1, y0:y1, x0:x1]

            bi_z, bi_y, bi_x = block_index
            output[bi_z, bi_y, bi_x] = reducer_func(block_data)
        return output

    def reduce_blockwise(data, block_shape, data_box=None):
        """
        Reduce the given 3D array block-by-block, returning a smaller array of scalars (one per block).
        
        Args:
         data:
             3D array, whose shape need not be exactly divisible by the block_shape
         
         block_shape:
             tuple (Bz,By,Bx)
         
         data_box:
             bounding box pair: [(z0, y0, x0), (z1, y1, x1)]
             
             If block_shape does not cleanly divide into block_shape, blocks on the edge
             of the full data array will be appropriately truncated before they are sent
             to the reducer function.  This is true for blocks on *any* side of the volume.
             
             It is assumed that blocks are aligned to some global coordinate grid,
             starting at (0,0,0), but the 'data' array might not be aligned with that grid.
             For example, the first element of the 'data' array may correspond to voxel (0, 0, 1),
             and therefore the first block will be smaller than most other blocks in the volume.
        """
        assert data.ndim == 3
    
        if data_box is None:
            data_box = np.array([(0,0,0), data.shape])
        else:
            data_box = np.asarray(data_box)
        
        assert data_box.shape == (2,3)
             
        # If the block_shape is an int, convert it to a shape.
        if np.issubdtype(type(block_shape), np.integer):
            block_shape = (block_shape, block_shape, block_shape)
        
        block_shape = np.array(block_shape)
        assert block_shape.shape == (3,)
        
        if (block_shape == 1).all():
            # Shortcut: Nothing to do.
            return data, data_box.copy()
        
        reduced_box = downsample_box(data_box, block_shape)
        reduced_output = _reduce_blockwise_compiled(data, block_shape, data_box, reduced_box)
        return reduced_output, reduced_box

    return reduce_blockwise


@jit(nopython=True, cache=True)
def flat_mode_except_zero(data):
    """
    Given an array, flatten it and return the mode, without including
    zeros, if possible.
    
    If (data == 0).all(), then 0 is returned.
    """
    data = data.copy().reshape(-1)
    data = data[data != 0]
    if data.size == 0:
        return 0
    return _flat_mode(data)


@jit(nopython=True, cache=True)
def flat_mode(data):
    """
    Given an ND array, flatten it and return the mode.
    """
    data = data.copy().reshape(-1)
    return _flat_mode(data)


@jit(nopython=True, cache=True)
def _flat_mode(data):
    """
    Given an contiguous flat array, return the mode.
    
    Note: We could have used scipy.stats.mode() here,
          but that implementation is insanely slow for large arrays,
          especially if there are many label values in the array.
    """
    data.sort()
    diff = np.diff(data)
    diff_bool = np.ones((len(diff)+2,), dtype=np.uint8)
    diff_bool[1:-1] = (diff != 0)

    diff_nonzero = diff_bool.nonzero()[0]
    run_lengths = diff_nonzero[1:] - diff_nonzero[:-1]
    max_run = np.argmax(run_lengths)
    return data[diff_nonzero[max_run]]


@jit(nopython=True, cache=True)
def flat_binary_mode(data):
    nonzero = 0
    for index in np.ndindex(data.shape):
        z,y,x = index
        if data[z,y,x] != 0:
            nonzero += 1

    if nonzero > data.size // 2:
        return 1
    return 0

# Signature:
# reduced_output, reduced_box = f(data, block_shape, data_box=None)
downsample_labels_3d = make_blockwise_reducer_3d(flat_mode)
downsample_binary_3d = make_blockwise_reducer_3d(flat_binary_mode)

# These variants will not return zero as the block mode UNLESS it's the only value in the block.
downsample_labels_3d_suppress_zero = make_blockwise_reducer_3d(flat_mode_except_zero)
downsample_binary_3d_suppress_zero = make_blockwise_reducer_3d(np.any)
