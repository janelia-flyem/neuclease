import numpy as np
import pandas as pd

from .util import ndrange, tqdm_proxy, compute_parallel
from .box import box_intersection, round_box, overwrite_subvol

class Grid:
    """
    Describes a blocking scheme, which consists of the following:
    
    - A block shape representing the Grid spacing
    - An offset coordinate, i.e. the start of the first Grid block
    - An optional halo shape, indicating that blocks within the grid
      should extend outside the block shape, causing neighboring
      blocks to overlap.
    """
    def __init__(self, block_shape, offset=None, halo=0):
        if offset is None:
            offset = (0,)*len(block_shape)
        assert len(block_shape) == len(offset)

        self.block_shape = np.asarray(block_shape)
        assert (self.block_shape > 0).all(), f"block_shape must be non-zero, not {self.block_shape}"

        self.offset = np.asarray(offset)
        self.modulus_offset = self.offset % block_shape
        
        self.halo_shape = np.zeros_like(self.block_shape)
        self.halo_shape[:] = halo
        
        # FIXME: Why is this assertion necessary?
        assert (self.halo_shape <= self.block_shape).all(), \
            f"Halo shape must be <= than the block shape in all dimensions: {self.halo_shape} vs {self.block_shape}"

    def equivalent_to(self, other_grid):
        """
        Returns True if the other grid is equivalent to this one, meaning: 
        - it has the same block shape
        - it has the same halo shape
        - it's offset is the same (after modulus by block shape).
        """
        return (self.block_shape == other_grid.block_shape).all() and \
               (self.modulus_offset == other_grid.modulus_offset).all() and \
               (self.halo_shape == other_grid.halo_shape).all()

    def compute_logical_box(self, point):
        """
        Return the so-called "logical box" that encompasses the given point.
        A logical box is defined by only the block shape and offset,
        NOT the halo.
        """
        block_index = (point - self.offset) // self.block_shape
        block_start = self.offset + (block_index * self.block_shape)
        return np.asarray( (block_start, block_start + self.block_shape) )


class boxes_from_grid:
    """
    Iterable.
    
    Given a bounding box and a Grid, this class can be used to iterate
    over the boxes of the Grid that intersect the bounding box.
    See __init__ for options.
    
    This iterable declares __len__, so it can be used with tqdm, for instance.
    """
    def __init__(self, bounding_box, grid, include_halos=True, clipped=False):
        """
        Args:
            bounding_box:
                The extents of the region to pull boxes from.
                Note:
                    If clipped=False and this bounding box is not aligned to the grid,
                    then some boxes will extend beyond the the bounding_box.
            grid:
                An instance of Grid, or a block shape (tuple).
            
            include_halos:
                If True, yielded boxes will include the Grid halo, otherwise the Grid
                halo is ignored and only the 'internal' portion of each Grid box
                is yielded (as if the Grid has no halo).
            
            clipped:
                If True, boxes that intersect the bounding box edges will be "clipped"
                (truncated) to their intersecting portion, and thus no yielded boxes
                will extend outside of the bounding box (but some boxes may be smaller
                than others).
        
        Examples:

            >>> for box in boxes_from_grid([[0,0], [10,10]], (5,6)):
            ...     print(box.tolist())
            [[0, 0], [5, 6]]
            [[0, 6], [5, 12]]
            [[5, 0], [10, 6]]
            [[5, 6], [10, 12]]

            >>> for box in boxes_from_grid([[0,0], [10,10]], (5,6), clipped=True):
            ...     print(box.tolist())
            [[0, 0], [5, 6]]
            [[0, 6], [5, 10]]
            [[5, 0], [10, 6]]
            [[5, 6], [10, 10]]
        """
        # If necessary, auto-convert blockshape into a Grid
        if not isinstance(grid, Grid):
            if not hasattr(grid, '__len__'):
                # If we were given an int, convert to isotropic blockshape
                grid = (grid,)*len(bounding_box[0])
            grid = Grid(grid)

        self.bounding_box = bounding_box
        self.grid = grid
        self.include_halos = include_halos
        self.clipped = clipped

    def __iter__(self):
        if self.clipped:
            return _clipped_boxes_from_grid(self.bounding_box, self.grid, self.include_halos)
        else:
            return _boxes_from_grid(self.bounding_box, self.grid, self.include_halos)
    
    def __len__(self):
        offset_bounding_box = self.bounding_box - self.grid.offset
        aligned_bounding_box = round_box(offset_bounding_box, self.grid.block_shape, 'out')
        num_boxes = np.prod((aligned_bounding_box[1] - aligned_bounding_box[0]) // self.grid.block_shape)
        return num_boxes


def clipped_boxes_from_grid(bounding_box, grid, include_halos=True):
    """
    Convenience function for boxes_from_grid(..., clipped=True).
    (Mostly here for backwards compatibility with old code.)
    """
    return boxes_from_grid(bounding_box, grid, include_halos, clipped=True)


def _boxes_from_grid(bounding_box, grid, include_halos=True):
    """
    Generator.
    
    Assuming an ND grid with boxes of size grid.block_shape, and aligned at the given grid.offset,
    iterate over all boxes of the grid that fall within or intersect the given bounding_box.
    
    Note: The returned boxes are not clipped to fall within the bounding_box.
          If either bounding_box[0] or bounding_box[1] is not aligned with the grid,
          some returned boxes will extend beyond the bounding_box.
    """
    if include_halos:
        halo = grid.halo_shape
    else:
        halo = 0
    
    if grid.offset is None or not any(grid.offset):
        # Shortcut
        yield from _boxes_from_grid_no_offset(bounding_box, grid.block_shape, halo)
    else:
        grid_offset = np.asarray(grid.offset)
        bounding_box = bounding_box - grid.offset
        for box in _boxes_from_grid_no_offset(bounding_box, grid.block_shape, halo):
            box += grid_offset
            yield box


def _boxes_from_grid_no_offset(bounding_box, block_shape, halo):
    """
    Generator.
    
    Assuming an ND grid with boxes of size block_shape, and aligned at the origin (0,0,...),
    iterate over all boxes of the grid that fall within or intersect the given bounding_box.
    
    For the purposes of determining whether or not a grid section intersects the bounding_box,
    the halo is not used.  The halo is supplied only if the box's internal ("logical") portion
    intersects the bounding box.
    
    Note: The returned boxes are not clipped to fall within the bounding_box.
          If either bounding_box[0] or bounding_box[1] is not aligned with the grid
          (i.e. they are not a multiple of block_shape),
          some returned boxes will extend beyond the bounding_box.
    """
    bounding_box = np.asarray(bounding_box, dtype=int)
    block_shape = np.asarray(block_shape)
    halo_shape = np.zeros((len(block_shape),), dtype=np.int32)
    halo_shape[:] = halo

    # round down, round up
    aligned_start = ((bounding_box[0]) // block_shape) * block_shape
    aligned_stop = ((bounding_box[1] + block_shape-1) // block_shape) * block_shape

    for block_start in ndrange( aligned_start, aligned_stop, block_shape ):
        yield np.array((block_start - halo_shape,
                        block_start + halo_shape + block_shape))


def _clipped_boxes_from_grid(bounding_box, grid, include_halos=True):
    """
    Generator.
    
    Assuming an ND grid with boxes of size grid.block_shape, and aligned at the given grid.offset,
    iterate over all boxes of the grid that fall within or intersect the given bounding_box.
    
    Returned boxes that would intersect the edge of the bounding_box are clipped so as not
    to extend beyond the bounding_box.
    """
    for box in _boxes_from_grid(bounding_box, grid, include_halos):
        yield box_intersection(box, bounding_box)


def slabs_from_box( full_res_box, slab_depth, scale=0, scaling_policy='round-out', slab_cutting_axis=0 ):
    """
    Generator.
    
    Divide a bounding box into several 'slabs' stacked along a particular axis,
    after optionally reducing the bounding box to a reduced scale.
    
    Note: The output slabs are aligned to multiples of the slab depth.
          For example, if full_res_box starts at 3 and slab_depth=10,
          then the first slab will span [3,10], and the second slab will span from [10,20].
    
    full_res_box: (start, stop)
        The original bounding-box, in full-res coordinates
    
    slab_depth: (int)
        The desired width of the output slabs.
        This will be the size of the output slabs, regardless of any scaling applied.
    
    scale:
        Reduce the bounding-box to a smaller scale before computing the output slabs.
        
    scaling_policy:
        For scale > 0, the input bounding box is reduced.
        For bounding boxes that aren't divisible by 2*scale, the start/stop coordinates must be rounded up or down.
        Choices are:
            'round-out': Expand full_res_box to the next outer multiple of 2**scale before scaling down.
            'round-in': Shrink full_res_box to the next inner multiple of 2**scale before scaling down.
            'round-down': Round down on full_res_box (both start and stop) before scaling down.
    
    slab_cutting_axes:
        Which axis to cut across to form the stacked slabs. Default is Z (assuming ZYX order).
    """
    assert scaling_policy in ('round-out', 'round-in', 'round-down')
    full_res_box = np.asarray(full_res_box)

    round_method = scaling_policy[len('round-'):]
    scaled_input_bb_zyx = round_box(full_res_box, 2**scale, round_method) // 2**scale

    slab_shape_zyx = scaled_input_bb_zyx[1] - scaled_input_bb_zyx[0]
    slab_shape_zyx[slab_cutting_axis] = slab_depth

    # This grid outlines the slabs -- each box in slab_grid is a full slab
    grid_offset = scaled_input_bb_zyx[0].copy()
    grid_offset[slab_cutting_axis] = 0 # See note about slab alignment, above.
    
    slab_grid = Grid(slab_shape_zyx, grid_offset)
    slab_boxes = clipped_boxes_from_grid(scaled_input_bb_zyx, slab_grid)
    
    return slab_boxes



def boxes_from_mask(mask, mask_offset=(0,0,0), grid=(64,64,64), clipped=False):
    """
    Given a boolean mask and a grid to overlay on it,
    emit boxes for each grid space which contains at least
    one nonzero voxel from the mask.

    If no nonzero voxels fall within a paritcular grid space,
    no box will be emitted for that space.
    
    Note:
        The current implementation of this function works by calling
        mask.nonzero() and creating a DataFrame from the resulting
        coordiantes.  That will be expensive for large masks.
        Use with caution.  In the future, a more RAM-friendly implementation
        could be written that operates directly on the mask voxels,
        one grid block at a time.
    
    Args:
        mask:
            3D array, will be treated as boolean.
        
        mask_offset:
            The mask need not be aligned to (0,0,0).
            Give the coordiante of the mask's first voxel here.
        
        grid:
            Either a Grid, block tuple, or block width

        clipped:
            If True, reduce the size of each emitted box to minimally
            encapsulate the coordinates that fall within it.
            Otherwise, emit a box for the complete grid space.

    Returns:
        ndarray with shape (N, 2, 3) for N boxes.
    """
    assert mask.ndim == 3, \
        f"mask must be 3D.  Your mask has shape: {mask.shape}"
    
    coords = np.array(mask.nonzero()).transpose()
    coords += mask_offset
    return boxes_from_coords(coords, grid, clipped)


def boxes_from_coords(coords, grid, clipped=False):
    """
    Group the given coordinates according to a Grid,
    and emit the set of boxes that the coordinates fall into.

    If no coordinates fall within a paritcular grid space,
    no box will be emitted for that space.
    
    Args:
        coords:
            2D array, [[z,y,x],
                       [z,y,x],
                       ...]
        
        grid:
            Either a Grid, block tuple, or block width
        
        clipped:
            If True, reduce the size of each emitted box to minimally
            encapsulate the coordinates that fall within it.
            Otherwise, emit a box for the complete grid space.
    
    Returns:
        ndarray with shape (N, 2, 3) for N boxes.
    """
    coords = np.asarray(coords)
    assert coords.ndim == 2
    assert coords.shape[1] == 3, "This function only works for 3D coordinates"

    # If necessary, auto-convert blockshape into a Grid
    if not isinstance(grid, Grid):
        if not hasattr(grid, '__len__'):
            # If we were given an int, convert to isotropic blockshape
            grid = (grid,)*coords.shape[1]
        grid = Grid(grid)
    
    assert not grid.offset.any(), \
        "Grids with offsets are not yet supported by this function."
    
    blocks = coords // grid.block_shape
    
    if not clipped:
        blocks = pd.DataFrame(blocks).drop_duplicates().values
        starts = (blocks * grid.block_shape)
        stops = starts + grid.block_shape
        return np.array((starts, stops)).transpose(1,0,2)
    
    table = np.concatenate((blocks, coords), axis=1)
    df = pd.DataFrame(table, columns=['bz', 'by', 'bx', 'z', 'y', 'x'])
    df = df.sort_values(['bz', 'by', 'bx', 'z', 'y', 'x'])
    boxes_df = df.groupby(['bz', 'by', 'bx']).agg({'z': ['min', 'max'], 'y': ['min', 'max'], 'x': ['min', 'max']})
    boxes_df.columns = ['z_min', 'z_max', 'y_min', 'y_max', 'x_min', 'x_max']
    boxes_df[['z_max', 'y_max', 'x_max']] += (1,1,1)
    boxes = boxes_df.values.reshape((-1, 3, 2)).transpose((0, 2, 1))
    return boxes


def _fetch_volume_in_chunks_singlethreaded(bounding_box, grid, fetch_fn):
    """
    Fetch a large volume, one chunk at a time.
    
    Args:
        bounding_box:
            The location of the voxels to fetch
        
        grid:
            The chunking scheme
        
        fetch_fn:
            The function to use to fetch each chunk.
            Should have signature:
            
                subvol = fetch_fn(box)
    
    Returns:
        ndarray, shaped according to bounding box stop - start
    """
    # Initialize volume after the first fetch, to determine dtype
    vol = None
    shape = bounding_box[1] - bounding_box[0]

    for box in tqdm_proxy(boxes_from_grid(bounding_box, grid, clipped=True)):
        subvol = fetch_fn(box)
        if vol is None:
            vol = np.zeros(shape, subvol.dtype)
        overwrite_subvol(vol, box - bounding_box[0], subvol)

    return vol


def fetch_volume_in_chunks(bounding_box, grid, fetch_fn, threads=0, dtype=None):
    """
    Fetch a large volume in chunks, and return the assembled result.
    The chunks can optionally be fetched in parallel, via a threadpool.
    
    Args:
        bounding_box:
            The location of the voxels to fetch
        
        grid:
            The chunking scheme
        
        fetch_fn:
            The function to use to fetch each chunk.
            Should have signature:
            
                subvol = fetch_fn(box)
        
        threads:
            If nonzero, fetch chunks in parallel, using a threadpool.
            For completely synchronous operation (no threadpool), use threads=0.
        
        dtype:
            Optional. If you know the dtype of the volume,
            you can pass it here as a tiny optimization.
            Otherwise, the first chunk is fetched before
            the threadpool is started, to determine the dtype.
    
    Returns:
        ndarray, shaped according to bounding box (stop - start)
    """
    if threads == 0:
        return _fetch_volume_in_chunks_singlethreaded(bounding_box, grid, fetch_fn)
    
    shape = bounding_box[1] - bounding_box[0]
    
    boxes = boxes_from_grid(bounding_box, grid, clipped=True)
    num_boxes = len(boxes)
    prog_start = 0
    boxes = iter(boxes)
    
    # If the user provided a dtype, allocate the volume right away.
    # Otherwise, fetch the first subvol before the threadpool starts,
    # just to determine the dtype.
    if dtype is not None:
        vol = np.empty(shape, dtype)
    else:
        box = next(boxes)
        prog_start = 1
        subvol = fetch_fn(box)
        dtype = subvol.dtype
        vol = np.empty(shape, dtype)
        overwrite_subvol(vol, box - bounding_box[0], subvol)

    def fetch_and_overwrite(box):
        subvol = fetch_fn(box)
        overwrite_subvol(vol, box - bounding_box[0], subvol)
    
    compute_parallel( fetch_and_overwrite,
                      boxes,
                      threads=threads,
                      ordered=False,
                      leave_progress=True,
                      total=num_boxes,
                      initial=prog_start )
    return vol
