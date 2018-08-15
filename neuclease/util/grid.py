import numpy as np

from .box import ndrange
from .box import box_intersection, round_box

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
        assert (self.halo_shape < self.block_shape).all(), \
            f"Halo shape must be smaller than the block shape in all dimensions: {self.halo_shape} vs {self.block_shape}"

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


def boxes_from_grid(bounding_box, grid, include_halos=True):
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
    aligned_start = ((bounding_box[0] - halo_shape) // block_shape) * block_shape
    aligned_stop = ((bounding_box[1] + halo_shape + block_shape-1) // block_shape) * block_shape

    for block_start in ndrange( aligned_start, aligned_stop, block_shape ):
        yield np.array((block_start - halo_shape,
                        block_start + halo_shape + block_shape))


def clipped_boxes_from_grid(bounding_box, grid, include_halos=True):
    """
    Generator.
    
    Assuming an ND grid with boxes of size grid.block_shape, and aligned at the given grid.offset,
    iterate over all boxes of the grid that fall within or intersect the given bounding_box.
    
    Returned boxes that would intersect the edge of the bounding_box are clipped so as not
    to extend beyond the bounding_box.
    """
    for box in boxes_from_grid(bounding_box, grid, include_halos):
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
