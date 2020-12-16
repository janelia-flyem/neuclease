import collections
import numpy as np

from .view_as_blocks import view_as_blocks
from .box import box_intersection, box_to_slicing, extract_subvol, round_box
from .grid import Grid, boxes_from_grid
from .segmentation import compute_nonzero_box
from .util import downsample_mask

class SparseBlockMask:
    """
    Tiny class to hold a low-resolution binary mask and the box it corresponds to.

    In other words, a simple way of sparsely marking blocks of a large volume.

    If your blocks of interest are VERY sparse, it would be cheaper to simply
    store the list of block coordinate tuples.  But for semi-sparse data, storing
    a binary mask as done here is cheaper, assuming your chosen block 'resolution'
    is reasonably large (typically (64,64,64), or (32,32,32) for DVID ROIs).
    """
    def __init__(self, lowres_mask, box, resolution):
        """
        Args:
            lowres_mask:
                boolean ndarray, where each voxel represents a block of full-res data.
            box:
                The volume of space covered by the mask, in FULL-RES coordinates
            resolution:
                The width (or shape) of each lowres voxel in FULL-RES coordinates.
        """
        self.lowres_mask = lowres_mask.astype(bool, copy=False)

        self.box = np.asarray(box) # full-res

        self.resolution = resolution
        if isinstance(self.resolution, collections.abc.Iterable):
            self.resolution = np.asarray(resolution)
        else:
            self.resolution = np.array( [resolution]*lowres_mask.ndim )

        assert (((self.box[1] - self.box[0]) // self.resolution) == self.lowres_mask.shape).all(), \
            f"Inconsistent mask shape ({lowres_mask.shape}) and box {self.box.tolist()} for the given resolution ({resolution}).\n"\
            "Note: box should be specified in FULL resolution coordinates."

        # Nonzero box (full-res coordinates)
        self.nonzero_box = compute_nonzero_box(self.lowres_mask)
        self.nonzero_box *= resolution
        self.nonzero_box += self.box[0]

    @classmethod
    def create_empty(cls, resolution, corner=[0,0,0]):
        empty_box = np.array([corner, corner])
        empty_mask = np.zeros((0,)*len(empty_box[0]), dtype=bool)
        return SparseBlockMask(empty_mask, empty_box, resolution)

    @classmethod
    def create_from_sbm_box(cls, sbm, sub_box):
        """
        Create a SparseBlockMask by extract a portion of an existing SparseBlockMask.
        If sub_box isn't aligned to the given sbm resolution, it will be expanded to make it aligned.
        """
        sub_box = round_box(sub_box, sbm.resolution, 'out')

        new_box = box_intersection(sub_box, sbm.box)
        new_resolution = sbm.resolution.copy()
        new_lowres_box = new_box // new_resolution

        old_lowres_box = sbm.box // sbm.resolution
        new_lowres_mask = extract_subvol(sbm.lowres_mask, new_lowres_box - old_lowres_box[0]).copy()
        return SparseBlockMask(new_lowres_mask, new_box, new_resolution)


    @classmethod
    def create_from_highres_mask(cls, highres_mask, highres_resolution, fullres_box, lowres_resolution):
        """
        Given a mask that is not necessarily at full-res,
        but still at a higher res than the SBM you want to create,
        create a SBM at a lower resolution.

        Args:
            highres_mask:
                A binary mask which will be downsampled before creating the SBM
            highres_resolution:
                The resolution of the input mask, i.e. the width of each mask voxel, in FULL-RES coordinates.
            fullres_box:
                The volume of space covered by the high-res mask, in FULL-RES coordinates.
            lowres_resolution:
                The resolution of the returned SBM, in FULL-RES coordinates.
        Returns:
            SparseBlockMask
        """
        assert not (np.array(lowres_resolution) % highres_resolution).any()
        lowres_mask = downsample_mask(highres_mask, lowres_resolution // highres_resolution)
        return SparseBlockMask(lowres_mask, fullres_box, lowres_resolution)


    @classmethod
    def create_from_lowres_coords(cls, coords_zyx, resolution):
        """
        Convenience constructor.
        Create a SparseBlockMask from an array of block coordinates
        (i.e. 'lowres' coordinates).

        Args:
            coords_zyx:
                ndarray, shape (N,3), indicating the 'blocks' that are masked.
            resolution:
                The width (or shape) of each lowres voxel in FULL-RES coordinates.
        Returns:
            SparseBlockMask
        """
        if len(coords_zyx) == 0:
            return SparseBlockMask.create_empty(resolution)

        min_coord = coords_zyx.min(axis=0)
        max_coord = coords_zyx.max(axis=0)
        mask_box = np.array((min_coord, 1+max_coord))
        mask_shape = mask_box[1] - mask_box[0]
        mask = np.zeros(mask_shape, dtype=bool)

        mask_coords = coords_zyx - mask_box[0]
        mask[(*mask_coords.transpose(),)] = True

        return SparseBlockMask(mask, resolution*mask_box, resolution)


    def change_resolution(self, new_resolution):
        """
        Without changing the mask data,
        change interpretation of the mask data's resolution and bounding box.
        """
        new_resolution = np.asarray(new_resolution)
        if (new_resolution == self.resolution).all():
            return

        if (new_resolution > self.resolution).all():
            factor = (new_resolution // self.resolution)
            self.resolution = new_resolution
            self.box[:] *= factor
            self.nonzero_box[:] *= factor
        elif (new_resolution <= self.resolution).all():
            factor = (self.resolution // new_resolution)
            self.resolution = new_resolution
            self.box[:] //= factor
            self.nonzero_box[:] //= factor
        else:
            raise RuntimeError("Can't change to a resolution that is bigger in some axes and smaller in others.")


    def get_fullres_mask(self, requested_box_fullres):
        """
        Sample a subvolume of the mask, using full-resolution
        coordinates and returning a full-resolution mask subvolume.

        This means creating an array of the requested shape, and overwriting
        the portion of it that intersects (if any) with our stored lowres_mask.

        Any box can be requested.  If the requested box does not intersect
        with the stored voxels, the returned mask will be zeros.
        """
        # Here's a guide to the variable names below:
        #
        # +-----------------------+
        # | stored                |
        # | (self.lowres_mask)    |
        # |      +----------------+--------+
        # |      |  intersecting  |        |
        # |      |  ("clipped")   |        |
        # +------+----------------+        |
        #        |              requested  |
        #        |              ("result") |
        #        +-------------------------+

        req_box_fullres = np.asarray(requested_box_fullres)
        assert (req_box_fullres % self.resolution == 0).all(), \
            "FIXME: Currently this function requires that the requested box is aligned to the lowres grid."

        req_shape_fullres = req_box_fullres[1] - req_box_fullres[0]
        result_mask_fullres = np.zeros(req_shape_fullres, bool)

        # Unless otherwise noted, all boxes/masks below are in low-res coordinates/voxels.
        req_box = req_box_fullres // self.resolution
        stored_box = (self.box // self.resolution)

        clipped_box = box_intersection(req_box, stored_box)
        if (clipped_box[1] <= clipped_box[0]).any():
            # No intersection; return zeros
            return result_mask_fullres

        # Compute relative boxes as shown in diagram above.
        clipped_within_stored = clipped_box - stored_box[0]
        clipped_within_req = clipped_box - req_box[0]

        # Extract intersecting region from within stored voxels
        clipped_mask = self.lowres_mask[box_to_slicing(*clipped_within_stored)]

        # Use the view_as_blocks() trick to create a 6D view,
        # which we will write to via broadcasting from low-res voxels
        result_mask_view = view_as_blocks(result_mask_fullres, tuple(self.resolution))

        # Overwrite within result view (broadcasting achieves upsampling here)
        slicing = (Ellipsis,) + (None,)*self.lowres_mask.ndim
        result_mask_view[box_to_slicing(*clipped_within_req)] = clipped_mask[slicing]

        # Return the full-res voxels
        return result_mask_fullres


    def sparse_boxes( self, brick_grid, halo=0, return_logical_boxes=False ):
        """
        Overlay a coarse grid (brick_grid) on top of this SparseBlockMask
        and extract the list of non-empty boxes from the given coarse grid.

        Args:
            brick_grid:
                The desired grid to use for the output.
                Does not need to be equivalent to the lowres grid that this SBM corresponds to.

            halo:
                If nonzero, expand each box by the given width in all dimensions.
                Note: This will result in boxes that are wider than the brick grid's native block shape.

            return_logical_boxes:
                If True, the result is returned as a list of full-size "logical" boxes.
                Otherwise, each box is shrunken to the minimal size while still
                encompassing all non-zero mask voxels with its grid box (i.e. a physical box),
                plus halo, if given.
                Note: It is not valid to use this option if halo is nonzero.

        Returns:
            boxes, shape=(N,2,3) of non-empty bricks, as indicated by block_mask.
        """
        if brick_grid is None:
            brick_grid = Grid(self.resolution)

        if not isinstance(brick_grid, Grid):
            assert isinstance(brick_grid, collections.abc.Iterable)
            brick_grid = Grid(brick_grid)

        assert (brick_grid.modulus_offset == (0,0,0)).all(), \
            "TODO: This function doesn't yet support brick grids with non-zero offsets"
        assert ((brick_grid.block_shape % self.resolution) == 0).all(), \
            "Brick grid must be a multiple of the block grid"
        assert not (halo > 0 and return_logical_boxes), \
            "The return_logical_boxes option makes no sense if halo > 0"

        block_mask_box = np.asarray(self.box)

        lowres_brick_grid = Grid( brick_grid.block_shape // self.resolution )
        lowres_block_mask_box = block_mask_box // self.resolution

        lowres_logical_boxes = boxes_from_grid(lowres_block_mask_box, lowres_brick_grid)
        lowres_clipped_boxes = boxes_from_grid(lowres_block_mask_box, lowres_brick_grid, clipped=True)

        lowres_boxes = []
        for logical_lowres_box, clipped_lowres_box in zip(lowres_logical_boxes, lowres_clipped_boxes):
            box_within_mask = clipped_lowres_box - lowres_block_mask_box[0]
            brick_mask = self.lowres_mask[box_to_slicing(*box_within_mask)]
            brick_coords = np.transpose(brick_mask.nonzero()).astype(np.int32)
            if len(brick_coords) == 0:
                continue

            if return_logical_boxes:
                lowres_boxes.append( logical_lowres_box )
            else:
                # Find the smallest box that still encompasses the non-zero
                # lowres voxels in this brick (denoted by brick_coords)
                physical_lowres_box = ( brick_coords.min(axis=0),
                                        brick_coords.max(axis=0) + 1 )

                # Translate back to global coordinates.
                # Offset by this brick's location within the overall mask volume,
                # and by the mask volume's location in global coordinates.
                physical_lowres_box += box_within_mask[0] + lowres_block_mask_box[0]

                lowres_boxes.append( physical_lowres_box )

        if len(lowres_boxes) == 0:
            nonempty_boxes = np.zeros((0,2,3), dtype=np.int32)
        else:
            nonempty_boxes = np.array(lowres_boxes, dtype=np.int32) * self.resolution

            halo_shape = np.zeros((3,), dtype=np.int32)
            halo_shape[:] = halo
            if halo_shape.any():
                nonempty_boxes[:] += (-halo_shape, halo_shape)

        return nonempty_boxes
