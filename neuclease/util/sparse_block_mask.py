from collections.abc import Collection
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

        self.box = np.asarray(box)  # full-res

        self.resolution = resolution
        if isinstance(self.resolution, Collection):
            self.resolution = np.asarray(resolution)
        else:
            self.resolution = np.array( [resolution]*lowres_mask.ndim )

        assert self.resolution.ndim == 1
        assert len(self.resolution) == lowres_mask.ndim

        assert (((self.box[1] - self.box[0]) // self.resolution) == self.lowres_mask.shape).all(), \
            f"Inconsistent mask shape ({lowres_mask.shape}) and box {self.box.tolist()} for the given resolution ({resolution}).\n"\
            "Note: box should be specified in FULL resolution coordinates."

        # Nonzero box (full-res coordinates)
        self.nonzero_box = compute_nonzero_box(self.lowres_mask)
        self.nonzero_box *= resolution
        self.nonzero_box += self.box[0]

    @classmethod
    def create_empty(cls, resolution, corner=None):
        if isinstance(resolution, Collection):
            D = len(resolution)
        else:
            D = 3

        if corner is None:
            corner = [0] * D
        empty_box = np.array([corner, corner])
        empty_mask = np.zeros((0,)*len(empty_box[0]), dtype=bool)
        return SparseBlockMask(empty_mask, empty_box, resolution)

    @classmethod
    def create_from_sbm_box(cls, sbm, sub_box):
        """
        Create a SparseBlockMask by extracting a portion of an existing SparseBlockMask.
        If sub_box isn't aligned to the given sbm resolution, it will be expanded to make it aligned.
        """
        sub_box = round_box(sub_box, sbm.resolution, 'out')

        new_box = box_intersection(sub_box, sbm.box)
        if (new_box[0] >= new_box[1]).any():
            return None  # The sub_box doesn't overlap with the sbm box

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
                ndarray, shape (N,D), indicating the 'blocks' that are masked.
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

    def sparse_boxes( self, brick_grid, return_logical_boxes=False ):
        """
        Overlay a coarse grid (brick_grid) on top of this SparseBlockMask
        and extract the list of non-empty boxes from the given coarse grid.

        Args:
            brick_grid:
                The desired grid to use for the output, in full-res coordinates.
                Does not need to be equivalent to the lowres grid that this SBM corresponds to.
                May include a halo, in which case the halo is applied AFTER the boxes are computed
                and clipped to the overall bounding box of the SBM.

            return_logical_boxes:
                If True, the result is returned as a list of full-size "logical" boxes.
                Otherwise, each box is shrunken to the minimal size while still
                encompassing all non-zero mask voxels with its grid box (i.e. a physical box),
                plus halo, if the given Grid has one.

        Returns:
            boxes, shape=(N,2,D) of non-empty bricks, as indicated by block_mask.
        """
        if brick_grid is None:
            brick_grid = Grid(self.resolution)

        if np.issubdtype(type(brick_grid), np.integer):
            brick_grid = Grid([brick_grid]*self.lowres_mask.ndim)

        if not isinstance(brick_grid, Grid):
            assert isinstance(brick_grid, Collection)
            brick_grid = Grid(brick_grid)

        sbm_fullres_box = self.nonzero_box
        sbm_lowres_box = self.nonzero_box // self.resolution
        sbm_lowres_mask = extract_subvol(self.lowres_mask, (self.nonzero_box - self.box[0]) // self.resolution)

        logical_boxes = boxes_from_grid(sbm_fullres_box, brick_grid)
        clipped_boxes = boxes_from_grid(sbm_fullres_box, brick_grid, clipped=True)

        boxes = []
        for logical_box, clipped_box in zip(logical_boxes, clipped_boxes):
            lowres_clipped_box = round_box(clipped_box, self.resolution, 'out') // self.resolution
            lowres_box_within_mask = lowres_clipped_box - sbm_lowres_box[0]
            brick_lowres_mask = extract_subvol(sbm_lowres_mask, lowres_box_within_mask)
            if brick_lowres_mask.sum() == 0:
                continue

            if return_logical_boxes:
                boxes.append( logical_box )
            else:
                # Find the smallest box that still encompasses the non-zero
                # lowres voxels in this brick (denoted by brick_coords)
                brick_lowres_coords = np.array(brick_lowres_mask.nonzero()).T.astype(np.int32)
                physical_lowres_box = np.array([ brick_lowres_coords.min(axis=0),
                                                 brick_lowres_coords.max(axis=0) + 1] )

                # Translate back to global coordinates.
                # Offset by this brick's location within the overall nonzero mask,
                # and again by the nonzero-mask's location in global SBM coordinates.
                physical_lowres_box += lowres_box_within_mask[0] + sbm_lowres_box[0]
                physical_fullres_box = physical_lowres_box * self.resolution

                assert (physical_fullres_box[0] < physical_fullres_box[1]).all()
                physical_fullres_box = box_intersection(physical_fullres_box, clipped_box)

                boxes.append( physical_fullres_box )

        D = self.lowres_mask.ndim
        if len(boxes) == 0:
            return np.zeros((0,2,D), dtype=np.int32)

        nonempty_boxes = np.array(boxes, dtype=np.int32)
        if brick_grid.halo_shape.any() and not return_logical_boxes:
            nonempty_boxes[:] += (-brick_grid.halo_shape, brick_grid.halo_shape)

        return nonempty_boxes
