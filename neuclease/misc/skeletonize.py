import json
import threading
from functools import partial
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import vigra
import skimage.measure
import networkx as nx
from scipy.spatial import KDTree

from neuclease.util import compute_parallel, compute_nonzero_box, extract_subvol, box_intersection
from neuclease.util.segmentation import distance_transform
from neuclease.util.skeleton import skeleton_to_neuroglancer
from neuclease.dvid import fetch_sparsevol, fetch_instance_info, fetch_lastmod
from neuclease.dvid.rle import blockwise_masks_from_ranges

class HaloComponentTracker:
    """
    Resolves connected-component equivalences across the boundaries between
    neighboring blocks during blockwise skeletonization.

    Each block computes a local connected-component labeling of its (halo-padded)
    mask.  Because neighboring blocks overlap in a halo region, the same physical
    component may be assigned different local labels in each block.  This tracker
    compares the labels present in the mutual overlap slab between a pair of
    neighboring blocks and records which ``(block_id, local_label)`` pairs refer
    to the same physical component.

    A face is registered for a block only if a neighbor block actually exists on
    that side (see ``skeleton_coords()``).  Neighbor presence is symmetric, so
    every shared boundary is registered exactly twice: the first registration is
    stored in ``self.pending``; the second one finds its partner, compares the two
    label slabs, records the equivalences, and discards both slabs.  Consequently
    ``self.pending`` is empty once all blocks have been processed.

    After all blocks are processed, ``resolve()`` maps every skeleton point's
    ``(block_id, local_label)`` to a contiguous global connected-component id.

    Thread-safe: ``register()`` may be called concurrently from multiple threads.
    """
    def __init__(self):
        self.uf = nx.utils.UnionFind()
        self.pending = {}
        self.lock = threading.Lock()

    def register(self, block_id, face_entries):
        """
        Register the overlap-slab labels for one block's faces.

        Args:
            block_id:
                The id of the block being registered.
            face_entries:
                A dict of ``{boundary_key: (slab, slab_box)}`` where ``slab`` is a
                3D array of local component labels covering that face's mutual
                overlap region (possibly empty if the block's content didn't reach
                the overlap region) and ``slab_box`` is its box in global
                coordinates.  ``boundary_key`` is block-independent, so the two
                blocks sharing a boundary produce the same key.
        """
        with self.lock:
            for key, (slab, slab_box) in face_entries.items():
                if key not in self.pending:
                    self.pending[key] = (block_id, slab, slab_box)
                    continue

                other_id, other_slab, other_box = self.pending.pop(key)
                self._union_overlap(block_id, slab, slab_box, other_id, other_slab, other_box)

    def _union_overlap(self, id_a, slab_a, box_a, id_b, slab_b, box_b):
        """
        Union the (block_id, label) pairs that co-occur on foreground voxels
        within the physical intersection of two neighboring blocks' face slabs.
        """
        if slab_a.size == 0 or slab_b.size == 0:
            return

        inter = box_intersection(box_a, box_b)
        if (inter[1] <= inter[0]).any():
            return

        sub_a = extract_subvol(slab_a, inter - box_a[0])
        sub_b = extract_subvol(slab_b, inter - box_b[0])

        both = (sub_a != 0) & (sub_b != 0)
        if not both.any():
            return

        pairs = np.unique(np.stack([sub_a[both], sub_b[both]], axis=1), axis=0)
        for label_a, label_b in pairs:
            self.uf.union((id_a, int(label_a)), (id_b, int(label_b)))

    def resolve(self, block_ids, local_labels):
        """
        Map each skeleton point's ``(block_id, local_label)`` to a contiguous
        global connected-component id, honoring the equivalences discovered during
        registration.  Must be called after all blocks have been registered.

        Returns:
            An int32 array of global connected-component ids, one per point.
        """
        global_ids = {}
        cc = np.empty(len(block_ids), dtype=np.int32)
        for i, key in enumerate(zip(block_ids.tolist(), local_labels.tolist())):
            root = self.uf[key]
            if root not in global_ids:
                global_ids[root] = len(global_ids)
            cc[i] = global_ids[root]
        return cc


def skeletonize_neuron(
    dvid_server,
    uuid,
    segmentation_instance,
    body,
    scale=2,
    block_shape=(128, 128, 128),
    halo=16,
    closing_radius=5,
    return_radii=False,
    heal_max_distance=None,
    voxel_size_xyz=None,
    first_node=1,
    tracker=None,
    format='pandas',
    output_path=None,
    threads=12
):
    """
    Skeletonize a body in DVID by from its sparsevol.

    The body's sparsevol is fetched from DVID at the given scale.  The sparsevol is processed in big
    overlapping chunks ("blocks"), each of which is skeletonized with scikit-image after a bit of
    preprocessing (morphological closing, followed by hole filling).

    Args:
        dvid_server:
            The DVID server to connect to.
        uuid:
            The UUID of the DVID instance to connect to.
        segmentation_instance:
            The name of the segmentation volume to use.
        body:
            The ID of the body to skeletonize.
        scale:
            The scale at which to fetch the sparsevol.
        block_shape:
            The processing will be split into to blocks of this shape (not including the halo).
        halo:
            The amount of overlap between blocks.
            Since we skeletonize each block independently, this halo minimizes edge effects between blocks.
            The skeleton points in the halo region are discarded before returning.
        closing_radius:
            The radius of the morphological closing operation, applied before hole-filling and skeletonization.
        return_radii:
            If True, also compute an approximate "radius" for each skeleton point.
            This is computed via the distance transform within the neuron mask.
            The radii account for anisotropic voxels (see voxel_size_xyz) and are
            in physical units (e.g. nanometers).  (Radii are always computed for
            format='swc', regardless of this setting.)
        heal_max_distance:
            Optional. If provided, the connected components of the skeleton are
            reconnected to each other via bridging edges no longer than this
            distance (see treeify_coords).  This can recover connections that were
            missed across block boundaries without joining components that are
            genuinely far apart.  If None, distinct connected components are left
            unconnected.

            NOTE: This distance is specified in NANOMETERS (physical units), not
            voxels, so it is independent of the scale at which the sparsevol was
            fetched.  The neuron's physical voxel size is used to convert.
        voxel_size_xyz:
            Optional. The physical voxel size (nm) in XYZ order, as reported by
            DVID (info['Extended']['VoxelSize']).  Used to make radius estimation
            anisotropy-aware and to interpret heal_max_distance in nanometers.
            If not provided, it is fetched from DVID automatically when needed
            (i.e. when return_radii is True or heal_max_distance is given).
        tracker:
            Optional. A HaloComponentTracker used to reconcile per-block
            connected-component labels across block boundaries.  If not provided,
            a fresh one is created.
        format:
            The format of the returned skeleton:
              'pandas':       A DataFrame (see below).  This is the default.
              'swc':          SWC-formatted text (str), with a comment header
                              recording the segmentation uuid, instance, mutation
                              id, generation parameters, and creation timestamp.
              'neuroglancer': The binary "precomputed" skeleton format (bytes),
                              with vertex positions in physical nanometers.
        output_path:
            Optional. If given (only valid for the 'swc' and 'neuroglancer'
            formats), also write the result to this file path.
        threads:
            Blocks are processed in parallel using a thread pool of this size.

    Returns:
        Depends on 'format':
          'pandas':       A DataFrame with columns
                          ['node', 'x', 'y', 'z', 'parent', 'cc', 'radius']
                          (the 'radius' column is omitted if return_radii=False).
          'swc':          SWC text (str).
          'neuroglancer': The precomputed skeleton (bytes).
    """
    assert format in ('pandas', 'swc', 'neuroglancer'), \
        f"Unknown format: {format}"

    # SWC requires radii; neuroglancer needs the voxel size to scale to nm.
    need_radii = return_radii or (format == 'swc')
    need_voxel_size = need_radii or (heal_max_distance is not None) or (format == 'neuroglancer')

    dvid_seg = (dvid_server, uuid, segmentation_instance)

    # Perform the DVID fetches here (sparsevol, and, if needed, voxel size and
    # mutation id), then hand off to skeletonize_neuron_from_ranges() for the
    # (fetch-free) skeletonization.  Callers who want to manage the fetch
    # themselves (e.g. within a resource-manager context) can call that function
    # directly instead.
    if voxel_size_xyz is None and need_voxel_size:
        voxel_size_xyz = fetch_instance_info(*dvid_seg)['Extended']['VoxelSize']

    ranges = fetch_sparsevol(*dvid_seg, body, scale=scale, format='ranges')

    mutid = None
    if format == 'swc':
        mutid = fetch_lastmod(*dvid_seg, body)["mutation id"]

    return skeletonize_neuron_from_ranges(
        ranges,
        scale=scale,
        block_shape=block_shape,
        halo=halo,
        closing_radius=closing_radius,
        return_radii=return_radii,
        heal_max_distance=heal_max_distance,
        voxel_size_xyz=voxel_size_xyz,
        first_node=first_node,
        tracker=tracker,
        format=format,
        output_path=output_path,
        threads=threads,
        uuid=uuid,
        segmentation_instance=segmentation_instance,
        mutid=mutid,
    )


def skeletonize_neuron_from_ranges(
    ranges,
    scale=2,
    block_shape=(128, 128, 128),
    halo=16,
    closing_radius=5,
    return_radii=False,
    heal_max_distance=None,
    voxel_size_xyz=None,
    first_node=1,
    tracker=None,
    format='pandas',
    output_path=None,
    threads=12,
    *,
    uuid=None,
    segmentation_instance=None,
    mutid=None,
):
    """
    Like skeletonize_neuron(), but operates on a pre-fetched sparsevol (in DVID
    'ranges' format) and performs no DVID access of its own.  This lets the caller
    fetch the sparsevol separately -- e.g. within a resource-manager context --
    without holding that resource for the (CPU-bound) duration of skeletonization.

    Any metadata that skeletonize_neuron() would normally fetch from DVID must be
    supplied by the caller when the corresponding output is requested:

        - voxel_size_xyz: needed for anisotropy-aware radii, nm-based
          heal_max_distance, and neuroglancer output.  If omitted, isotropic unit
          voxels are assumed.
        - uuid, segmentation_instance, mutid: required for format='swc'
          (they populate the SWC comment header).

    See skeletonize_neuron() for a description of the remaining arguments and the
    return value.
    """
    assert format in ('pandas', 'swc', 'neuroglancer'), \
        f"Unknown format: {format}"
    assert output_path is None or format in ('swc', 'neuroglancer'), \
        "output_path is only supported for the 'swc' and 'neuroglancer' formats"
    if format == 'swc':
        assert None not in (uuid, segmentation_instance, mutid), \
            "For format='swc', you must provide uuid, segmentation_instance, and mutid."

    need_radii = return_radii or (format == 'swc')

    # DVID reports VoxelSize in XYZ order; the rest of this code uses ZYX.
    pixel_pitch_zyx = ()
    anisotropy_zyx = None
    if voxel_size_xyz is not None:
        anisotropy_zyx = np.array(voxel_size_xyz, dtype=np.float64)[::-1]
        pixel_pitch_zyx = tuple(float(v) for v in anisotropy_zyx)

    all_coords, radii, block_ids, point_labels, tracker = skeleton_coords_from_ranges(
        ranges,
        scale=scale,
        block_shape=block_shape,
        halo=halo,
        closing_radius=closing_radius,
        return_radii=need_radii,
        tracker=tracker,
        pixel_pitch_zyx=pixel_pitch_zyx,
        threads=threads,
    )
    cc_ids = tracker.resolve(block_ids, point_labels)
    df = treeify_coords(
        all_coords, radii, cc_ids=cc_ids,
        heal_max_distance=heal_max_distance, anisotropy_zyx=anisotropy_zyx, first_node=first_node
    )

    if format == 'pandas':
        return df

    if format == 'swc':
        swc = _skeleton_df_to_swc(df, uuid, segmentation_instance, mutid, scale, block_shape, halo, closing_radius)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(swc)
        return swc

    # format == 'neuroglancer'
    orig_resolution_nm = voxel_size_xyz if voxel_size_xyz is not None else 8
    return skeleton_to_neuroglancer(df, orig_resolution_nm=orig_resolution_nm, output_path=output_path)


def _skeleton_df_to_swc(df, uuid, seg_instance, mutid, scale, block_shape, halo, closing_radius):
    """
    Serialize a skeleton DataFrame (as produced by skeletonize_neuron) to SWC text,
    with a header of comment lines describing how the skeleton was generated.

    The header preserves the NeuTu-style comment keys that downstream tools still
    rely on -- 'mutation id', 'downresLevel' (the scale), 'dataName' (the
    segmentation instance), and 'uuid' -- and adds a line recording the full set
    of skeletonization parameters and the creation timestamp.
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # NOTE: The 'mutation id' comment is parsed elsewhere (e.g. the derived-data
    # update script) via a regex that expects json.dumps()'s exact spacing, so
    # don't reformat it.
    header_lines = [
        "#Generated by neuclease (https://github.com/janelia-flyem/neuclease)",
        "#$" + json.dumps({"downresLevel": scale, "uuid": uuid, "dataName": seg_instance}),
        "#$" + json.dumps({"mutation id": mutid}),
        "#$" + json.dumps({
            "scale": scale,
            "block_shape": list(block_shape) if hasattr(block_shape, '__len__') else block_shape,
            "halo": halo,
            "closing_radius": closing_radius,
            "timestamp": now,
        }),
    ]

    swc_df = df.assign(kind=0)[['node', 'kind', *'xyz', 'radius', 'parent']]
    body_csv = swc_df.to_csv(sep=' ', header=False, index=False)
    return '\n'.join(header_lines) + '\n' + body_csv


def skeleton_coords(
    dvid_server,
    uuid,
    segmentation_instance,
    body,
    scale=2,
    block_shape=(128, 128, 128),
    halo=16,
    closing_radius=5,
    trim_halo=True,
    return_radii=False,
    tracker=None,
    pixel_pitch_zyx=(),
    threads=12
):
    """
    For a given body in DVID, produce a set of coordinates which can be joined together to form a skeleton.

    The body's sparsevol is fetched from DVID at the given scale.  The sparsevol is processed in big
    overlapping chunks ("blocks"), each of which is skeletonized with scikit-image after a bit of
    preprocessing (morphological closing, followed by hole filling).

    Args:
        dvid_server:
            The DVID server to connect to.
        uuid:
            The UUID of the DVID instance to connect to.
        segmentation_instance:
            The name of the segmentation volume to use.
        body:
            The ID of the body to skeletonize.
        scale:
            The scale at which to fetch the sparsevol.
        block_shape:
            The processing will be split into to blocks of this shape (not including the halo).
        halo:
            The amount of overlap between blocks.
            Since we skeletonize each block independently, this halo minimizes edge effects between blocks.
            The skeleton points in the halo region are discarded before returning unless trim_halo is False.
        closing_radius:
            The radius of the morphological closing operation, applied before hole-filling and skeletonization.
        trim_halo:
            If False, DON'T discard coordinates in the halo region.
            (Useful for debugging.)
        return_radii:
            If True, also return an approximate "radius" for each skeleton point.
            This is computed via the distance transform within the neuron mask.
        tracker:
            Optional. A HaloComponentTracker instance to use for reconciling the
            per-block connected-component labels across block boundaries.
            If not provided, a fresh one is created.  The tracker is returned so
            the caller can use it to resolve global component ids for each point.
        pixel_pitch_zyx:
            Optional. The physical size of a voxel, in ZYX order, used when
            computing the distance transform for radius estimation (see
            return_radii).  Provide it in native (scale-0) units; the resulting
            radii are scaled by 2**scale along with the coordinates.  An empty
            tuple (the default) assumes isotropic unit voxels.
        threads:
            Blocks are processed in parallel using a thread pool of this size.
    Returns:
        (coords_zyx, radii, block_ids, point_labels, tracker)
            coords_zyx has shape (N, 3) and contains the skeleton points. Note that the coordinates are given in Z,Y,X order.
            If return_radii is True, then radii has shape (N,) and contains the mask radius for each skeleton point.
            If return_radii is False, then radii is None.
            block_ids has shape (N,) and gives the source block of each point.
            point_labels has shape (N,) and gives the (block-local) connected-component
            label of each point.  Together, (block_ids, point_labels) uniquely
            identifies a per-block component, which tracker.resolve() maps to a
            global connected-component id.
    """
    dvid_seg = (dvid_server, uuid, segmentation_instance)
    ranges = fetch_sparsevol(*dvid_seg, body, scale=scale, format='ranges')
    return skeleton_coords_from_ranges(
        ranges,
        scale=scale,
        block_shape=block_shape,
        halo=halo,
        closing_radius=closing_radius,
        trim_halo=trim_halo,
        return_radii=return_radii,
        tracker=tracker,
        pixel_pitch_zyx=pixel_pitch_zyx,
        threads=threads,
    )


def skeleton_coords_from_ranges(
    ranges,
    scale=2,
    block_shape=(128, 128, 128),
    halo=16,
    closing_radius=5,
    trim_halo=True,
    return_radii=False,
    tracker=None,
    pixel_pitch_zyx=(),
    threads=12
):
    """
    Like skeleton_coords(), but operates on a pre-fetched sparsevol (in DVID
    'ranges' format) instead of fetching it from DVID.  This lets the caller
    perform the (DVID) fetch separately -- for example, within a resource-manager
    context -- without holding that resource for the duration of skeletonization.

    See skeleton_coords() for a description of the arguments and return value.
    """
    if tracker is None:
        tracker = HaloComponentTracker()

    if not hasattr(block_shape, '__len__'):
        block_shape = 3 * (block_shape,)
    block_shape = np.asarray(block_shape)

    mask_boxes, mask_iterator = blockwise_masks_from_ranges(ranges, block_shape=block_shape, halo=halo)
    mask_boxes = np.asarray(mask_boxes)

    # The set of occupied (non-empty) block grid coordinates, used to determine
    # which faces of a block have a neighbor block (and thus a shared boundary to
    # reconcile).  A face without a neighbor is never registered with the tracker.
    grid_coords = (mask_boxes[:, 0] + halo) // block_shape
    occupied = {tuple(int(c) for c in gc) for gc in grid_coords}

    def threadsafe_generator(g):
        lock = threading.Lock()
        with lock:
            yield from g

    # I don't think it's safe for multiple threads to use
    # mask_iterator simultaneously, so I wrapped it with a lock.
    mask_iterator = threadsafe_generator(mask_iterator)

    # The function to process one block.
    _process_block = partial(
        _skeleton_coords_for_mask,
        block_shape=block_shape,
        occupied=occupied,
        tracker=tracker,
        closing_radius=closing_radius,
        halo=halo,
        trim_halo=trim_halo,
        return_radii=return_radii,
        pixel_pitch_zyx=pixel_pitch_zyx
    )

    # Use a thread pool.  Each item is (box, mask, block_id).
    results = compute_parallel(
        _process_block,
        zip(mask_boxes, mask_iterator, range(len(mask_boxes))),
        starmap=True,
        total=len(mask_boxes),
        leave_progress=True,
        threads=threads,
    )

    all_coords, all_radii, all_block_ids, all_point_labels = zip(*results)
    all_coords = np.concatenate(all_coords)
    all_coords *= 2**scale

    # Offset by half a downscaled voxel to place the points into the middle of
    # the downscaled voxel.  (At scale 0 there's no downscaling, so no offset --
    # and 2**(scale-1) would be a non-integer 0.5, which the integer coords can't hold.)
    if scale > 0:
        all_coords += 2**(scale-1)

    all_block_ids = np.concatenate(all_block_ids)
    all_point_labels = np.concatenate(all_point_labels)

    if not return_radii:
        return all_coords, None, all_block_ids, all_point_labels, tracker

    all_radii = np.concatenate(all_radii)
    all_radii *= 2**scale
    return all_coords, all_radii, all_block_ids, all_point_labels, tracker


def _skeleton_coords_for_mask(box, mask, block_id, block_shape, occupied, tracker,
                              closing_radius, halo, trim_halo, return_radii, pixel_pitch_zyx=()):
    """
    Helper for skeleton_coords().  Processes a single block of the sparsevol.

    In addition to the skeleton coordinates, this computes the connected-component
    labeling of the block's mask and assigns a (block-local) component label to
    each skeleton point.  The overlap slabs shared with neighboring blocks are
    registered with ``tracker`` so that per-block labels can be reconciled into
    global connected components afterward (see HaloComponentTracker).

    Returns:
        (coords, radii, block_ids, point_labels)
        where block_ids is a constant array (== block_id) and point_labels holds
        the block-local component label for each skeleton point.
    """
    box = np.asarray(box)
    block_shape = np.asarray(block_shape)

    # We only care about the part that actually contains non-zero values.
    # (Note that the closing operation would not result in pixels outside the bounding box.)
    nz_box = compute_nonzero_box(mask)
    mask = extract_subvol(mask, nz_box)

    if closing_radius > 0:
        # After trying skimage and scipy, vigra seems ~2x faster at closing.
        mask = vigra.filters.multiBinaryClosing(mask, radius=closing_radius)

    mask = fill_holes(mask)

    # Connected components of the (post-fill) mask, using face-connectivity, which
    # is fast and sufficient here (we're already skeletonizing an approximation).
    # Label 0 is background.  Skeleton points and halo overlap slabs are keyed by
    # these labels, which are unique only within this block.
    cc = skimage.measure.label(mask, connectivity=1)

    skeleton = skimage.morphology.skeletonize(mask)
    coords = np.array(skeleton.nonzero()).T

    # Component label for each skeleton point, in the trimmed frame (before offset).
    point_labels = cc[tuple(coords.T)].astype(np.int32)

    if return_radii:
        # pixel_pitch_zyx accounts for anisotropic voxels so the radius estimate
        # reflects true distance-to-boundary.  It is given here in native (scale-0)
        # nm-per-voxel; the resulting radii are scaled by 2**scale (as are the
        # coords) in skeleton_coords(), yielding physical nanometers.
        dt = distance_transform(mask, pixel_pitch=pixel_pitch_zyx)
        radii = np.array(dt[tuple(coords.T)])

    # Register this block's halo overlap slabs so that component labels can be
    # matched to those of neighboring blocks.
    grid_coord = (box[0] + halo) // block_shape
    cc_origin = box[0] + nz_box[0]
    cc_box = np.array([cc_origin, cc_origin + cc.shape])
    tracker.register(
        block_id,
        _halo_face_entries(cc, cc_box, box, grid_coord, occupied, halo)
    )

    coords += box[0] + nz_box[0]
    block_ids = np.full(len(coords), block_id, dtype=np.int32)

    if trim_halo:
        non_halo = (coords >= box[0] + halo).all(axis=1) & (coords < box[1] - halo).all(axis=1)
        coords = coords[non_halo]
        point_labels = point_labels[non_halo]
        block_ids = block_ids[non_halo]
        if return_radii:
            radii = radii[non_halo]

    if return_radii:
        return coords, radii, block_ids, point_labels
    else:
        return coords, None, block_ids, point_labels


def _halo_face_entries(cc, cc_box, box, grid_coord, occupied, halo):
    """
    Build the ``{boundary_key: (slab, slab_box)}`` dict for the faces of a block
    that have a neighbor block.

    For each of the 6 faces whose neighbor exists (per ``occupied``), extract the
    slice of the label image ``cc`` that lies within the mutual overlap region
    shared with that neighbor.  The overlap region is ``2*halo`` wide along the
    face axis and spans the full block on the other two axes.  The slab is clipped
    to this block's (trimmed) content, so it may be empty if the content doesn't
    reach the face.

    ``boundary_key`` is constructed so that this block's face and the neighbor's
    opposite face produce the SAME key: it uses the block's grid coords with the
    face-axis component set to the higher of the two coords sharing the boundary.
    """
    entries = {}
    for axis in range(3):
        for direction in (-1, 1):
            neighbor = list(grid_coord)
            neighbor[axis] += direction
            if tuple(neighbor) not in occupied:
                continue

            # Mutual overlap region shared with the neighbor, in global coords.
            overlap = box.copy()
            if direction < 0:
                overlap[1, axis] = box[0, axis] + 2 * halo
            else:
                overlap[0, axis] = box[1, axis] - 2 * halo

            slab_box = box_intersection(cc_box, overlap)
            if (slab_box[1] <= slab_box[0]).any():
                slab = cc[:0, :0, :0]
                slab_box = np.array([cc_box[0], cc_box[0]])
            else:
                slab = extract_subvol(cc, slab_box - cc_box[0])

            # Key on the boundary plane (higher of the two adjacent block coords).
            bkey = list(grid_coord)
            if direction > 0:
                bkey[axis] += 1
            key = (axis, int(bkey[0]), int(bkey[1]), int(bkey[2]))
            entries[key] = (slab, slab_box)
    return entries


def fill_holes(mask):
    """
    Fill holes in a binary mask.
    Computes the connected components of the mask (inside and out),
    and fills all non-masked regions except for the largest one
    (which we presume is the true background).
    """
    if mask.all():
        return mask
    cc = skimage.measure.label(mask.astype(np.uint8), background=99, connectivity=1)
    nonmask_cc_counts = np.bincount(cc[~mask])
    background_cc = np.argmax(nonmask_cc_counts)
    return (cc != background_cc)


def treeify_coords(all_coords, radii=None, cc_ids=None, heal_max_distance=None, anisotropy_zyx=None, first_node=1):
    """
    Given an array of coordinates, join them into a minimum spanning tree.
    We only consider possible edges between each point and its N closest neighbors,
    so it is possible for the output to consist of multiple connected components.

    Args:
        all_coords:
            An array of shape (N, 3) containing the coordinates of the points to join.
        radii:
            Optional column of radii to include in the result as column.
            The radii are not used in the computation of the MST.
        cc_ids:
            Optional array of shape (N,) assigning each point to a connected
            component (e.g. as produced by ``HaloComponentTracker.resolve()``).
            If provided, nearest-neighbor edges are only considered BETWEEN points
            that share a common cc_id, so the tree is never allowed to join points
            from distinct components.  (Components may still be reconnected by the
            ``heal_max_distance`` pass, below.)
        heal_max_distance:
            Optional. If provided, perform a final pass that reconnects the
            resulting components to each other, but only via bridging edges whose
            length does not exceed this distance.  This can recover connections
            that were missed across block boundaries without joining components
            that are genuinely far apart.  The distance is measured in the same
            units as ``anisotropy_zyx`` (e.g. nanometers if ``anisotropy_zyx`` is
            the physical voxel size); if ``anisotropy_zyx`` is not given, it is
            measured in raw coordinate (voxel) units.
        anisotropy_zyx:
            Optional. The physical size of one coordinate unit along each axis, in
            ZYX order (e.g. the nm-per-voxel voxel size).  When provided, all
            distances used internally (nearest-neighbor selection and the
            ``heal_max_distance`` threshold) are measured in these physical units,
            which matters for anisotropic data.  The output coordinates themselves
            are left unchanged (still in raw voxel units).
        first_node:
            Either 0 or 1, depending on whether you desired 0-based or 1-based indexing.
            (Either way, parentless nodes always have parent -1.)

    Returns:
        A pandas DataFrame with columns:
            ['node', 'x', 'y', 'z', 'parent', 'cc', 'radius']

        (If radii are not provided, the 'radius' column is omitted.)
    """
    # Select edges for every node's N closest neighbors
    # (Ignore the first "neighbor", which is the node itself.)
    # TODO:
    #   Consider using a max radius instead (or in addition to)
    #   of a fixed number of neighbors.
    num_neighbors = 7
    nodes = np.arange(len(all_coords))

    if len(all_coords) == 0:
        # No skeleton points at all (e.g. a tiny body that skeletonized to nothing).
        # Return a well-formed but empty result rather than choking downstream.
        cols = ['node', *'xyz', 'parent', 'cc']
        if radii is not None:
            cols.append('radius')
        return pd.DataFrame(columns=cols)

    # For anisotropic data, measure distances in physical units by scaling the
    # coordinates.  The output coordinates (below) still use the raw values.
    if anisotropy_zyx is None:
        dist_coords = all_coords
    else:
        dist_coords = all_coords * np.asarray(anisotropy_zyx, dtype=np.float64)

    if cc_ids is None:
        groups = [nodes]
    else:
        cc_ids = np.asarray(cc_ids)
        # Process each connected component separately so that no edge is ever
        # proposed between points of different components.
        groups = [np.where(cc_ids == c)[0] for c in np.unique(cc_ids)]

    edges = pd.concat(
        [_knn_edges(dist_coords, group, num_neighbors) for group in groups],
        ignore_index=True
    )

    # Load into nx.Graph and compute MST.
    # (Add all nodes first, so that isolated single-point components are retained.)
    g = nx.Graph()
    g.add_nodes_from(nodes.tolist())
    for row in edges.itertuples():
        g.add_edge(row.u, row.v, weight=row.distance)
    mst = nx.minimum_spanning_tree(g, weight='weight')

    # Optionally reconnect the resulting components, subject to a max bridging distance.
    if heal_max_distance is not None:
        _heal_graph(mst, dist_coords, heal_max_distance)

    dfs_edges = []
    for component in nx.connected_components(mst):
        root = min(component)
        dfs_edges.extend(nx.dfs_edges(mst, source=root))
        dfs_edges.append((-1, root))

    df = pd.DataFrame(dfs_edges, columns=['parent', 'node'])
    df[[*'zyx']] = all_coords[df['node']]

    df = df.set_index('node').sort_index()

    # Assign a component id to each node, based on the (possibly healed) tree.
    df['cc'] = np.int32(-1)
    for i, component in enumerate(nx.connected_components(mst)):
        df.loc[list(component), 'cc'] = i

    cols = ['node', *'xyz', 'parent', 'cc']
    if radii is not None:
        df['radius'] = radii[df.index]
        cols.append('radius')
    df = df.reset_index()[cols]

    if first_node != 0:
        # Switch from 0-based to 1-based indexing
        df.loc[df['node'] != -1, 'node'] += first_node
        df.loc[df['parent'] != -1, 'parent'] += first_node
    return df


def _knn_edges(all_coords, group, num_neighbors):
    """
    Construct candidate MST edges between each point in ``group`` and its nearest
    neighbors WITHIN the same group.  ``group`` is an array of node ids (indices
    into ``all_coords``).  Returns a DataFrame with columns ['u', 'v', 'distance']
    whose 'u'/'v' values are global node ids.
    """
    group = np.asarray(group)
    m = len(group)
    if m <= 1:
        return pd.DataFrame({
            'u': np.array([], dtype=np.int64),
            'v': np.array([], dtype=np.int64),
            'distance': np.array([], dtype=np.float64),
        })

    pts = all_coords[group]
    kdtree = KDTree(pts)

    # We can't ask for more neighbors than exist in this group.
    k = min(num_neighbors, m - 1)
    distances, neighbors = kdtree.query(pts, k=list(range(2, 2 + k)))

    # KDTree may return 1D arrays when k == 1; normalize to 2D.
    distances = distances.reshape(m, k)
    neighbors = neighbors.reshape(m, k)

    return pd.DataFrame({
        'u': np.repeat(group, k),
        'v': group[neighbors.ravel()],
        'distance': distances.ravel(),
    })


def _heal_graph(g, all_coords, max_distance):
    """
    Reconnect the connected components of graph ``g`` (in place) by adding
    bridging edges between them, but only where the bridge length does not exceed
    ``max_distance``.

    Uses the same fragment-quotient-MST strategy as neuprint's ``heal_skeleton()``:
    treat each component as a single node, connect components at their nearest
    points, take the MST of that (small) quotient graph, and add the corresponding
    fine-grained edges back to ``g`` (subject to max_distance).
    """
    components = [np.fromiter(c, dtype=np.int64) for c in nx.connected_components(g)]
    if len(components) <= 1:
        return

    # Larger fragments first, so each pairwise query is run against the larger set.
    components.sort(key=len, reverse=True)
    kdtrees = [KDTree(all_coords[idx]) for idx in components]

    frag_graph = nx.Graph()
    frag_graph.add_nodes_from(range(len(components)))
    for a, b in combinations(range(len(components)), 2):
        idx_a, idx_b = components[a], components[b]
        distances, locs_a = kdtrees[a].query(all_coords[idx_b])
        j = np.argmin(distances)
        i = locs_a[j]
        frag_graph.add_edge(
            a, b,
            node_a=int(idx_a[i]),
            node_b=int(idx_b[j]),
            distance=float(distances[j]),
        )

    for _u, _v, d in nx.minimum_spanning_edges(frag_graph, weight='distance', data=True):
        if d['distance'] <= max_distance:
            g.add_edge(d['node_a'], d['node_b'], weight=d['distance'])
