"""
Generate a neuroglancer multi-resolution (multi-LOD) Draco mesh for a single
object directly from a DVID sparse volume.

The mesh is produced one grid block at a time: the object's RLEs are split
into block-aligned per-block range descriptors, and each block is inflated
to a mask and meshed (marching cubes + optional smoothing/decimation)
independently, becoming a fragment of a single-LOD multires mesh object
written via ``vol2mesh.multires``.

Fragment generation can be parallelized over threads or processes via
``compute_parallel``.  Crucially, the parallel work-items are lightweight
per-block range descriptors (RLE-sized), not inflated 3D masks: each worker
inflates its own block's mask and frees it after meshing, so only a
pool-sized number of masks are ever resident at once (rather than inflating
the entire object up front).

``vol2mesh`` is imported lazily inside the functions, since it is not a
required dependency of neuclease.

Coordinate conventions
-----------------------
- ``block_shape`` and the DVID ``ranges`` are in ZYX order (the neuclease
  convention).
- The multires "stored model" coordinate space used here is full-resolution
  (scale-0) voxels, so the output ``info`` ``transform`` carries the
  voxel->nm scale.  ``voxel_size_nm`` is therefore given in XYZ order, to
  match both DVID's ``VoxelSize`` and the XYZ transform.
"""
from functools import partial

import numpy as np

from neuclease.util import compute_parallel
from neuclease.dvid.rle import split_ranges_for_grid, _write_mask_from_ranges


def _block_mask_specs(ranges, block_shape, halo):
    """
    Split RLE ranges into lightweight, per-block descriptors suitable for
    distributing to a worker pool *without* inflating any 3D masks here.

    Each descriptor is ``(cell_index_zyx, box_zyx, local_ranges, full_block_shape)``:

    - ``cell_index_zyx``: (3,) integer grid-cell index.
    - ``box_zyx``: (2, 3) spatial box of the (haloed) block, in the ranges' scale.
    - ``local_ranges``: the block's RLEs, offset to block-local coordinates and
      converted to the EXCLUSIVE-X convention expected by
      ``_write_mask_from_ranges`` (so a worker can inflate the mask cheaply).
    - ``full_block_shape``: (3,) tuple, the inflated mask shape (block + 2*halo).

    These descriptors are RLE-sized; the (potentially large) 3D masks are
    inflated later, per block, by :func:`_inflate_block_mask`.

    Note: this mirrors the per-block inflation performed by
    ``neuclease.dvid.rle.blockwise_masks_from_ranges``; the
    ``test_block_mask_specs_match_blockwise`` test guards against drift.
    """
    block_shape = np.asarray(block_shape)
    if block_shape.ndim == 0:
        block_shape = np.array(3 * [int(block_shape)])
    BZ, BY, BX = (int(x) for x in block_shape)
    full_block_shape = tuple(int(x) for x in (block_shape + 2 * halo))

    df = split_ranges_for_grid(ranges, block_shape, halo)
    specs = []
    for (Bz, By, Bx), block_df in df.groupby(['Bz', 'By', 'Bx'], sort=True):
        local_ranges = block_df[['z', 'y', 'x1', 'x2']].to_numpy(copy=True)
        # Offset to block-local coords; the X2 column also gets (X_offset - 1)
        # subtracted, which converts DVID's inclusive X2 to the exclusive
        # convention _write_mask_from_ranges expects.
        local_ranges -= (Bz * BZ - halo, By * BY - halo, Bx * BX - halo, Bx * BX - halo - 1)
        coords = np.array([Bz, By, Bx], dtype=np.int64)
        box = np.array([block_shape * coords - halo,
                        block_shape * (coords + 1) + halo])
        specs.append((coords, box, local_ranges, full_block_shape))
    return specs


def _inflate_block_mask(local_ranges, full_block_shape):
    """Inflate a single block's mask from its (exclusive-X) local RLEs."""
    mask = np.zeros(full_block_shape, dtype=bool)
    _write_mask_from_ranges(local_ranges, mask)
    return mask


def _mesh_fragment_from_spec(spec, scale, method, smoothing, preserve_border,
                             decimation, trim, cell_size_zyx):
    """
    Worker: inflate one block's mask and turn it into a (trimmed) fragment mesh.

    Returns ``(fragment_position_xyz, Mesh)`` or ``None`` if the block yields
    no geometry.  Defined at module scope (and importing vol2mesh internally)
    so it is usable from a process pool.
    """
    from vol2mesh import Mesh
    from vol2mesh import multires as v2m_multires

    cell_index_zyx, box_zyx, local_ranges, full_block_shape = spec

    mask = _inflate_block_mask(local_ranges, full_block_shape)
    if not mask.any():
        return None

    # Mesh the block in full-resolution voxel coordinates.
    scaled_box_zyx = box_zyx * (2 ** scale)
    mesh = Mesh.from_binary_vol(mask, scaled_box_zyx, method=method)

    if smoothing:
        mesh.laplacian_smooth(smoothing, preserve_border=preserve_border)
    if decimation < 1.0:
        mesh.simplify(decimation)
    if len(mesh.faces) == 0:
        return None

    if trim:
        cell_size_zyx = np.asarray(cell_size_zyx)
        cell_lo_zyx = cell_index_zyx * cell_size_zyx
        tv, tf = v2m_multires.trim_mesh_to_box(
            mesh.vertices_zyx, mesh.faces, cell_lo_zyx, cell_lo_zyx + cell_size_zyx)
        if len(tf) == 0:
            return None
        mesh = Mesh(tv, tf)

    fragment_position_xyz = tuple(int(c) for c in cell_index_zyx[::-1])
    return (fragment_position_xyz, mesh)


def _split_mesh_into_cells(mesh, cell_size_zyx):
    """
    Partition a mesh into per-grid-cell fragments.

    Each face is binned to the cell(s) its bounding box overlaps, and each
    cell's faces are then geometrically trimmed to the cell box.  A face that
    straddles a cell boundary is therefore cut and appears (trimmed) in every
    cell it touches, so the partition tiles space with no gaps.  The grid is
    origin-aligned, with cells of size ``cell_size_zyx``.

    Args:
        mesh:
            A vol2mesh ``Mesh`` (ZYX vertices).
        cell_size_zyx:
            (3,) cell extents in the mesh's coordinate space, ZYX order.

    Returns:
        ``{(x, y, z): Mesh}`` keyed by integer grid-cell index, XYZ order.
    """
    from vol2mesh import Mesh
    from vol2mesh import multires as v2m_multires

    v = np.asarray(mesh.vertices_zyx, dtype=np.float64)
    f = np.asarray(mesh.faces)
    if len(f) == 0:
        return {}

    cell_size_zyx = np.asarray(cell_size_zyx, dtype=np.float64)
    fv = v[f]                                                       # (F, 3, 3) zyx
    lo = np.floor(fv.min(axis=1) / cell_size_zyx).astype(np.int64)  # (F, 3) cell index
    hi = np.floor(fv.max(axis=1) / cell_size_zyx).astype(np.int64)

    # Build (cell_index, face_index) pairs. Single-cell faces (the common
    # case) are assigned in bulk; the few boundary-straddling faces are
    # enumerated over their (small) cell ranges.
    straddles = (lo != hi).any(axis=1)
    cell_rows = [lo[~straddles]]
    face_rows = [np.flatnonzero(~straddles)]
    for i in np.flatnonzero(straddles):
        for cz in range(lo[i, 0], hi[i, 0] + 1):
            for cy in range(lo[i, 1], hi[i, 1] + 1):
                for cx in range(lo[i, 2], hi[i, 2] + 1):
                    cell_rows.append(np.array([[cz, cy, cx]], dtype=np.int64))
                    face_rows.append(np.array([i]))
    cells = np.concatenate(cell_rows)                               # (P, 3)
    face_ids = np.concatenate(face_rows)                            # (P,)

    # Group face indices by cell.
    order = np.lexsort(cells.T)
    cells = cells[order]
    face_ids = face_ids[order]
    group_starts = np.r_[0,
                         1 + np.flatnonzero((cells[1:] != cells[:-1]).any(axis=1)),
                         len(cells)]

    fragments = {}
    for s, e in zip(group_starts[:-1], group_starts[1:]):
        cz, cy, cx = (int(c) for c in cells[s])
        sub_faces = f[face_ids[s:e]]
        used = np.unique(sub_faces)
        sub_v = v[used]
        sub_f = np.searchsorted(used, sub_faces)
        cell_lo = np.array([cz, cy, cx], dtype=np.float64) * cell_size_zyx
        tv, tf = v2m_multires.trim_mesh_to_box(sub_v, sub_f, cell_lo, cell_lo + cell_size_zyx)
        if len(tf) == 0:
            continue
        fragments[(cx, cy, cz)] = Mesh(tv, tf)

    return fragments


def multires_mesh_from_ranges(
    ranges,
    block_shape,
    voxel_size_nm,
    output_dir,
    segment_id=1,
    scale=0,
    halo=1,
    smoothing=0,
    preserve_border=True,
    decimation=1.0,
    final_decimation=1.0,
    trim=True,
    method='skimage',
    vertex_quantization_bits=16,
    lod_scale_multiplier=1.0,
    threads=0,
    processes=0,
    write_info=True,
    progress=True,
):
    """
    Build a single-LOD neuroglancer multires Draco mesh from RLE ranges.

    Each grid block of the sparse volume is meshed independently and becomes
    one fragment of the output object.  Blocks are inflated with an optional
    ``halo`` so that adjacent block meshes overlap slightly (reducing seams
    after smoothing/decimation); vertices that stray outside a block's grid
    cell are clipped back onto the cell boundary during quantization.

    Args:
        ranges:
            RLEs in DVID 'ranges' form, shape (N, 4), as ``[[Z, Y, X1, X2]]``
            with an INCLUSIVE X2 (e.g. from
            ``fetch_sparsevol(..., format='ranges')``).  Coordinates are in
            scale-``scale`` voxel units.
        block_shape:
            Grid block shape in ZYX order (or a scalar for a cubic block),
            in scale-``scale`` voxel units.  This defines the multires grid:
            one fragment per occupied block.
        voxel_size_nm:
            Full-resolution voxel size in nm, XYZ order (or a scalar for
            isotropic).  Written into the ``info`` transform.
        output_dir:
            Directory to write the multires mesh into (``info``,
            ``<segment_id>.index``, ``<segment_id>``).  Created if needed.
        segment_id:
            Integer object label used for the output filenames.
        scale:
            Scale at which ``ranges`` (and therefore the masks) are defined.
            Used to convert block coordinates to full-resolution voxels.
        halo:
            Halo (in scale-``scale`` voxels) added around each block before
            meshing, so blocks overlap.
        smoothing:
            Number of Laplacian smoothing iterations applied to each block
            mesh (0 to disable).
        preserve_border:
            If True (and smoothing is enabled), hold each block mesh's open
            border vertices fixed during smoothing.  Because each block is
            meshed from a haloed mask, the open border is the halo cut, so
            this prevents smoothing from pulling that cut inward and
            distorting the cell-boundary region that gets clipped/quantized.
        decimation:
            Initial, per-fragment decimation: fraction of faces to keep when
            simplifying each block mesh individually (1.0 to disable).  This
            is a cheap, parallelizable pre-reduction; it decimates every
            fragment equally regardless of its complexity.
        final_decimation:
            Final, whole-mesh decimation: fraction of faces to keep when
            decimating all (trimmed) fragments together as one mesh
            (1.0 to disable).  Because the fragments are merged and decimated
            jointly, the face budget is allocated across them by geometric
            complexity -- complex fragments keep more detail, simple ones
            less -- which is what you want when targeting a total mesh size
            rather than a per-fragment size.  The decimated mesh is then split
            back into per-cell fragments.  Requires ``trim=True`` (the merge
            is only valid once each fragment has been trimmed to its cell).
            The overall reduction is approximately ``decimation *
            final_decimation`` of the original face count.
        trim:
            If True, geometrically trim each block mesh to its grid cell
            (cutting the haloed overhang at the cell planes) before encoding,
            rather than clipping overhanging vertices onto the cell face.
            With a wide halo this lets neighboring fragments align closely at
            their shared boundary.  Requires the ``trimesh`` package.
        method:
            Marching-cubes method passed to ``Mesh.from_binary_vol``.
        vertex_quantization_bits:
            Draco vertex quantization, 10 or 16 (per the neuroglancer spec).
        lod_scale_multiplier:
            ``lod_scale_multiplier`` written into the ``info`` file.
        threads:
            If nonzero, generate fragments in a thread pool of this size (via
            ``compute_parallel``).  Note: per-fragment ``decimation`` is
            serialized by pyfqmr's global lock, so use ``processes`` (not
            ``threads``) if you want decimation to run in parallel.
        processes:
            If nonzero, generate fragments in a process pool of this size.
            Best for CPU-bound meshing/decimation.  (Stage-1 work-items are
            lightweight per-block range descriptors and each worker inflates
            its own mask, so inter-process payloads and peak memory stay
            bounded.)  Specify either ``threads`` or ``processes``, not both.
        write_info:
            If True, (re)write the dataset-level ``info`` file.  Set False
            when writing many objects into the same directory and the
            ``info`` file already exists.
        progress:
            If True, show a progress bar over the blocks.

    Returns:
        The number of fragments actually written (blocks that produced no
        geometry are skipped).
    """
    from vol2mesh import Mesh
    from vol2mesh import multires as v2m_multires

    block_shape = np.asarray(block_shape)
    if block_shape.ndim == 0:
        block_shape = np.array(3 * [int(block_shape)])
    assert block_shape.shape == (3,), "block_shape must be a scalar or a length-3 (ZYX) sequence"

    voxel_size_nm = np.asarray(voxel_size_nm, dtype=float)
    if voxel_size_nm.ndim == 0:
        voxel_size_nm = np.array(3 * [float(voxel_size_nm)])
    assert voxel_size_nm.shape == (3,), "voxel_size_nm must be a scalar or a length-3 (XYZ) sequence"

    if final_decimation < 1.0 and not trim:
        raise ValueError(
            "final_decimation requires trim=True: the whole-mesh decimation "
            "pass merges all fragments, which is only valid once each fragment "
            "has been trimmed to its cell (otherwise the haloed fragments "
            "overlap and the merged mesh has doubled surfaces)."
        )

    # Stored-model space is full-resolution voxels; the grid is origin-aligned.
    cell_size_zyx = block_shape * (2 ** scale)
    chunk_shape_xyz = cell_size_zyx[::-1].astype(float)
    grid_origin_xyz = np.zeros(3, dtype=float)

    # Stage 1: generate each fragment independently (optionally in parallel),
    # then trim it to its grid cell so the fragments tile space without overlap.
    # The work-items are lightweight per-block range descriptors; each worker
    # inflates its own mask, so masks aren't all materialized at once.
    specs = _block_mask_specs(ranges, block_shape, halo)
    mesh_fragment = partial(
        _mesh_fragment_from_spec,
        scale=scale, method=method, smoothing=smoothing,
        preserve_border=preserve_border, decimation=decimation,
        trim=trim, cell_size_zyx=cell_size_zyx)

    results = compute_parallel(
        mesh_fragment, specs, threads=threads, processes=processes,
        ordered=False, show_progress=progress)
    fragments = {pos: mesh for pos, mesh in filter(None, results)}

    # Stage 2: optional whole-mesh decimation. Merge the trimmed fragments,
    # decimate jointly (preserving cell boundaries so neighbors stay aligned
    # and the result can be re-split), then split back into per-cell fragments.
    if final_decimation < 1.0 and fragments:
        merged = Mesh.concatenate_meshes(list(fragments.values()), keep_normals=False)
        merged.simplify(final_decimation, preserve_border=True)
        fragments = _split_mesh_into_cells(merged, cell_size_zyx)

    transform = [voxel_size_nm[0], 0, 0, 0,
                 0, voxel_size_nm[1], 0, 0,
                 0, 0, voxel_size_nm[2], 0]

    if write_info:
        v2m_multires.write_info(
            output_dir,
            vertex_quantization_bits=vertex_quantization_bits,
            transform=transform,
            lod_scale_multiplier=lod_scale_multiplier,
        )

    # Fragments are already trimmed to their cells (in stage 1, and again
    # when split after whole-mesh decimation), so no trimming is needed here.
    return v2m_multires.write_object_mesh(
        output_dir,
        segment_id,
        fragments,
        chunk_shape_xyz,
        grid_origin_xyz,
        vertex_quantization_bits=vertex_quantization_bits,
        trim=False,
    )


def multires_mesh_from_sparsevol(
    dvid_server,
    uuid,
    instance,
    body,
    scale,
    block_shape,
    output_dir,
    supervoxels=False,
    halo=1,
    smoothing=0,
    preserve_border=True,
    decimation=1.0,
    final_decimation=1.0,
    trim=True,
    method='skimage',
    vertex_quantization_bits=16,
    lod_scale_multiplier=1.0,
    threads=0,
    processes=0,
    write_info=True,
    progress=True,
):
    """
    Fetch a body's sparse volume from DVID and write a single-LOD multires
    mesh for it.

    The object's RLE ranges are fetched at the given ``scale``, the voxel
    size (nm) is read from the instance info, and the work is delegated to
    :func:`multires_mesh_from_ranges`.

    Args:
        dvid_server, uuid, instance:
            DVID labelmap (segmentation) instance to read from.
        body:
            Body (or supervoxel) label to mesh.  Also used as the output
            ``segment_id``.
        scale:
            Scale at which to fetch the sparse volume and generate the mesh.
        block_shape:
            Grid block shape (ZYX) in scale-``scale`` voxels.  See
            :func:`multires_mesh_from_ranges`.
        output_dir:
            Output directory for the multires mesh.
        supervoxels:
            If True, treat ``body`` as a supervoxel id.
        halo, smoothing, preserve_border, decimation, final_decimation, trim,
        method, vertex_quantization_bits, lod_scale_multiplier, threads,
        processes, write_info, progress:
            Forwarded to :func:`multires_mesh_from_ranges`.

    Returns:
        The number of fragments written.
    """
    from neuclease.dvid.labelmap import fetch_sparsevol
    from neuclease.dvid.node import fetch_instance_info

    ranges = fetch_sparsevol(
        dvid_server, uuid, instance, body,
        scale=scale, supervoxels=supervoxels, format='ranges')

    # DVID reports VoxelSize in XYZ order, at full resolution (scale 0),
    # which is exactly what the multires transform wants.
    voxel_size_nm_xyz = fetch_instance_info(dvid_server, uuid, instance)['Extended']['VoxelSize']

    return multires_mesh_from_ranges(
        ranges,
        block_shape,
        voxel_size_nm_xyz,
        output_dir,
        segment_id=body,
        scale=scale,
        halo=halo,
        smoothing=smoothing,
        preserve_border=preserve_border,
        decimation=decimation,
        final_decimation=final_decimation,
        trim=trim,
        method=method,
        vertex_quantization_bits=vertex_quantization_bits,
        lod_scale_multiplier=lod_scale_multiplier,
        threads=threads,
        processes=processes,
        write_info=write_info,
        progress=progress,
    )
