"""
Generate a neuroglancer multi-resolution (multi-LOD) Draco mesh for a single
object directly from a DVID sparse volume.

The mesh is produced one grid block at a time: ``blockwise_masks_from_ranges``
splits the object's RLEs into block-aligned masks, each mask is meshed
(marching cubes + optional smoothing/decimation) independently, and the
resulting per-block meshes become the fragments of a single-LOD multires
mesh object written via ``vol2mesh.multires``.

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
import numpy as np

from neuclease.dvid.rle import blockwise_masks_from_ranges


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
    method='skimage',
    vertex_quantization_bits=16,
    lod_scale_multiplier=1.0,
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
            meshing, so blocks overlap.  Passed to
            ``blockwise_masks_from_ranges``.
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
            Fraction of faces to keep when simplifying each block mesh
            (1.0 to disable).
        method:
            Marching-cubes method passed to ``Mesh.from_binary_vol``.
        vertex_quantization_bits:
            Draco vertex quantization, 10 or 16 (per the neuroglancer spec).
        lod_scale_multiplier:
            ``lod_scale_multiplier`` written into the ``info`` file.
        write_info:
            If True, (re)write the dataset-level ``info`` file.  Set False
            when writing many objects into the same directory and the
            ``info`` file already exists.
        progress:
            If True, show a tqdm progress bar over the blocks (if tqdm is
            installed).

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

    boxes, masks = blockwise_masks_from_ranges(ranges, block_shape, halo)

    # Stored-model space is full-resolution voxels; the grid is origin-aligned.
    chunk_shape_xyz = block_shape[::-1].astype(float) * (2 ** scale)
    grid_origin_xyz = np.zeros(3, dtype=float)

    block_iter = zip(boxes, masks)
    if progress:
        try:
            from tqdm import tqdm
            block_iter = tqdm(block_iter, total=len(boxes))
        except ImportError:
            pass

    fragments = {}
    for box, mask in block_iter:
        if not mask.any():
            continue

        # Mesh the block in full-resolution voxel coordinates.
        scaled_box_zyx = box * (2 ** scale)
        mesh = Mesh.from_binary_vol(mask, scaled_box_zyx, method=method)

        if smoothing:
            mesh.laplacian_smooth(smoothing, preserve_border=preserve_border)
        if decimation < 1.0:
            mesh.simplify(decimation)
        if len(mesh.faces) == 0:
            continue

        # The block's grid-cell index. box[0] == block_shape*coords - halo,
        # so (box[0] + halo) // block_shape recovers the (ZYX) cell index.
        cell_index_zyx = (box[0] + halo) // block_shape
        fragment_position_xyz = tuple(int(c) for c in cell_index_zyx[::-1])
        fragments[fragment_position_xyz] = mesh

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

    return v2m_multires.write_object_mesh(
        output_dir,
        segment_id,
        fragments,
        chunk_shape_xyz,
        grid_origin_xyz,
        vertex_quantization_bits=vertex_quantization_bits,
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
    method='skimage',
    vertex_quantization_bits=16,
    lod_scale_multiplier=1.0,
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
        halo, smoothing, preserve_border, decimation, method,
        vertex_quantization_bits, lod_scale_multiplier, write_info, progress:
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
        method=method,
        vertex_quantization_bits=vertex_quantization_bits,
        lod_scale_multiplier=lod_scale_multiplier,
        write_info=write_info,
        progress=progress,
    )
