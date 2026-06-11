import tempfile

import numpy as np
import pytest

from neuclease.dvid.rle import runlength_encode_mask_to_ranges, blockwise_masks_from_ranges
from neuclease.misc.sparsevol_multires_mesh import (
    multires_mesh_from_ranges,
    _block_mask_specs,
    _inflate_block_mask,
)

# vol2mesh (and its DracoPy dependency) are optional for neuclease.
pytest.importorskip("vol2mesh")
pytest.importorskip("DracoPy")

from vol2mesh.multires import split_mesh_into_cells


@pytest.fixture
def sphere_ranges():
    """A solid sphere spanning several 32-voxel blocks, as DVID 'ranges'."""
    N = 128
    zz, yy, xx = np.ogrid[:N, :N, :N]
    c = N // 2
    vol = ((zz - c)**2 + (yy - c)**2 + (xx - c)**2) <= 44**2
    return runlength_encode_mask_to_ranges(vol)


def _read_fragments(output_dir, segment_id):
    from vol2mesh import multires as v2m
    return v2m.read_object_mesh(output_dir, segment_id)


def _total_faces_and_out_of_cell(res):
    chunk = res['chunk_shape_xyz']
    origin = res['grid_origin_xyz']
    total_faces = 0
    out_of_cell = 0
    for frag in res['fragments']:
        lo = origin + frag['position'] * chunk
        hi = lo + chunk
        v = frag['vertices_xyz']
        total_faces += len(frag['faces'])
        if (v < lo - 1e-3).any() or (v > hi + 1e-3).any():
            out_of_cell += 1
    return total_faces, out_of_cell


def test_per_fragment_only(sphere_ranges):
    """No whole-mesh pass: each block becomes a trimmed, cell-confined fragment."""
    d = tempfile.mkdtemp()
    n = multires_mesh_from_ranges(
        sphere_ranges, block_shape=32, voxel_size_nm=8.0, output_dir=d,
        segment_id=1, scale=0, halo=2, smoothing=2, decimation=1.0,
        final_decimation=1.0, trim=True, progress=False)
    assert n > 1

    res = _read_fragments(d, 1)
    assert res['num_lods'] == 1
    assert np.allclose(res['chunk_shape_xyz'], [32, 32, 32])
    total_faces, out_of_cell = _total_faces_and_out_of_cell(res)
    assert total_faces > 0
    assert out_of_cell == 0


def test_final_decimation_reduces_total(sphere_ranges):
    """final_decimation reduces the *total* face count across all fragments."""
    d0 = tempfile.mkdtemp()
    multires_mesh_from_ranges(
        sphere_ranges, 32, 8.0, d0, segment_id=1, scale=0, halo=2,
        smoothing=2, decimation=1.0, final_decimation=1.0, trim=True, progress=False)
    base_faces, base_bad = _total_faces_and_out_of_cell(_read_fragments(d0, 1))

    d1 = tempfile.mkdtemp()
    multires_mesh_from_ranges(
        sphere_ranges, 32, 8.0, d1, segment_id=1, scale=0, halo=2,
        smoothing=2, decimation=1.0, final_decimation=0.25, trim=True, progress=False)
    dec_faces, dec_bad = _total_faces_and_out_of_cell(_read_fragments(d1, 1))

    assert base_bad == 0 and dec_bad == 0
    # Whole-mesh decimation should substantially reduce the total face count.
    # (Not exactly 0.25x: preserve_border pins cell boundaries and the
    # re-split re-cuts straddling faces, so the realized ratio is higher.)
    assert dec_faces < base_faces
    assert dec_faces < 0.6 * base_faces


def test_final_decimation_requires_trim(sphere_ranges):
    """Merging untrimmed (haloed) fragments would overlap, so it's disallowed."""
    with pytest.raises(ValueError):
        multires_mesh_from_ranges(
            sphere_ranges, 32, 8.0, tempfile.mkdtemp(),
            final_decimation=0.5, trim=False, progress=False)


@pytest.mark.parametrize("halo", [0, 1, 3])
def test_block_mask_specs_match_blockwise(sphere_ranges, halo):
    """
    The lightweight per-block descriptors must inflate to exactly the same
    boxes/masks that blockwise_masks_from_ranges produces (guards against the
    replicated offset/X-convention logic drifting from the rle implementation).
    """
    boxes, masks = blockwise_masks_from_ranges(sphere_ranges, 32, halo)
    blockwise = list(zip(boxes, masks))
    specs = _block_mask_specs(sphere_ranges, 32, halo)

    assert len(specs) == len(blockwise)
    for (coords, box, local_ranges, full_block_shape), (bbox, bmask) in zip(specs, blockwise):
        assert np.array_equal(box, bbox)
        assert np.array_equal(_inflate_block_mask(local_ranges, full_block_shape), bmask)


def _fragment_signature(output_dir, segment_id=1):
    from vol2mesh import multires as v2m
    res = v2m.read_object_mesh(output_dir, segment_id)
    return {tuple(int(c) for c in f['position']): (len(f['vertices_xyz']), len(f['faces']))
            for f in res['fragments']}


def test_parallel_matches_serial(sphere_ranges):
    """Serial, thread-pool, and process-pool generation produce identical fragments."""
    kw = dict(block_shape=32, voxel_size_nm=8.0, scale=0, halo=2, smoothing=2,
              decimation=0.5, final_decimation=1.0, trim=True, progress=False)

    d_serial = tempfile.mkdtemp()
    multires_mesh_from_ranges(sphere_ranges, output_dir=d_serial, segment_id=1, **kw)
    serial = _fragment_signature(d_serial)
    assert len(serial) > 1

    d_threads = tempfile.mkdtemp()
    multires_mesh_from_ranges(sphere_ranges, output_dir=d_threads, segment_id=1, threads=4, **kw)
    assert _fragment_signature(d_threads) == serial

    d_procs = tempfile.mkdtemp()
    multires_mesh_from_ranges(sphere_ranges, output_dir=d_procs, segment_id=1, processes=3, **kw)
    assert _fragment_signature(d_procs) == serial


def test_split_mesh_into_cells_straddling_face():
    """
    A face straddling cell boundaries is cut and appears (trimmed) in every
    cell it touches, with no part left outside any cell.
    """
    from vol2mesh import Mesh

    cell_size_zyx = np.array([32.0, 32.0, 32.0])
    # A thin triangle spanning x (last axis) across three cells: 0, 1, 2.
    verts_zyx = np.array([[0.0, 0.0, 10.0],
                          [0.0, 8.0, 10.0],
                          [4.0, 4.0, 80.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    mesh = Mesh(verts_zyx, faces)

    cells = split_mesh_into_cells(mesh, cell_size_zyx)

    # The triangle spans x-cells 0, 1, 2 (y=z=cell 0).
    assert set(cells.keys()) == {(0, 0, 0), (1, 0, 0), (2, 0, 0)}

    for (cx, cy, cz), frag in cells.items():
        lo = np.array([cz, cy, cx]) * cell_size_zyx
        hi = lo + cell_size_zyx
        v = frag.vertices_zyx
        assert len(frag.faces) > 0
        # Every vertex of every cell fragment lies within that cell.
        assert (v >= lo - 1e-3).all() and (v <= hi + 1e-3).all()


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_sparsevol_multires_mesh'])
