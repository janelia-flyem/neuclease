import json
import tempfile

import numpy as np
import pytest

from neuclease.misc.bodymesh import mesh_instance_names, _assemble_multires_object, BodyMeshParametersSchema

# vol2mesh (and its DracoPy dependency) are optional for neuclease.
pytest.importorskip("vol2mesh")
pytest.importorskip("DracoPy")

from confiddler import validate
from vol2mesh import multires as v2m


def test_mesh_instance_names():
    n = mesh_instance_names("segmentation")
    assert (n.body, n.info, n.chunk, n.sv) == (
        "segmentation_meshes", "segmentation_mesh_info",
        "segmentation_chunk_meshes", "segmentation_sv_meshes")

    m = mesh_instance_names("segmentation", multires=True)
    assert (m.body, m.info, m.chunk) == (
        "segmentation_multires_meshes", "segmentation_multires_mesh_info",
        "segmentation_multires_chunk_meshes")
    # The supervoxel instance is single-res only (no multires variant).
    assert m.sv == "segmentation_sv_meshes"


def test_format_config():
    # Single-res is the default; multi-res is opt-in via the 'format' field.
    cfg = {}
    validate(cfg, BodyMeshParametersSchema, inject_defaults=True)
    assert cfg['format'] == 'neuroglancer-single-res'

    cfg2 = {'format': 'neuroglancer-multi-res'}
    validate(cfg2, BodyMeshParametersSchema, inject_defaults=True)
    assert cfg2['format'] == 'neuroglancer-multi-res'

    with pytest.raises(Exception):
        validate({'format': 'bogus'}, BodyMeshParametersSchema, inject_defaults=True)


def _cube(corner, size):
    c = np.asarray(corner, dtype=float)
    off = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    v = c + size * off
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                  [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                  [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], dtype=np.uint32)
    return v, f


def _make_fragment_bytes(chunk_shape_xyz, cells, bits=16):
    chunk_shape_xyz = np.asarray(chunk_shape_xyz, dtype=float)
    frag_bytes = {}
    for cell in cells:
        corner = np.array(cell) * chunk_shape_xyz
        # A cube mostly inside the cell, with a little overhang to exercise trim.
        v, f = _cube(corner + 0.1 * chunk_shape_xyz, 0.8 * chunk_shape_xyz[0])
        frag_bytes[cell] = v2m.encode_fragment(
            (v, f), cell, chunk_shape_xyz, [0, 0, 0], vertex_quantization_bits=bits, trim=True)
    return frag_bytes


def _body_config(small_dec, large_dec, smoothing, bits=16, num_lods=1, lod_factor=0.25):
    cfg = {
        'small-body-overall-decimation-s0': small_dec,
        'large-body-overall-decimation-s0': large_dec,
        'smoothing': smoothing,
        'vertex-quantization-bits': bits,
        'num-lods': num_lods,
        'lod-decimation-factor': lod_factor,
    }
    validate(cfg, BodyMeshParametersSchema, inject_defaults=True)
    return cfg


def _parse_positions_by_lod(index_bytes):
    buf = index_bytes
    pos = 0

    def take(dt, c):
        nonlocal pos
        a = np.frombuffer(buf, dtype=dt, count=c, offset=pos)
        pos += a.nbytes
        return a

    take("<f4", 3); take("<f4", 3)
    num_lods = int(take("<u4", 1)[0])
    take("<f4", num_lods); take("<f4", num_lods * 3)
    nfr = take("<u4", num_lods).astype(int)
    by_lod = {}
    for lod in range(num_lods):
        n = int(nfr[lod])
        p = take("<u4", n * 3).reshape(3, n).T
        take("<u4", n)
        by_lod[lod] = {tuple(int(c) for c in row) for row in p}
    return num_lods, by_lod


def _read(data_bytes, index_bytes, info_json):
    d = tempfile.mkdtemp()
    with open(f"{d}/info", "w") as fp:
        json.dump(info_json, fp)
    with open(f"{d}/9", "wb") as fp:
        fp.write(data_bytes)
    with open(f"{d}/9.index", "wb") as fp:
        fp.write(index_bytes)
    return v2m.read_object_mesh(d, 9)


def _all_in_cell(res):
    ch = res['chunk_shape_xyz']
    for frag in res['fragments']:
        lo = res['grid_origin_xyz'] + frag['position'] * ch
        v = frag['vertices_xyz']
        if (v < lo - 1e-3).any() or (v > lo + ch + 1e-3).any():
            return False
    return True


def test_assemble_fast_path_reuses_fragment_bytes():
    chunk = np.array([256.0, 256.0, 256.0])
    cells = [(0, 0, 0), (1, 0, 0), (2, 1, 3)]
    frag_bytes = _make_fragment_bytes(chunk, cells)

    # No body smoothing/decimation -> fast path: cached fragment bytes reused verbatim.
    cfg = _body_config(1.0, 1.0, 0)
    data, index, info, stats = _assemble_multires_object(frag_bytes, chunk, [8.0, 8.0, 8.0], 1.0, cfg)

    assert stats['applied_decimation'] == 1.0
    assert stats['num_fragments'] == 3
    assert info['transform'][0] == 8.0 and info['transform'][5] == 8.0

    # The fast path must produce byte-identical output to directly encoding the
    # same fragment bytes (i.e. no decode/re-encode happened).
    exp_data, exp_index, _ = v2m.encode_object_mesh(frag_bytes, chunk, [0, 0, 0], vertex_quantization_bits=16)
    assert data == exp_data and index == exp_index

    assert _all_in_cell(_read(data, index, info))


def test_assemble_decimation_path():
    chunk = np.array([256.0, 256.0, 256.0])
    cells = [(0, 0, 0), (1, 0, 0), (2, 1, 3)]
    frag_bytes = _make_fragment_bytes(chunk, cells)

    # Aggressive body decimation -> decode/merge/decimate/re-split path.
    cfg = _body_config(0.3, 0.3, 0)
    data, index, info, stats = _assemble_multires_object(frag_bytes, chunk, [8.0, 8.0, 8.0], 1.0, cfg)

    assert stats['applied_decimation'] < 1.0
    assert stats['final_vertices'] < stats['orig_vertices']
    assert _all_in_cell(_read(data, index, info))


def test_assemble_multilod():
    # Many small cells so there's a real octree across LODs.
    chunk = np.array([32.0, 32.0, 32.0])
    cells = [(cx, cy, cz)
             for cx in range(4) for cy in range(4) for cz in range(2)]
    frag_bytes = _make_fragment_bytes(chunk, cells)

    cfg = _body_config(1.0, 1.0, 0, num_lods=3, lod_factor=0.5)
    data, index, info, stats = _assemble_multires_object(frag_bytes, chunk, [8.0, 8.0, 8.0], 1.0, cfg)

    assert stats['num_lods'] == 3
    assert len(stats['num_fragments_per_lod']) == 3
    assert all(n > 0 for n in stats['num_fragments_per_lod'])
    # Coarser LODs have fewer (or equal) fragments than finer ones.
    nf = stats['num_fragments_per_lod']
    assert nf[0] >= nf[1] >= nf[2]

    # Octree ancestor closure holds in the written manifest.
    num_lods, pos_by_lod = _parse_positions_by_lod(index)
    assert num_lods == 3
    for lod in range(num_lods - 1):
        for (x, y, z) in pos_by_lod[lod]:
            assert (x // 2, y // 2, z // 2) in pos_by_lod[lod + 1]

    # Round-trips, and every fragment stays within its (LOD-scaled) cell.
    res = _read(data, index, info)
    assert res['num_lods'] == 3
    for frag in res['fragments']:
        cell_size = res['chunk_shape_xyz'] * (2 ** frag['lod'])
        lo = res['grid_origin_xyz'] + frag['position'] * cell_size
        v = frag['vertices_xyz']
        assert (v >= lo - 1e-3).all() and (v <= lo + cell_size + 1e-3).all()


def test_assemble_empty():
    chunk = np.array([256.0, 256.0, 256.0])
    cfg = _body_config(1.0, 1.0, 0)
    data, index, info, stats = _assemble_multires_object({}, chunk, [8.0, 8.0, 8.0], 1.0, cfg)
    assert data == b'' and index == b'' and stats['num_fragments'] == 0


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_bodymesh_multires'])
