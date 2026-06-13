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
    frag_bytes = {}
    for cell in cells:
        corner = np.array(cell) * chunk_shape_xyz
        v, f = _cube(corner + 40.0, 150.0)  # overhangs the cell -> exercises trim
        frag_bytes[cell] = v2m.encode_fragment(
            (v, f), cell, chunk_shape_xyz, [0, 0, 0], vertex_quantization_bits=bits, trim=True)
    return frag_bytes


def _body_config(small_dec, large_dec, smoothing, bits=16):
    cfg = {
        'small-body-overall-decimation-s0': small_dec,
        'large-body-overall-decimation-s0': large_dec,
        'smoothing': smoothing,
        'vertex-quantization-bits': bits,
    }
    validate(cfg, BodyMeshParametersSchema, inject_defaults=True)
    return cfg


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


def test_assemble_empty():
    chunk = np.array([256.0, 256.0, 256.0])
    cfg = _body_config(1.0, 1.0, 0)
    data, index, info, stats = _assemble_multires_object({}, chunk, [8.0, 8.0, 8.0], 1.0, cfg)
    assert data == b'' and index == b'' and stats['num_fragments'] == 0


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_bodymesh_multires'])
