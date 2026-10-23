"""
Tests for the chunk-based (single-resolution) body mesh machinery in neuclease.misc.bodymesh.
(For multi-resolution meshes, see test_multiresmesh.py.)
"""
import numpy as np
import pytest

from confiddler import validate

# vol2mesh is optional for neuclease.
pytest.importorskip("vol2mesh")

from neuclease.misc.bodymesh import mesh_instance_names  # noqa: E402


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



def test_chunk_table_anisotropic_chunk_shape(monkeypatch):
    """
    _chunk_table() must assign blocks to the right chunks, even for a non-cubic
    chunk shape (chunk-shape-s0 is in XYZ order, but block coordinates are ZYX).
    """
    import pandas as pd
    from neuclease.misc import bodymesh
    from neuclease.dvid.labelmap import PandasLabelIndex

    block_zyx = np.array([[0, 0, 0], [64, 128, 192], [192, 64, 320], [320, 320, 64]])
    blocks_df = pd.DataFrame(block_zyx, columns=[*'zyx'])
    blocks_df['sv'] = np.uint64(1)
    blocks_df['count'] = 1
    surface_mutids = pd.DataFrame(block_zyx, columns=[*'zyx'])
    surface_mutids['surface_mutid'] = 10
    pli = PandasLabelIndex(blocks_df, 1, 10, '', '', surface_mutids)

    key_cols = ['config_name', 'body', 'chunk_id', 'mesh_mutid', 'mesh_block_hash', 'quality', 'key']
    no_keys = pd.DataFrame({c: pd.Series(dtype=object) for c in key_cols})
    no_keys = no_keys.astype({'mesh_mutid': int}).assign(
        z=pd.Series(dtype=int), y=pd.Series(dtype=int), x=pd.Series(dtype=int))

    monkeypatch.setattr(bodymesh, 'resolve_ref', lambda *args, **kwargs: 'abc123')
    monkeypatch.setattr(bodymesh, 'fetch_labelindex', lambda *args, **kwargs: pli)
    monkeypatch.setattr(bodymesh, 'fetch_stored_chunk_keys', lambda *args, **kwargs: no_keys)

    # x, y, z
    chunk_cfg = {'config-name': 'test', 'chunk-shape-s0': [256, 128, 64], 'base-uuid': 'abc123'}
    chunk_df = bodymesh._chunk_table('http://fake-dvid', 'abc123', 'segmentation', 1, chunk_cfg, None)

    # z is chunked by 64, y by 128, x by 256
    corners = chunk_df[[*'zyx']].sort_values([*'zyx']).values
    assert (corners == [[0, 0, 0], [64, 128, 0], [192, 0, 256], [320, 256, 0]]).all()


@pytest.mark.parametrize("check_stored", [True, False])
def test_meshes_for_chunks_check_stored(monkeypatch, check_stored):
    """
    With check_stored=True, chunks that another process has already stored are fetched
    rather than generated, and the results are still returned in the caller's order.
    """
    import pandas as pd
    import requests
    from requests import HTTPError
    from neuclease.misc import bodymesh

    zz, yy, xx = np.ogrid[:192, :192, :192]
    vol_s0 = ((zz - 96)**2 + (yy - 96)**2 + (xx - 96)**2) <= 70**2
    vol_s1 = vol_s0[::2, ::2, ::2]

    def not_found():
        r = requests.Response()
        r.status_code = 404
        return HTTPError("404", response=r)

    kv = {}
    sparsevol_requests = []

    def fake_fetch_sparsevol(server, uuid, instance, label, scale=0, supervoxels=False, *, mask_box=None, **kwargs):
        sparsevol_requests.append(mask_box)
        box = np.asarray(mask_box)
        mask = np.zeros(box[1] - box[0], bool)
        lo, hi = np.maximum(box[0], 0), np.minimum(box[1], vol_s1.shape)
        mask[tuple(slice(a - b, c - b) for a, c, b in zip(lo, hi, box[0]))] = \
            vol_s1[tuple(slice(a, c) for a, c in zip(lo, hi))]
        if not mask.any():
            raise not_found()
        return mask, box

    def fake_fetch_key(server, uuid, instance, key, **kwargs):
        if (instance, key) not in kv:
            raise not_found()
        return kv[(instance, key)]

    def fake_post_key(server, uuid, instance, key, data=None, **kwargs):
        kv[(instance, key)] = data

    monkeypatch.setattr(bodymesh, 'fetch_sparsevol', fake_fetch_sparsevol)
    monkeypatch.setattr(bodymesh, 'fetch_key', fake_fetch_key)
    monkeypatch.setattr(bodymesh, 'post_key', fake_post_key)

    chunk_cfg = {
        'config-name': 'test',
        'chunk-shape-s0': [64, 64, 64],
        'chunk-halo': 2,
        'quality-configs': [{'name': 'q', 'source-scale': 1, 'morphological-closing-s0': 0,
                             'fill-holes': False, 'smoothing': 1, 'decimation-s0': 1.0}],
    }
    validate(chunk_cfg, bodymesh.MeshChunkConfigSchema, inject_defaults=True)

    corners = [(z, y, x) for z in (0, 64, 128) for y in (0, 64, 128) for x in (0, 64, 128)]
    chunk_df = pd.DataFrame(corners, columns=[*'zyx'])
    chunk_df['surface_mutid'] = 10
    chunk_df['block_hash'] = '0xabc'
    chunk_instance = 'segmentation_chunk_meshes'

    def key(i):
        z, y, x = corners[i]
        return bodymesh.CHUNK_KEY_FMT.format(
            config_name='test', body=1, chunk_id=f'{x},{y},{z}',
            mesh_mutid=10, mesh_block_hash='0xabc', quality='q')

    # Pretend that another process has already stored every other chunk
    # (as empty meshes, so we can distinguish them from generated ones).
    prestored = range(0, len(corners), 2)
    for i in prestored:
        kv[(chunk_instance, key(i))] = b''

    results = bodymesh.meshes_for_chunks(
        'http://fake-dvid', 'abc123', 'segmentation', 1, chunk_cfg, 'q', True, chunk_df,
        0, bodymesh.DummyResourceMgr(), check_stored=check_stored)

    assert len(results) == len(corners)
    if check_stored:
        # Pre-stored chunks were used as-is (in the right positions); only the others were generated.
        assert len(sparsevol_requests) == len(corners) - len(prestored)
        for i in prestored:
            assert len(results[i].faces) == 0
    else:
        assert len(sparsevol_requests) == len(corners)

    # Generated chunks were stored, and returned in the right positions.
    generated = range(1, len(corners), 2) if check_stored else range(len(corners))
    for i in generated:
        # (Every chunk except the center one is on the sphere's surface.)
        assert (len(results[i].faces) > 0) == (corners[i] != (64, 64, 64))
        assert kv[(chunk_instance, key(i))] == results[i].serialize(fmt='ngmesh')



def test_chunk_table_body_not_in_base_uuid(monkeypatch):
    """
    If some blocks lack a surface_mutid but the body didn't exist in the base uuid,
    _chunk_table() still works: chunks without a stored mesh count as missing,
    and stored chunk meshes with a matching block hash count as up-to-date.
    """
    import pandas as pd
    import requests
    from requests import HTTPError
    from neuclease.misc import bodymesh
    from neuclease.dvid.labelmap import PandasLabelIndex

    block_zyx = np.array([[0, 0, 0], [0, 0, 64], [0, 0, 128]])
    blocks_df = pd.DataFrame(block_zyx, columns=[*'zyx'])
    blocks_df['sv'] = np.uint64(1)
    blocks_df['count'] = 1
    surface_mutids = pd.DataFrame(block_zyx, columns=[*'zyx'])
    surface_mutids['surface_mutid'] = [0, 0, 20]
    pli = PandasLabelIndex(blocks_df, 1, 20, '', '', surface_mutids)

    def fetch_lastmod(*args, **kwargs):
        response = requests.Response()
        response.status_code = 404
        raise HTTPError("404 Client Error: Not Found", response=response)

    # Chunk (0,0,0) has a stored mesh; the others don't.
    block_hash_0 = hex(bodymesh.CityHash64(np.array([[0, 0, 0]]).copy('C')))
    stored_keys = pd.DataFrame({
        'config_name': ['test'], 'body': [1], 'chunk_id': ['0,0,0'], 'mesh_mutid': [1],
        'mesh_block_hash': [block_hash_0], 'quality': ['q'], 'key': ['test-1-0,0,0-1-hash-q'],
        'z': [0], 'y': [0], 'x': [0],
    })

    monkeypatch.setattr(bodymesh, 'resolve_ref', lambda *args, **kwargs: 'abc123')
    monkeypatch.setattr(bodymesh, 'fetch_labelindex', lambda *args, **kwargs: pli)
    monkeypatch.setattr(bodymesh, 'fetch_lastmod', fetch_lastmod)
    monkeypatch.setattr(bodymesh, 'fetch_stored_chunk_keys', lambda *args, **kwargs: stored_keys)

    chunk_cfg = {'config-name': 'test', 'chunk-shape-s0': [64, 64, 64], 'base-uuid': 'abc123'}
    chunk_df = bodymesh._chunk_table('http://fake-dvid', 'abc123', 'segmentation', 1, chunk_cfg, None)
    chunk_df = chunk_df.sort_values('x', ignore_index=True)

    missing = chunk_df.eval('mesh_mutid < surface_mutid or quality != "q"')
    assert missing.tolist() == [False, True, True]


def test_meshes_for_chunks_with_chunk_fn():
    """
    If a chunk_fn is given, it's called (from multiple threads) instead of mesh_for_chunk(),
    with plain-python chunk specs, and the results are returned in the caller's order.
    """
    import threading
    import pandas as pd
    from neuclease.misc import bodymesh

    corners = [(z, y, x) for z in (0, 64) for y in (0, 64) for x in (0, 64)]
    chunk_df = pd.DataFrame(corners, columns=[*'zyx'])
    chunk_df['surface_mutid'] = np.int64(10)
    chunk_df['block_hash'] = [f'0x{i}' for i in range(len(corners))]

    calls = []
    lock = threading.Lock()

    def chunk_fn(chunk_spec):
        with lock:
            calls.append(chunk_spec)
        z, y, x, mutid, block_hash = chunk_spec
        return f'{z},{y},{x}-{mutid}-{block_hash}'.encode()

    results = bodymesh.meshes_for_chunks(
        'http://fake-dvid', 'abc123', 'segmentation', 1, {'config-name': 'test'}, 'q', True, chunk_df,
        0, bodymesh.DummyResourceMgr(), check_stored=True,
        chunk_fn=chunk_fn, chunk_fn_threads=4)

    assert len(calls) == len(corners)
    assert all(type(v) in (int, str) for spec in calls for v in spec)
    assert results == [f'{z},{y},{x}-10-0x{i}'.encode() for i, (z, y, x) in enumerate(corners)]



def test_init_mesh_instances_tags(monkeypatch):
    """The mesh instances are tagged (so DVID can store them separately), and versioned."""
    from neuclease.misc import bodymesh

    created = []
    monkeypatch.setattr(bodymesh, 'resolve_ref', lambda server, uuid, expand=False: uuid)
    monkeypatch.setattr(bodymesh, 'fetch_repo_instances', lambda *args, **kwargs: {})
    monkeypatch.setattr(bodymesh, 'create_instance', lambda *args, **kwargs: created.append((args, kwargs)))
    monkeypatch.setattr(bodymesh, 'create_tarsupervoxel_instance', lambda *args, **kwargs: None)

    bodymesh.init_mesh_instances('http://fake-dvid', 'abc123', 'segmentation')
    by_name = {args[2]: (args, kwargs) for args, kwargs in created}
    assert set(by_name) == {'segmentation_meshes', 'segmentation_mesh_info', 'segmentation_chunk_meshes'}
    for name in ('segmentation_meshes', 'segmentation_chunk_meshes'):
        args, kwargs = by_name[name]
        assert args[3:] == ('keyvalue',)
        assert kwargs == {'tags': {'type': 'meshes'}}


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_bodymesh_chunks'])
