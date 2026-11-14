"""
Tests for neuclease.misc.multiresmesh.

DVID is faked (see FakeDVID), so the real update machinery runs against an in-memory
keyvalue store and a fake segmentation: a single body (label 1) shaped like a sphere,
spanning a 3x3x3 grid of 64-voxel chunks.
"""
import json
import threading

import numpy as np
import pandas as pd
import pytest
import requests
from requests import HTTPError

pytest.importorskip("vol2mesh")
pytest.importorskip("DracoPy")
pytest.importorskip("trimesh")

from vol2mesh import multires as v2m  # noqa: E402

from neuclease.misc import bodymesh, multiresmesh  # noqa: E402
from neuclease.misc.meshstore import MemoryStore, open_store  # noqa: E402
from neuclease.misc.multiresmesh import (  # noqa: E402
    validate_config, generate_chunk_pieces, encode_chunk_pieces, decode_chunk_pieces,
    assemble_body_mesh, update_multires_body_mesh, locate_multires_body_mesh)

UUID = 'abc123def456abc123def456abc12345'
MESH_INSTANCE = 'segmentation_multires_meshes'
INFO_INSTANCE = 'segmentation_multires_mesh_info'


def _config(**kwargs):
    cfg = {
        'chunk-shape-s0': [64, 64, 64],
        'chunk-halo': 2,
        'source-scale': 1,
        'fill-holes': False,
        'smoothing': 1,
        'lod-error-thresholds': [0.5, 5.0],
        **kwargs
    }
    return validate_config(cfg)


def _dvid_error(status):
    r = requests.Response()
    r.status_code = status
    return HTTPError(f"{status} Error", response=r)


def _crop(vol, box):
    """Extract box from vol, zero-padding wherever the box extends beyond the volume."""
    box = np.asarray(box)
    out = np.zeros(box[1] - box[0], dtype=bool)
    lo = np.maximum(box[0], 0)
    hi = np.minimum(box[1], vol.shape)
    if (hi > lo).all():
        out[tuple(slice(a - b, c - b) for a, c, b in zip(lo, hi, box[0]))] = \
            vol[tuple(slice(a, c) for a, c in zip(lo, hi))]
    return out


class FakeDVID:
    """Just enough of DVID for multiresmesh.update_multires_body_mesh()."""

    def __init__(self, monkeypatch):
        zz, yy, xx = np.ogrid[:192, :192, :192]
        self.set_volume(((zz - 96)**2 + (yy - 96)**2 + (xx - 96)**2) <= 70**2)
        self.mutid = 10
        self.kv = {}
        self.sparsevol_requests = []
        self.head = UUID
        self._lock = threading.Lock()

        # Supervoxels: 7 is the half of the sphere with x < 96,
        # and 8 is a single voxel, which vanishes at scale 1.
        self.sv_vols = {7: self.vols[0] & (xx < 96)}
        self.sv_vols[8] = np.zeros_like(self.vols[0])
        self.sv_vols[8][101, 101, 101] = True
        self.instances = {MESH_INSTANCE, INFO_INSTANCE}
        self.created_instances = []

        for mod in (bodymesh, multiresmesh):
            for name in ('resolve_ref', 'fetch_lastmod', 'fetch_labelindex', 'fetch_sparsevol',
                         'fetch_sparsevol_coarse', 'fetch_key', 'post_key', 'delete_key',
                         'fetch_repo_instances', 'create_instance'):
                if hasattr(mod, name):
                    monkeypatch.setattr(mod, name, getattr(self, name))
        monkeypatch.setattr(multiresmesh, 'fetch_resolution_zyx', lambda *args: (8.0, 8.0, 8.0))
        multiresmesh._ensure_sv_mesh_instance.cache_clear()

    def set_volume(self, vol_s0):
        self.vols = {0: vol_s0, 1: vol_s0[::2, ::2, ::2]}
        block_zyx = np.unique((np.argwhere(vol_s0) // 64) * 64, axis=0)
        old = getattr(self, 'surface_mutids', None)
        self.blocks_df = pd.DataFrame(block_zyx, columns=[*'zyx'])
        self.blocks_df['sv'] = np.uint64(1)
        self.blocks_df['count'] = 1
        sm = pd.DataFrame(block_zyx, columns=[*'zyx'])
        sm['surface_mutid'] = 10
        if old is not None:
            sm = sm.merge(old, 'left', on=[*'zyx'], suffixes=('_new', ''))
            sm['surface_mutid'] = sm['surface_mutid'].fillna(sm['surface_mutid_new']).astype(int)
            sm = sm[[*'zyx', 'surface_mutid']]
        self.surface_mutids = sm

    def edit_chunk(self, chunk_zyx):
        """Simulate an edit to the body's surface within the given chunk."""
        self.mutid += 1
        sm = self.surface_mutids
        sm.loc[(sm[[*'zyx']].values == chunk_zyx).all(axis=1), 'surface_mutid'] = self.mutid

    def resolve_ref(self, server, ref, expand=False, **kwargs):
        if ref.endswith(':master'):
            return self.head
        return ref

    def fetch_lastmod(self, server, uuid, instance, label, **kwargs):
        if label != 1:
            raise _dvid_error(404)
        return {'mutation id': self.mutid}

    def fetch_labelindex(self, server, uuid, instance, label, format='protobuf', **kwargs):
        from neuclease.dvid.labelmap import PandasLabelIndex
        if label != 1:
            raise _dvid_error(404)
        return PandasLabelIndex(self.blocks_df.copy(), 1, self.mutid, '', '', self.surface_mutids.copy())

    def fetch_sparsevol(self, server, uuid, instance, label, scale=0, supervoxels=False, *,
                        format='coords', mask_box=None, **kwargs):
        with self._lock:
            self.sparsevol_requests.append(tuple(np.asarray(mask_box[0])))
        if supervoxels:
            if label not in self.sv_vols:
                raise _dvid_error(404)
            vol = self.sv_vols[label][::2**scale, ::2**scale, ::2**scale]
        else:
            vol = self.vols[scale]
        mask = _crop(vol, mask_box)
        if not mask.any():
            raise _dvid_error(404)
        return mask, np.asarray(mask_box)

    def fetch_sparsevol_coarse(self, server, uuid, instance, label, supervoxels=False, **kwargs):
        """Scale-6 (block) coordinates, ZYX"""
        assert supervoxels
        if label not in self.sv_vols:
            raise _dvid_error(404)
        return np.unique(np.argwhere(self.sv_vols[label]) // 64, axis=0)

    def fetch_repo_instances(self, server, uuid, **kwargs):
        return {name: 'keyvalue' for name in self.instances}

    def create_instance(self, server, uuid, instance, typename, versioned=True, compression=None, tags=[], **kwargs):
        self.instances.add(instance)
        self.created_instances.append((instance, typename, versioned, tags))

    def fetch_key(self, server, uuid, instance, key, *, as_json=False, check_head=False, **kwargs):
        value = self.kv.get((instance, str(key)))
        if check_head:
            return value is not None
        if value is None:
            raise _dvid_error(404)
        return json.loads(value) if as_json else value

    def post_key(self, server, uuid, instance, key, data=None, json=None, **kwargs):
        import json as _json
        self.kv[(instance, str(key))] = _json.dumps(json).encode() if json is not None else data

    def delete_key(self, server, uuid, instance, key, **kwargs):
        self.kv.pop((instance, str(key)), None)

    def mesh_files(self, body=1):
        return {suffix: self.kv.get((MESH_INSTANCE, f'{body}{suffix}')) for suffix in ('', '.index', '.chunks')}

    def mesh_info(self, body=1):
        return json.loads(self.kv[(INFO_INSTANCE, str(body))])


@pytest.fixture
def dvid(monkeypatch):
    MemoryStore.clear_all()
    multiresmesh._verified_store_infos.clear()
    multiresmesh._verified_sv_stores.clear()
    return FakeDVID(monkeypatch)


def _read_mesh(data, index, info_json=None):
    import tempfile
    d = tempfile.mkdtemp()
    with open(f"{d}/1", "wb") as f:
        f.write(data)
    with open(f"{d}/1.index", "wb") as f:
        f.write(index)
    return v2m.read_object_mesh(d, 1, vertex_quantization_bits=16)


def test_config_validation():
    cfg = _config()
    assert cfg['vertex-quantization-bits'] == 16
    assert cfg['storage'] == 'dvid'

    with pytest.raises(ValueError):
        _config(**{'lod-error-thresholds': [5.0, 1.0]})
    with pytest.raises(ValueError):
        _config(**{'chunk-shape-s0': [64, 64, 100]})


def test_encode_decode_chunk_pieces():
    pieces = [b'abc', b'', b'defgh']
    assert decode_chunk_pieces(encode_chunk_pieces(pieces)) == pieces
    assert decode_chunk_pieces(encode_chunk_pieces([b''])) == [b'']
    with pytest.raises(ValueError):
        decode_chunk_pieces(encode_chunk_pieces(pieces) + b'x')


def test_generate_chunk_pieces(dvid):
    cfg = _config()
    chunk_shape = np.array([64.0, 64.0, 64.0])

    # A chunk on the sphere's surface (cell (2, 1, 1))
    pieces = generate_chunk_pieces('fake', UUID, 'segmentation', 1, cfg, (64, 64, 128))
    assert len(pieces) == 2
    assert dvid.sparsevol_requests == [(30, 30, 62)]   # scale 1, with a halo of 2

    v0, f0 = v2m.decode_fragment(pieces[0], (2, 1, 1), chunk_shape, [0, 0, 0], lod=0)
    v1, f1 = v2m.decode_fragment(pieces[1], (1, 0, 0), chunk_shape, [0, 0, 0], lod=1)
    assert 0 < len(f1) < len(f0)

    # Each LOD's piece is trimmed to the chunk.
    for v in (v0, v1):
        assert (v >= [128, 64, 64]).all() and (v <= [192, 128, 128]).all()

    # The vertices lie close to the sphere's surface
    # (at most about sqrt(threshold) away, plus the meshing/smoothing error).
    for v, threshold in [(v0, 0.5), (v1, 5.0)]:
        r = np.linalg.norm(v - 96, axis=1)
        assert np.abs(r - 70).max() < 2 + 2 * np.sqrt(threshold)

    # A chunk with no voxels at the source scale
    assert generate_chunk_pieces('fake', UUID, 'segmentation', 1, cfg, (0, 0, 256)) == [b'', b'']


def _all_pieces(cfg):
    """The pieces of every chunk of the fake body, keyed by cell."""
    pieces = {}
    for z in range(3):
        for y in range(3):
            for x in range(3):
                pieces[(x, y, z)] = generate_chunk_pieces(
                    'fake', UUID, 'segmentation', 1, cfg, (64 * z, 64 * y, 64 * x))
    return pieces


def test_assemble_body_mesh(dvid):
    cfg = _config()
    chunk_shape = np.array([64.0, 64.0, 64.0])
    pieces = _all_pieces(cfg)

    data, index, info, stats = assemble_body_mesh(pieces, chunk_shape, [8.0, 8.0, 8.0], cfg)
    assert info['@type'] == 'neuroglancer_multilod_draco'
    assert info['transform'] == [8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0]
    assert stats['num_lods'] == 2

    m = v2m.parse_manifest(index)
    lod0 = {tuple(int(c) for c in p): (o, s) for p, o, s in
            zip(m['lods'][0]['positions'], m['lods'][0]['offsets'], m['lods'][0]['sizes'])}

    # The LOD-0 fragments are the LOD-0 pieces, verbatim.
    nonempty = {cell for cell, p in pieces.items() if p[0]}
    assert {cell for cell, (o, s) in lod0.items() if s} == nonempty
    for cell in nonempty:
        o, s = lod0[cell]
        assert data[o:o + s] == pieces[cell][0]

    # The LOD-1 fragments contain all of their chunks' LOD-1 pieces.
    mesh = _read_mesh(data, index)
    lod1 = {tuple(int(c) for c in f['position']): f for f in mesh['fragments'] if f['lod'] == 1}
    assert set(lod1) == {(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)}
    for parent, frag in lod1.items():
        expected_faces = sum(
            len(v2m.decode_fragment(p[1], parent, chunk_shape, [0, 0, 0], lod=1)[1])
            for cell, p in pieces.items() if tuple(c // 2 for c in cell) == parent and p[1])
        assert len(frag['faces']) == expected_faces

    # No triangle crosses an LOD-1 fragment's mid-planes (a neuroglancer requirement).
    for parent, frag in lod1.items():
        mid = (np.array(parent) * 128 + 64)
        side = frag['vertices_xyz'][frag['faces']] >= mid - 1e-3
        on_plane = np.abs(frag['vertices_xyz'][frag['faces']] - mid) < 1e-3
        # Each triangle's vertices are all on one side (or on the plane).
        assert ((side | on_plane).all(axis=1) | (~side | on_plane).all(axis=1)).all()


def test_update_body_mesh(dvid):
    cfg = _config()
    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert mesh_info == dvid.mesh_info()
    assert mesh_info['lastmod'] == 10
    assert mesh_info['mesh-settings']['lod-error-thresholds'] == [0.5, 5.0]
    assert mesh_info['num-lods'] == 2

    # Every chunk which contains part of the body was generated.
    assert len(dvid.sparsevol_requests) == 27
    files = dvid.mesh_files()
    assert files[''] and files['.index']
    assert json.loads(dvid.kv[(MESH_INSTANCE, 'info')])['@type'] == 'neuroglancer_multilod_draco'

    mesh = _read_mesh(files[''], files['.index'])
    assert {f['lod'] for f in mesh['fragments']} == {0, 1}

    loc = locate_multires_body_mesh('fake', UUID, 'segmentation', 1)
    assert loc.store is None and (loc.data_name, loc.index_name) == ('1', '1.index')

    # Up-to-date meshes aren't regenerated...
    assert update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg) is None
    assert len(dvid.sparsevol_requests) == 27

    # ...unless forced, or the settings change.
    assert update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg, force=True)
    assert len(dvid.sparsevol_requests) == 54
    assert update_multires_body_mesh('fake', UUID, 'segmentation', 1, _config(smoothing=2))
    assert len(dvid.sparsevol_requests) == 81

    # Deleted bodies' meshes are deleted.
    assert update_multires_body_mesh('fake', UUID, 'segmentation', 2, cfg) is None


def test_deleted_body(dvid, monkeypatch):
    cfg = _config()
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert dvid.mesh_files()['']

    def no_body(*args, **kwargs):
        raise _dvid_error(404)

    monkeypatch.setattr(multiresmesh, 'fetch_lastmod', no_body)
    assert update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg) is None
    assert dvid.mesh_files() == {'': None, '.index': None, '.chunks': None}
    assert dvid.mesh_info()['method'] == 'deleted'
    assert locate_multires_body_mesh('fake', UUID, 'segmentation', 1) is None


def test_chunk_fn(dvid):
    cfg = _config()
    requested = []

    def chunk_fn(chunk_zyx):
        requested.append(chunk_zyx)
        return generate_chunk_pieces('fake', UUID, 'segmentation', 1, cfg, chunk_zyx)

    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg, chunk_fn=chunk_fn, chunk_fn_threads=4)
    assert len(requested) == 27
    assert all(isinstance(c, int) for zyx in requested for c in zyx)


def test_mesh_in_object_store(dvid):
    cfg = _config(storage='mem://bucket/meshes')
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    store = open_store('mem://bucket/meshes')
    assert store.stat('1').metadata == {'uuid': UUID, 'lastmod': '10'}
    assert json.loads(store.get('info'))['@type'] == 'neuroglancer_multilod_draco'
    assert (MESH_INSTANCE, '1') not in dvid.kv

    loc = locate_multires_body_mesh('fake', UUID, 'segmentation', 1)
    assert loc.store is store and loc.data_name == '1'
    assert v2m.parse_manifest(store.get(loc.index_name))['num_lods'] == 2


def _triangles_by_lod(data, index):
    """{lod: set of triangles (as tuples of rounded vertex coordinates)}"""
    tris = {}
    for frag in _read_mesh(data, index)['fragments']:
        t = np.round(frag['vertices_xyz'][frag['faces']], 3)
        # Canonical vertex order within each triangle (preserving orientation)
        t = np.array([np.roll(tri, -np.lexsort(tri.T[::-1])[0], axis=0) for tri in t])
        tris.setdefault(frag['lod'], set()).update(map(lambda a: a.tobytes(), t))
    return tris


def _lod_fragments(data, index, lod):
    m = v2m.parse_manifest(index)
    frags = m['lods'][lod]
    return {tuple(int(c) for c in p): data[o:o + s]
            for p, o, s in zip(frags['positions'], frags['offsets'], frags['sizes']) if s}


def test_unchanged_chunks_are_reused(dvid):
    cfg = _config()
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    before = dvid.mesh_files()
    chunks_file = json.loads(before['.chunks'])
    assert chunks_file['lastmod'] == 10
    assert len(chunks_file['chunks']) == 27
    assert len(dvid.sparsevol_requests) == 27

    # The body changed, but not within any of its chunks
    # (e.g. a supervoxel split which didn't change the body's surface).
    dvid.mutid += 1
    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert mesh_info['lastmod'] == 11
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (0, 27)
    assert len(dvid.sparsevol_requests) == 27

    # The mesh is identical (the fragments were copied verbatim).
    after = dvid.mesh_files()
    assert after[''] == before['']
    assert after['.index'] == before['.index']
    assert json.loads(after['.chunks'])['lastmod'] == 11


def test_only_edited_chunks_are_regenerated(dvid):
    cfg = _config()
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    before = dvid.mesh_files()

    # Carve a dent into the body within chunk (x, y, z) = (2, 1, 1)
    vol = dvid.vols[0].copy()
    vol[80:110, 75:115, 150:180] = False
    dvid.set_volume(vol)
    dvid.edit_chunk((64, 64, 128))
    dvid.sparsevol_requests.clear()

    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (1, 26)
    assert dvid.sparsevol_requests == [(30, 30, 62)]
    after = dvid.mesh_files()

    # The other chunks' LOD-0 fragments are unchanged, as are the LOD-1 fragments that don't contain the edited chunk.
    for lod, edited in [(0, (2, 1, 1)), (1, (1, 0, 0))]:
        old = _lod_fragments(before[''], before['.index'], lod)
        new = _lod_fragments(after[''], after['.index'], lod)
        assert set(old) == set(new)
        assert {p for p in old if old[p] != new[p]} == {edited}

    # The result has the same geometry as regenerating everything.
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg, force=True)
    full = dvid.mesh_files()
    assert full[''] != after['']  # (The LOD-1 pieces are concatenated in a different order.)
    assert _triangles_by_lod(full[''], full['.index']) == _triangles_by_lod(after[''], after['.index'])


def test_removed_chunks_are_dropped(dvid):
    cfg = _config()
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    before = dvid.mesh_files()

    # Remove the part of the body in chunk (0, 0, 0)
    vol = dvid.vols[0].copy()
    vol[:64, :64, :64] = False
    dvid.set_volume(vol)
    dvid.mutid += 1
    dvid.sparsevol_requests.clear()

    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (0, 26)
    after = dvid.mesh_files()

    old0 = _lod_fragments(before[''], before['.index'], 0)
    new0 = _lod_fragments(after[''], after['.index'], 0)
    assert set(old0) - set(new0) == {(0, 0, 0)}

    # The LOD-1 fragment which contained it no longer has any triangles within it.
    mesh = _read_mesh(after[''], after['.index'])
    frag = next(f for f in mesh['fragments'] if f['lod'] == 1 and tuple(f['position']) == (0, 0, 0))
    centroids = frag['vertices_xyz'][frag['faces']].mean(axis=1)
    assert len(centroids) > 0
    assert not (centroids < 64).all(axis=1).any()


def test_inconsistent_chunks_file_is_ignored(dvid):
    cfg = _config()
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)

    # The '.chunks' file doesn't match the mesh (as if it were written by another process).
    contents = json.loads(dvid.kv[(MESH_INSTANCE, '1.chunks')])
    contents['data-bytes'] += 1
    dvid.kv[(MESH_INSTANCE, '1.chunks')] = json.dumps(contents).encode()

    dvid.mutid += 1
    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (27, 0)


def test_stale_chunks():
    from types import SimpleNamespace
    from neuclease.misc.multiresmesh import stale_chunks

    chunk_df = pd.DataFrame({
        'z': [0] * 7,
        'y': [0] * 7,
        'x': [0, 64, 128, 192, 256, 320, 384],
        'surface_mutid': [5, 5, 11, 5, 0, 7, 12],
        'block_hash': ['0xa', '0xb', '0xc', '0xdd', '0xe', '0xf', '0x9'],
    })
    chunks = {
        (0, 0, 0): ('0xa', None),   # unchanged (unknown recorded mutid)
        (2, 0, 0): ('0xc', None),   # newer surface
        (3, 0, 0): ('0xd', None),   # different blocks (e.g. a merge)
        (4, 0, 0): ('0xe', None),   # unknown current surface_mutid
        (5, 0, 0): ('0xf', 7),      # unchanged (same recorded mutid)
        (6, 0, 0): ('0x9', 4),      # changed since (a different recorded mutid, and newer than lastmod)
    }

    # Same lineage: unknown mutids are compared with the previous mesh's lastmod.
    previous = SimpleNamespace(lastmod=10, same_lineage=True, chunks=chunks)
    stale = stale_chunks(chunk_df, [64, 64, 64], previous)
    assert stale.tolist() == [False, True, True, True, True, False, True]
    assert stale_chunks(chunk_df, [64, 64, 64], None).all()

    # Other lineage: only chunks with the same recorded mutid can be reused.
    previous = SimpleNamespace(lastmod=10, same_lineage=False, chunks=chunks)
    assert stale_chunks(chunk_df, [64, 64, 64], previous).tolist() == [True, True, True, True, True, False, True]

    # A recorded mutid which differs (e.g. the previous mesh is from a *newer* state, which
    # changed this chunk) means the chunk is stale, even if its surface_mutid is <= lastmod.
    chunks = {(0, 0, 0): ('0xa', 8)}
    previous = SimpleNamespace(lastmod=10, same_lineage=False, chunks=chunks)
    assert stale_chunks(chunk_df.iloc[:1], [64, 64, 64], previous).tolist() == [True]


def test_object_store_reuses_chunks(dvid):
    cfg = _config(storage='mem://bucket/meshes')
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    store = open_store('mem://bucket/meshes')
    assert store.list('1') == ['1', '1.chunks', '1.index']

    dvid.edit_chunk((64, 64, 128))
    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (1, 26)
    assert store.stat('1.chunks').metadata == {'uuid': UUID, 'lastmod': '11'}


SV_INSTANCE = 'segmentation_multires_sv_meshes'


def test_sv_mesh_in_dvid(dvid):
    from neuclease.misc.multiresmesh import update_multires_sv_mesh, locate_multires_sv_mesh
    cfg = _config(**{'supervoxel-storage': 'dvid'})

    index = update_multires_sv_mesh('fake', UUID, 'segmentation', 7, cfg)
    assert index

    # The instance was created unversioned (so the meshes are visible from every UUID).
    assert dvid.created_instances == [(SV_INSTANCE, 'keyvalue', False, {'type': 'meshes'})]
    assert dvid.kv[(SV_INSTANCE, '7.index')] == index
    assert json.loads(dvid.kv[(SV_INSTANCE, 'mesh-settings')]) == multiresmesh.mesh_settings(cfg)
    assert json.loads(dvid.kv[(SV_INSTANCE, 'info')])['@type'] == 'neuroglancer_multilod_draco'

    # The supervoxel's chunks were meshed (and only the supervoxel's voxels).
    num_requests = len(dvid.sparsevol_requests)
    assert num_requests == 2 * 3 * 3   # x < 96 spans two chunks in x
    mesh = _read_mesh(dvid.kv[(SV_INSTANCE, '7')], index)
    v = np.concatenate([f['vertices_xyz'] for f in mesh['fragments']])
    assert v[:, 0].max() <= 96 + 1
    assert {f['lod'] for f in mesh['fragments']} == {0, 1}

    # The mesh is never regenerated...
    assert update_multires_sv_mesh('fake', UUID, 'segmentation', 7, cfg) == index
    assert update_multires_sv_mesh('fake', 'some-other-uuid', 'segmentation', 7, cfg) == index
    assert len(dvid.sparsevol_requests) == num_requests

    loc = locate_multires_sv_mesh('fake', UUID, 'segmentation', 7, cfg)
    assert (loc.store.instance, loc.data_name, loc.index_name) == (SV_INSTANCE, '7', '7.index')
    assert loc.store.public_url('7') is None

    # ...so meshes with different settings can't be stored in the same place.
    multiresmesh._verified_sv_stores.clear()
    with pytest.raises(RuntimeError, match='different settings'):
        update_multires_sv_mesh('fake', UUID, 'segmentation', 7, _config(**{'supervoxel-storage': 'dvid', 'smoothing': 3}))

    # But they can be stored elsewhere.
    other = _config(**{'supervoxel-storage': 'dvid:other_sv_meshes', 'smoothing': 3})
    assert update_multires_sv_mesh('fake', UUID, 'segmentation', 7, other)
    assert ('other_sv_meshes', '7.index') in dvid.kv


def test_sv_mesh_without_geometry(dvid):
    from neuclease.misc.multiresmesh import update_multires_sv_mesh
    cfg = _config(**{'supervoxel-storage': 'dvid'})

    # Supervoxel 8 vanishes at the source scale, so its mesh is empty (and that's stored, too).
    assert update_multires_sv_mesh('fake', UUID, 'segmentation', 8, cfg) == b''
    assert dvid.kv[(SV_INSTANCE, '8.index')] == b''
    num_requests = len(dvid.sparsevol_requests)
    assert update_multires_sv_mesh('fake', UUID, 'segmentation', 8, cfg) == b''
    assert len(dvid.sparsevol_requests) == num_requests

    # A supervoxel which doesn't exist (e.g. it was split) isn't stored.
    with pytest.raises(HTTPError):
        update_multires_sv_mesh('fake', UUID, 'segmentation', 9, cfg)
    assert (SV_INSTANCE, '9.index') not in dvid.kv


def test_sv_mesh_in_object_store(dvid):
    from neuclease.misc.multiresmesh import update_multires_sv_mesh, locate_multires_sv_mesh
    cfg = _config(**{'storage': 'mem://bucket/bodies', 'supervoxel-storage': 'mem://bucket/svs'})
    index = update_multires_sv_mesh('fake', UUID, 'segmentation', 7, cfg)
    store = open_store('mem://bucket/svs')
    assert store.list('') == ['7', '7.index', 'info', 'mesh-settings']
    assert store.get('7.index') == index
    assert open_store('mem://bucket/bodies').list('') == []
    assert dvid.created_instances == []

    loc = locate_multires_sv_mesh('fake', UUID, 'segmentation', 7, cfg)
    assert loc.store is store

    with pytest.raises(ValueError):
        _config(**{'storage': 'mem://bucket/bodies', 'supervoxel-storage': 'mem://bucket/bodies/'})
    with pytest.raises(ValueError, match="no 'supervoxel-storage'"):
        update_multires_sv_mesh('fake', UUID, 'segmentation', 7, _config())


def _chunks_contents(store_or_dvid, name='1.chunks'):
    if isinstance(store_or_dvid, FakeDVID):
        return json.loads(store_or_dvid.kv[(MESH_INSTANCE, name)])
    return json.loads(store_or_dvid.get(name))


def _hide_mesh_info(dvid):
    """Simulate a request in a UUID in which the body's mesh_info isn't visible (e.g. an older UUID)."""
    dvid.kv.pop((INFO_INSTANCE, '1'), None)


def test_chunks_file_records_surface_mutids(dvid):
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, _config())
    contents = _chunks_contents(dvid)
    assert contents['version'] == 2
    assert len(contents['chunks']) == 27
    assert all(len(e) == 5 and e[4] == 10 for e in contents['chunks'])


def test_mutids_not_recorded_if_body_changed_during_update(dvid, monkeypatch):
    real_fetch_lastmod = dvid.fetch_lastmod
    calls = []

    def fetch_lastmod(*args, **kwargs):
        calls.append(1)
        result = real_fetch_lastmod(*args, **kwargs)
        if len(calls) > 1:
            result = {'mutation id': result['mutation id'] + 1}   # An edit happened meanwhile
        return result

    monkeypatch.setattr(multiresmesh, 'fetch_lastmod', fetch_lastmod)
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, _config())
    assert all(e[4] is None for e in _chunks_contents(dvid)['chunks'])


def test_chunks_reused_from_live_mesh_in_other_uuids(dvid):
    """Without a visible mesh_info, the live mesh's chunks are reused if their recorded mutids match."""
    cfg = _config(storage='mem://bucket/meshes')
    store = open_store('mem://bucket/meshes')
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)

    _hide_mesh_info(dvid)
    num_requests = len(dvid.sparsevol_requests)
    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (0, 27)
    assert len(dvid.sparsevol_requests) == num_requests


def test_live_mesh_from_newer_state(dvid):
    """
    If the live mesh is from a newer state of the body (e.g. the head), and we're updating
    an older state (e.g. an older UUID), the chunks changed since then are regenerated.
    """
    cfg = _config(storage='mem://bucket/meshes')
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    old_surface_mutids = dvid.surface_mutids.copy()

    # A newer state, with one chunk edited
    dvid.edit_chunk((64, 64, 128))
    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert mesh_info['generated-chunk-count'] == 1

    # Now update the older state (where that chunk still has its old surface_mutid) in an
    # older UUID, with only the newer mesh available.
    old_uuid = 'd' * 32
    dvid.surface_mutids = old_surface_mutids
    dvid.mutid = 10
    _hide_mesh_info(dvid)
    dvid.sparsevol_requests.clear()
    mesh_info = update_multires_body_mesh('fake', old_uuid, 'segmentation', 1, cfg)
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (1, 26)
    assert dvid.sparsevol_requests == [(30, 30, 62)]

    # (The older UUID's mesh is stored under its own name, leaving the live mesh alone.)
    store = open_store('mem://bucket/meshes')
    assert store.stat('1').metadata == {'uuid': UUID, 'lastmod': '11'}
    assert store.stat(f'1-{old_uuid}').metadata == {'uuid': old_uuid, 'lastmod': '10'}


def _make_v1(contents):
    contents = dict(contents, version=1)
    contents['chunks'] = [e[:4] for e in contents['chunks']]
    return contents


def test_v1_chunks_files(dvid):
    """Version-1 '.chunks' files (without mutids) are reused in the same lineage, but not otherwise."""
    cfg = _config(storage='mem://bucket/meshes')
    store = open_store('mem://bucket/meshes')
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    meta = store.stat('1.chunks').metadata
    store.put('1.chunks', json.dumps(_make_v1(_chunks_contents(store))).encode(), metadata=meta)

    # Same lineage (the mesh_info is visible): reused
    dvid.edit_chunk((64, 64, 128))
    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (1, 26)

    # The new file records all of the mutids (including the reused chunks').
    contents = _chunks_contents(store)
    assert contents['version'] == 2 and all(e[4] is not None for e in contents['chunks'])

    # Other lineage: a v1 file's chunks can't be reused.
    store.put('1.chunks', json.dumps(_make_v1(contents)).encode(), metadata=store.stat('1.chunks').metadata)
    _hide_mesh_info(dvid)
    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (27, 0)


def test_chunks_file_missing_chunks_is_not_used(dvid):
    """If the '.chunks' file doesn't list every chunk with geometry, the previous mesh isn't reused."""
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, _config())
    contents = _chunks_contents(dvid)
    contents['chunks'] = [e for e in contents['chunks'] if e[:3] != [0, 0, 0]]
    dvid.kv[(MESH_INSTANCE, '1.chunks')] = json.dumps(contents).encode()

    dvid.mutid += 1
    mesh_info = update_multires_body_mesh('fake', UUID, 'segmentation', 1, _config())
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (27, 0)


def test_backfill_chunks_file(dvid):
    from neuclease.misc.multiresmesh import backfill_chunks_file
    cfg = _config()
    update_multires_body_mesh('fake', UUID, 'segmentation', 1, cfg)
    complete = _chunks_contents(dvid)
    v1 = json.dumps(_make_v1(complete)).encode()

    # Unchanged body: all mutids are recovered.
    new, status = backfill_chunks_file('fake', 'segmentation', cfg, v1)
    assert status == 'complete'
    assert json.loads(new) == complete
    assert backfill_chunks_file('fake', 'segmentation', cfg, new) == (None, 'already-complete')

    # Changed body: only the unchanged chunks' mutids are recovered.
    dvid.edit_chunk((64, 64, 128))
    new, status = backfill_chunks_file('fake', 'segmentation', cfg, v1)
    assert status == 'partial'
    entries = {tuple(e[:3]): e[4] for e in json.loads(new)['chunks']}
    assert entries.pop((128, 64, 64)) is None
    assert set(entries.values()) == {10}

    # ...which can be used in any UUID.
    store_contents = json.loads(new)
    assert store_contents['lastmod'] == 10

    # The body hasn't changed, but the hashes don't match (shouldn't happen)
    dvid.mutid = 10
    bad = _make_v1(complete)
    bad['chunks'][0][3] = '0xbad'
    assert backfill_chunks_file('fake', 'segmentation', cfg, json.dumps(bad).encode()) == (None, 'hash-mismatch')

    # Different settings
    assert backfill_chunks_file('fake', 'segmentation', _config(smoothing=3), v1) == (None, 'settings-mismatch')

    # No such body (anymore)
    gone = dict(_make_v1(complete), body=2)
    assert backfill_chunks_file('fake', 'segmentation', cfg, json.dumps(gone).encode()) == (None, 'body-not-found')


def test_init_supervoxel_mesh_store(dvid):
    from neuclease.misc.multiresmesh import init_supervoxel_mesh_store, DvidKeyValueStore
    cfg = _config(**{'supervoxel-storage': 'dvid'})
    store = init_supervoxel_mesh_store('fake', UUID, 'segmentation', cfg)
    assert isinstance(store, DvidKeyValueStore) and store.instance == SV_INSTANCE
    assert json.loads(dvid.kv[(SV_INSTANCE, 'mesh-settings')]) == multiresmesh.mesh_settings(cfg)

    # (Idempotent)
    init_supervoxel_mesh_store('fake', UUID, 'segmentation', cfg)
    assert dvid.created_instances == [(SV_INSTANCE, 'keyvalue', False, {'type': 'meshes'})]

    multiresmesh._verified_sv_stores.clear()
    with pytest.raises(multiresmesh.SupervoxelMeshSettingsMismatch):
        init_supervoxel_mesh_store('fake', UUID, 'segmentation', _config(**{'supervoxel-storage': 'dvid', 'smoothing': 3}))
