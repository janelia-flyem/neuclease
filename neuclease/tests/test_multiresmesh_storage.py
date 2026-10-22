"""
Tests for storing multires body meshes in an object store (see multiresmesh.py and meshstore.py).
"""
import json

import pytest
import requests
from requests import HTTPError

from neuclease.misc import multiresmesh
from neuclease.misc.meshstore import MemoryStore, open_store, PreconditionFailed

UUID_A = 'a' * 32
UUID_B = 'b' * 32   # A child of UUID_A
UUID_C = 'c' * 32   # A side branch
INFO_JSON = {'@type': 'neuroglancer_multilod_draco'}


def _not_found():
    r = requests.Response()
    r.status_code = 404
    return HTTPError("404", response=r)


class FakeRepo:
    """
    Just enough of DVID for the object store protocol: the head of the master branch,
    and the mesh_info for each UUID (including inheritance from parent UUIDs).
    """
    def __init__(self, monkeypatch):
        self.head = UUID_A
        self.parents = {UUID_B: UUID_A, UUID_C: UUID_A}
        self.mesh_info = {}

        monkeypatch.setattr(multiresmesh, 'resolve_ref', self.resolve_ref)
        monkeypatch.setattr(multiresmesh, 'fetch_key', self.fetch_key)

    def resolve_ref(self, server, ref, expand=False, **kwargs):
        assert ref.endswith(':master')
        return self.head

    def fetch_key(self, server, uuid, instance, key, as_json=False, **kwargs):
        assert instance == 'segmentation_multires_mesh_info'
        while uuid:
            if (uuid, int(key)) in self.mesh_info:
                return self.mesh_info[(uuid, int(key))]
            uuid = self.parents.get(uuid)
        raise _not_found()

    def update(self, store, uuid, body, lastmod):
        """Store a mesh as update_multires_body_mesh() would, and return True if it was stored."""
        files = {suffix: f'{suffix or "data"}-{uuid[0]}-{lastmod}'.encode()
                 for suffix in multiresmesh.MESH_FILE_SUFFIXES}
        stored = multiresmesh._store_body_mesh_in_object_store(
            'http://fake-dvid', uuid, 'segmentation', body, store, 'master', files, INFO_JSON, lastmod)
        if stored:
            self.mesh_info[(uuid, body)] = {
                'body': body, 'lastmod': lastmod, 'written-uuid': uuid,
                'method': multiresmesh.MESH_METHOD, 'storage': store.url,
            }
        return stored

    def read(self, store, uuid, body):
        """Return the (data, index, chunks) contents of the mesh which a reader in the given UUID would find."""
        loc = multiresmesh.locate_multires_body_mesh('http://fake-dvid', uuid, 'segmentation', body)
        if loc is None:
            return None
        return store.get(loc.data_name), store.get(loc.index_name), store.get(loc.chunks_name)


def _contents(uuid, lastmod):
    return tuple(f'{name}-{uuid[0]}-{lastmod}'.encode() for name in ('data', '.index', '.chunks'))


@pytest.fixture
def store():
    MemoryStore.clear_all()
    multiresmesh._verified_store_infos.clear()
    return open_store('mem://test-bucket/meshes')


def test_live_mesh_and_history(monkeypatch, store):
    repo = FakeRepo(monkeypatch)

    # The first mesh (in the head UUID) is stored under the standard names.
    assert repo.update(store, UUID_A, 1, 10)
    assert store.list('1') == ['1', '1.chunks', '1.index']
    assert json.loads(store.get('info')) == INFO_JSON
    assert repo.read(store, UUID_A, 1) == _contents(UUID_A, 10)

    # An update within the same UUID simply replaces it (intermediate states aren't kept).
    assert repo.update(store, UUID_A, 1, 12)
    assert repo.read(store, UUID_A, 1) == _contents(UUID_A, 12)
    assert store.list('1') == ['1', '1.chunks', '1.index']

    # A new head UUID inherits the mesh until the body changes.
    repo.head = UUID_B
    assert repo.read(store, UUID_B, 1) == _contents(UUID_A, 12)

    # When the body is updated in the new head, the old mesh is preserved for its UUID.
    assert repo.update(store, UUID_B, 1, 20)
    assert store.list('1') == ['1', f'1-{UUID_A}', f'1-{UUID_A}.chunks', f'1-{UUID_A}.index', '1.chunks', '1.index']
    assert repo.read(store, UUID_B, 1) == _contents(UUID_B, 20)
    assert repo.read(store, UUID_A, 1) == _contents(UUID_A, 12)

    # A mesh for a UUID other than the head never replaces the live mesh.
    assert repo.update(store, UUID_C, 1, 15)
    assert store.get(f'1-{UUID_C}') == b'data-c-15'
    assert repo.read(store, UUID_C, 1) == _contents(UUID_C, 15)
    assert repo.read(store, UUID_B, 1) == _contents(UUID_B, 20)

    # Likewise, regenerating an old UUID's mesh replaces only its own copy.
    assert repo.update(store, UUID_A, 1, 13)
    assert repo.read(store, UUID_A, 1) == _contents(UUID_A, 13)
    assert repo.read(store, UUID_B, 1) == _contents(UUID_B, 20)


def test_live_mesh_from_unreferenced_state_is_not_preserved(monkeypatch, store):
    """
    If the live mesh doesn't match its UUID's mesh_info (e.g. that UUID's mesh was
    regenerated since then), it isn't copied when it is replaced.
    """
    repo = FakeRepo(monkeypatch)
    assert repo.update(store, UUID_A, 1, 10)

    # UUID A's mesh_info now refers to a different mesh
    repo.mesh_info[(UUID_A, 1)]['lastmod'] = 11

    repo.head = UUID_B
    assert repo.update(store, UUID_B, 1, 20)
    assert store.list('1') == ['1', '1.chunks', '1.index']


def test_newer_mesh_in_same_uuid_is_not_replaced(monkeypatch, store):
    repo = FakeRepo(monkeypatch)
    assert repo.update(store, UUID_A, 1, 20)
    assert not repo.update(store, UUID_A, 1, 15)
    assert repo.read(store, UUID_A, 1) == _contents(UUID_A, 20)


def test_concurrent_modification_is_retried(monkeypatch, store):
    """If another process changes the live mesh while we're replacing it, we start over."""
    repo = FakeRepo(monkeypatch)
    assert repo.update(store, UUID_A, 1, 10)

    repo.head = UUID_B
    real_put = type(store).put
    interfered = []

    def put(self, name, data, **kwargs):
        if name == '1' and not interfered:
            # Another process (in the same new head) stores its own mesh first.
            interfered.append(True)
            for suffix, contents in zip(('', '.index', '.chunks'), _contents(UUID_B, 18)):
                real_put(self, f'1{suffix}', contents, metadata={'uuid': UUID_B, 'lastmod': '18'})
        return real_put(self, name, data, **kwargs)

    monkeypatch.setattr(type(store), 'put', put)
    assert repo.update(store, UUID_B, 1, 20)
    assert interfered
    assert repo.read(store, UUID_B, 1) == _contents(UUID_B, 20)


def test_info_written_only_when_necessary(monkeypatch, store):
    """
    The dataset-wide 'info' file is only written if it's missing or different,
    since Cloud Storage permits only ~1 write per second to a single object.
    """
    repo = FakeRepo(monkeypatch)
    info_writes = []
    real_put = type(store).put

    def put(self, name, data, **kwargs):
        if name == 'info':
            info_writes.append(data)
        return real_put(self, name, data, **kwargs)

    monkeypatch.setattr(type(store), 'put', put)
    for body in range(1, 6):
        assert repo.update(store, UUID_A, body, 10)
    assert len(info_writes) == 1
    assert json.loads(store.get('info')) == INFO_JSON

    # Another process (with no memory of having verified it) doesn't rewrite it either.
    multiresmesh._verified_store_infos.clear()
    assert repo.update(store, UUID_A, 6, 10)
    assert len(info_writes) == 1

    # But an out-of-date info is replaced.
    real_put(store, 'info', b'{"old": "info"}')
    multiresmesh._verified_store_infos.clear()
    assert repo.update(store, UUID_A, 7, 10)
    assert len(info_writes) == 2
    assert json.loads(store.get('info')) == INFO_JSON


def test_info_conflict_is_reported(monkeypatch, store):
    """If another process concurrently writes a *different* info, we complain."""
    real_put = type(store).put

    def put(self, name, data, **kwargs):
        if name == 'info':
            real_put(self, 'info', b'{"from": "another config"}')
        return real_put(self, name, data, **kwargs)

    monkeypatch.setattr(type(store), 'put', put)
    with pytest.raises(RuntimeError, match="doesn't match the current config"):
        multiresmesh._ensure_store_info(store, INFO_JSON)


def test_memory_store_preconditions(store):
    store.put('x', b'1', if_generation_match=0)
    with pytest.raises(PreconditionFailed):
        store.put('x', b'2', if_generation_match=0)
    gen = store.stat('x').generation
    store.put('x', b'2', if_generation_match=gen)
    assert store.get('x') == b'2'
    with pytest.raises(PreconditionFailed):
        store.put('x', b'3', if_generation_match=gen)


@pytest.fixture
def gcs_emulator(monkeypatch):
    """A local GCS emulator.  (It doesn't enforce preconditions or keep old generations.)"""
    pytest.importorskip("google.cloud.storage")
    emulator = pytest.importorskip("gcp_storage_emulator.server")
    import socket
    with socket.socket() as sock:
        sock.bind(('localhost', 0))
        port = sock.getsockname()[1]
    server = emulator.create_server("localhost", port, in_memory=True, default_bucket="test-bucket")
    server.start()
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", f"http://localhost:{port}")
    yield
    server.stop()


def test_gcs_store(gcs_emulator):
    from neuclease.misc.meshstore import GcsStore
    store = GcsStore('gs://test-bucket/some/prefix')

    assert store.get('1') is None
    assert store.stat('1') is None
    store.put('1', b'data', metadata={'uuid': UUID_A, 'lastmod': '10'}, cache_control='no-cache')
    store.put('1.index', b'index')
    store.put('12', b'other body')
    store.put('empty', b'')

    assert store.get('1') == b'data'
    assert store.get('empty') == b''
    assert store.stat('1').metadata == {'uuid': UUID_A, 'lastmod': '10'}
    assert sorted(store.list('1')) == ['1', '1.index', '12']
    assert store.get_many(['1', '12', 'missing']) == {'1': b'data', '12': b'other body'}

    store.copy('1', f'1-{UUID_A}', source_generation=store.stat('1').generation)
    assert store.get(f'1-{UUID_A}') == b'data'
    assert store.public_url('1') == 'https://storage.googleapis.com/test-bucket/some/prefix/1'
