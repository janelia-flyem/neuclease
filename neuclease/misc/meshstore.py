"""
Object storage (e.g. a Google Cloud Storage bucket) for chunk meshes and body meshes,
as an alternative to storing them in DVID keyvalue instances.  (See bodymesh.py.)

A store is identified by a URL:

    gs://<bucket>/<prefix>    Google Cloud Storage
    mem://<name>              An in-memory store, for testing (shared within a process)

Object names are given relative to the store's prefix.
"""
import os
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

ObjectInfo = namedtuple('ObjectInfo', 'generation metadata')


class PreconditionFailed(Exception):
    """A conditional write failed because the object's generation didn't match."""


def open_store(url):
    """
    Return the store for the given URL.
    Stores are cached (per process), so this is cheap to call repeatedly.
    """
    return _open_store(url, os.getpid())


_stores = {}
_stores_lock = threading.Lock()


def _open_store(url, pid):
    with _stores_lock:
        try:
            return _stores[(url, pid)]
        except KeyError:
            pass
        if url.startswith('gs://'):
            store = GcsStore(url)
        elif url.startswith('mem://'):
            store = MemoryStore(url)
        else:
            raise ValueError(f"Unsupported mesh storage URL: {url!r}")
        _stores[(url, pid)] = store
        return store


def _split_url(url, scheme):
    path = url[len(scheme):].strip('/')
    bucket, _, prefix = path.partition('/')
    if not bucket:
        raise ValueError(f"Invalid storage URL: {url!r}")
    return bucket, prefix


class GcsStore:
    """
    A Google Cloud Storage bucket (or a 'directory' within one).

    Credentials are found in the usual way (Application Default Credentials).
    For testing, the STORAGE_EMULATOR_HOST environment variable is honored.
    """

    def __init__(self, url):
        self.url = url.rstrip('/')
        self.bucket_name, self.prefix = _split_url(self.url, 'gs://')
        self._client = None
        self._client_lock = threading.Lock()

    def _path(self, name):
        return f'{self.prefix}/{name}' if self.prefix else name

    def _bucket(self):
        with self._client_lock:
            if self._client is None:
                from google.cloud import storage
                self._client = storage.Client()
            return self._client.bucket(self.bucket_name)

    def list(self, prefix):
        """Return the names of all objects whose name starts with the given prefix."""
        start = len(self._path(''))
        blobs = self._bucket().client.list_blobs(self.bucket_name, prefix=self._path(prefix))
        return [b.name[start:] for b in blobs]

    def get(self, name):
        """Return the object's contents, or None if it doesn't exist."""
        from google.api_core.exceptions import NotFound
        try:
            return self._bucket().blob(self._path(name)).download_as_bytes()
        except NotFound:
            return None

    def get_many(self, names, threads=16):
        """Return {name: contents} for the given objects (omitting those that don't exist)."""
        return _get_many(self, names, threads)

    def stat(self, name):
        """Return the object's ObjectInfo, or None if it doesn't exist."""
        blob = self._bucket().get_blob(self._path(name))
        if blob is None:
            return None
        return ObjectInfo(blob.generation, dict(blob.metadata or {}))

    def put(self, name, data, *, metadata=None, cache_control=None,
            content_type='application/octet-stream', if_generation_match=None):
        """
        Write an object.
        If if_generation_match is given, the write only succeeds if the object's
        current generation matches (0 means the object must not exist yet).
        Otherwise, PreconditionFailed is raised.
        """
        from google.api_core.exceptions import PreconditionFailed as GcsPreconditionFailed
        blob = self._bucket().blob(self._path(name))
        blob.metadata = metadata
        blob.cache_control = cache_control
        try:
            blob.upload_from_string(data, content_type=content_type, if_generation_match=if_generation_match)
        except GcsPreconditionFailed as ex:
            raise PreconditionFailed(f"{self.url}/{name}: {ex}") from ex

    def copy(self, src, dst, *, source_generation=None):
        """Copy an object (optionally a particular generation of it), on the server side."""
        bucket = self._bucket()
        bucket.copy_blob(bucket.blob(self._path(src)), bucket, self._path(dst),
                         source_generation=source_generation)

    def public_url(self, name):
        """The HTTPS URL of the object (readable without credentials if the bucket is public)."""
        from urllib.parse import quote
        return f'https://storage.googleapis.com/{self.bucket_name}/{quote(self._path(name))}'


class MemoryStore:
    """
    An in-memory store with the same interface (and generation semantics) as GcsStore.
    All MemoryStore objects with the same URL share the same contents (within a process).
    """
    _contents = {}
    _lock = threading.Lock()

    def __init__(self, url):
        self.url = url.rstrip('/')
        with self._lock:
            self._objects = self._contents.setdefault(self.url, {})

    @classmethod
    def clear_all(cls):
        with cls._lock:
            for objects in cls._contents.values():
                objects.clear()

    def list(self, prefix):
        with self._lock:
            return sorted(n for n in self._objects if n.startswith(prefix))

    def get(self, name):
        with self._lock:
            obj = self._objects.get(name)
            return obj and obj['data']

    def get_many(self, names, threads=16):
        return _get_many(self, names, threads)

    def stat(self, name):
        with self._lock:
            obj = self._objects.get(name)
            return obj and ObjectInfo(obj['generation'], dict(obj['metadata']))

    def put(self, name, data, *, metadata=None, cache_control=None,
            content_type='application/octet-stream', if_generation_match=None):
        with self._lock:
            current = self._objects.get(name)
            if if_generation_match is not None:
                current_generation = current['generation'] if current else 0
                if current_generation != if_generation_match:
                    raise PreconditionFailed(f"{self.url}/{name}")
            self._objects[name] = self._new_object(bytes(data), metadata, cache_control)

    def copy(self, src, dst, *, source_generation=None):
        with self._lock:
            obj = self._objects[src]
            if source_generation is not None and obj['generation'] != source_generation:
                # (Unlike GCS, we don't keep old generations.)
                raise KeyError(f"{self.url}/{src} generation {source_generation}")
            self._objects[dst] = self._new_object(obj['data'], obj['metadata'], obj['cache_control'])

    def _new_object(self, data, metadata, cache_control):
        MemoryStore._generation_counter = getattr(MemoryStore, '_generation_counter', 0) + 1
        return {'data': data, 'metadata': dict(metadata or {}), 'cache_control': cache_control,
                'generation': MemoryStore._generation_counter}

    def public_url(self, name):
        return f'https://memory-store.invalid/{self.url[len("mem://"):]}/{name}'


def _get_many(store, names, threads):
    names = list(names)
    if len(names) <= 1 or threads <= 1:
        values = map(store.get, names)
    else:
        with ThreadPoolExecutor(min(threads, len(names))) as executor:
            values = list(executor.map(store.get, names))
    return {n: v for n, v in zip(names, values) if v is not None}
