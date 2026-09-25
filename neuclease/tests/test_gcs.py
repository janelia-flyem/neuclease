import os
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import NotFound, Forbidden

from neuclease.util import gcs


@pytest.fixture(autouse=True)
def clear_client_cache():
    gcs._client.cache_clear()
    yield
    gcs._client.cache_clear()


def _mock_bucket():
    bucket = MagicMock()
    blobs_by_name = {}

    def blob(name):
        if name not in blobs_by_name:
            blobs_by_name[name] = MagicMock()
        return blobs_by_name[name]

    bucket.blob.side_effect = blob
    bucket._blobs_by_name = blobs_by_name
    return bucket


def test_split_gs_path():
    assert gcs.split_gs_path('gs://mybucket/some/path') == ('mybucket', 'some/path')
    assert gcs.split_gs_path('mybucket/some/path') == ('mybucket', 'some/path')
    assert gcs.split_gs_path('mybucket') == ('mybucket', '')


def test_upload_string(monkeypatch):
    bucket = _mock_bucket()
    monkeypatch.setattr(gcs, '_client', lambda: MagicMock(bucket=lambda name: bucket))
    url = gcs.upload_string('mybucket', 'foo.json', '{"a": 1}', disable_cache=True)

    blob = bucket._blobs_by_name['foo.json']
    assert blob.cache_control == 'public, no-store'
    blob.upload_from_string.assert_called_once_with('{"a": 1}', 'application/json')
    assert url == blob.public_url


def test_upload_directory(tmp_path, monkeypatch):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'a.txt').write_text('a')
    (tmp_path / 'sub' / 'b.txt').write_text('b')

    bucket = _mock_bucket()
    monkeypatch.setattr(gcs, '_client', lambda: MagicMock(bucket=lambda name: bucket))
    gcs.upload_directory('mybucket', 'dest', str(tmp_path), disable_cache=True, unset_content_type=True)

    uploaded_names = set(bucket._blobs_by_name.keys())
    assert uploaded_names == {'dest/a.txt', 'dest/sub/b.txt'}
    for blob in bucket._blobs_by_name.values():
        assert blob.cache_control == 'public, no-store'
        blob.upload_from_filename.assert_called_once()
        _, kwargs = blob.upload_from_filename.call_args
        assert kwargs['content_type'] == 'application/octet-stream'


def test_download_to_file_missing(tmp_path, monkeypatch):
    bucket = _mock_bucket()
    bucket._blobs_by_name['missing.txt'] = MagicMock()
    bucket._blobs_by_name['missing.txt'].download_to_filename.side_effect = NotFound('nope')
    monkeypatch.setattr(gcs, '_client', lambda: MagicMock(bucket=lambda name: bucket))

    local_path = str(tmp_path / 'out.txt')
    result = gcs.download_to_file('mybucket', 'missing.txt', local_path)
    assert result is False
    assert not os.path.exists(local_path)


def test_download_to_file_success(tmp_path, monkeypatch):
    bucket = _mock_bucket()
    monkeypatch.setattr(gcs, '_client', lambda: MagicMock(bucket=lambda name: bucket))

    local_path = str(tmp_path / 'out.txt')
    result = gcs.download_to_file('mybucket', 'present.txt', local_path)
    assert result is True
    bucket._blobs_by_name['present.txt'].download_to_filename.assert_called_once_with(local_path)


def test_check_bucket_access_success(monkeypatch):
    bucket = _mock_bucket()
    monkeypatch.setattr(gcs, '_client', lambda: MagicMock(bucket=lambda name: bucket))
    gcs.check_bucket_access('mybucket/some/path')

    test_blob = bucket._blobs_by_name['some/path/_neuclease_test_file.txt']
    test_blob.upload_from_string.assert_called_once()
    test_blob.delete.assert_called_once()


def test_check_bucket_access_failure(monkeypatch):
    bucket = _mock_bucket()
    bucket._blobs_by_name['_neuclease_test_file.txt'] = MagicMock()
    bucket._blobs_by_name['_neuclease_test_file.txt'].upload_from_string.side_effect = Forbidden('nope')
    monkeypatch.setattr(gcs, '_client', lambda: MagicMock(bucket=lambda name: bucket))

    with pytest.raises(RuntimeError, match="Can't access gs://mybucket"):
        gcs.check_bucket_access('mybucket')


def test_set_bucket_public(monkeypatch):
    client = MagicMock()
    bucket = MagicMock()
    client.bucket.return_value = bucket
    policy = MagicMock()
    policy.bindings = []
    bucket.get_iam_policy.return_value = policy
    monkeypatch.setattr(gcs, '_client', lambda: client)

    gcs.set_bucket_public('mybucket')

    assert policy.bindings == [{"role": "roles/storage.objectViewer", "members": {"allUsers"}}]
    bucket.set_iam_policy.assert_called_once_with(policy)


def test_set_bucket_cors(monkeypatch):
    client = MagicMock()
    bucket = MagicMock()
    client.bucket.return_value = bucket
    monkeypatch.setattr(gcs, '_client', lambda: client)

    cors_config = [{"maxAgeSeconds": 3600, "method": ["GET"], "origin": ["*"]}]
    gcs.set_bucket_cors('mybucket', cors_config)

    assert bucket.cors == cors_config
    bucket.patch.assert_called_once()
