"""
Shared helpers for talking to Google Cloud Storage via the google-cloud-storage
SDK, replacing ad-hoc `gsutil` subprocess calls used elsewhere in this package.
"""
import os
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from google.api_core.exceptions import GoogleAPIError, NotFound

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _client():
    from google.cloud import storage
    return storage.Client()


def split_gs_path(path):
    """
    Parse a 'gs://bucket/blob/path' URL into (bucket_name, blob_path).
    The leading 'gs://' prefix is optional.
    """
    if path.startswith('gs://'):
        path = path[len('gs://'):]
    bucket_name, _, blob_path = path.partition('/')
    return bucket_name, blob_path


def _bucket(bucket):
    """
    Accept either a bucket name (str) or an existing google.cloud.storage.Bucket,
    and return a Bucket object.
    """
    if isinstance(bucket, str):
        bucket_name, _ = split_gs_path(bucket)
        return _client().bucket(bucket_name)
    return bucket


def upload_string(bucket, blob_name, contents, content_type='application/json', disable_cache=False):
    """
    Upload a string (or bytes) of data to the given blob.
    Returns the blob's public URL.
    """
    bucket = _bucket(bucket)
    blob = bucket.blob(blob_name)
    if disable_cache:
        blob.cache_control = 'public, no-store'
    blob.upload_from_string(contents, content_type)
    return blob.public_url


def upload_file(bucket, blob_name, local_path, content_type=None, disable_cache=False):
    """
    Upload the contents of a local file to the given blob.
    Returns the blob's public URL.
    """
    bucket = _bucket(bucket)
    blob = bucket.blob(blob_name)
    if disable_cache:
        blob.cache_control = 'public, no-store'
    blob.upload_from_filename(local_path, content_type=content_type)
    return blob.public_url


def download_to_file(bucket, blob_name, local_path):
    """
    Download the given blob to a local file.
    Returns True if the blob existed and was downloaded, False if it didn't exist.
    """
    bucket = _bucket(bucket)
    blob = bucket.blob(blob_name)
    try:
        blob.download_to_filename(local_path)
        return True
    except NotFound:
        if os.path.exists(local_path):
            os.unlink(local_path)
        return False


def list_blobs(bucket_name, prefix, max_results=None):
    """
    List blobs in the given bucket whose name starts with prefix.
    """
    bucket_name, _ = split_gs_path(bucket_name)
    return list(_client().list_blobs(bucket_name, prefix=prefix, max_results=max_results))


def delete_blob(bucket, blob_name):
    """
    Delete the given blob.
    """
    bucket = _bucket(bucket)
    bucket.blob(blob_name).delete()


def upload_directory(bucket, blob_prefix, local_dir, disable_cache=False, unset_content_type=False, max_workers=8):
    """
    Recursively upload every file under local_dir to the given bucket,
    preserving relative paths underneath blob_prefix.
    """
    bucket = _bucket(bucket)
    blob_prefix = blob_prefix.rstrip('/')

    local_paths = []
    for dirpath, _dirnames, filenames in os.walk(local_dir):
        for filename in filenames:
            local_paths.append(os.path.join(dirpath, filename))

    def _upload(local_path):
        rel_path = os.path.relpath(local_path, local_dir)
        blob_name = f"{blob_prefix}/{rel_path}"
        blob = bucket.blob(blob_name)
        if disable_cache:
            blob.cache_control = 'public, no-store'
        # unset_content_type mirrors gsutil's `-h 'Content-Type:'` (empty value),
        # which forces browsers to download the file rather than display it inline
        # (e.g. for .json files that would otherwise be shown as text).
        content_type = 'application/octet-stream' if unset_content_type else None
        blob.upload_from_filename(local_path, content_type=content_type)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Force evaluation (and propagate exceptions) by consuming the iterator.
        list(executor.map(_upload, local_paths))


def check_bucket_access(path):
    """
    Verify that we have write/delete permission at the given location by
    uploading and then deleting a small test blob there.

    Args:
        path: A bucket name, or 'bucket/path/prefix', or 'gs://bucket/path/prefix'.
    Raises:
        RuntimeError if access fails.
    """
    bucket_name, blob_prefix = split_gs_path(path)
    test_blob_name = f"{blob_prefix}/_neuclease_test_file.txt" if blob_prefix else "_neuclease_test_file.txt"
    try:
        upload_string(bucket_name, test_blob_name, "Just testing my bucket access...\n", content_type='text/plain')
        delete_blob(bucket_name, test_blob_name)
    except GoogleAPIError as ex:
        raise RuntimeError(f"Can't access gs://{path}") from ex


def set_bucket_public(bucket_name):
    """
    Grant allUsers the 'roles/storage.objectViewer' IAM role on the given bucket,
    i.e. the equivalent of `gsutil iam ch allUsers:objectViewer gs://bucket`.
    """
    bucket_name, _ = split_gs_path(bucket_name)
    bucket = _client().bucket(bucket_name)
    policy = bucket.get_iam_policy(requested_policy_version=3)
    policy.bindings.append({"role": "roles/storage.objectViewer", "members": {"allUsers"}})
    bucket.set_iam_policy(policy)


def set_bucket_cors(bucket_name, cors_config):
    """
    Set the CORS configuration on the given bucket,
    i.e. the equivalent of `gsutil cors set <file> gs://bucket`.
    """
    bucket_name, _ = split_gs_path(bucket_name)
    bucket = _client().bucket(bucket_name)
    bucket.cors = cors_config
    bucket.patch()
