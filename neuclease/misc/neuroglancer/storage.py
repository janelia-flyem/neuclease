"""
Utility functions related to storing neuroglancer states in cloud storage.
"""
import json
import argparse
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def upload_ngstates(bucket_dir, states, threads=0, processes=0, disable_cache=False, return_prefix='https://neuroglancer-demo.appspot.com'):
    """
    Use multithreading or multiprocessing to upload many files in parallel,
    similar to `gsutil -m cp []...]`, except that in this case you must choose
    between multithreading or multiprocessing (not a combination of the two).

    For the return values, neuroglancer links are returned (using the given prefix).
    If no prefix is given, then the URLs to the uploaded files are returned.

    Args:
        bucket_dir: str
            The bucket directory to upload the states to, such as 'gs://mybucket/mydir'
        states: dict
            A dictionary of neuroglancer states, keyed by the filename.
        threads: int
            The number of threads to use for uploading.
        processes: int
            The number of processes to use for uploading.
        disable_cache: bool
            If True, set `public, no-store` metadata on the uploaded files to disable caching.
        return_prefix: str
            If provided, return neuroglancer links using the given neuroglancer deployment.
            If None, then return the direct URLs to the uploaded files in google storage.

    Returns:
        list of str
            A list of links.
    """
    assert bucket_dir.startswith('gs://')
    assert processes == 0 or threads == 0, \
        "Pick multiprocessing or multithreading (or neither), but not both."

    bucket_dir = bucket_dir[len('gs://'):]
    bucket = bucket_dir.split('/')[0]
    dirpath = bucket_dir[1 + len(bucket):]

    blob_names = [dirpath + '/' + str(name) for name in states.keys()]
    blobs = map(json.dumps, states.values())
    args = [
        (bucket, blobname, blob, 'application/json', disable_cache)
        for blobname, blob in zip(blob_names, blobs)
    ]

    if processes > 0:
        with ProcessPoolExecutor(max_workers=processes) as executor:
            urls = list(executor.map(lambda args: upload_to_bucket(*args), args))
    elif threads > 0:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            urls = list(executor.map(lambda args: upload_to_bucket(*args), args))
    else:
        # Sequential execution
        urls = [upload_to_bucket(*arg_tuple) for arg_tuple in args]

    if return_prefix:
        old_prefix = 'https://storage.googleapis.com/'
        assert all(url.startswith(old_prefix) for url in urls)
        return_prefix = return_prefix.rstrip('/') + '/#!gs://'
        urls = [url.replace(old_prefix, return_prefix) for url in urls]
    return urls


def upload_ngstate(bucket_path, state, disable_cache=False, return_prefix='https://neuroglancer-demo.appspot.com'):
    """
    Upload the given JSON state to a gbucket location,
    such as 'gs://flyem-user-links/short/foobar.json'

    For the return value, a neuroglancer link is returned (using the given prefix).
    If no prerix is given, then the URL to the uploaded file is returned.

    Args:
        bucket_path: str
            The bucket directory to upload the state to, such as 'gs://mybucket/mydir'
        state: dict
            The neuroglancer state to upload.
        disable_cache: bool
            If True, set `public, no-store` metadata on the uploaded file to disable caching.
        return_prefix: str
            If provided, return neuroglancer links using the given neuroglancer deployment.
            If None, then return the direct URLs to the uploaded files in google storage.

    Returns:
        str
            A neuroglancer link to the uploaded file or the direct URL to the
            uploaded file in google storage, depending on whether `return_prefix`
            was provided.
    """
    assert bucket_path.startswith('gs://')
    bucket_path = bucket_path[len('gs://'):]

    bucket = bucket_path.split('/')[0]
    filename = bucket_path[1 + len(bucket):]

    state_string = json.dumps(state, indent=2)
    url = upload_to_bucket(bucket, filename, state_string, disable_cache=disable_cache)

    if return_prefix:
        return_prefix = return_prefix.rstrip('/') + '/#!gs://'
        url = url.replace('https://storage.googleapis.com/', return_prefix)
    return url


def upload_json(obj, bucket_path, disable_cache=True):
    """
    Upload the given JSON data to a gbucket location,
    such as 'gs://flyem-user-links/short/foobar.json'
    """
    assert bucket_path.startswith('gs://')
    bucket_path = bucket_path[len('gs://'):]

    bucket = bucket_path.split('/')[0]
    filename = bucket_path[1 + len(bucket):]

    return upload_to_bucket(bucket, filename, json.dumps(obj), 'application/json', disable_cache)


def upload_to_bucket(bucket, blob_name, blob_contents, content_type='application/json', disable_cache=False):
    """
    Upload a blob of data to the specified google storage bucket.

    Args:
        bucket: str or google.cloud.storage.Bucket
            The bucket to upload the data to.
        blob_name: str
            The name of the blob to upload.
        blob_contents: str
            The data to upload.
    """
    if isinstance(bucket, str):
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket)

    blob = bucket.blob(blob_name)
    if disable_cache:
        blob.cache_control = 'public, no-store'
    blob.upload_from_string(blob_contents, content_type)
    return blob.public_url


def make_bucket_public(bucket=None):
    """
    To host data in google storage for neuroglancer for general access,
    you need to make the bucket public and configure permissive CORS settings.

    This function does that.

    This feature is also installed along with ngsidekick as a command-line tool:

        ..code-block:: bash

            make-bucket-public gs://mybucket/mydir

    Args:
        bucket: str or None
            The bucket to make public, such as 'gs://mybucket/mydir'
            If None, then the bucket is read from the command line.
    """
    if bucket is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('bucket')
        bucket = parser.parse_args().bucket
    if bucket.startswith('gs://'):
        bucket = bucket[len('gs://'):]

    subprocess.run(f'gsutil iam ch allUsers:objectViewer gs://{bucket}', shell=True, check=True)

    with tempfile.NamedTemporaryFile('w') as f:
        cors_settings = [{
            "maxAgeSeconds": 3600,
            "method": ["GET"],
            "origin": ["*"],
            "responseHeader": ["Content-Type", "Range"]
        }]
        json.dump(cors_settings, f)
        f.flush()
        subprocess.run(f'gsutil cors set {f.name} gs://{bucket}', shell=True, check=True)

    print(f"Configured bucket for public neuroglancer access: gs://{bucket}")
