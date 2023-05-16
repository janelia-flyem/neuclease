import os
import glob
import tarfile
from io import BytesIO
from functools import partial
from contextlib import closing

import numpy as np
import pandas as pd

from ..util import fetch_file, post_file, tqdm_proxy, compute_parallel
from . import dvid_api_wrapper, fetch_generic_json
from .repo import create_instance


@dvid_api_wrapper
def create_tarsupervoxel_instance(server, uuid, instance, sync_instance, extension, tags=[], *, session=None):
    """
    Create a tarsupervoxel instance and sync it to a labelmap instance.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid tarsupervoxels instance name, e.g. 'segmentation_sv_meshes'
        
        sync_instance:
            dvid labelmap instance name, e.g. 'segmentation',
            to which the tarsupervoxels instance will be synchronized.
        
        extension:
            tarsupervoxels instances store one file per supervoxel,
            and the file extensions for all supervoxels must match.
            For example, mesh files are typically stored with extension 'drc' or 'obj'.
        
        tags:
            Optional 'tags' to initialize the instance with.
    """
    if extension[0] == '.':
        extension = extension[1:]

    create_instance(server, uuid, instance, "tarsupervoxels", versioned=False, tags=tags,
                    type_specific_settings={"Extension": extension}, session=session)
    
    post_tarsupervoxel_sync(server, uuid, instance, sync_instance, session=session)


@dvid_api_wrapper
def post_tarsupervoxel_sync(server, uuid, instance, sync_instance, replace=False, *, session=None):
    r = session.post(f'{server}/api/node/{uuid}/{instance}/sync',
                     params={ "replace": str(bool(replace)).lower() },
                     json={ "sync": sync_instance } )
    r.raise_for_status()


@dvid_api_wrapper
def fetch_tarfile(server, uuid, instance, body_id, output=None, *, check_head=False, session=None):
    """
    Fetch a .tar file from a tarsupervoxels instance for the given body,
    and save it to bytes, a file object, or a file path.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid tarsupervoxels instance name, e.g. 'segmentation_sv_meshes'
        
        body_id:
            The body whose supervoxel files will be fetched in the tar.
        
        output:
            If None, tarfile is returned in-memory, as bytes.
            If str, it is interpreted as a path to which the .tar file will be written.
            Otherwise, must be a file object to write the bytes to (e.g. a BytesIO object).
        
        check_head:
            If True, don't fetch the tarfile.
            Instead, send a HEAD requests to determine if all the supervoxel files for the
            given tarfile actually exist on the server, and thus the tarfile would be
            complete if you actually requested it.
            Returns True if the HEAD check passed (status 200),
            or False if the HEAD check failed (status 400).
    
    Returns:
        None, unless no output file object/path is provided,
        in which case the tarfile bytes are returned.
        Or, if check_head=True, returns True or False,
        depending on the status of the HEAD check. 

    See also: ``tar_to_dict()``
    """
    url = f'{server}/api/node/{uuid}/{instance}/tarfile/{body_id}'

    # TODO:
    #   Maybe verify that the instance is actually tarsupervoxels,
    #   or otherwise distinguish between check failures and true errors.

    if check_head:
        r = session.head(url)
        if r.status_code not in (200, 400):
            r.raise_for_status()
        return (r.status_code == 200)
    else:
        return fetch_file(url, output, session=session)


def tar_to_dict(tar_bytes, exts=None, exclude_empty=False):
    """
    Utility function.
    Convert a tarfile (given as bytes) into a dict of {name: bytes}
    Avoids a common mis-use of Python's tarfile API that could lead
    to O(N**2) behavior when reading tarfiles with many files.

    Args:
        tar_bytes:
            bytes which encode a tar file.
            The tar must have completely 'flat' structure,
            i.e. it does not contain any directories.

        exts:
            Optional. Extensions of the files of interest.
            Only files with the given extensions will be read;
            others will be ignored.
            If not provided, all files are read.

        exclude_empty:
            If True, drop 0-byte files
    Returns:
      dict of {filename: bytes}
    """
    if isinstance(exts, str):
        exts = [exts]

    tf = tarfile.TarFile(fileobj=BytesIO(tar_bytes))
    members = sorted(tf.getmembers(), key=lambda m: m.name)

    if exts is not None:
        exts = [ext if ext.startswith('.') else f'.{ext}' for ext in exts]

    data = {}
    for member in members:
        ext = os.path.splitext(member.name)[1]
        if (exts is None or ext in exts) and (member.size > 0 or not exclude_empty):
            data[member.name] = tf.extractfile(member).read()
    return data


@dvid_api_wrapper
def post_load(server, uuid, instance, tar, *, session=None):
    """
    Load a group of supervoxel files (e.g. .drc files) into the tarsupervoxels filestore.

    The given tar data should contain files named with the pattern <supervoxel-id>.<extension>,
    where the extension matches the extension specified in the tarsupervoxels instance metadata.
    For example:1234.drc
    (The tarfile should contain no directories.)

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid tarsupervoxels instance name, e.g. 'segmentation_sv_meshes'
        
        tar:
            Tarfile contents.  For convenience, the following types are supported:
              - A path to a .tar file
              - A (binary) file object (such as BytesIO) containing the bytes of a .tar file
              - A dictionary of {name: bytes} with which to construct a tarfile
              - A dictionary of {name: (binary) file object} with which to construct a tarfile
              - A dictionary of {name: path strings} with which to construct a tarfile

        Examples:
            tsv_instance = ('mydvid:8000', 'abc123', 'segmentation_sv_meshes')

            # Provide a .tar file from disk:
            subprocess.check_call('tar -cf ../supervoxel-meshes.tar 123.drc 456.drc 789.drc',
                                  shell=True, cwd=/path/to/meshes)
            post_load(*tsv_instance, 'supervoxel-meshes.tar')
            
            # Provide the .tar binary contents as a stream:
            post_load(*tsv_instance, open('supervoxel-meshes.tar', 'rb'))

            # Provide a dict of names and paths:
            post_load(*tsv_instance, {'123.drc': '/path/to/meshes/123.drc',
                                      '456.drc': '/path/to/meshes/456.drc'})
    """
    url = f'{server}/api/node/{uuid}/{instance}/load'
    
    if isinstance(tar, str):
        if os.path.isdir(tar):
            paths = sorted(glob.glob(tar))
            names = [os.path.basename(p) for p in paths]

            if len(paths) == 0:
                raise RuntimeError(f"Won't make tarfile from empty directory: {tar}")
            exts = [os.path.splitext(p)[1] for p in paths]
            assert len(set(exts)) == 1, \
                f"All files in the tar must have the same extension.  Found extensions: {set(exts)}"
            tar = dict(zip(names, paths))
        else:
            assert tar.endswith('.tar'), "Data to .../load must be a .tar file"

    if isinstance(tar, dict):
        tar = create_tar_from_dict(tar, 'meshes.tar', in_memory=True)

    post_file(url, tar, session=session)


def create_tar_from_dict(d, tar_name, in_memory=False):
    """
    Given a dictionary in one of the formats described below,
    create an in-memory tarfile and return its contents as bytes.
    The tar file with have a flat structure, with no internal
    directories whatsoever. If you were to open this tarfile with
    'tar xf foo.tar', it would barf its entire contents into your
    current working directory, rather than creating a new directory 'foo'.

    d:
        Either ``{item_name: bytes}`` or ``{item_name: file_path}``.
        If the latter, the file_path will be read from disk (as bytes)
        and the contents will be loaded into the tarfile.

    tar_name:
        The pathname of the tarfile.

    Returns:
        bytes
    """
    if in_memory:
        tar_stream = BytesIO()
        _create_tar_from_dict(d, tar_name, tar_stream)
        return tar_stream.getbuffer()

    with open(tar_name, 'wb') as f:
        _create_tar_from_dict(d, tar_name, f)


def _create_tar_from_dict(d, tar_name, f):
    with closing(tarfile.open(tar_name, 'w', f)) as tf:
        for (name, data) in d.items():
            f_info = tarfile.TarInfo(name)
            f_info.size = len(data)

            if isinstance(data, bytes):
                tf.addfile(f_info, BytesIO(data))
            elif isinstance(data, str):
                tf.addfile(f_info, open(data, 'rb'))


@dvid_api_wrapper
def post_supervoxel(server, uuid, instance, supervoxel_id, sv_file, *, session=None):
    """
    Post a supervoxel file (e.g. a mesh file) to a tarsupervoxels instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid tarsupervoxels instance name, e.g. 'segmentation_sv_meshes'
        
        sv:
            The supervoxel ID corresponding to the posted file.
            
        sv_file:
            The file to post.
            Either a path to a file, a (binary) file object, or bytes.
    """
    url = f'{server}/api/node/{uuid}/{instance}/supervoxel/{supervoxel_id}'
    post_file(url, sv_file, session=session)


@dvid_api_wrapper
def fetch_supervoxel(server, uuid, instance, supervoxel_id, output=None, *, session=None):
    """
    Fetch an individual supervoxel file from a tarsupervoxels instance,
    and save it to bytes, a file object, or a file path.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid tarsupervoxels instance name, e.g. 'segmentation_sv_meshes'
        
        supervoxel_id:
            The supervoxel ID whose file will be retrieved.
        
        output:
            If None, tarfile is returned in-memory, as bytes.
            If str, it is interpreted as a path to which the file will be written.
            Otherwise, must be a file object to write the bytes to (e.g. a BytesIO object).
    
    Returns:
        None, unless no output file object/path is provided,
        in which case the file bytes are returned.
    """
    url = f'{server}/api/node/{uuid}/{instance}/supervoxel/{supervoxel_id}'
    return fetch_file(url, output, session=session)


@dvid_api_wrapper
def delete_supervoxel(server, uuid, instance, supervoxel_id, *, session=None):
    """
    Delete a particular supervoxel file (e.g. a mesh file) from a tarsupervoxels instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid tarsupervoxels instance name, e.g. 'segmentation_sv_meshes'
        
        sv:
            The supervoxel ID corresponding to the posted file.
    """
    url = f'{server}/api/node/{uuid}/{instance}/supervoxel/{supervoxel_id}'
    r = session.delete(url)
    r.raise_for_status()


@dvid_api_wrapper
def fetch_exists(server, uuid, instance, supervoxels, batch_size=None, *, session=None, processes=1, show_progress=True):
    """
    Determine if the given supervoxels have associated files
    stored in the given tarsupervoxels instance.
    
    See also: ``fetch_tarfile(..., check_head=True)``
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid tarsupervoxels instance name, e.g. 'segmentation_sv_meshes'
        
        supervoxels:
            list of supervoxel IDs for which to look for files.
        
        batch_size:
            If not None, given, split the list into batches and issue multiple requests.
        
        processes:
            If given, fetch batches in parallel using multiple processes.
        
        show_progress:
            If True, display/log progress information.

    Returns:
        pd.Series of bool, indexed by supervoxel
    """
    url = f'{server}/api/node/{uuid}/{instance}/exists'
    supervoxels = np.asarray(supervoxels, np.uint64)
    
    if batch_size is None:
        batch_size = len(supervoxels)

    sv_chunks = [supervoxels[a:a+batch_size] for a in range(0, len(supervoxels), batch_size)]

    show_progress &= (len(sv_chunks) > 1)

    result_chunks = []
    if processes <= 1 or len(sv_chunks) <= 1:
        for sv_chunk in tqdm_proxy(sv_chunks, disable=not show_progress):
            result = fetch_generic_json(url, json=sv_chunk.tolist(), session=session)
            result = pd.Series(result, dtype=bool, index=sv_chunk)
            result.name = 'exists'
            result.index.name = 'sv'
            result_chunks.append( result )
    else:
        _fetch_batch = partial(fetch_exists, server, uuid, instance)
        result_chunks = compute_parallel(_fetch_batch, sv_chunks, processes=processes, ordered=True, show_progress=show_progress)

    results = pd.concat(result_chunks)
    assert results.index.dtype == np.uint64
    return results


@dvid_api_wrapper
def fetch_missing(server, uuid, instance, body_id, *, session=None):
    """
    For a given body, fetch the list of supervoxels that are missing from a tarsupervoxels instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid tarsupervoxels instance name, e.g. 'segmentation_sv_meshes'
        
        body_id:
            The body ID for which to check for missing supervoxel files.

    Returns:
        np.ndarray of supervoxels that are missing for the given body.
        If no supervoxels are missing, the array will be empty (len 0).
    """
    r = session.get(f'{server}/api/node/{uuid}/{instance}/missing/{body_id}')
    r.raise_for_status()
    return np.array(r.json(), np.uint64)


def fetch_missing_from_bodies(server, uuid, instance, body_ids, *, processes=4, session=None):
    """
    For the given list of bodies, fetch the list of supervoxels
    that are missing from the given tarsupervoxels instance.
    The supervoxels are returned in a dataframe (columns: sv, body).
    Also, if any of the given bodies no longer exist (due to merges),
    a second list is returned.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid tarsupervoxels instance name, e.g. 'segmentation_sv_meshes'

        body_id:
            The body ID for which to check for missing supervoxel files.

    Returns:
        (missing_df, retired_bodies), where

        missing_df:
            DataFrame of missing sv files and the bodies to which they belong.

        retired_bodies:
            An array of the body IDs from the input list which no longer exist in
            the sync'd segmentation instance.
    """
    fn = partial(_fetch_missing, server, uuid, instance)
    dfs = compute_parallel(fn, body_ids, processes=processes)
    df = pd.concat(dfs, ignore_index=True)
    missing_df = df.query('sv != 0').copy()
    retired_bodies = df.query('sv == 0')['body'].values
    return missing_df, retired_bodies


def _fetch_missing(server, uuid, instance, body):
    try:
        missing_svs = fetch_missing(server, uuid, instance, body)
        return pd.DataFrame({'sv': missing_svs, 'body': body}, dtype=np.uint64)
    except Exception as ex:
        if 'has no supervoxels' in str(ex):
            return pd.DataFrame({'sv': [0], 'body': [body]}, dtype=np.uint64)
        else:
            raise
