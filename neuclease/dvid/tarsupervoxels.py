import numpy as np
import pandas as pd

from ..util import fetch_file
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
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/sync',
                     params={ "replace": str(bool(replace)).lower() },
                     json={ "sync": sync_instance } )
    r.raise_for_status()


@dvid_api_wrapper
def fetch_tarfile(server, uuid, instance, body_id, output=None, *, session=None):
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
    
    Returns:
        None, unless no output file object/path is provided,
        in which case the tarfile bytes are returned.
    """
    url = f'http://{server}/api/node/{uuid}/{instance}/tarfile/{body_id}'
    return fetch_file(url, output, session=session)


@dvid_api_wrapper
def fetch_exists(server, uuid, instance, supervoxels, *, session=None):
    """
    Determine if the given supervoxels have files loaded into the given tarsupervoxels instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'segmentation'
        
        supervoxels:
            list of supervoxel IDs for which to look for files.
    
    Returns:
        pd.Series of bool, indexed by supervoxel
    """
    url = f'http://{server}/api/node/{uuid}/{instance}/exists'
    supervoxels = np.asarray(supervoxels, np.uint64)
    exists = fetch_generic_json(url, json=supervoxels.tolist(), session=session)

    result = pd.Series(exists, dtype=bool, index=supervoxels)
    result.name = 'exists'
    result.index.name = 'sv'
    return result

