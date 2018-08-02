import requests
from . import dvid_api_wrapper
from .repo import create_instance

@dvid_api_wrapper
def create_tarsupervoxel_instance(server, uuid, instance, sync_instance, extension, tags=[], *, session=None):
    """
    Create a tarsupervoxel instance and sync it to a labelmap instance.
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
            dvid instance name, e.g. 'segmentation'
        
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


# FIXME: Put this in util
def fetch_file(url, output=None, chunksize=2**10, *, session=None):
    """
    Fetch a file from the given endpoint,
    and save it to bytes, a file object, or a file path.

    Args:
        url:
            Complete url to fetch from.
        
        output:
            If None, file is returned in-memory, as bytes.
            If str, it is interpreted as a path to which the file will be written.
            Otherwise, must be a file object to write the bytes to (e.g. a BytesIO object).
        
        chunksize:
            Data will be streamed in chunks, with the given chunk size.

    Returns:
        None, unless no output file object/path is provided,
        in which case the fetched bytes are returned.
    """
    session = session or requests.Session()
    with session.get(url, stream=True) as r:
        r.raise_for_status()

        if output is None:
            return r.content

        if isinstance(output, str):
            # Create a file on disk and write to it.
            with open(output, 'wb') as f:
                for chunk in r.iter_content(chunksize):
                    f.write(chunk)
        else:
            # output is a file object
            for chunk in r.iter_content(chunksize):
                output.write(chunk)


