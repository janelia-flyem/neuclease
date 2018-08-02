from . import sanitize_server, default_dvid_session
from .repo import create_instance

def create_tarsupervoxel_instance(instance_info, sync_instance, extension, tags=[]):
    """
    Create a tarsupervoxel instance and sync it to a labelmap instance.
    """
    if extension[0] == '.':
        extension = extension[1:]

    create_instance(instance_info, "tarsupervoxels", versioned=False, tags=tags,
                    type_specific_settings={"Extension": extension})
    
    post_tarsupervoxel_sync(instance_info, sync_instance)


@sanitize_server
def post_tarsupervoxel_sync(instance_info, sync_instance, replace=False):
    server, uuid, instance = instance_info
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/sync',
                     params={ "replace": str(bool(replace)).lower() },
                     json={ "sync": sync_instance } )
    r.raise_for_status()


@sanitize_server
def fetch_tarfile(instance_info, body_id, output=None):
    """
    Fetch a .tar file from a tarsupervoxels instance for the given body,
    and save it to bytes, a file object, or a file path.
    
    Args:
        instance_info:
            server, uuid, instance
        
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
    server, uuid, instance = instance_info
    url = f'http://{server}/api/node/{uuid}/{instance}/tarfile/{body_id}'
    return fetch_file(url, output)


def fetch_file(url, output=None, chunksize=2**10):
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
    session = default_dvid_session()

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


