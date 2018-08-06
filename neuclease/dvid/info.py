from . import dvid_api_wrapper
from .server import fetch_server_info
from .repo import fetch_repo_info
from .node import fetch_instance_info

@dvid_api_wrapper
def fetch_info(server, uuid=None, instance=None, *, session=None):
    """
    Convenience wrapper to call either fetch_server_info(),
    fetch_repo_info(), or fetch_instance_info(),
    depending on which parameters you pass.
    """
    assert instance is None or uuid is not None, \
        "Can't request instance info without a UUID"
    
    if instance is not None:
        return fetch_instance_info(server, uuid, instance, session=session)
    if uuid is not None:
        return fetch_repo_info(server, uuid, session=session)

    return fetch_server_info(server, session=session)
