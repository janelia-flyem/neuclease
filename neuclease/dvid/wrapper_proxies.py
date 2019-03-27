"""
Usually, each of our wrapper functions for dvid endpoints
is simply named after the final endpoint name.

For example:
    
    .. code-block:: python
    
        # GET <server>/api/node/<uuid>/<instance>/keys
        fetch_keys(server, uuid, instance)

But this can lead to ambiguity.  Sometimes different API calls
happen to use the same name in the final position:

For example:

    .. code-block::
        
        # GET <server>/api/server/info
        # GET <server>/api/repo/<uuid>/info
        # GET <server>/api/node/<uuid>/<instance>/info

Without special handling, our wrapper function naming convention would lead to ambiguity:

    .. code-block::

        from neuclease.dvid import fetch_info
        info = fetch_info(...) # Which fetch_info() will this call?


We offer three alternatives to allow clients to resolve the ambiguity:

1. The client can import (only) the exact module that defines the wrapper of interest:

    .. code-block::

        from neuclease.dvid.server import fetch_info
        server_info = fetch_info(server)

        from neuclease.dvid.repo import fetch_info
        repo_info = fetch_info(server, instance)

        from neuclease.dvid.node import fetch_info
        node_info = fetch_info(server, instance, node)

2. In ambiguous cases, we also define synonyms for the given wrapper
   functions, with more precise names:
 
    .. code-block:: python

        from neuclease.dvid import fetch_node_info
        node_info = fetch_node_info(server, instance, node)

3. In some cases, the intended wrapper function can be easily inferred from the arguments,
   in which case we simply define a proxy function of the same name, which dispatches
   to the intended wrapper:

    .. code-block:: python

        from neuclease.dvid import fetch_info # proxy function
        server_info = fetch_info(server)
        repo_info = fetch_info(server, instance)
        node_info = fetch_info(server, instance, node)

    Such proxy functions are defined below, in this module (``wrapper_proxies.py``).

    But in cases where the intended wrapper function cannot be easily inferred,
    we will define a proxy function that merely raises an ``AssertionError``, asking
    the client to explicitly disambiguate the call.
"""
from . import dvid_api_wrapper
from .server import fetch_server_info
from .repo import fetch_repo_info
from .node import fetch_instance_info

@dvid_api_wrapper
def fetch_info(server, uuid=None, instance=None, *, session=None):
    """
    Convenience wrapper to call either ``fetch_server_info()``,
    ``fetch_repo_info()``, or ``fetch_instance_info()``,
    depending on which parameters you pass.
    """
    assert instance is None or uuid is not None, \
        "Can't request instance info without a UUID"
    
    if instance is not None:
        return fetch_instance_info(server, uuid, instance, session=session)
    if uuid is not None:
        return fetch_repo_info(server, uuid, session=session)

    return fetch_server_info(server, session=session)
