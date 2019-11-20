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
from .repo import fetch_repo_instances, fetch_repo_info
from .node import fetch_instance_info
from .roi import fetch_roi_roi
from .annotation import fetch_annotation_label, post_annotation_sync, post_annotation_reload, fetch_annotation_roi
from .labelmap import fetch_labelmap_label


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


@dvid_api_wrapper
def post_sync(server, uuid, instance, sync_instances, replace=False, *, session=None):
    """
    Convenience wrapper for POST .../sync, which is an endpoint that is supported by multiple instance types.
    
    See also:
        - ``neuclease.dvid.annotation.post_sync()``
        - ``neuclease.dvid.labelsz.post_sync()``
    """
    # It turns out that the POST .../sync arguments and options are the same for annotations and labelsz,
    # so it doesn't actually matter which one we call.
    post_annotation_sync(server, uuid, instance, sync_instances, replace=replace, session=session)


@dvid_api_wrapper
def post_reload(server, uuid, instance, *, session=None, **kwargs):
    """
    Convenience wrapper for both ``labelsz.post_reload()`` and ``annotation.post_reload()``
    """
    # Only the annotation version of post_reload() supports extra options,
    # so if the user supplied any, they must be posting to an annotations instance.
    # Aside from that, labelsz reload and annotation reload calls are identical,
    # so it doesn't matter which wrapper we call.
    post_annotation_reload(server, uuid, instance, **kwargs, session=session)


def fetch_roi(server, uuid, instance, *args, session=None, **kwargs):
    """
    Convenience wrapper for both ``annotation.fetch_roi()`` and ``roi.fetch_roi()``
    """
    instance_type = fetch_repo_instances(server, uuid, session=session)[instance]
    assert instance_type in ('roi', 'annotation'), \
        f"Unexpected instance type for instance '{instance}': '{instance_type}'"

    if instance_type == 'roi':
        return fetch_roi_roi(server, uuid, instance, *args, **kwargs, session=session)
    elif instance_type == 'annotation':
        return fetch_annotation_roi(server, uuid, instance, *args, **kwargs, session=session)


def fetch_label(server, uuid, instance, *args, session=None, **kwargs):
    """
    Convenience wrapper for both ``annotations.fetch_label()`` and ``labelmap.fetch_label()``
    """
    instance_type = fetch_repo_instances(server, uuid, session=session)[instance]
    assert instance_type in ('labelmap', 'annotation'), \
        f"Unexpected instance type for instance '{instance}': '{instance_type}'"

    if instance_type == 'labelmap':
        return fetch_labelmap_label(server, uuid, instance, *args, **kwargs, session=session)
    elif instance_type == 'annotation':
        return fetch_annotation_label(server, uuid, instance, *args, **kwargs, session=session)
