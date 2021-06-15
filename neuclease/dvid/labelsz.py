import logging

import pandas as pd

from . import dvid_api_wrapper, fetch_generic_json
from .repo import create_instance

logger = logging.getLogger(__name__)


@dvid_api_wrapper
def create_labelsz_instance(server, uuid, instance, *, ROI=None, session=None):
    """
    Create a new labelsz instance.

    Equivalent to:
    
        .. code-block:: python
        
            create_instance(server, uuid, instance, type_specific_settings={'ROI': ROI})
    """
    settings = {}
    if ROI:
        settings["ROI"] = ROI
    create_instance(server, uuid, instance, 'labelsz', type_specific_settings=settings, session=session)


@dvid_api_wrapper
def fetch_count(server, uuid, instance, body, element_type, *, session=None):
    """
    Returns the count of the given annotation element type for the given body.

    For synapse indexing, the labelsz data instance must be synced with an annotations instance.
    (future) For number-of-voxels indexing, the labelsz data instance must be synced with a labelvol instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata4:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid labelsz instance name
        
        body:
            The body ID of interest
        
        element_type:
            An indexed element type supported by the labelsz instance,
            i.e. one of: "PostSyn", "PreSyn", "Gap", "Note",
            or the catch-all for synapses "AllSyn", or the number of voxels "Voxels".
    
    Returns:
        JSON. For example: ``{ "Label": 21847,  "PreSyn": 81 }``
    """
    return fetch_generic_json(f'{server}/api/node/{uuid}/{instance}/count/{body}/{element_type}', session=session)


@dvid_api_wrapper
def fetch_counts(server, uuid, instance, bodies, element_type, format='pandas', *, session=None):
    """
    Returns the count of the given annotation element type for the given labels.

    For synapse indexing, the labelsz data instance must be synced with an annotations instance.
    (future) For number-of-voxels indexing, the labelsz data instance must be synced with a labelvol instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata4:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid labelsz instance name
        
        bodies:
            A list of body IDs
        
        element_type:
            An indexed element type supported by the labelsz instance,
            i.e. one of: "PostSyn", "PreSyn", "Gap", "Note",
            or the catch-all for synapses "AllSyn", or the number of voxels "Voxels".
    
    Returns:
        JSON. For example: ``[{ "Label": 21847,  "PreSyn": 81 }, { "Label": 23, "PreSyn": 65 }, ...]``
    """
    assert format in ('json', 'pandas')
    bodies = [int(body) for body in bodies]
    counts = fetch_generic_json(f'{server}/api/node/{uuid}/{instance}/counts/{element_type}', json=bodies, session=session)
    if format == 'json':
        return counts
    
    counts = pd.DataFrame(counts).set_index('Label').rename_axis('body')[element_type]
    return counts


@dvid_api_wrapper
def fetch_top(server, uuid, instance, n, element_type, *, session=None):
    """
    Returns a list of the top N labels with respect to number of the specified index type.

    Args:
        server:
            dvid server, e.g. 'emdata4:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid labelsz instance name
        
        n:
            The number of labels to return counts for.
        
        element_type:
            An indexed element type supported by the labelsz instance,
            i.e. one of: "PostSyn", "PreSyn", "Gap", "Note",
            or the catch-all for synapses "AllSyn", or the number of voxels "Voxels".
    
    Returns:
        JSON. For example: ``[{ "Label": 21847,  "PreSyn": 81 }, { "Label": 23, "PreSyn": 65 }, ...]``
    """
    return fetch_generic_json(f'{server}/api/node/{uuid}/{instance}/top/{n}/{element_type}', session=session)


@dvid_api_wrapper
def fetch_threshold(server, uuid, instance, threshold, element_type, offset=0, n=10_000, *, session=None):
    """
    Returns a list of up to 10,000 labels per request that have # given element types >= T.
    The "page" size is 10,000 labels so a call without any query string will return the 
    largest labels with # given element types >= T.  If there are more than 10,000 labels,
    you can access the next 10,000 by setting ``offset=10_000``

    The index type may be any annotation element type ("PostSyn", "PreSyn", "Gap", "Note"),
    the catch-all for synapses "AllSyn", or the number of voxels "Voxels".

    For synapse indexing, the labelsz data instance must be synced with an annotations instance.
    (future) For number-of-voxels indexing, the labelsz data instance must be synced with a labelvol instance.

    Args:
        server:
            dvid server, e.g. 'emdata4:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid labelsz instance name

        threshold:
            bodies of this size or greater will be returned

        offset:
            Which rank to start returning labels for (0 by default)
        
        element_type:
            An indexed element type supported by the labelsz instance,
            i.e. one of: "PostSyn", "PreSyn", "Gap", "Note",
            or the catch-all for synapses "AllSyn", or the number of voxels "Voxels".
    
        n:
            The number of labels to return counts for.
        
    Returns:
        JSON. For example: ``[{ "Label": 21847,  "PreSyn": 81 }, { "Label": 23, "PreSyn": 65 }, ...]``
    """
    params = {}
    if offset is not None:
        params['offset'] = int(offset)

    if n is not None:
        params['n'] = int(n)
    
    r = session.get(f'{server}/api/node/{uuid}/{instance}/threshold/{threshold}/{element_type}', params=params)
    r.raise_for_status()
    return r.json()


@dvid_api_wrapper
def post_sync(server, uuid, instance, sync_instances, replace=False, *, session=None):
    """
    Appends to list of data instances with which the labelsz are synced.
    
    See also: ``neuclease.dvid.wrapper_proxies.post_sync``
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid labelsz instance name
            
        sync_instances:
            list of dvid instances to which the labelsz instance should be synchronized,
            e.g. ['segmentation']
        
        replace:
            If True, replace existing sync instances with the given sync_instances.
            Otherwise append the sync_instances.
    """
    body = { "sync": ",".join(sync_instances) }

    params = {}
    if replace:
        params['replace'] = str(bool(replace)).lower()

    r = session.post(f'{server}/api/node/{uuid}/{instance}/sync', json=body, params=params)
    r.raise_for_status()

# Synonym
post_labelsz_sync = post_sync


@dvid_api_wrapper
def post_reload(server, uuid, instance, *, session=None): # Note: See wrapper_proxies.post_reload()
    """
    Forces asynchornous denormalization from its synced annotations instance.  Can be 
    used to initialize a newly added instance.

    Args:
        server:
            dvid server, e.g. 'emdata4:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid labelsz instance name
    """
    r = session.post(f'{server}/api/node/{uuid}/{instance}/reload')
    r.raise_for_status()


# Synonym
post_labelsz_reload = post_reload
