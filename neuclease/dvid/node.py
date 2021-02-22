from . import dvid_api_wrapper, fetch_generic_json

@dvid_api_wrapper
def fetch_info(server, uuid, instance, *, session=None):
    """
    Returns the full JSON instance info from DVID

    See also: ``neuclease.dvid.wrapper_proxies.fetch_info()``
    """
    return fetch_generic_json(f'{server}/api/node/{uuid}/{instance}/info', session=session)

# Synonym
fetch_instance_info = fetch_info


@dvid_api_wrapper
def post_info(server, uuid, instance, info, *, session=None):
    """
    Replace the instance info.  Use with caution. 
    """
    r = session.post(f'{server}/api/node/{uuid}/{instance}/info', json=info)
    r.raise_for_status()

@dvid_api_wrapper
def post_commit(server, uuid, note, log=[], *, session=None):
    """
    Commit (a.k.a. lock) the given node with a commit note and optional log.
    Returns the full UUID of the committed node.
    """
    body = { "note": note }
    if log:
        assert isinstance(log, list)
        for item in log:
            assert isinstance(item, str)
        body["log"] = log
    
    r = session.post(f'{server}/api/node/{uuid}/commit', json=body)
    r.raise_for_status()
    return r.json()["committed"]


@dvid_api_wrapper
def fetch_commit(server, uuid, *, session=None):
    """
    Returns the locked status (True/False) if the given node.
    
    Returns True if the node is locked.
    """
    r = session.get(f'{server}/api/node/{uuid}/commit')
    r.raise_for_status()
    return r.json()["Locked"]
    

@dvid_api_wrapper
def post_branch(server, uuid, branch_name, note, custom_uuid=None, *, session=None):
    """
    Create a branch from the given UUID with the given new branch name.
    Branch name must be unique (not used previously in the repo).
    
    If a custom UUID is provided, it will be used by DVID instead of auto-generating one.
    Note: Providing a custom UUID is unusual and should probably be avoided.

    Returns:
        The uuid of the new node (the start of the new branch).
    """
    body = {"branch": branch_name}
    body["note"] = note

    if custom_uuid:
        body["uuid"] = custom_uuid
        
    r = session.post(f'{server}/api/node/{uuid}/branch', json=body)
    r.raise_for_status()
    return r.json()["child"]


@dvid_api_wrapper
def post_newversion(server, uuid, note, custom_uuid=None, *, session=None):
    """
    Creates a new node on the same branch as the given uuid.

    If a custom UUID is provided, it will be used by DVID instead of auto-generating one.
    Note: Providing a custom UUID is unusual and should probably be avoided.

    Returns:
        uuid of the new node
    """
    body = { "note": note }
    if custom_uuid:
        body["uuid"] = custom_uuid
        
    r = session.post(f'{server}/api/node/{uuid}/newversion', json=body)
    r.raise_for_status()
    return r.json()["child"]


@dvid_api_wrapper
def post_blob(server, uuid, instance, data=None, json=None, *, session=None):
    """
    Post a 'blob' of arbitrary data to the DVID server's blobstore.
    
    The server will create a reference for the blob, which is returned.
    The reference is a URL-friendly content hash (FNV-128) of the blob data.
    """
    assert (data is not None) ^ (json is not None), "Must provide either data or json (but not both)"
    r = session.post(f'{server}/api/node/{uuid}/{instance}/blobstore', data=data, json=json)
    r.raise_for_status()
    
    return r.json()["reference"]


@dvid_api_wrapper
def fetch_blob(server, uuid, instance, reference, as_json=False, *, session=None):
    """
    Fetch a previously-stored 'blob' of data from the DVID server's blobstore.
    
    Blobs are stored by various operations, such as supervoxel splits,
    annotation elements POSTs, and others.
    
    Returns:
        Either bytes or parsed JSON data, depending on as_json.
    """
    r = session.get(f'{server}/api/node/{uuid}/{instance}/blobstore/{reference}')
    r.raise_for_status()
    if as_json:
        return r.json()
    return r.content


@dvid_api_wrapper
def fetch_note(server, uuid, *, session=None):
    """
    Fetch the node "note" stored in DVID for the given uuid.
    """
    r = session.get(f"{server}/api/node/{uuid}/note")
    r.raise_for_status()
    return r.json()["note"]


@dvid_api_wrapper
def post_note(server, uuid, note, *, session=None):
    """
    Set the node "note" stored in DVID for the given uuid.
    """
    assert isinstance(note, str)
    body = {"note": note}
    r = session.post(f"{server}/api/node/{uuid}/note", json=body)
    r.raise_for_status()


@dvid_api_wrapper
def fetch_log(server, uuid, *, session=None):
    """
    Fetch the node log stored in DVID.

    The node log is a list of strings associated with the node.
    The messages should be usable by clients to reconstruct the
    types of operations done to that version of data.

    Note:
        This is the node log.  For the repo log, see
        ``dvid.repo.fetch_log()``

    Note:
        Not to be confused with other logs produced by dvid,
        such as the node note, the repo log, the http log,
        the kafka log, or the mutation log.
    """
    r = session.get(f"{server}/api/node/{uuid}/log")
    r.raise_for_status()
    return r.json()["log"]


# Synonym
fetch_node_log = fetch_log


@dvid_api_wrapper
def post_log(server, uuid, messages, *, session=None):
    """
    Append messages to the node log stored in DVID for the given uuid.

    The node log is a list of strings associated with the node.
    The messages should be usable by clients to reconstruct the
    types of operations done to that version of data.

    Note:
        This is the node log.  For the repo log, see
        ``dvid.repo.fetch_log()``

    Note:
        Not to be confused with other logs produced by dvid,
        such as the node note, the repo log, the http log,
        the kafka log, or the mutation log.
    """
    if isinstance(messages, str):
        messages = [messages]
    assert all(isinstance(s, str) for s in messages)
    body = {"log": [*messages]}
    r = session.post(f"{server}/api/node/{uuid}/log", json=body)
    r.raise_for_status()


# Synonym
post_node_log = post_log
