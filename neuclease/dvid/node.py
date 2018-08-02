from . import dvid_api_wrapper, fetch_generic_json

@dvid_api_wrapper
def fetch_full_instance_info(server, uuid, instance, *, session=None):
    #FIXME: Rename this function to 'fetch_info' or maybe 'fetch_instance_info'
    """
    Returns the full JSON instance info from DVID
    """
    return fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/info', session=session)


@dvid_api_wrapper
def create_branch(server, uuid, branch_name, note=None, custom_uuid=None, *, session=None):
    """
    Create a branch from the given UUID with the given new branch name.
    Branch name must be unique (not used previously in the repo).
    
    If a custom UUID is provided, it will be used by DVID instead of auto-generating one.
    Note: Providing a custom UUID is unusual and should probably be avoided.

    Returns:
        The uuid of the new branch.
    """
    body = {"branch": branch_name}
    if note:
        body["note"] = note

    if custom_uuid:
        body["uuid"] = custom_uuid
        
    r = session.post(f'http://{server}/api/node/{uuid}/branch', json=body)
    r.raise_for_status()
    return r.json()["child"]



