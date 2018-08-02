from . import sanitize_server, default_dvid_session, fetch_generic_json

@sanitize_server
def fetch_full_instance_info(instance_info):
    """
    Returns the full JSON instance info from DVID
    """
    server, uuid, instance = instance_info
    return fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/info')


@sanitize_server
def create_branch(server, uuid, branch_name, note=None, custom_uuid=None):
    """
    Create a branch from the given UUID with the given new branch name.
    Branch name must be unique (not used previously in the repo).
    
    If a custom UUID is provided, it will be used by DVID instead of auto-generating one.
    Note: Providing a custom UUID is unusual and should probably be avoided.

    Returns:
        The uuid of the new branch.
    """
    session = default_dvid_session()
    body = {"branch": branch_name}
    if note:
        body["note"] = note

    if custom_uuid:
        body["uuid"] = custom_uuid
        
    r = session.post(f'http://{server}/api/node/{uuid}/branch', json=body)
    r.raise_for_status()
    return r.json()["child"]



