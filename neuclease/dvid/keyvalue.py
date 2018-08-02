from . import sanitize_server, default_dvid_session, fetch_generic_json

@sanitize_server
def fetch_keys(instance_info):
    """
    Fetches the complete list of keys in the instance (not their values).
    
    WARNING: In the current version of DVID, which uses the basholeveldb backend,
             this will be VERY SLOW for instances with a lot of data.
             (The speed depends on the total size of the values, not the number of keys.)
    """
    server, uuid, instance = instance_info
    return fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/keys')


@sanitize_server
def fetch_keyrange(instance_info, key1, key2):
    """
    Returns all keys between 'key1' and 'key2' for
    the given data instance (not their values).
    
    WARNING: This can be slow for large ranges.
    """
    server, uuid, instance = instance_info
    url = f'http://{server}/api/node/{uuid}/{instance}/keyrange/{key1}/{key2}'
    return fetch_generic_json(url)
    

@sanitize_server
def fetch_key(instance_info, key, as_json=False):
    server, uuid, instance = instance_info
    session = default_dvid_session()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/key/{key}')
    r.raise_for_status()
    if as_json:
        return r.json()
    return r.content


@sanitize_server
def post_key(instance_info, key, data):
    server, uuid, instance = instance_info
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/key/{key}', data=data)
    r.raise_for_status()
    

