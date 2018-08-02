from . import dvid_api_wrapper, default_dvid_session, fetch_generic_json

@dvid_api_wrapper
def fetch_keys(server, uuid, instance):
    """
    Fetches the complete list of keys in the instance (not their values).
    
    WARNING: In the current version of DVID, which uses the basholeveldb backend,
             this will be VERY SLOW for instances with a lot of data.
             (The speed depends on the total size of the values, not the number of keys.)
    """
    return fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/keys')


@dvid_api_wrapper
def fetch_keyrange(server, uuid, instance, key1, key2):
    """
    Returns all keys between 'key1' and 'key2' for
    the given data instance (not their values).
    
    WARNING: This can be slow for large ranges.
    """
    url = f'http://{server}/api/node/{uuid}/{instance}/keyrange/{key1}/{key2}'
    return fetch_generic_json(url)
    

@dvid_api_wrapper
def fetch_key(server, uuid, instance, key, as_json=False):
    session = default_dvid_session()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/key/{key}')
    r.raise_for_status()
    if as_json:
        return r.json()
    return r.content


@dvid_api_wrapper
def post_key(server, uuid, instance, key, data):
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/key/{key}', data=data)
    r.raise_for_status()
    

