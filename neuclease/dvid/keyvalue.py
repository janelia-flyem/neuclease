import json
from io import BytesIO
from tarfile import TarFile
import numpy as np
from . import dvid_api_wrapper, fetch_generic_json


@dvid_api_wrapper
def fetch_keys(server, uuid, instance, *, session=None):
    """
    Fetches the complete list of keys in the instance (not their values).
    
    WARNING: In the current version of DVID, which uses the basholeveldb backend,
             this will be VERY SLOW for instances with a lot of data.
             (The speed depends on the total size of the values, not the number of keys.)
    """
    return fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/keys', session=session)


@dvid_api_wrapper
def fetch_keyrange(server, uuid, instance, key1, key2, *, session=None):
    """
    Returns all keys between 'key1' and 'key2' for
    the given data instance (not their values).
    
    WARNING: This can be slow for large ranges.
    """
    url = f'http://{server}/api/node/{uuid}/{instance}/keyrange/{key1}/{key2}'
    return fetch_generic_json(url, session=session)
    

@dvid_api_wrapper
def fetch_key(server, uuid, instance, key, as_json=False, *, session=None):
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/key/{key}')
    r.raise_for_status()
    if as_json:
        return r.json()
    return r.content


@dvid_api_wrapper
def post_key(server, uuid, instance, key, data, *, session=None):
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/key/{key}', data=data)
    r.raise_for_status()
    

@dvid_api_wrapper
def fetch_keyvalues(server, uuid, instance, keys, as_json=False, *, session=None):
    params = {'jsontar': 'true'}
    
    assert not isinstance(keys, str), "keys should be a list (or array) of strings"
    if isinstance(keys, np.ndarray):
        keys = keys.tolist()
    else:
        keys = list(keys)
    
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/keyvalues', params=params, json=keys)
    r.raise_for_status()
    
    with open('/tmp/test.tar', 'wb') as f:
        f.write(r.content)
    
    tf = TarFile(f'{instance}.tar', fileobj=BytesIO(r.content))
    
    keyvalues = {}
    for key in keys:
        val = tf.extractfile(key).read()
        if as_json:
            val = json.loads(val)
        keyvalues[key] = val

    return keyvalues
