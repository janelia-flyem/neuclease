import json
from io import BytesIO
from tarfile import TarFile
from collections.abc import Mapping

import numpy as np

from .. import dvid_api_wrapper, fetch_generic_json

# $ protoc --python_out=. neuclease/dvid/keyvalue/ingest.proto
from .ingest_pb2 import Keys, KeyValue, KeyValues


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
def post_key(server, uuid, instance, key, data=None, json=None, *, session=None):
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/key/{key}', data=data, json=json)
    r.raise_for_status()
    

@dvid_api_wrapper
def fetch_keyvalues(server, uuid, instance, keys, as_json=False, *, use_jsontar=False, session=None):
    """
    Fetch a list of values from a keyvalue instance in a single batch call.
    The result is returned as a dict `{ key : value }`
        
    Internally, this function can use either the 'jsontar' option to
    fetch the keys as a tarball, or via the default protobuf implementation (faster).
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            keyvalue instance name, e.g. 'focused_merged'
        
        keys:
            A list of keys (strings) to fetch values for
        
        as_json:
            If True, parse the returned values as JSON.
            Otherwise, return bytes.
        
        use_jsontar:
            If True, fetch the data via the 'jsontar' mechanism, rather
            than the default protobuf implementation, which is faster.
    
    Returns:
        dict of `{ key: value }`
    """
    if use_jsontar:
        return _fetch_keyvalues_jsontar_via_jsontar(server, uuid, instance, keys, as_json, session=session)
    else:
        return _fetch_keyvalues_via_protobuf(server, uuid, instance, keys, as_json, session=session)
        

@dvid_api_wrapper
def _fetch_keyvalues_via_protobuf(server, uuid, instance, keys, as_json=False, *, use_jsontar=False, session=None):
    assert not isinstance(keys, str), "keys should be a list (or array) of strings"

    proto_keys = Keys()
    for key in keys:
        proto_keys.keys.append(key)
    
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/keyvalues', data=proto_keys.SerializeToString())
    r.raise_for_status()

    proto_keyvalues = KeyValues()
    proto_keyvalues.ParseFromString(r.content)
    
    try:
        keyvalues = {}
        for kv in proto_keyvalues.kvs:
            if as_json:
                keyvalues[kv.key] = json.loads(kv.value)
            else:
                keyvalues[kv.key] = kv.value
    except json.JSONDecodeError as ex:
        raise RuntimeError(f"Error decoding key '{kv.key}' from value {kv.value}") from ex

    return keyvalues


@dvid_api_wrapper
def _fetch_keyvalues_jsontar_via_jsontar(server, uuid, instance, keys, as_json=False, *, session=None):
    params = {'jsontar': 'true'}
    
    assert not isinstance(keys, str), "keys should be a list (or array) of strings"
    if isinstance(keys, np.ndarray):
        keys = keys.tolist()
    else:
        keys = list(keys)
    
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/keyvalues', params=params, json=keys)
    r.raise_for_status()
    
    tf = TarFile(f'{instance}.tar', fileobj=BytesIO(r.content))
    
    # Note: It is important to iterate over TarInfo *members* (not string names),
    #       since calling extractfile() with a name causes an iteration over all filenames.
    #       That is, looping over names (instead of members) results in quadratic behavior.
    keyvalues = {}
    for member in tf:
        try:
            val = tf.extractfile(member).read()
            if as_json:
                val = json.loads(val)
            keyvalues[member.name] = val
        except json.JSONDecodeError as ex:
            raise RuntimeError(f"Error decoding key '{member.name}' from value {val}") from ex

    return keyvalues


@dvid_api_wrapper
def post_keyvalues(server, uuid, instance, keyvalues, *, session=None):
    """
    Post a batch of key-value pairs to a keyvalue instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            keyvalue instance name, e.g. 'focused_merged'
        
        keyvalues:
            A dictionary of { key : value }, in which key is a string and value is bytes.
            If the value is not bytes, it will be treated as JSON data and encoded to bytes.
    """
    assert isinstance(keyvalues, Mapping)

    kvs = []
    for key, value in keyvalues.items():
        if not isinstance(value, (bytes, str)):
            value = json.dumps(value)
        if isinstance(value, str):
            value = value.encode('utf-8')

        kvs.append( KeyValue(key=key, value=value) )

    proto_keyvalues = KeyValues()
    proto_keyvalues.kvs.extend(kvs)

    url = f'http://{server}/api/node/{uuid}/{instance}/keyvalues'
    r = session.post(url, data=proto_keyvalues.SerializeToString())
    r.raise_for_status()
