import logging
from io import BytesIO
from tarfile import TarFile
from collections.abc import Mapping

import ujson
import numpy as np
import pandas as pd
import requests

from ...util import tqdm_proxy
from .. import dvid_api_wrapper, fetch_generic_json
from ..node import fetch_instance_info
from ..repo import create_instance

# $ protoc --python_out=. neuclease/dvid/keyvalue/ingest.proto
from .ingest_pb2 import Keys, KeyValue, KeyValues

logger = logging.getLogger(__name__)

@dvid_api_wrapper
def fetch_keys(server, uuid, instance, *, session=None):
    """
    Fetch the complete list of keys in the instance (not their values).
    
    WARNING: In the current version of DVID, which uses the basholeveldb backend,
             this will be VERY SLOW for instances with a lot of data.
             (The speed depends on the total size of the values, not the number of keys.)

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            keyvalue instance name, e.g. 'focused_merged'

    Returns:
        list of strings
    """
    return fetch_generic_json(f'http://{server}/api/node/{uuid}/{instance}/keys', session=session)


@dvid_api_wrapper
def fetch_keyrange(server, uuid, instance, key1, key2, *, session=None):
    """
    Returns all keys between 'key1' and 'key2' for
    the given data instance (not their values).
    
    WARNING: This can be slow for large ranges.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            keyvalue instance name, e.g. 'focused_merged'
    
    Returns:
        List of keys (strings)
        
    Examples:
        
        # Everything from 0...999999999
        keys = fetch_keyrange('emdata3:8900', 'abc9', '0', '999999999')

        # This will catch everything from 'aaa...' to a single 'z', but not 'za'
        keys = fetch_keyrange('emdata3:8900', 'abc9', 'a', 'z')

        # This gets everything from 'aaa...' to 'zzzzz...'
        keys = fetch_keyrange('emdata3:8900', 'abc9', 'a', chr(ord('z')+1))
    """
    url = f'http://{server}/api/node/{uuid}/{instance}/keyrange/{key1}/{key2}'
    return fetch_generic_json(url, session=session)
    

@dvid_api_wrapper
def fetch_key(server, uuid, instance, key, as_json=False, *, session=None):
    """
    Fetch a single value from a DVID keyvalue instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            keyvalue instance name, e.g. 'focused_merged'
        
        key:
            A key (string) whose corresponding value will be fetched.
        
        as_json:
            If True, interpret the value as json and load it into a Python value
    
    Returns:
        Bytes or parsed json data (see ``as_json``)
    """
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/key/{key}')
    r.raise_for_status()
    if as_json:
        return r.json()
    return r.content


@dvid_api_wrapper
def post_key(server, uuid, instance, key, data=None, json=None, *, session=None):
    """
    Post a single value to a DVID keyvalue instance.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            keyvalue instance name, e.g. 'focused_merged'
        
        key:
            A key (string) whose corresponding value will be posted.
        
        data:
            The value to post to the given key, given as bytes.
        
        json:
            Instead of posting bytes, you can provide a JSON-serializable
            Python object via this argument.
            The object will be encoded as JSON and then posted.
    """
    assert data is not None or json is not None, "No data to post"
    if data is not None:
        assert isinstance(data, bytes), f"Raw data must be posted as bytes, not {type(data)}"
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/key/{key}', data=data, json=json)
    r.raise_for_status()
    

@dvid_api_wrapper
def delete_key(server, uuid, instance, key, *, session=None):
    """
    Delete a single key from a dvid keyvalue instance.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            keyvalue instance name, e.g. 'focused_merged'
        
        key:
            A key (string) whose corresponding value will be deleted.
    """
    r = session.delete(f'http://{server}/api/node/{uuid}/{instance}/key/{key}')
    r.raise_for_status()


@dvid_api_wrapper
def fetch_keyvalues(server, uuid, instance, keys, as_json=False, batch_size=None, *, use_jsontar=False, session=None):
    """
    Fetch a list of values from a keyvalue instance in a single batch call,
    or split across multiple batches.
    The result is returned as a dict `{ key : value }`.
    If as_json is True, any keys that do not exist in the instance will
    appear in the results with a value of None.
        
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
        
        batch_size:
            Optional.  Split the keys into batches to
            split the query across multiple DVID calls.
            Otherwise, the values are fetched in a single call (batch_size = len(keys)).
        
        use_jsontar:
            If True, fetch the data via the 'jsontar' mechanism, rather
            than the default protobuf implementation, which is faster.
    
    Returns:
        dict of `{ key: value }`
    """
    batch_size = batch_size or len(keys)

    keyvalues = {}
    for start in tqdm_proxy(range(0, len(keys), batch_size), leave=False, disable=(batch_size >= len(keys))):
        batch_keys = keys[start:start+batch_size]
        
        if use_jsontar:
            batch_kvs = _fetch_keyvalues_jsontar_via_jsontar(server, uuid, instance, batch_keys, as_json, session=session)
        else:
            batch_kvs = _fetch_keyvalues_via_protobuf(server, uuid, instance, batch_keys, as_json, session=session)
        
        keyvalues.update( batch_kvs )
    
    return keyvalues


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
            if not as_json:
                keyvalues[kv.key] = kv.value
            elif kv.value:
                keyvalues[kv.key] = ujson.loads(kv.value)
            else:
                keyvalues[kv.key] = None
    except ValueError as ex:
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
                if val:
                    val = ujson.loads(val)
                else:
                    val = None
            keyvalues[member.name] = val
        except ValueError as ex:
            raise RuntimeError(f"Error decoding key '{member.name}' from value {val}") from ex

    return keyvalues


@dvid_api_wrapper
def post_keyvalues(server, uuid, instance, keyvalues, batch_size=None, *, session=None, show_progress=True):
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
        
        batch_size:
            If provided, don't post all the values at once.
            Instead, break the values into smaller batches (of the given size),
            and post them one batch at a time.  Progress will be shown in the console.
    """
    assert isinstance(keyvalues, Mapping)
    batch_size = batch_size or len(keyvalues)
    
    keyvalues = list(keyvalues.items())

    batch_starts = range(0, len(keyvalues), batch_size)
    progress = tqdm_proxy(batch_starts, leave=False, disable=(batch_size >= len(keyvalues)) or not show_progress)
    for start in progress:
        kvs = []
        for key, value in keyvalues[start:start+batch_size]:
            if not isinstance(value, (bytes, str)):
                value = ujson.dumps(value)
            if isinstance(value, str):
                value = value.encode('utf-8')
    
            kvs.append( KeyValue(key=key, value=value) )
    
        proto_keyvalues = KeyValues()
        proto_keyvalues.kvs.extend(kvs)
    
        url = f'http://{server}/api/node/{uuid}/{instance}/keyvalues'
        r = session.post(url, data=proto_keyvalues.SerializeToString())
        r.raise_for_status()


def extend_list_value(server, uuid, instance, key, new_list, *, session=None):
    """
    Convenience function.
    
    For the list stored at the given keyvalue instance and key,
    extend it with the given new_list.
    
    If the keyvalue instance and/or key are missing from the server, create them.

    Note: Duplicates will be removed before posting the new list value.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            keyvalue instance name, e.g. '.meta'.
            If it doesn't exist yet, it will be created.
        
        key:
            key of the list to extend.  For example, this function is often
            used for the 'restrictions', or 'neuroglancer' lists.
        
        new_list: Items to append to the existing list on the server.
    """
    assert isinstance(new_list, list)

    # Ensure the instance exists, create it if not.
    try:
        _info = fetch_instance_info(server, uuid, instance, session=session)
    except requests.HTTPError as ex:
        if 'invalid data instance name' in str(ex):
            create_instance(server, uuid, instance, 'keyvalue', session=session)
        else:
            raise

    # Fetch the original value (if it exists)
    try:
        old_list = fetch_key(server, uuid, instance, key, as_json=True, session=session)
        if not isinstance(old_list, list):
            raise RuntimeError(f"Can't extend value: Stored value for key {key} is not a list.")
    except requests.HTTPError as ex:
        if ex.response.status_code == 404:
            old_list = []
        else:
            raise

    # Post the new value (if it's different)
    new_list = list(set(old_list + new_list))
    if set(new_list) != set(old_list):
        logger.debug(f"Updating '{instance}/{key}' list from: {old_list} to: {new_list}")
        post_key(server, uuid, instance, key, json=new_list, session=session)


@dvid_api_wrapper
def fetch_body_annotations(server, uuid, instance='segmentation_annotations', *, session=None):
    """
    Special convenience function for reading the 'segmentation_annotations' keyvalue from DVID,
    which is created and managed by NeuTu and maintains body status information.

    Fetches ALL values from the instance, and loads them into a DataFrame, indexed by body.
    
    >>> annotations_df = fetch_body_annotations('emdata4:8900', '0b0b')
    >>> traced_bodies = annotations_df.query('status == "Traced"').index
    """
    all_keys = fetch_keys(server, uuid, instance, session=session)
    kvs = fetch_keyvalues(server, uuid, instance, all_keys, as_json=True, batch_size=100_000)
    df = pd.DataFrame(list(filter(lambda v: v is not None and 'body ID' in v, kvs.values())))
    df['body'] = df['body ID']
    df = df.set_index('body')
    return df
