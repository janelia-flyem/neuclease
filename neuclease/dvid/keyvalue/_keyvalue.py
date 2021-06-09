import logging
from io import BytesIO
from functools import partial
from json import JSONDecodeError
from tarfile import TarFile
from collections.abc import Mapping

import ujson
import numpy as np
import pandas as pd
import requests

from ...util import tqdm_proxy, compute_parallel
from .. import dvid_api_wrapper, fetch_generic_json
from ..common import post_tags
from ..node import fetch_instance_info
from ..repo import create_instance

# $ protoc --python_out=. neuclease/dvid/keyvalue/ingest.proto
from .ingest_pb2 import Keys, KeyValue, KeyValues

logger = logging.getLogger(__name__)

# The common post_tags() function works for keyvalue instances.
#post_tags = post_tags


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
    return fetch_generic_json(f'{server}/api/node/{uuid}/{instance}/keys', session=session)


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

        key1:
            Minimal key in the queried range.

        key2:
            Maximal key in the queried range.

    Returns:
        List of keys (strings)

    Examples:

        # Everything from 0...999999999999999
        keys = fetch_keyrange('emdata3:8900', 'abc9', 'segmentation_annotations', '0', '999999999999999')

        # This will catch everything from 'aaa...' to a single 'z', but not 'za'
        keys = fetch_keyrange('emdata3:8900', 'abc9', 'my-kv-instance', 'a', 'z')

        # This gets everything from '0' to 'zzzzz...'
        keys = fetch_keyrange('emdata3:8900', 'abc9', 'my-kv-instance', '0', chr(ord('z')+1))
    """
    url = f'{server}/api/node/{uuid}/{instance}/keyrange/{key1}/{key2}'
    return fetch_generic_json(url, session=session)


@dvid_api_wrapper
def fetch_keyrangevalues(server, uuid, instance, key1, key2, as_json=False, *, check=None, serialization=None, session=None):
    """
    Fetch a set of keys and values from DVID via the ``/keyrangevalues`` endpoint.
    Instead of specifying the list of keys explicitly as in fetch_keyvalues(),
    here you specify a range of keys to query.  All keys within the range will
    be found and returned along with their values.

    WARNING: This can be slow for large ranges.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            keyvalue instance name, e.g. 'segmentation_annotations'

        key1:
            Minimal key in the queried range.

        key2:
            Maximal key in the queried range.

        as_json:
            If True, interpret each value as a JSON object, and parse it as such.

        check:
            When using the 'json' serialization protocol, this argument
            specifies whether or not the server should check that the returned values are valid JSON.
            Since we also parse the data here on the client, we use a default of False,
            to avoid duplicate work on the server.

        serialization:
            Specifies which data format this function will request from DVID internally.
            Uses 'protobuf' by default.

    Returns:
        dict ``{key: value}``, where value is bytes, unless ``as_json=True``.

    Examples:

        # Everything from 0...999999999999999
        kvs = fetch_keyrangevalues('emdata3:8900', 'abc9', 'segmentation_annotations', '0', '999999999999999')

        # This will catch everything from 'aaa...' to a single 'z', but not 'za'
        kvs = fetch_keyrangevalues('emdata3:8900', 'abc9', 'my-kv-instance', 'a', 'z')

        # This gets everything from '0' to 'zzzzz...'
        kvs = fetch_keyrangevalues('emdata3:8900', 'abc9', 'my-kv-instance', '0', chr(ord('z')+1))
    """
    if serialization is None:
        serialization = 'protobuf'
    assert serialization in ('protobuf', 'tar', 'json')

    assert as_json or serialization != 'json', \
        "If using serialization 'json', then the results must be json.\n"\
        "Use as_json=True or try a different serialization."

    assert not check or serialization == 'json', \
        "The 'check' parameter applies only to the 'json' serialization mechanism"

    url = f'{server}/api/node/{uuid}/{instance}/keyrangevalues/{key1}/{key2}'
    params = {serialization: 'true'}
    if check:
        params['check'] = 'true'
    else:
        params['check'] = 'false'

    r = session.get(url, params=params)
    r.raise_for_status()

    if serialization == 'protobuf':
        return _parse_protobuf_keyvalues(r.content, as_json)
    elif serialization == 'json':
        return r.json()
    elif serialization == 'tar':
        return _parse_tarfile_keyvalues(r.content, as_json)
    else:
        raise NotImplementedError(f"Unimplemented serialization choice: {serialization}")


@dvid_api_wrapper
def fetch_key(server, uuid, instance, key, as_json=False, *, check_head=False, session=None):
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
    url = f'{server}/api/node/{uuid}/{instance}/key/{key}'
    if check_head:
        r = session.head(url)
        if r.status_code == 200:
            return True
        if r.status_code == 404:
            return False
        r.raise_for_status()
    else:
        r = session.get(url)
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
    r = session.post(f'{server}/api/node/{uuid}/{instance}/key/{key}', data=data, json=json)
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
    r = session.delete(f'{server}/api/node/{uuid}/{instance}/key/{key}')
    r.raise_for_status()


@dvid_api_wrapper
def fetch_keyvalues(server, uuid, instance, keys, as_json=False, batch_size=None, *, check=None, serialization=None, session=None):
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

        check:
            When using the 'json' serialization protocol, this argument
            specifies whether or not the server should check that the returned values are valid JSON.
            Since we also parse the data here on the client, we use a default of False,
            to avoid duplicate work on the server.

        serialization:
            Specifies which data format this function will request from DVID internally.
            Uses 'protobuf' by default.

    Returns:
        dict of ``{ key: value }``
    """
    if serialization is None:
        serialization = 'protobuf'
    assert serialization in ('protobuf', 'tar', 'json')

    assert as_json or serialization != 'json', \
        "If using serialization 'json', then the results must be json.\n"\
        "Use as_json=True or try a different serialization."

    assert not check or serialization == 'json', \
        "The 'check' parameter applies only to the 'json' serialization mechanism"

    keyvalues = {}
    batch_size = batch_size or len(keys)
    for start in tqdm_proxy(range(0, len(keys), batch_size), leave=False, disable=(batch_size >= len(keys))):
        batch_keys = keys[start:start+batch_size]

        if serialization == 'json':
            batch_kvs = _fetch_keyvalues_via_json(server, uuid, instance, keys, check, session)
        elif serialization == 'protobuf':
            batch_kvs = _fetch_keyvalues_via_protobuf(server, uuid, instance, batch_keys, as_json, session=session)
        elif serialization == 'jsontar':
            batch_kvs = _fetch_keyvalues_via_jsontar(server, uuid, instance, batch_keys, as_json, session=session)
        else:
            raise NotImplementedError(f"Unimplemented serialization choice: {serialization}")

        keyvalues.update( batch_kvs )

    return keyvalues


def _fetch_keyvalues_via_json(server, uuid, instance, keys, check, session):
    params = {'json': 'true'}
    if check:
        params['check'] = 'true'
    else:
        params['check'] = 'false'

    assert not isinstance(keys, str), "keys should be a list (or array) of strings"
    if isinstance(keys, np.ndarray):
        keys = keys.tolist()
    else:
        keys = list(keys)

    url = f'{server}/api/node/{uuid}/{instance}/keyvalues'
    r = session.get(url, params=params, json=keys)
    r.raise_for_status()

    try:
        return r.json()
    except JSONDecodeError:
        # Workaround for:
        # https://github.com/janelia-flyem/dvid/issues/356
        return ujson.loads(r.content + b'}')


@dvid_api_wrapper
def _fetch_keyvalues_via_protobuf(server, uuid, instance, keys, as_json=False, *, use_jsontar=False, session=None):
    assert not isinstance(keys, str), "keys should be a list (or array) of strings"

    proto_keys = Keys()
    for key in keys:
        proto_keys.keys.append(key)

    r = session.get(f'{server}/api/node/{uuid}/{instance}/keyvalues', data=proto_keys.SerializeToString())
    r.raise_for_status()
    return _parse_protobuf_keyvalues(r.content, as_json)


def _parse_protobuf_keyvalues(buf, as_json):
    """
    Parse the given bytes from protobuf KeyValues
    (neuclease.dvid.keyvalue.ingest_pb2.KeyValues)

    If as_json=True, also decode each value as a json object.
    """
    proto_keyvalues = KeyValues()
    proto_keyvalues.ParseFromString(buf)

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
def _fetch_keyvalues_via_jsontar(server, uuid, instance, keys, as_json, *, session=None):
    params = {'jsontar': 'true'}

    assert not isinstance(keys, str), "keys should be a list (or array) of strings"
    if isinstance(keys, np.ndarray):
        keys = keys.tolist()
    else:
        keys = list(keys)

    r = session.get(f'{server}/api/node/{uuid}/{instance}/keyvalues', params=params, json=keys)
    r.raise_for_status()
    return _parse_tarfile_keyvalues(r.content, as_json)


def _parse_tarfile_keyvalues(buf, as_json):
    tf = TarFile('keyvalues.tar', fileobj=BytesIO(buf))

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

    TODO:
        - This can fail if a batch ends up larger than 2GB.
          Would be nice to automatically split batches if necessary.
    
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
    
            kvs.append( KeyValue(key=str(key), value=value) )
    
        proto_keyvalues = KeyValues()
        proto_keyvalues.kvs.extend(kvs)
    
        url = f'{server}/api/node/{uuid}/{instance}/keyvalues'
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
        fetch_instance_info(server, uuid, instance, session=session)
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


# Copied from the following URL, but I prepended the empty status ("").
# http://emdata5.janelia.org:8400/api/node/b31220/neutu_config/key/body_status_v2
DEFAULT_BODY_STATUS_CATEGORIES = [
    '',
    'Orphan-artifact',
    'Orphan',
    'Orphan hotknife',
    'Not examined',
    '0.5assign',
    'Leaves',
    'Anchor',
    'Cervical Anchor',
    'Soma Anchor',
    'Hard to trace',
    'Unimportant',
    'Partially traced',
    'Prelim Roughly traced',
    'Roughly traced',
    'Traced in ROI',
    'Traced',
    'Finalized'
]


@dvid_api_wrapper
def fetch_body_annotations(server, uuid, instance='segmentation_annotations', bodies=None, *,
                           status_categories=DEFAULT_BODY_STATUS_CATEGORIES,
                           batch_size=None, session=None):
    """
    Special convenience function for reading the 'segmentation_annotations' keyvalue from DVID,
    which is created and managed by NeuTu and maintains body status information.

    If a list of bodies is provided, fetches annotation info for those bodies only.
    Otherwise, fetches ALL values from the instance.

    Loads the results into a DataFrame, indexed by body.

    Note: Any missing annotation entries are silently ignored.
          You should check the results to see if every body annotation
          you requested was returned. (Otherwise, it was missing from dvid.)

    Exmaple:

        .. code-block: python

            annotations_df = fetch_body_annotations('emdata4:8900', '0b0b')
            traced_bodies = annotations_df.query('status == "Traced"').index

    Args:
        server:
            dvid server
        uuid:
            dvid uuid
        instance:
            The keyvalue instance names where body annotations are stored.
            NeuTu expects this to be named with a formula, after name of
            the labelmap instance it corresponds to, i.e. f"{seg_instance}_annotations"
            For example: segmentation_annotations
        bodies:
            If provided, fetch only the annotations for the listed body IDs
        status_categories:
            The 'status' column is in the result is an ordered ``pd.Categorical``,
            so it can be sorted from "small" to "big".
            By default, our standard status categories are used, but if an unknown
            status is encountered, and error is raised.
            In that case, you have two choices:
                - You can specify the correct list of statuses you expect to see in the instance.
                - You can set ``status_categories=None`` to simply return strings instead of a Categorical
        batch_size:
            If there are a lot (100k) of body annotation values in the instance,
            it may be convenient to split the download into batches.
            Specify the batch size with this parameter.

    Returns:
        DataFrame, indexed by body.
        The DataFrame is constructed by treating each annotation JSON object
        as a 'record' from which the row should be created.
        Also, the raw json object from which the row was constructed
        is provided in a column named 'json'.
    """
    if bodies is not None:
        keys = [str(b) for b in bodies]
        try:
            max(int(b) for b in keys)
        except ValueError:
            raise RuntimeError(f"Malformed body list: {bodies}")

        batch_size = batch_size or 100_000
        # Due to https://github.com/janelia-flyem/dvid/issues/356,
        # we can't use serialization='json' yet.
        kvs = fetch_keyvalues(server, uuid, instance, keys, as_json=True, batch_size=batch_size, serialization='protobuf', session=session)
    elif batch_size is None:
        # This gets everything from '0' to 'zzzzz...'
        kvs = fetch_keyrangevalues(server, uuid, instance, '0', chr(ord('z')+1), as_json=True, session=session)
    else:
        keys = fetch_keys(server, uuid, instance, session=session)
        kvs = fetch_keyvalues(server, uuid, instance, keys, as_json=True, batch_size=batch_size, session=session)

    values = list(filter(lambda v: v is not None and 'body ID' in v, kvs.values()))
    if len(values) == 0:
        empty_index = pd.Series([], dtype=int, name='body')
        return pd.DataFrame({'status': [], 'json': []}, dtype=object, index=empty_index)

    df = pd.DataFrame(values)
    if 'body ID' in df:
        df['body'] = df['body ID']
        df = df.set_index('body')

    if 'status' in df.columns:
        df['status'].fillna('', inplace=True)
    else:
        df['status'] = ""

    if status_categories is not None:
        # status categories are ordered from small to large, so they
        # can be sorted in the same direction as size and synapses.
        unrecognized_statuses = set(df['status']) - set(status_categories)
        if unrecognized_statuses:
            msg = ("Can't create categorical statuses!\n"
                   f"Found unrecognized statuses: {unrecognized_statuses}\n"
                   "Try providing your own status_categories,\n"
                   "or status_categories=None to parse status as non-categorical strings.")
            raise RuntimeError(msg)

        df['status'] = pd.Categorical(df['status'], status_categories, ordered=True)

    # As a convenience, also provide the original json object for each row.
    # This is useful when you want to push changes back to dvid.
    df['json'] = values
    return df


def fetch_sphere_annotations(server, uuid, instance, seg_instance=None, *, session=None):
    """
    Convenience function for fetching sphere annotations from a keyvalue instance.

    Example keys/values which will be parsed:

        {
            'cordish25@gmail.com--10003_37134_46349-9999_36393_46292': {
                'Kind': 'Sphere',
                'Pos': ['10003', '37134', '46349', '9999', '36393', '46292'],
                'Prop': {'timestamp': ''}}

            'cordish25@gmail.com--10005_30224_47596-10014_29489_47504': {
                'Kind': 'Sphere',
                'Pos': ['10005', '30224', '47596', '10014', '29489', '47504'],
                'Prop': {'timestamp': ''}}
        }

    Args:
        server:
            dvid server
        uuid:
            dvid uuid
        instance:
            keyvalue instance containing annotations as shown in the example above
        seg_instance:
            Optional.  A labelmap instance from which fetch the label under each sphere annotation midpoint.

    Returns:
        DataFrame
        Columns for the user, start/end/midpoint coordinates, and diameter

    Example:

        ..code-block:: python

            fetch_sphere_annotations('emdata5.janelia.org:8400', 'b31220', 'soma-bookmarks')
    """
    # This gets everything from '0' to 'zzzzz...'
    kv = fetch_keyrangevalues(server, uuid, instance, '0', chr(ord('z')+1), as_json=True)

    users = []
    coords = []
    props = []
    for k, v in kv.items():
        if v.get('Kind') != 'Sphere':
            continue
        users.append(k.split('-')[0])
        pos = [int(p) for p in v['Pos']]
        coords.append((pos[:3], pos[3:]))
        props.append(v.get('Prop', None))

    cols = ['user', *'xyz', 'diameter', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'prop']
    if seg_instance:
        cols = ['body'] + cols

    if len(users) == 0:
        return pd.DataFrame([], columns=cols)

    coords = np.array(coords).astype(int)
    midpoints = coords.sum(axis=1) // 2

    df = pd.DataFrame(coords.reshape((-1, 6)), columns=['x0', 'y0', 'z0', 'x1', 'y1', 'z1'])
    df['x'] = midpoints[:, 0]
    df['y'] = midpoints[:, 1]
    df['z'] = midpoints[:, 2]

    radii = np.linalg.norm(coords[:, 1, :] - coords[:, 0, :], axis=1)
    df['diameter'] = radii

    df['user'] = users
    df['prop'] = props

    if seg_instance:
        from ..labelmap import fetch_labels_batched
        df['body'] = fetch_labels_batched(server, uuid, seg_instance, df[[*'zyx']].values, processes=4, batch_size=5_000)

    return df[cols]


def post_sphere_annotations(server, uuid, instance, df, *, session=None):
    """
    Post a set of sphere annotations to a key-value instance.

    Will post key-values in this format:

        {
            'cordish25@gmail.com--10003_37134_46349-9999_36393_46292': {
                'Kind': 'Sphere',
                'Pos': ['10003', '37134', '46349', '9999', '36393', '46292'],
                'Prop': {'timestamp': ''}}

            'cordish25@gmail.com--10005_30224_47596-10014_29489_47504': {
                'Kind': 'Sphere',
                'Pos': ['10005', '30224', '47596', '10014', '29489', '47504'],
                'Prop': {'timestamp': ''}}
        }

    Args:
        df:
            DataFrame with columns ['user', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1']
            and optionally a 'prop' column.
            The two coordinates represend endpoints of the sphere diameter.
    """
    assert {'user', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1'} <= set(df.columns)
    df = df.copy()

    kvs = {}
    for t in df.itertuples():
        key = f"{t.user}--{t.x0}_{t.y0}_{t.z0}-{t.x1}-{t.y1}-{t.z1}"
        value = {
            'Kind': 'Sphere',
            'Pos': [*map(str, [t.x0, t.y0, t.z0, t.x1, t.y1, t.z1])]
        }
        if 'prop' in df.columns:
            value['Prop'] = t.properties

        kvs[key] = value

    post_keyvalues(server, uuid, instance, kvs, session=session)


@dvid_api_wrapper
def fetch_skeleton_mutation(server, uuid, instance, body, *, session=None):
    """
    Fetch the skeleton for a given body and read its header comment to
    determine the mutation ID at which the skeleton was generated.

    Ting's skeletonization code prepends a header to the SWC file as shown below.
    This function parses the mutation id and returns it.

    If the skeleton can't be found, or does not contain the expected header,
    then -1 is returned.

    .. code-block:: swc

        #Generated by NeuTu (https://github.com/janelia-flyem/NeuTu)
        #${"ds_intv": [11, 11, 11], "min_length": 40.0, "final_min_length": 0.0, "vskl": 1}
        #${"mutation id": 1000009350}
        1 0 27952 35892 61824 37.2666 -1
        2 2 28019.3 35847.5 61708.7 54 1
        3 0 28117.9 35750.1 61674.3 75.1885 2
    """
    try:
        s = fetch_key(server, uuid, instance, f"{body}_swc").decode('utf-8')
    except requests.HTTPError:
        return (body, -1)

    for line in s.splitlines()[:10]:
        if 'mutation id' in line:
            return (body, ujson.loads(line[2:])["mutation id"])

    return (body, -1)


def fetch_skeleton_mutations(server, uuid, instance, bodies, processes=4):
    """
    Call fetch_skeleton_mutation() in parallel for a list of bodies.
    """
    assert processes >= 1
    fn = partial(fetch_skeleton_mutation, server, uuid, instance)
    skel_muts = compute_parallel(fn, bodies, processes=processes)
    skel_muts = pd.DataFrame(skel_muts, columns=['body', 'skeleton_mutid'])
    return skel_muts.set_index('body')['skeleton_mutid']
