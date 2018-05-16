import getpass
import logging
import threading
import functools
from io import BytesIO

import requests

import numpy as np
import pandas as pd

from .util import Timer

DEFAULT_DVID_SESSIONS = {}
DEFAULT_APPNAME = "neuclease"

logger = logging.getLogger(__name__)

def default_dvid_session(appname=None):
    """
    Return a default requests.Session() object that automatically appends the
    'u' and 'app' query string parameters to every request.
    """
    if appname is None:
        appname = DEFAULT_APPNAME
    # Technically, request sessions are not threadsafe,
    # so we keep one for each thread.
    thread_id = threading.current_thread().ident
    try:
        s = DEFAULT_DVID_SESSIONS[(appname, thread_id)]
    except KeyError:
        s = requests.Session()
        s.params = { 'u': getpass.getuser(),
                     'app': appname }
        DEFAULT_DVID_SESSIONS[(appname, thread_id)] = s

    return s

def sanitize_server_arg(f):
    """
    Decorator for functions whose first arg is 'server'.
    If the server begins with 'http://', that prefix is stripped from the argument.
    """
    @functools.wraps(f)
    def wrapper(server, *args, **kwargs):
        if server.startswith('http://'):
            server = server[len('http://'):]
        return f(server, *args, **kwargs)
    return wrapper


@sanitize_server_arg
def fetch_supervoxels_for_body(server, uuid, labelmap_instance, body_id, user=None):
    query_params = {}
    if user:
        query_params['u'] = user

    url = f'http://{server}/api/node/{uuid}/{labelmap_instance}/supervoxels/{body_id}'
    r = default_dvid_session().get(url, params=query_params)
    r.raise_for_status()
    return r.json()


@sanitize_server_arg
def fetch_label_for_coordinate(server, uuid, instance, coordinate_zyx, supervoxels=False):
    session = default_dvid_session()
    coord_xyz = np.array(coordinate_zyx)[::-1]
    coord_str = '_'.join(map(str, coord_xyz))
    supervoxels = str(bool(supervoxels)).lower()
    r = session.get(f'http://{server}/api/node/{uuid}/{instance}/label/{coord_str}?supervoxels={supervoxels}')
    r.raise_for_status()
    return np.uint64(r.json()["Label"])


@sanitize_server_arg
def split_supervoxel(server, uuid, instance, supervoxel, rle_payload_bytes):
    """
    Split the given supervoxel according to the provided RLE payload, as specified in DVID's split-supervoxel docs.
    
    Returns:
        The two new IDs resulting from the split: (split_sv_id, remaining_sv_id)
    """
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/split-supervoxel/{supervoxel}', data=rle_payload_bytes)
    r.raise_for_status()   
    
    results = r.json()
    return (results["SplitSupervoxel"], results["RemainSupervoxel"] )


@sanitize_server_arg
def fetch_mappings(server, uuid, labelmap_instance, include_identities=True, retired_supervoxels=[]):
    """
    Fetch the complete sv-to-label mapping table from DVID and return it as a pandas Series (indexed by sv).
    
    Args:
        include_identities:
            If True, add rows for identity mappings (which are not included in DVID's response).
        
        retired_supervoxels:
            A set of supervoxels NOT to automatically add as identity mappings,
            e.g. due to the fact that they were split.
    
    Returns:
        pd.Series(index=sv, data=body)
    """
    session = default_dvid_session()
    
    # This takes ~30 seconds so it's nice to log it.
    uri = f"http://{server}/api/node/{uuid}/{labelmap_instance}/mappings"
    with Timer(f"Fetching {uri}", logger):
        r = session.get(uri)
        r.raise_for_status()

    with Timer(f"Parsing mapping", logger), BytesIO(r.content) as f:
        df = pd.read_csv(f, sep=' ', header=None, names=['sv', 'body'], engine='c', dtype=np.uint64)

    if include_identities:
        missing_idents = set(df['body']) - set(df['sv']) - set(retired_supervoxels)
        missing_idents = np.fromiter(missing_idents, np.uint64)
        missing_idents.sort()
        
        idents_df = pd.DataFrame({'sv': missing_idents, 'body': missing_idents})
        df = pd.concat((df, idents_df), ignore_index=True)

    df.set_index('sv', inplace=True)

    return df['body']


@sanitize_server_arg
def fetch_mutation_id(server, uuid, labelmap_instance, body_id):
    session = default_dvid_session()
    r = session.get(f'http://{server}/api/node/{uuid}/{labelmap_instance}/lastmod/{body_id}')
    r.raise_for_status()
    return r.json()["mutation id"]

