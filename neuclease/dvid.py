import getpass
import threading

import requests

DEFAULT_DVID_SESSIONS = {}
DEFAULT_APPNAME = "neuclease"

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


def fetch_supervoxels_for_body(server, uuid, segmentation_instance, body_id, user=None):
    query_params = {}
    if user:
        query_params['u'] = user

    url = f'{server}/api/node/{uuid}/{segmentation_instance}/supervoxels/{body_id}'
    r = default_dvid_session().get(url, params=query_params)
    r.raise_for_status()
    return r.json()

def split_supervoxel(server, uuid, instance, supervoxel, rle_payload_bytes):
    session = default_dvid_session()
    r = session.post(f'http://{server}/api/node/{uuid}/{instance}/split-supervoxel/{supervoxel}', data=rle_payload_bytes)
    r.raise_for_status()    

