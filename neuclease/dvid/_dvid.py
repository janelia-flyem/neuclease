import os
import getpass
import functools
import threading
from collections import namedtuple

import requests
from libdvid import DVIDNodeService

DEFAULT_DVID_SESSIONS = {}
DEFAULT_DVID_NODE_SERVICES = {}
DEFAULT_APPNAME = "neuclease"
DvidInstanceInfo = namedtuple("DvidInstanceInfo", "server uuid instance")

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
    pid = os.getpid()
    try:
        s = DEFAULT_DVID_SESSIONS[(appname, thread_id, pid)]
    except KeyError:
        s = requests.Session()
        s.params = { 'u': getpass.getuser(),
                     'app': appname }
        DEFAULT_DVID_SESSIONS[(appname, thread_id, pid)] = s

    return s


def default_node_service(server, uuid, appname=None):
    if appname is None:
        appname = DEFAULT_APPNAME

    # One per thread/process
    thread_id = threading.current_thread().ident
    pid = os.getpid()

    try:
        ns = DEFAULT_DVID_NODE_SERVICES[(appname, thread_id, pid, server, uuid)]
    except KeyError:
        ns = DVIDNodeService(server, str(uuid), getpass.getuser(), appname)

    return ns


def dvid_api_wrapper(f):
    """
    Decorator for functions whose first arg is a dvid server address, and calls DVID via the requests module.
    - If the server address begins with 'http://', that prefix is stripped from it.
    - If an requests.HTTPError is raised, the response body (if any) is also included in the exception text.
      (DVID error responses often includes useful information in the response body,
      but requests doesn't show that by default.)
    """
    @functools.wraps(f)
    def wrapper(server, *args, **kwargs):
        assert isinstance(server, str)
        if server.startswith('http://'):
            server = server[len('http://'):]

        try:
            return f(server, *args, **kwargs)
        except requests.HTTPError as ex:
            # If the error response had content (and it's not super-long),
            # show that in the traceback, too.  DVID error messages are often helpful.
            if ( not hasattr(ex, 'response_content_appended')
                 and ex.response is not None
                 and ex.response.content
                 and len(ex.response.content) <= 200 ):
                
                msg = str(ex.args[0]) + "\n" + ex.response.content.decode('utf-8')
                new_ex = requests.HTTPError(msg, *args[1:])
                new_ex.response_content_appended = True
                raise new_ex from ex
            else:
                raise
    return wrapper


def fetch_generic_json(url, json=None):
    session = default_dvid_session()
    r = session.get(url, json=json)
    r.raise_for_status()
    return r.json()

