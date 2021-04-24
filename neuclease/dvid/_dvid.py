import os
import copy
import getpass
import inspect
import platform
import functools
import threading
from collections import namedtuple

import requests
from libdvid import DVIDNodeService

# On Mac, requests uses a system library which is not fork-safe,
# resulting in segfaults such as the following:
#
#   File ".../lib/python3.7/urllib/request.py", line 2588 in proxy_bypass_macosx_sysconf
#   File ".../lib/python3.7/urllib/request.py", line 2612 in proxy_bypass
#   File ".../lib/python3.7/site-packages/requests/utils.py", line 745 in should_bypass_proxies
#   File ".../lib/python3.7/site-packages/requests/utils.py", line 761 in get_environ_proxies
#   File ".../lib/python3.7/site-packages/requests/sessions.py", line 700 in merge_environment_settings
#   File ".../lib/python3.7/site-packages/requests/sessions.py", line 524 in request
#   File ".../lib/python3.7/site-packages/requests/sessions.py", line 546 in get
# ...

# The workaround is to set a special environment variable
# to avoid the particular system function in question.
# Details here:
# https://bugs.python.org/issue30385
if platform.system() == "Darwin" and 'no_proxy' not in os.environ:
    os.environ["no_proxy"] = "*"

DEFAULT_DVID_SESSIONS = {}
DEFAULT_DVID_NODE_SERVICES = {}
DEFAULT_APPNAME = "neuclease"
DEFAULT_ADMIN_TOKEN = os.environ.get("DVID_ADMIN_TOKEN", None)

# FIXME: This should be eliminated or at least renamed
DvidInstanceInfo = namedtuple("DvidInstanceInfo", "server uuid instance")


def default_dvid_session(appname=DEFAULT_APPNAME, user=getpass.getuser(), admintoken=None):
    """
    Return a default requests.Session() object that automatically appends the
    'u' and 'app' query string parameters to every request.
    The Session object is cached, so this function will return the same Session
    object if called again from the same thread with the same arguments.
    """
    # TODO:
    # Proper authentication will involve fetching a JWT from this endpoint:
    # https://hemibrain-dvid.janelia.org/api/server/token

    # Technically, request sessions are not threadsafe,
    # so we keep one for each thread.
    thread_id = threading.current_thread().ident
    pid = os.getpid()
    if admintoken is None:
        admintoken = DEFAULT_ADMIN_TOKEN

    try:
        s = DEFAULT_DVID_SESSIONS[(appname, user, admintoken, thread_id, pid)]
    except KeyError:
        s = requests.Session()
        s.params = { 'u': user, 'app': appname }
        if admintoken:
            s.params['admintoken'] = admintoken

        DEFAULT_DVID_SESSIONS[(appname, user, admintoken, thread_id, pid)] = s

    return s


def default_node_service(server, uuid, appname=DEFAULT_APPNAME, user=getpass.getuser()):
    """
    Return a DVIDNodeService for the given server and uuid.
    The object is cached, so this function will return the same service
    object if called again from the same thread with the same arguments.
    """
    # One per thread/process
    thread_id = threading.current_thread().ident
    pid = os.getpid()

    try:
        ns = DEFAULT_DVID_NODE_SERVICES[(appname, user, thread_id, pid, server, uuid)]
    except KeyError:
        ns = DVIDNodeService(server, str(uuid), user, appname)
        DEFAULT_DVID_NODE_SERVICES[(appname, user, thread_id, pid, server, uuid)] = ns

    return ns


def dvid_api_wrapper(f):
    """
    Decorator for functions whose first arg is a dvid server address,
    and accepts 'session' as a keyword-only argument.

    This decorator does the following:
    - If the server address doesn't begin with 'http://' or 'https://', it is prefixed with 'http://'
    - If 'session' was not provided by the caller, a default one is provided.
    - If an HTTPError is raised, the response body (if any) is also included in the exception text.
      (DVID error responses often include useful information in the response body,
      but requests doesn't normally include the error response body in the exception string.
      This fixes that.)
    """
    argspec = inspect.getfullargspec(f)
    assert 'session' in argspec.kwonlyargs, \
        f"Cannot wrap {f.__name__}: DVID API wrappers must accept 'session' as a keyword-only argument."

    @functools.wraps(f)
    def wrapper(server, *args, session=None, **kwargs):
        assert isinstance(server, str)
        if not server.startswith('http://') and not server.startswith('https://'):
            server = 'http://' + server

        if session is None:
            session = default_dvid_session()

        try:
            return f(server, *args, **kwargs, session=session)
        except requests.RequestException as ex:
            # If the error response had content (and it's not super-long),
            # show that in the traceback, too.  DVID error messages are often helpful.
            if not hasattr(ex, 'response_content_appended') and (ex.response is not None or ex.request is not None):
                msg = ""
                if (ex.request is not None):
                    msg += f"Error accessing {ex.request.method} {ex.request.url}\n"

                if ex.response is not None and ex.response.content:
                    # Decode up to 10_000 bytes of the content,
                    MAX_ERR_DISPLAY = 10_000
                    try:
                        err = ex.response.content[:MAX_ERR_DISPLAY].decode('utf-8')
                    except UnicodeDecodeError as unicode_err:
                        # Last byte cuts off a character by chance.
                        # Discard it.
                        err = ex.response.content[:unicode_err.start].decode('utf-8')

                    msg += str(ex.args[0]) + "\n" + err + "\n"

                new_ex = copy.copy(ex)
                new_ex.args = (msg, *ex.args[1:])
                new_ex.response_content_appended = True
                raise new_ex from ex
            else:
                raise
    return wrapper


@dvid_api_wrapper
def fetch_generic_json(url, json=None, *, params=None, session=None):
    # TODO: change this to use ujson?
    r = session.get(url, json=json, params=params)
    r.raise_for_status()
    return r.json()

