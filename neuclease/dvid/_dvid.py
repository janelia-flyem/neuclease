from multiprocessing import connection
import os
import copy
import getpass
import inspect
import platform
import functools
import threading
from collections import namedtuple

import requests
import requests.adapters
from urllib3.util.retry import Retry

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

# FIXME: This should be eliminated or at least renamed
DvidInstanceInfo = namedtuple("DvidInstanceInfo", "server uuid instance")


class DefaultTimeoutHTTPAdapter(requests.adapters.HTTPAdapter):
    """
    Transport adapter to set a default timeout on all of a Session's
    requests if the caller didn't provide one explicitly.
    Effectively injects the 'timeout' parameter to Session.get().
    """
    def __init__(self, *args, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = timeout

    def send(self, *args, **kwargs):
        timeout = kwargs.get('timeout', None)
        if timeout is None:
            kwargs['timeout'] = self.timeout
        return super().send(*args, **kwargs)

    def __getstate__(self):
        state = super().__getstate__()
        state['timeout'] = self.timeout
        return state

    def __repr__(self):
        return f"DefaultTimeoutHTTPAdapter(timeout={self.timeout})"


def clear_default_dvid_sessions(connection_timeout=None, timeout=None):
    global DEFAULT_DVID_SESSIONS
    global DEFAULT_DVID_NODE_SERVICES
    global DEFAULT_DVID_SESSION_TEMPLATE
    DEFAULT_DVID_SESSIONS.clear()
    DEFAULT_DVID_NODE_SERVICES.clear()
    DEFAULT_DVID_SESSION_TEMPLATE = _default_dvid_session_template()
    if connection_timeout is None:
        connection_timeout = DEFAULT_DVID_TIMEOUT[0]
    if timeout is None:
        timeout = DEFAULT_DVID_TIMEOUT[1]

    DEFAULT_DVID_SESSION_TEMPLATE.adapters['http://'].timeout = (connection_timeout, timeout)
    DEFAULT_DVID_SESSION_TEMPLATE.adapters['https://'].timeout = (connection_timeout, timeout)


# Medium timeout for connections, long timeout for data
# https://docs.python-requests.org/en/latest/user/advanced/#timeouts
DEFAULT_DVID_TIMEOUT = (3.05, 120.0)


def _default_dvid_session_template(appname=DEFAULT_APPNAME, user=getpass.getuser(), admintoken=None, timeout=None):
    """
    Note: To specify no timeout at all, set timeout=(None, None)
    """
    # If the connection fails, retry a couple times.
    retries = Retry(connect=2, backoff_factor=0.1)

    if timeout is None:
        timeout = DEFAULT_DVID_TIMEOUT
    adapter = DefaultTimeoutHTTPAdapter(max_retries=retries, timeout=timeout)

    s = requests.Session()
    s.mount('http://', adapter)
    s.mount('https://', adapter)

    s.params = { 'u': user, 'app': appname }

    if admintoken is None:
        admintoken = os.environ.get("DVID_ADMIN_TOKEN", None)

    if admintoken:
        s.params['admintoken'] = admintoken

    return s


# Note:
#   To change the settings for all new default sessions,
#   modify this global template and then clear the cached sessions:
#
#       clear_default_dvid_sessions(3.05, 600.0)
#       default_dvid_session_template().adapters['http://'].timeout = (3.05, 600.0)
#
DEFAULT_DVID_SESSION_TEMPLATE = _default_dvid_session_template()


def create_dvid_session(timeout=DEFAULT_DVID_TIMEOUT):
    s = copy.deepcopy(DEFAULT_DVID_SESSION_TEMPLATE)
    s.adapters['http://'].timeout = timeout
    s.adapters['https://'].timeout = timeout
    return s


def default_dvid_session_template():
    return DEFAULT_DVID_SESSION_TEMPLATE


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
        admintoken = os.environ.get("DVID_ADMIN_TOKEN", None)

    try:
        s = DEFAULT_DVID_SESSIONS[(appname, user, admintoken, thread_id, pid)]
    except KeyError:
        s = _default_dvid_session_template(appname, user, admintoken)
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

        is_default_session = False
        if session is None:
            is_default_session = True
            session = default_dvid_session()

        try:
            return f(server, *args, **kwargs, session=session)
        except requests.RequestException as ex:
            if hasattr(ex, 'response_content_appended'):
                # We already processed this exception (via a nested dvid_api_wrapper call)
                raise

            if isinstance(ex, requests.ConnectionError) and is_default_session:
                # If we're seeing connection errors, let's try discarding the old session.
                # I have no idea if sessions (and connections) can become 'tainted'
                # by failed/aborted connections, but this seems harmless enough.
                clear_default_dvid_sessions()

            if (ex.response is None and ex.request is None):
                # There's no additional info to show
                raise

            # If the error response had content (and it's not super-long),
            # show that in the traceback, too.  DVID error messages are often helpful.
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

    return wrapper


@dvid_api_wrapper
def fetch_generic_json(url, json=None, *, params=None, session=None):
    # TODO: change this to use ujson?
    r = session.get(url, json=json, params=params)
    r.raise_for_status()
    return r.json()
