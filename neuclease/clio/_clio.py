import copy
import inspect
import functools
import subprocess

import requests

DEFAULT_CLIO_SESSION = None
DEFAULT_CLIO_STORE_BASE = 'https://clio-store-vwzoicitea-uk.a.run.app'


def default_clio_session():
    #
    # http GET https://clio-store-vwzoicitea-uk.a.run.app/v2/pull-request?user_email=tansygarvey@gmail.com
    # "Authorization: Bearer $(gcloud auth print-identity-token)"
    #

    global DEFAULT_CLIO_SESSION
    if DEFAULT_CLIO_SESSION is None:
        p = subprocess.run("gcloud auth print-identity-token", shell=True, check=True, capture_output=True)
        token = p.stdout.decode('utf-8').strip()
        s = requests.Session()
        s.headers.update({"Authorization": f"Bearer {token}"})
        DEFAULT_CLIO_SESSION = s

    return DEFAULT_CLIO_SESSION


def clio_api_wrapper(f):
    """
    Decorator for functions which wrap Clio endpoints.
    The function must accept 'session' and 'base' as a keyword-only arguments.

    This decorator does the following:
    - If 'session' was not provided by the caller, a default one is provided.
    - If 'base' was not provided by the caller, the default one is used.
    - If the base address doesn't begin with 'https://', it is prefixed with 'https://'
    - If an HTTPError is raised, the response body (if any) is also included in the exception text.
      (Clio error responses often include useful information in the response body,
      but requests doesn't normally include the error response body in the exception string.
      This fixes that.)
    """
    argspec = inspect.getfullargspec(f)
    assert 'session' in argspec.kwonlyargs, \
        f"Cannot wrap {f.__name__}: Clio API wrappers must accept 'session' as a keyword-only argument."

    assert 'base' in argspec.kwonlyargs, \
        f"Cannot wrap {f.__name__}: Clio API wrappers must accept 'base' as a keyword-only argument."

    @functools.wraps(f)
    def wrapper(*args, base=None, session=None, **kwargs):
        if base is None:
            base = DEFAULT_CLIO_STORE_BASE
        assert isinstance(base, str)
        assert not base.startswith('http://')

        if not base.startswith('https://'):
            base = 'https://' + base

        if session is None:
            session = default_clio_session()

        try:
            return f(*args, **kwargs, session=session, base=base)
        except requests.RequestException as ex:
            # If the error response had content (and it's not super-long),
            # show that in the traceback, too.  Clio error messages are often helpful.
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
