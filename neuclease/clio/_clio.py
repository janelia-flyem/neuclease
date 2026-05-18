import os
import copy
import inspect
import functools
import subprocess

import requests

import google.auth
import google.auth.transport.requests
from google.auth.exceptions import DefaultCredentialsError

DEFAULT_CLIO_SESSION = None
DEFAULT_CLIO_STORE_BASE = 'https://clio-store-vwzoicitea-uk.a.run.app'


_ADC_HELP = (
    "No Google credentials found. To authenticate, run this in a terminal:\n"
    "\n"
    "    gcloud auth application-default login\n"
    "\n"
    "This only needs to be done once (you can run it while Python is already running).\n"
    "\n"
    "Note: 'gcloud auth login' is NOT sufficient — you need the 'application-default' variant."
)


def _get_google_id_creds() -> str:
    """Obtain a Google OAuth2 ID token for the default credentials.

    Works with:
    - ``gcloud auth application-default login`` (interactive / user credentials)
    - Service account keys (via ``GOOGLE_APPLICATION_CREDENTIALS``)
    - Workload identity on GCE/Cloud Run
    """
    try:
        creds, _ = google.auth.default()
    except DefaultCredentialsError:
        raise RuntimeError(_ADC_HELP) from None

    request = google.auth.transport.requests.Request()
    try:
        creds.refresh(request)
    except google.auth.exceptions.RefreshError as e:
        raise RuntimeError(
            f"Google credentials found but could not be refreshed: {e}\n\n"
            "Try re-running:  gcloud auth application-default login"
        ) from None

    return creds


def reset_default_clio_session():
    global DEFAULT_CLIO_SESSION
    DEFAULT_CLIO_SESSION = None
    return default_clio_session()


def default_clio_session():
    #
    # http GET https://clio-store-vwzoicitea-uk.a.run.app/v2/pull-request?user_email=tansygarvey@gmail.com
    # "Authorization: Bearer $(gcloud auth print-identity-token)"
    #

    # URL=https://clio-store-vwzoicitea-uk.a.run.app/v2/pull-request?user_email=tansygarvey@gmail.com
    # curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" ${URL}

    global DEFAULT_CLIO_SESSION
    if DEFAULT_CLIO_SESSION is None or (DEFAULT_CLIO_SESSION.creds is not None and DEFAULT_CLIO_SESSION.creds.expired):
        if os.environ.get('GOOGLE_IDENTITY_TOKEN'):
            creds = None
            token = os.environ['GOOGLE_IDENTITY_TOKEN']
        else:
            creds = _get_google_id_creds()
            token = creds.id_token

        s = requests.Session()
        s.headers.update({"Authorization": f"Bearer {token}"})
        s.creds = creds
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
