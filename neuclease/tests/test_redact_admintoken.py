import io
import logging

import pytest
import requests

from neuclease import redact_admintoken, _RedactingFormatter
from neuclease.dvid._dvid import dvid_api_wrapper

URL = 'http://dvid-server/api/node/abc123/segmentation/lastmod/1?u=me&admintoken=SECRET&app=neuclease'


def test_redact_admintoken():
    assert redact_admintoken(URL) == URL.replace('SECRET', '<redacted>')
    assert redact_admintoken("'...?admintoken=SECRET'") == "'...?admintoken=<redacted>'"
    assert redact_admintoken("no token here") == "no token here"


@dvid_api_wrapper
def _fetch_with_timeout(server, *, session=None):
    raise requests.ConnectTimeout(
        f"Max retries exceeded with url: {URL} (Caused by ConnectTimeoutError())",
        request=requests.Request('GET', URL).prepare())


def test_dvid_api_wrapper_keeps_cause_and_redacts():
    with pytest.raises(requests.ConnectTimeout) as exc_info:
        _fetch_with_timeout('dvid-server', session=object())

    msg = str(exc_info.value)
    assert msg.startswith('Error accessing GET')
    assert 'Max retries exceeded' in msg
    assert 'admintoken=<redacted>' in msg
    assert 'SECRET' not in msg


def test_redacting_formatter_covers_tracebacks():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_RedactingFormatter('%(levelname)s %(message)s'))
    logger = logging.getLogger('test_redacting_formatter')
    logger.addHandler(handler)
    try:
        try:
            _fetch_with_timeout('dvid-server', session=object())
        except requests.ConnectTimeout as ex:
            # The original (unredacted) exception is chained to the one we log.
            logger.warning(f"Failed: {ex.__cause__}", exc_info=True)
    finally:
        logger.removeHandler(handler)

    output = stream.getvalue()
    assert 'Traceback' in output
    assert 'admintoken=<redacted>' in output
    assert 'SECRET' not in output
