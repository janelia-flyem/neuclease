import time
import logging
import contextlib
from datetime import timedelta


@contextlib.contextmanager
def Timer(msg=None, logger=None):
    if msg:
        logger = logger or logging.getLogger(__name__)
        logger.info(msg + '...')
    result = _TimerResult()
    start = time.time()
    yield result
    result.seconds = time.time() - start
    result.timedelta = timedelta(seconds=result.seconds)
    if msg:
        logger.info(msg + f' took {result.timedelta}')

class _TimerResult(object):
    seconds = -1.0

