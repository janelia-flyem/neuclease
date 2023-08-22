import sys

try:
    import faulthandler
    if not faulthandler.is_enabled():
        faulthandler.enable()
except Exception:
    print("Failed to enable faulthandlder module", file=sys.stderr)

import inspect
import logging
import warnings
from functools import wraps
from contextlib import contextmanager
from contextvars import ContextVar

logging.getLogger('pykafka').setLevel(logging.WARNING)

# Don't show the following warning from within pandas:
# FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
warnings.filterwarnings("ignore", module=r"pandas\..*", category=FutureWarning)


def configure_default_logging():
    """
    Simple logging configuration.
    Useful for interactive terminal sessions.
    """
    logging.captureWarnings(True)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(PrefixFilter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


_logging_prefix = ContextVar('logging_prefix')


class PrefixFilter(logging.Filter):
    """
    A logging filter that inserts a globally-stored prefix onto all logging messages that it sees.
    To apply it to all log messages, it is attached to the stdout handler in configure_default_logging().

    Use either the context manager (PrefixFilter.context) or the decorator (PrefixFilter.with_context).

    Note:
        If you create additional handlers, you must call handler.addFilter(PrefixFilter())
        to them or they won't see the prefixes.

    Example:

        from neuclease import PrefixFilter

        @PrefixFilter.with_context('howdy_{name}')
        def howdy(*, name):
            logger.info(f'howdy {name}')

        logger.info("Good Morning!")
        with PrefixFilter.context("hello_context"):
            logger.info('hello')
            howdy(name='Guido')
            logger.info('bye')
        logger.info("Goodnight!")

    Example Output:

        [2023-08-22 11:01:33,873] INFO Good Morning!
        [2023-08-22 11:01:33,874] INFO hello_context: hello
        [2023-08-22 11:01:33,874] INFO hello_context: howdy_func: howdy
        [2023-08-22 11:01:33,874] INFO hello_context: bye
        [2023-08-22 11:01:33,874] INFO Goodnight!
    """

    @classmethod
    def push(cls, s):
        p = _logging_prefix.get([])
        p.append(s)
        _logging_prefix.set(p)

    @classmethod
    def pop(cls):
        p = _logging_prefix.get([])
        if p:
            p.pop()
            _logging_prefix.set(p)

    @classmethod
    @contextmanager
    def context(cls, s):
        cls.push(s)
        try:
            yield
        finally:
            cls.pop()

    @classmethod
    def with_context(cls, s):
        def _decorator(f):
            argspec = inspect.getfullargspec(f)
            testargs = {arg: '' for arg in argspec.kwonlyargs}
            try:
                s.format(**testargs)
            except KeyError as ex:
                msg = (
                    f"Can't use logging context decorator with prefix '{s}' because "
                    "it contains a format string identifier which isn't "
                    "named in the function signature as a keyword-only argument!"
                )
                raise RuntimeError(msg) from ex

            @wraps(f)
            def _f(*args, **kwargs):
                with PrefixFilter.context(s.format(**kwargs)):
                    return f(*args, **kwargs)
            return _f
        return _decorator

    def filter(self, record):
        p = _logging_prefix.get([])
        if p:
            record.msg = ': '.join(p) + ': ' + record.msg
        return True


try:
    import numexpr
except ImportError:
    pass
else:
    import textwrap
    _msg = textwrap.dedent("""\
        Cannot use this library in an environment in which numexpr is installed.

        Please do not use this library in an environment in which numexpr is installed.

        This library relies on pandas eval() and pandas query() to compare uint64 types,
        which is not supported by the numexpr library.
        
        If numexpr is present, pandas will use it by default in eval() and query(),
        rather than using query(..., parser='python').
        
        Using numexpr leads to errors like this one:

            >>> df = pd.DataFrame([[1,2]], columns=list('ab'), dtype=np.uint64)
            >>> df.query('a == b')
            TypeError: Iterator operand 1 dtype could not be cast from dtype('uint64') to dtype('int64') according to the rule 'safe'
        
        In the future, this library will be patched to avoid using uint64 types altogether,
        and this error will be removed.
        
        See: https://github.com/janelia-flyem/neuclease/issues/3
    """)
    raise AssertionError(_msg)

from . import _version
__version__ = _version.get_versions()['version']
