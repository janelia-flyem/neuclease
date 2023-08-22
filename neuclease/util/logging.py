"""
Miscellaneous logging utilities.
"""
import logging
import logging.handlers
import traceback
import functools
from io import StringIO

class PrefixedLogger(logging.Logger):
    """
    Logger subclass that prepends a pre-configured string to every log message.

    DEPRECATED.  Use neuclease.PrefixFilter instead.
    """
    def __init__(self, base_logger, msg_prefix):
        super().__init__(base_logger.name, base_logger.level)
        self.base_logger = base_logger
        self.msg_prefix = msg_prefix

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        """
        This is a little dicey since we're overriding a "private" method of logging.Logger.
        I don't remember why I didn't just override Logger.log().
        """
        msg = self.msg_prefix + msg
        self.base_logger._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

    def __eq__(self, other):
        return (self.base_logger.name == other.base_logger.name
                and self.msg_prefix == other.msg_prefix)  # noqa

    def __hash__(self):
        return hash((self.base_logger.name, self.msg_prefix))


class ExceptionLogger:
    """
    Context manager.
    Any exceptions that occur while the context is active
    will be logged to the given logger and re-raised.
    """
    def __init__(self, logger):
        self.logger = logger
        self.last_traceback = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            sio = StringIO()
            traceback.print_exception( exc_type, exc_value, exc_tb, file=sio )
            self.logger.error( sio.getvalue() )
            self.last_traceback = sio.getvalue()


def log_exceptions(logger):
    """
    Returns a decorator.
    Any exceptions that occur in the decorated function
    will be logged to the given logger and re-raised.
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with ExceptionLogger(logger):
                return f(*args, **kwargs)
        return wrapper
    return decorator
