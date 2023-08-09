import os
import sys
import logging
import logging.handlers
import threading
import traceback
import functools
from io import StringIO


def init_logging(logger, log_dir, db_path, stdout_logging=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    db_name = os.path.basename(db_path)
    db_name = db_name.replace(':', '_') # For OSX, which treats ':' like '/'
    logfile_path = os.path.join(log_dir, os.path.splitext(db_name)[0]) + '.log'

    # Don't log ordinary GET, POST, etc.
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    rootLogger = logging.getLogger()

    # Clear any handlers that were automatically added (by werkzeug)
    rootLogger.handlers = []
    logger.handlers = []

    formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s')
    handler = logging.handlers.RotatingFileHandler(logfile_path, maxBytes=int(10e6), backupCount=1000)
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)

    rootLogger.setLevel(logging.INFO)
    rootLogger.addHandler(handler)

    if stdout_logging:
        rootLogger.addHandler( logging.StreamHandler(sys.stdout) )

    # FIXME: For some reason monkey-patching threading.Thread.run()
    #        doesn't seem to work properly in a Flask app,
    #        so perhaps this custom excepthook isn't really helping us any
    #        except for exceptions during main()
    initialize_excepthook(logger)

    return logfile_path


class PrefixedLogger(logging.Logger):
    """
    Logger subclass that prepends a pre-configured string to every log message.
    """
    def __init__(self, base_logger, msg_prefix):
        super().__init__(base_logger.name, base_logger.level)
        self.base_logger = base_logger
        self.msg_prefix = msg_prefix
    
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        msg = self.msg_prefix + msg
        self.base_logger._log(level, msg, args, exc_info, extra, stack_info)

    def __eq__(self, other):
        return (self.base_logger.name == other.base_logger.name
                and self.msg_prefix == other.msg_prefix)

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


def initialize_excepthook(logger=logging.getLogger()):
    """
    This excepthook simply logs all unhandled exception tracebacks with Logger.error()
    """
    orig_excepthook = sys.excepthook
    def _log_exception(*exc_info):
        # Write traceback to logger.error
        thread_name = threading.current_thread().name
        logger.error( "Unhandled exception in thread: '{}'".format(thread_name) )
        sio = StringIO()
        traceback.print_exception( exc_info[0], exc_info[1], exc_info[2], file=sio )
        logger.error( sio.getvalue() )

        # Also call the original exception hook
        orig_excepthook(*exc_info)

    sys.excepthook = _log_exception
    _install_thread_excepthook()


def _install_thread_excepthook():
    # This function was copied from: http://bugs.python.org/issue1230540
    # It is necessary because sys.excepthook doesn't work for unhandled exceptions in other threads.
    """
    Workaround for sys.excepthook thread bug
    (https://sourceforge.net/tracker/?func=detail&atid=105470&aid=1230540&group_id=5470).
    Call once from __main__ before creating any threads.
    If using psyco, call psycho.cannotcompile(threading.Thread.run)
    since this replaces a new-style class method.
    """
    run_old = threading.Thread.run
    def run(*args, **kwargs):
        try:
            run_old(*args, **kwargs)
        except SystemExit as ex:
            if ex.code != 0:
                sys.excepthook(*sys.exc_info())
                raise
        except:
            sys.excepthook(*sys.exc_info())
            raise
    threading.Thread.run = run
