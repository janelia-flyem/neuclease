import os
import sys
import logging
import logging.handlers
import threading
import traceback
import functools
import multiprocessing
from io import StringIO


def init_logging(logger, log_dir, db_path, debug_mode=False):
    if not log_dir:
        log_dir = os.path.dirname(db_path)
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
    handler = logging.handlers.RotatingFileHandler(logfile_path, maxBytes=int(10e9), backupCount=10)
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)

    rootLogger.setLevel(logging.INFO)
    rootLogger.addHandler(handler)
    
    if debug_mode:
        rootLogger.addHandler( logging.StreamHandler(sys.stdout) )

    # FIXME: For some reason monkey-patching threading.Thread.run()
    #        doesn't seem to work properly in a Flask app,
    #        so perhaps this custom excepthook isn't really helping us any
    #        except for exceptions during main()
    initialize_excepthook(logger)

    return logfile_path


class ProtectedLogger:
    """
    A simple wrapper around logging.Logger that protects the log() method
    (and therefore also protects info(), warning(), etc.) with a multiprocessing.Lock,
    to avoid intermingled log messages when writing from multiple processes.
    """
    def __init__(self, name):
        self.name = name
        self.lock = multiprocessing.Lock()
    
    def __getattr__(self, attrname):
        if attrname == 'log':
            return object.__getattr__(self, attrname)
        else:
            return getattr(logging.getLogger(self.name), attrname)

    def log(self, *args, **kwargs):
        with self.lock:
            return logging.getLogger(self.name).log(*args, **kwargs)


class ExceptionLogger:
    """
    Context manager.
    Any exceptions that occur while the context is active
    will be logged to the given logger and re-raised.
    """
    def __init__(self, logger):
        self.logger = logger
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            sio = StringIO()
            traceback.print_exception( exc_type, exc_value, exc_tb, file=sio )
            self.logger.error( sio.getvalue() )


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
