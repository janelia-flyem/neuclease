import sys
import faulthandler
if not faulthandler.is_enabled():
    faulthandler.enable()

import logging
logging.getLogger('kafka').setLevel(logging.WARNING)

def configure_default_logging():
    """
    Simple logging configuration.
    Useful for interactive terminal sessions.
    """
    formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
