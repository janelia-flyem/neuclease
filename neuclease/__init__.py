import sys
import faulthandler
if not faulthandler.is_enabled():
    faulthandler.enable()

import logging
logging.getLogger('pykafka').setLevel(logging.WARNING)

## Don't show the following warning from within pandas:
## FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
import warnings
warnings.filterwarnings("ignore", module=r"pandas\..*", category=FutureWarning)

def configure_default_logging():
    """
    Simple logging configuration.
    Useful for interactive terminal sessions.
    """
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
