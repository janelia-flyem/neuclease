import faulthandler
if not faulthandler.is_enabled():
    faulthandler.enable()

import logging
logging.getLogger('kafka').setLevel(logging.WARNING)
