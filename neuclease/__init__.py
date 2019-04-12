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

try:
    import numexpr
except ImportError:
    pass
else:
    import textwrap
    msg = textwrap.dedent("""\
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
    raise AssertionError(msg)
