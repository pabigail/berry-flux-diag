# src/berry_flux_diag/__init__.py

import logging as _logging


def configure_logging(level="INFO", stream=None):
    """Send this package's log messages to the console.

    The package logs rather than prints, so by default only warnings
    surface and progress messages are silent. Call this once - in a
    notebook or a script - to see them:

        import berry_flux_diag as bfd
        bfd.configure_logging()          # INFO: progress and results
        bfd.configure_logging("DEBUG")   # adds per-direction detail

    This touches only the berry_flux_diag logger, never the root logger,
    so it cannot disturb the logging of a program that imports this
    package. Calling it again replaces the handler rather than adding a
    second one, so messages are not duplicated.
    """
    logger = _logging.getLogger(__name__)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = _logging.StreamHandler(stream)
    handler.setFormatter(_logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger


# Core modules
from . import constants
from . import preprocess
from . import utils
from . import Overlaps

# Optional modules (only if installed)
try :
    from . import QEParser
except ModuleNotFoundError:
    QEParser = None

try:
    from . import VASPParser
except ModuleNotFoundError:
    VASPParser = None

try:
    from . import BFDMaker
except ModuleNotFoundError:
    BFDMaker = None

try:
    from . import BFDJobs
except ModuleNotFoundError:
    BFDJobs = None

# VASPParser_unnormalized is deliberately absent from this list. It reads a
# WAVECAR without pawpyseed, so it runs where MKL is unavailable, but it
# omits the PAW augmentation terms and the polarization it produces is not
# physically correct. Importing it here would let a user reach those numbers
# by accident whenever pawpyseed is missing, so it must be named explicitly:
#
#     from berry_flux_diag import VASPParser_unnormalized
#
# See that module's docstring before using it.

