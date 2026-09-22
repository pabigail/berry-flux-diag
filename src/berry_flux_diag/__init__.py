# src/berry_flux_diag/__init__.py

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

