# src/berry_flux_diag/__init__.py

# Core modules
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

