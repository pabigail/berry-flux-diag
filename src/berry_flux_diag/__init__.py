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


# --- Optional modules -----------------------------------------------------
#
# Each of these needs a dependency that may not be installed, so a failure to
# import is tolerated. But only *its own* dependency being absent is a benign
# reason to be missing. A ModuleNotFoundError naming anything else - a
# mistyped or unqualified import inside this package, a half-installed
# pymatgen - is a defect, and used to be indistinguishable from a missing
# extra: the module became None either way, and the reason was discarded. The
# reason is now kept, and reported loudly when it is not one of the expected
# dependencies.

#: For each optional module: the dependencies whose absence is benign, and
#: the extra that installs them.
_OPTIONAL_MODULES = {
    "QEParser": (("qeschema", "h5py"), "QE"),
    "VASPParser": (("pawpyseed",), "VASP"),
    "BFDMaker": (("jobflow", "atomate2", "monty"), "VASP_atomate2"),
    "BFDJobs": (("jobflow", "atomate2", "monty"), "VASP_atomate2"),
}

#: Why each unavailable optional module could not be imported, keyed by module
#: name - the name of the module that was actually missing. Empty when every
#: optional module imported. Consult it when one of them is None:
#:
#:     >>> import berry_flux_diag as bfd
#:     >>> bfd.VASPParser is None
#:     True
#:     >>> bfd.unavailable
#:     {'VASPParser': 'pawpyseed'}
unavailable = {}


def _import_optional(name):
    """Import an optional submodule, or return None and record why."""
    import importlib

    expected, extra = _OPTIONAL_MODULES[name]

    try:
        return importlib.import_module("." + name, __name__)
    except ModuleNotFoundError as exc:
        missing = exc.name or "<unknown>"
        unavailable[name] = missing
        logger = _logging.getLogger(__name__)

        if missing.split(".")[0] in expected:
            # The ordinary case: an extra was not installed.
            logger.debug(
                "%s is unavailable because %r is not installed; "
                "install it with: pip install '.[%s]'",
                name, missing, extra,
            )
        else:
            # Not one of this module's dependencies, so the install is not
            # the problem. Say so, rather than let it read as a missing extra.
            logger.warning(
                "%s could not be imported because there is no module named "
                "%r. That is not one of its optional dependencies (%s), so "
                "this is a defect in berry_flux_diag or a broken environment "
                "rather than a missing extra. berry_flux_diag.%s is None.",
                name, missing, ", ".join(expected), name,
            )
        return None


QEParser = _import_optional("QEParser")
VASPParser = _import_optional("VASPParser")
BFDMaker = _import_optional("BFDMaker")
BFDJobs = _import_optional("BFDJobs")

# VASPParser_unnormalized is deliberately absent from this list. It reads a
# WAVECAR without pawpyseed, so it runs where MKL is unavailable, but it
# omits the PAW augmentation terms and the polarization it produces is not
# physically correct. Importing it here would let a user reach those numbers
# by accident whenever pawpyseed is missing, so it must be named explicitly:
#
#     from berry_flux_diag import VASPParser_unnormalized
#
# See that module's docstring before using it.

