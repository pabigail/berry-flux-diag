"""Unit tests for how __init__ handles an optional module that will not import.

A missing extra and a bug inside this package both raise
ModuleNotFoundError. The old code caught both, set the module to None and
discarded the reason, so a mistyped import inside BFDJobs was
indistinguishable from atomate2 not being installed - and in this repo,
that is exactly what was happening.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

import berry_flux_diag as bfd


def fails_with(missing):
    """An import_module that raises as if `missing` were not installed."""
    return patch("importlib.import_module",
                 side_effect=ModuleNotFoundError(
                     f"No module named {missing!r}", name=missing))


# --- a genuinely missing extra is benign ---------------------------------

def test_a_missing_dependency_returns_none(caplog):
    with fails_with("pawpyseed"), caplog.at_level(logging.DEBUG, "berry_flux_diag"):
        assert bfd._import_optional("VASPParser") is None


def test_a_missing_dependency_does_not_warn(caplog):
    """Not installing an extra is a choice, not a fault."""
    with fails_with("pawpyseed"), caplog.at_level(logging.DEBUG, "berry_flux_diag"):
        bfd._import_optional("VASPParser")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_missing_dependency_names_the_extra_that_installs_it(caplog):
    with fails_with("qeschema"), caplog.at_level(logging.DEBUG, "berry_flux_diag"):
        bfd._import_optional("QEParser")
    assert "[QE]" in caplog.text


def test_a_submodule_of_a_dependency_counts_as_that_dependency(caplog):
    with fails_with("pawpyseed.core.momentum"), \
            caplog.at_level(logging.DEBUG, "berry_flux_diag"):
        bfd._import_optional("VASPParser")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- anything else is a defect and must be loud --------------------------

def test_an_unexpected_missing_module_warns(caplog):
    """The case this exists for: a bad import inside our own package."""
    with fails_with("bfd"), caplog.at_level(logging.DEBUG, "berry_flux_diag"):
        assert bfd._import_optional("BFDJobs") is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "bfd" in warnings[0].getMessage()


def test_the_warning_says_it_is_not_a_missing_extra(caplog):
    with fails_with("bfd"), caplog.at_level(logging.DEBUG, "berry_flux_diag"):
        bfd._import_optional("BFDJobs")
    assert "rather than a missing extra" in caplog.text


def test_a_broken_base_dependency_warns(caplog):
    """pymatgen is not optional, so its absence is never benign."""
    with fails_with("pymatgen.core.entries"), \
            caplog.at_level(logging.DEBUG, "berry_flux_diag"):
        bfd._import_optional("BFDJobs")
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_near_miss_on_a_dependency_name_warns(caplog):
    """'jobflows' is not 'jobflow'; a typo must not read as a missing extra."""
    with fails_with("jobflows"), caplog.at_level(logging.DEBUG, "berry_flux_diag"):
        bfd._import_optional("BFDMaker")
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- the reason is recorded either way -----------------------------------

def test_the_reason_is_recorded(caplog):
    with fails_with("pawpyseed"), caplog.at_level(logging.DEBUG, "berry_flux_diag"):
        bfd._import_optional("VASPParser")
    assert bfd.unavailable["VASPParser"] == "pawpyseed"


def test_every_optional_module_is_either_importable_or_explained():
    """No optional module may be None without a recorded reason."""
    for name in bfd._OPTIONAL_MODULES:
        if getattr(bfd, name) is None:
            assert name in bfd.unavailable, f"{name} is None for no recorded reason"


def test_each_optional_module_declares_its_dependencies():
    for name, (expected, extra) in bfd._OPTIONAL_MODULES.items():
        assert expected, f"{name} declares no optional dependencies"
        assert isinstance(extra, str) and extra


def test_a_successful_import_returns_the_module():
    assert bfd._import_optional("QEParser") is not None
