"""Shared fixtures and paths for the berry-flux-diag test suite."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Run against the working tree without requiring an editable install.
sys.path.insert(0, str(REPO / "src"))

QE_DIR = REPO / "tests" / "BaTiO3_QE_IO_nospin"
REFERENCE_DIR = REPO / "tests" / "reference"

# VASP references need WAVECAR and POTCAR files, which are too large and (for
# POTCAR) copyrighted to commit. Point these at run directories on disk.
VASP_POL_ENV = "BFD_VASP_BATIO3_POL"
VASP_NP_ENV = "BFD_VASP_BATIO3_NP"


def load_reference(name: str) -> dict:
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"no reference file at {path}; run tests/capture_reference.py first")
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def qe_reference() -> dict:
    return load_reference("batio3_qe_nospin.json")


@pytest.fixture(scope="session")
def vasp_reference() -> dict:
    return load_reference("batio3_vasp_nospin.json")


@pytest.fixture(scope="session")
def vasp_run_dirs() -> tuple[Path, Path]:
    """Polar and nonpolar VASP run directories, or skip."""
    pol, npol = os.environ.get(VASP_POL_ENV), os.environ.get(VASP_NP_ENV)
    if not (pol and npol):
        pytest.skip(f"set {VASP_POL_ENV} and {VASP_NP_ENV} to the BaTiO3 VASP run directories")
    pol_path, np_path = Path(pol), Path(npol)
    for p in (pol_path, np_path):
        if not p.is_dir():
            pytest.skip(f"{p} is not a directory")
    return pol_path, np_path
