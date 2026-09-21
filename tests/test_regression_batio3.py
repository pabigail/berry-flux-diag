"""BaTiO3 regression tests against captured reference results.

These pin the numbers the current code produces so that later work -
vectorizing the overlaps, consolidating the polarization assembly, spin
polarization, IBZ unfolding - can be checked for correctness rather than
only for not crashing.

The QE and VASP references are captured from different calculations and are
not expected to agree with each other; each is only compared against itself.

Regenerate a reference deliberately, never to make a red test pass:

    python tests/capture_reference.py qe
    python tests/capture_reference.py vasp --pol-dir <dir> --np-dir <dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import berry_flux_diag as bfd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_reference import capture_qe, capture_vasp  # noqa: E402

# Tolerances. The string sums are a sum of ~N^3 independent plaquette phases,
# so reordering the arithmetic (vectorizing, parallelizing) moves the last
# couple of digits; anything larger is a real change. The polarization is in
# uC/cm^2, where 1e-6 is far below any physically meaningful difference.
STRING_SUM_ATOL = 1e-10
POLARIZATION_ATOL = 1e-6


def assert_matches_reference(result: dict, reference: dict) -> None:
    """Compare a fresh run against a stored reference, field by field."""
    # Structural facts first: a mismatch here means the run is not comparable,
    # and reporting it as a polarization difference would be misleading.
    assert result["num_kpoints"] == reference["num_kpoints"]
    assert result["spin_polarized"] == reference["spin_polarized"]
    assert result["string_lens"] == reference["string_lens"]
    assert result["zval_dict"] == reference["zval_dict"]
    assert result["pol_formula"] == reference["pol_formula"]

    fill_keys = (
        ["band_fill_up", "band_fill_down"] if reference["spin_polarized"] else ["band_fill"]
    )
    for key in fill_keys:
        assert result[key] == reference[key], f"{key} changed"

    np.testing.assert_allclose(
        result["string_sums"],
        reference["string_sums"],
        atol=STRING_SUM_ATOL,
        err_msg="electronic string sums drifted from the reference",
    )
    np.testing.assert_allclose(
        result["polarization_norm"],
        reference["polarization_norm"],
        atol=POLARIZATION_ATOL,
        err_msg="polarization magnitude drifted from the reference",
    )


@pytest.mark.skipif(bfd.QEParser is None, reason="QEParser needs qeschema and h5py")
def test_batio3_qe_nospin(qe_reference):
    """Quantum ESPRESSO path, using the inputs committed under tests/."""
    assert_matches_reference(capture_qe(), qe_reference)


@pytest.mark.skipif(bfd.VASPParser is None, reason="VASPParser needs pawpyseed")
def test_batio3_vasp_nospin(vasp_reference, vasp_run_dirs):
    """VASP path, using run directories named by the environment."""
    pol_dir, np_dir = vasp_run_dirs
    assert_matches_reference(capture_vasp(pol_dir, np_dir), vasp_reference)
