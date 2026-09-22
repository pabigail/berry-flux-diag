"""Unit tests for the occupied-band count.

The BaTiO3 regression fixtures cannot cover this: they are insulators with
the same number of filled bands at every k-point, which is exactly the case
the old off-by-one comparison got right. The cases below are written as bare
occupation arrays so the k-dependent ones can be tested without a DFT run.
"""

from __future__ import annotations

import numpy as np
import pytest

from berry_flux_diag import utils

TOL = 1e-6


def occ(n_filled: int, n_bands: int) -> list[float]:
    """Occupations for one k-point: n_filled full bands, the rest empty."""
    return [1.0] * n_filled + [0.0] * (n_bands - n_filled)


def test_uniform_filling():
    """An insulator: every k-point has the same count."""
    assert utils.max_filled_bands([occ(20, 24)] * 5, TOL) == 20


def test_one_extra_filled_band_at_one_kpoint():
    """The regression case for the off-by-one.

    The old comparison was `(fill - 1) > max_fill`, so it only accepted a
    k-point with at least two more filled bands than the running maximum.
    A single extra band - the common case when a band dips below E_F at one
    k-point - was dropped, and those states were left out of the Berry flux.
    """
    occupations = [occ(20, 24), occ(21, 24), occ(20, 24)]
    assert utils.max_filled_bands(occupations, TOL) == 21


def test_extra_band_at_the_first_kpoint():
    """Order must not matter: the maximum is over all k-points."""
    assert utils.max_filled_bands([occ(21, 24), occ(20, 24)], TOL) == 21


def test_two_extra_filled_bands():
    """The case the old code did handle, kept so the fix stays symmetric."""
    assert utils.max_filled_bands([occ(20, 24), occ(22, 24)], TOL) == 22


def test_all_bands_filled():
    """No band below tol means every band counts, rather than StopIteration."""
    assert utils.max_filled_bands([occ(24, 24)] * 3, TOL) == 24


def test_partial_occupation_below_tol_is_empty():
    """A band at 1e-9 is numerical noise, not an occupied state."""
    occupations = [[1.0, 1.0, 1e-9, 0.0]]
    assert utils.max_filled_bands(occupations, TOL) == 2


def test_partial_occupation_above_tol_is_filled():
    """A genuinely partly filled band counts; metallicity is not diagnosed here."""
    occupations = [[1.0, 1.0, 0.5, 0.0]]
    assert utils.max_filled_bands(occupations, TOL) == 3


def test_no_occupied_bands_raises():
    """An empty lowest band means the tolerance or the parse is wrong."""
    with pytest.raises(ValueError, match="no occupied bands at k-point 1"):
        utils.max_filled_bands([occ(20, 24), occ(0, 24)], TOL)


def test_accepts_numpy_arrays():
    """The VASP path passes columns of wavecar.band_energy, not lists."""
    occupations = [np.array(occ(20, 24)), np.array(occ(21, 24))]
    assert utils.max_filled_bands(occupations, TOL) == 21


def test_accepts_a_generator():
    """All four call sites pass a generator, so it must not be re-iterated."""
    occupations = (occ(n, 24) for n in (20, 21, 20))
    assert utils.max_filled_bands(occupations, TOL) == 21
