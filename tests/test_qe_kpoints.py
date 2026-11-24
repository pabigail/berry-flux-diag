"""QEParser's k-points are in crystal coordinates.

QE writes k-points in Cartesian units of 2 pi / alat; VASP writes them
fractional. Symmetry operations - and comparing the k-points of two runs
whose cells differ - need the fractional ones.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from berry_flux_diag import QEParser

pytestmark = pytest.mark.skipif(QEParser is None, reason="QEParser needs qeschema and h5py")


def qe_dict(cell_bohr, kpoints_crystal):
    """The parts of a QE XML dict get_kpt_list_from_qeschema reads, with the
    k-points written as QE writes them."""
    cell = np.asarray(cell_bohr, dtype=float)
    alat = float(np.linalg.norm(cell[0]))
    recip = 2 * np.pi * np.linalg.inv(cell).T            # rows b_i, 1/bohr
    cart = np.asarray(kpoints_crystal) @ recip / (2 * np.pi / alat)
    return {"qes:espresso": {"output": {
        "atomic_structure": {"@alat": alat,
                             "cell": {"a1": cell[0].tolist(), "a2": cell[1].tolist(),
                                      "a3": cell[2].tolist()}},
        "band_structure": {"ks_energies": [{"k_point": {"$": k.tolist()}} for k in cart]},
    }}}


def test_the_batio3_fixture_mesh_is_fractional():
    """5x5x5 with a half-step shift. QE writes it centred on Gamma, so every
    coordinate is one of -0.5, -0.3, -0.1, 0.1, 0.3."""
    import qeschema
    from capture_reference import QE_DIR

    doc = qeschema.PwDocument()
    doc.read(str(QE_DIR / "pol" / "batio3_pol.xml"), validation="lax")
    kpts = np.array(QEParser.get_kpt_list_from_qeschema(doc.to_dict(validation="lax")))
    assert kpts.shape == (125, 3)
    axis = [-0.5, -0.3, -0.1, 0.1, 0.3]
    np.testing.assert_allclose(np.sort(np.unique(np.round(kpts, 9))), axis, rtol=0, atol=1e-12)
    assert {tuple(np.round(k, 9)) for k in kpts} == \
        {tuple(np.round(k, 9)) for k in product(axis, repeat=3)}


def test_a_skewed_cell_converts_exactly():
    cell = [[7.5, 0.0, 0.0], [1.2, 7.9, 0.0], [0.4, -0.7, 8.3]]
    k = np.array([[0.25, -0.5, 1 / 3], [0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    got = np.array(QEParser.get_kpt_list_from_qeschema(qe_dict(cell, k)))
    np.testing.assert_allclose(got, k, rtol=0, atol=1e-12)


def test_runs_on_different_cells_give_the_same_kpoints():
    """In Cartesian units the same mesh sits at different points; in crystal
    coordinates it does not, so check_kpoints_match can compare them."""
    k = np.array(list(product([0.0, 0.5], repeat=3)))
    a = QEParser.get_kpt_list_from_qeschema(qe_dict(np.diag([7.6, 7.6, 7.9]), k))
    b = QEParser.get_kpt_list_from_qeschema(qe_dict(np.diag([7.6, 7.6, 7.6]), k))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
