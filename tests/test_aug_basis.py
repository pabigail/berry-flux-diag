"""Unit tests for MomentumMatrix._aug_basis after the Y_lm reuse.

_aug_basis used to evaluate Y_lm afresh for every site and projector
channel, and each element's radial factors afresh for every site of that
element: for BaTiO3 that was 26040 _ylm calls, 8.6 s of a 21 s parse. Both
depend only on k + G (and the element), so they are now computed once per
k-point. The result must not change at all - not to a tolerance - so these
compare against the old loop, kept below verbatim, on a synthetic cell with
an element on several sites, several channels sharing each l, and two
elements with the same channel layout but different partial waves.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from berry_flux_diag import paw

R0, RATIO, NPTS = 2.92861757568e-05, 1.0325175053072226, 377
LOG_GRID = R0 * RATIO ** np.arange(NPTS)
LATTICE = np.array([[4.0, 0.0, 0.0], [0.1, 4.1, 0.0], [0.0, 0.2, 4.2]])
ENCUT = 150.0


def reference_aug_basis(mm, k_idx):
    """MomentumMatrix._aug_basis as of b2e455d, before the reuse."""
    wav = mm.wf.wav
    G = mm._grid
    gcart = -(G + wav["kpts"][k_idx]) @ wav["reclattice"]
    r = np.sqrt(np.einsum("ij,ij->i", gcart, gcart))
    theta, phi = paw._angles(gcart, r)
    pref = 4 * paw.PI * paw._det(wav["lattice"]) ** -0.5
    cols = []
    for coord, el in zip(mm.wf.coords, mm.wf.site_elems):
        phase = np.exp(-2 * paw.PI * 1j * (G @ coord))
        radial = [paw.wave_interpolate(r, el["kwave_grid"], f["kwave"], f["kwave_spline"])
                  for f in el["funcs"]]
        for n, l, m in el["lm"]:
            cols.append(radial[n] * paw._ylm(l, m, theta, phi) * (1j ** l) * phase * pref)
    return np.array(cols).T


def fake_element(ls, width):
    """What _element_setup reads from a Pseudopotential, with Gaussian
    partial waves; width sets how the elements differ."""
    rmax, ndata = 2.0, 100
    grid = rmax / ndata * np.arange(ndata)
    waves = [LOG_GRID ** (l + 1) * np.exp(-(LOG_GRID / (width * (1 + 0.2 * i))) ** 2)
             for i, l in enumerate(ls)]
    pp = SimpleNamespace(
        ndata=ndata, rmax=rmax, grid=LOG_GRID, ls=ls,
        realprojs=[np.exp(-grid ** 2) * (1 + i) for i in range(len(ls))],
        aewaves=waves, pswaves=[0.5 * w for w in waves])
    return paw._element_setup(pp)


def make_momentum_matrix(kpts):
    a = fake_element([0, 0, 1, 1, 2, 2], width=0.6)
    b = fake_element([0, 0, 1, 1, 2, 2], width=0.9)   # same (l, m)s as a
    c = fake_element([1, 1], width=0.4)
    site_elems = [a, b, c, c, c]
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.52],
                       [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    b_rec, _ = paw._setup(ENCUT, LATTICE)
    wav = {"lattice": LATTICE, "reclattice": b_rec, "kpts": [np.array(k) for k in kpts]}
    wf = SimpleNamespace(wav=wav, encut=ENCUT, coords=coords, site_elems=site_elems)
    return paw.MomentumMatrix(wf)


KPTS = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.25, -0.25, 0.5], [0.5, 0.5, 0.5]]


@pytest.mark.parametrize("k_idx", range(len(KPTS)))
def test_bit_identical_to_the_old_loop(k_idx):
    mm = make_momentum_matrix(KPTS)
    new, old = mm._aug_basis(k_idx), reference_aug_basis(mm, k_idx)
    assert new.dtype == old.dtype and new.shape == old.shape
    assert new.tobytes() == old.tobytes()


def test_the_cell_exercises_both_reuses():
    """Repeated (l, m)s across channels and sites, and an element on
    several sites; otherwise the test above would prove little."""
    mm = make_momentum_matrix(KPTS)
    lms = [(l, m) for el in mm.wf.site_elems for _, l, m in el["lm"]]
    assert len(set(lms)) < len(lms)
    ids = [id(el) for el in mm.wf.site_elems]
    assert len(set(ids)) < len(ids)


def test_elements_with_the_same_channels_keep_their_own_radial_factors():
    """a and b share every (l, m) but not their partial waves: a radial
    cache keyed on anything but the element itself would mix them up."""
    mm = make_momentum_matrix(KPTS)
    basis = mm._aug_basis(1)
    na = len(mm.wf.site_elems[0]["lm"])
    a_cols, b_cols = basis[:, :na], basis[:, na:2 * na]
    phase_a = np.exp(-2 * paw.PI * 1j * (mm._grid @ mm.wf.coords[0]))
    phase_b = np.exp(-2 * paw.PI * 1j * (mm._grid @ mm.wf.coords[1]))
    assert not np.allclose(a_cols / phase_a[:, None], b_cols / phase_b[:, None])


def test_ylm_is_evaluated_once_per_l_m(monkeypatch):
    mm = make_momentum_matrix(KPTS)
    calls = []
    orig = paw._ylm

    def counting(l, m, theta, phi):
        calls.append((l, m))
        return orig(l, m, theta, phi)

    monkeypatch.setattr(paw, "_ylm", counting)
    mm._aug_basis(2)
    assert sorted(calls) == sorted(set(calls))
    assert set(calls) == {(l, m) for el in mm.wf.site_elems for _, l, m in el["lm"]}


def test_nothing_carries_over_between_kpoints():
    """The caches are per call: a k-point's basis must not depend on which
    k-point was built before it."""
    mm = make_momentum_matrix(KPTS)
    first = mm._aug_basis(3)
    mm._aug_basis(0)
    assert mm._aug_basis(3).tobytes() == first.tobytes()
    assert mm._aug_basis(3).tobytes() == reference_aug_basis(mm, 3).tobytes()
