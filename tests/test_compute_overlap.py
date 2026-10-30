"""Unit tests for Overlaps.compute_overlap, the plane-wave inner product.

compute_overlap is the innermost kernel of the Berry phase calculation:
every plaquette calls it four times. These pin it to the definition,
overlap[i, j] = vdot(band i of state 0, band j of state 1), on random
coefficients where the result has no symmetry to hide a misplaced
conjugate or transpose behind.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from berry_flux_diag import Overlaps


def reference_overlap(c0, c1):
    """The definition, one band pair at a time."""
    return np.array([[np.vdot(a, b) for b in c1] for a in c0])


def random_coeffs(rng, nbands, npw):
    return rng.normal(size=(nbands, npw)) + 1j * rng.normal(size=(nbands, npw))


def make_overlaps(pol_dict, np_dict, kpoints, nbands):
    struct = Structure(Lattice.cubic(4.0), ["Ba"], [[0, 0, 0]])
    return Overlaps.Overlaps({
        "pol_struct": struct, "np_struct": struct,
        "pol_wfcn_dict": pol_dict, "np_wfcn_dict": np_dict,
        "kpoint_list": kpoints, "zval_dict": {"Ba": 10.0},
        "ES_code": "test", "spin_pol": False,
        "pol_band_fill": nbands, "np_band_fill": nbands,
    })


@pytest.fixture
def shared_grid():
    """Two k-points whose states share one G grid, as VASPParser's do."""
    rng = np.random.default_rng(0)
    gvecs = rng.integers(-3, 4, size=(40, 3))
    k0, k1 = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5])
    states = {(l, tuple(k)): random_coeffs(rng, 5, len(gvecs))
              for l in (0, 1) for k in (k0, k1)}
    pol = {tuple(k): {"wfcn": states[(1, tuple(k))], "gvecs": gvecs} for k in (k0, k1)}
    npd = {tuple(k): {"wfcn": states[(0, tuple(k))], "gvecs": gvecs} for k in (k0, k1)}
    return make_overlaps(pol, npd, [k0, k1], 5), states, (k0, k1)


@pytest.mark.parametrize("l0,l1", [(0, 1), (1, 0), (0, 0), (1, 1)])
def test_shared_grid_matches_the_definition(shared_grid, l0, l1):
    ov, states, (k0, k1) = shared_grid
    got = ov.compute_overlap(l0, k0, l1, k1, "z")
    expected = reference_overlap(states[(l0, tuple(k0))], states[(l1, tuple(k1))])
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)


def test_the_overlap_is_not_symmetric_here(shared_grid):
    """So that the test above can tell a missing conjugate or transpose."""
    ov, states, (k0, k1) = shared_grid
    m = ov.compute_overlap(0, k0, 1, k1, "z")
    assert not np.allclose(m, m.T)
    assert not np.allclose(m, m.conj().T)


def test_swapping_the_states_gives_the_adjoint(shared_grid):
    ov, _, (k0, k1) = shared_grid
    np.testing.assert_allclose(ov.compute_overlap(1, k1, 0, k0, "z"),
                               ov.compute_overlap(0, k0, 1, k1, "z").conj().T,
                               rtol=1e-13, atol=1e-13)


def test_rectangular_when_the_band_counts_differ():
    rng = np.random.default_rng(1)
    gvecs = rng.integers(-3, 4, size=(30, 3))
    k = np.zeros(3)
    c_pol, c_np = random_coeffs(rng, 4, 30), random_coeffs(rng, 6, 30)
    ov = make_overlaps({tuple(k): {"wfcn": c_pol, "gvecs": gvecs}},
                       {tuple(k): {"wfcn": c_np, "gvecs": gvecs}}, [k], 4)
    got = ov.compute_overlap(0, k, 1, k, "z")
    assert got.shape == (6, 4)
    np.testing.assert_allclose(got, reference_overlap(c_np, c_pol), rtol=1e-13, atol=1e-13)


def test_mapped_grids_match_the_definition_on_common_gvectors():
    """When the G sets differ, coefficients are matched by G vector first.

    Each state's G list is a different permutation of a different subset of
    one sphere, so the overlap is the sum over the common G vectors only.
    """
    rng = np.random.default_rng(2)
    sphere = np.array([g for g in np.ndindex(5, 5, 5)]) - 2
    idx0 = rng.permutation(len(sphere))[:90]
    idx1 = rng.permutation(len(sphere))[:80]
    g0, g1 = sphere[idx0], sphere[idx1]
    c0, c1 = random_coeffs(rng, 3, len(g0)), random_coeffs(rng, 3, len(g1))
    k0, k1 = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.25])
    ov = make_overlaps({tuple(k1): {"wfcn": c1, "gvecs": g1}},
                       {tuple(k0): {"wfcn": c0, "gvecs": g0}}, [k0, k1], 3)

    common = {tuple(g): (i, None) for i, g in enumerate(g0)}
    pairs = [(common[tuple(g)][0], j) for j, g in enumerate(g1) if tuple(g) in common]
    a = np.array([c0[:, i] for i, _ in pairs]).T
    b = np.array([c1[:, j] for _, j in pairs]).T

    got = ov.compute_overlap(0, k0, 1, k1, "z")
    np.testing.assert_allclose(got, reference_overlap(a, b), rtol=1e-12, atol=1e-12)
