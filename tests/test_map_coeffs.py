"""Unit tests for utils.map_coeffs and gvec_to_index.

map_coeffs lines up two states' plane-wave coefficients by G vector when
their G lists differ, which happens at the wrapped end of every k-string.
It was a Python loop over G vectors - 65 million gvec_to_index calls for
BaTiO3 at 4 x ENCUT, three quarters of compute_string_sums - and is now
array arithmetic. These tests hold the vectorized version to the loop it
replaced, kept below verbatim as the reference, bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from berry_flux_diag import utils


# --- the loop implementation this replaced, verbatim ----------------------

def reference_gvec_to_index(gvec, kpt, max_rad, base):
    kpt_centered_gvec = gvec # + kpt
    shift = np.full(3, max_rad)
    shifted_gvec = kpt_centered_gvec + shift
    index = round(shifted_gvec[0])*base**2 + round(shifted_gvec[1])*base + round(shifted_gvec[2])
    return index


def reference_map_coeffs(coeffs0, gvecs0, kpt0, coeffs1, gvecs1, kpt1):
    gvec_min0, gvec_max0, gvec_len0 = utils.gvec_extrema(gvecs0)
    gvec_min1, gvec_max1, gvec_len1 = utils.gvec_extrema(gvecs1)

    max_g_rad = np.max([np.max(gvec_max0), np.max(gvec_max1)])
    min_g_rad = np.abs(np.min([np.min(gvec_min0), np.min(gvec_min1)]))
    max_rad = int(np.max([max_g_rad, min_g_rad]))
    base = (2*max_rad + 1)

    coeffs0_mapped = np.zeros((len(coeffs0), base**3 + base**2 + base + 1), dtype=complex)
    coeffs1_mapped = np.zeros((len(coeffs1), base**3 + base**2 + base + 1), dtype=complex)

    for i in range(gvec_len0):
        mapped_i = reference_gvec_to_index(gvecs0[i], kpt0, max_rad, base)
        coeffs0_mapped[:, mapped_i] = coeffs0[:, i]
    for i in range(gvec_len1):
        mapped_i = reference_gvec_to_index(gvecs1[i], kpt1, max_rad, base)
        coeffs1_mapped[:, mapped_i] = coeffs1[:, i]

    return coeffs0_mapped, coeffs1_mapped


# --- helpers ---------------------------------------------------------------

def sphere(radius):
    r = np.arange(-radius, radius + 1)
    g = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    return g[np.sum(g ** 2, axis=1) <= radius ** 2]


def random_coeffs(rng, nbands, npw):
    return rng.normal(size=(nbands, npw)) + 1j * rng.normal(size=(nbands, npw))


def random_pair(seed, radius=4, n0=None, n1=None, nbands=5, dtype=np.int32):
    rng = np.random.default_rng(seed)
    s = sphere(radius).astype(dtype)
    n0 = n0 or len(s) - 3
    n1 = n1 or len(s) - 7
    g0 = s[rng.permutation(len(s))[:n0]]
    g1 = s[rng.permutation(len(s))[:n1]]
    return random_coeffs(rng, nbands, n0), g0, random_coeffs(rng, nbands + 1, n1), g1


K0, K1 = np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, -0.5])


# --- bit-for-bit agreement with the loop ----------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_matches_the_loop_exactly(seed):
    c0, g0, c1, g1 = random_pair(seed)
    got = utils.map_coeffs(c0, g0, K0, c1, g1, K1)
    expected = reference_map_coeffs(c0, g0, K0, c1, g1, K1)
    for a, b in zip(got, expected):
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


def test_matches_the_loop_on_the_wrapped_string_end():
    """What compute_overlap actually passes: one G list shifted by the
    closing vector, as an int64 array, against an unshifted int32 one."""
    c0, g0, c1, g1 = random_pair(7)
    _, _, gvec_shift = utils.direction_to_vals("z")
    g1_shifted = np.asarray(g1) - gvec_shift
    got = utils.map_coeffs(c0, g0, K0, c1, g1_shifted, K1)
    expected = reference_map_coeffs(c0, g0, K0, c1, list(g1_shifted), K1)
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)


def test_matches_the_loop_with_a_list_of_gvecs():
    """compute_overlap used to pass the shifted G vectors as a list."""
    c0, g0, c1, g1 = random_pair(8)
    got = utils.map_coeffs(c0, list(g0), K0, c1, list(g1), K1)
    expected = reference_map_coeffs(c0, g0, K0, c1, g1, K1)
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)


def test_matches_the_loop_with_float_gvecs():
    c0, g0, c1, g1 = random_pair(9, dtype=np.float64)
    got = utils.map_coeffs(c0, g0, K0, c1, g1, K1)
    expected = reference_map_coeffs(c0, g0, K0, c1, g1, K1)
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)


def test_a_repeated_gvector_keeps_the_last_column_as_the_loop_did():
    c0, g0, c1, g1 = random_pair(10)
    g0 = np.vstack([g0, g0[:2]])
    c0 = np.hstack([c0, random_coeffs(np.random.default_rng(11), len(c0), 2)])
    got = utils.map_coeffs(c0, g0, K0, c1, g1, K1)
    expected = reference_map_coeffs(c0, g0, K0, c1, g1, K1)
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)


# --- what the mapping means ------------------------------------------------

def test_the_overlap_pairs_coefficients_by_gvector():
    """After mapping, the inner product sums over the common G vectors."""
    c0, g0, c1, g1 = random_pair(12)
    m0, m1 = utils.map_coeffs(c0, g0, K0, c1, g1, K1)
    where1 = {tuple(g): j for j, g in enumerate(g1)}
    pairs = [(i, where1[tuple(g)]) for i, g in enumerate(g0) if tuple(g) in where1]
    expected = sum(np.outer(c0[:, i].conj(), c1[:, j]) for i, j in pairs)
    np.testing.assert_allclose(m0.conj() @ m1.T, expected, rtol=1e-12, atol=1e-12)


def test_every_coefficient_lands_somewhere_distinct():
    c0, g0, c1, g1 = random_pair(13)
    m0, _ = utils.map_coeffs(c0, g0, K0, c1, g1, K1)
    assert np.count_nonzero(np.any(m0 != 0, axis=0)) == len(g0)
    np.testing.assert_allclose(np.sort(np.abs(m0[m0 != 0])), np.sort(np.abs(c0.ravel())))


# --- gvec_to_index ---------------------------------------------------------

def test_index_of_one_gvector_is_an_int_matching_the_loop():
    for g in ([0, 0, 0], [-3, 2, 1], [3, -3, -3]):
        got = utils.gvec_to_index(np.array(g), K0, 3, 7)
        assert isinstance(got, int)
        assert got == reference_gvec_to_index(np.array(g), K0, 3, 7)


def test_index_of_many_gvectors_matches_the_loop():
    g = sphere(5)
    got = utils.gvec_to_index(g, K0, 5, 11)
    assert got.shape == (len(g),)
    assert got.tolist() == [reference_gvec_to_index(v, K0, 5, 11) for v in g]


def test_indices_are_collision_free_within_the_radius():
    g = sphere(6)
    idx = utils.gvec_to_index(g, K0, 6, 13)
    assert len(np.unique(idx)) == len(g)


def test_half_integers_round_to_even_like_the_builtin():
    g = np.array([[0.5, 1.5, -0.5], [2.5, -1.5, 0.5]])
    got = utils.gvec_to_index(g, K0, 3, 7)
    assert got.tolist() == [reference_gvec_to_index(v, K0, 3, 7) for v in g]
