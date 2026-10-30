"""Unit tests for the numerical pieces of berry_flux_diag.paw.

The port as a whole was checked coefficient by coefficient against
pawpyseed's C, and test_regression_batio3 pins the polarization it
produces, but both need VASP run directories. These need nothing, and check
each building block against something known in closed form, so that a
broken piece is named rather than showing up only as a drifted
polarization.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from berry_flux_diag import paw

# A logarithmic radial grid shaped like a VASP POTCAR's (this one is O's).
R0, RATIO, NPTS = 2.92861757568e-05, 1.0325175053072226, 377
LOG_GRID = R0 * RATIO ** np.arange(NPTS)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_bessel_transform_of_a_gaussian(l):
    """int_0^inf r^2 [r^l e^{-r^2}] j_l(kr) dr = sqrt(pi) / 2^(l+2) k^l e^{-k^2/4}.

    wave_sbt takes r times the radial function, as POTCAR partial waves
    are stored.
    """
    d, k = paw.sbt_setup(1e7, 0, l, NPTS, LOG_GRID)
    transform = paw.wave_sbt(d, LOG_GRID ** (l + 1) * np.exp(-LOG_GRID ** 2), l)
    exact = math.sqrt(math.pi) / 2 ** (l + 2) * k ** l * np.exp(-k ** 2 / 4)
    window = (k > 0.05) & (k < 12)
    np.testing.assert_allclose(transform[window], exact[window], rtol=0, atol=1e-8)


def test_ylm_are_orthonormal():
    """Normalisation and the Condon-Shortley phase together: <Y_lm|Y_l'm'> = delta."""
    x, w = np.polynomial.legendre.leggauss(24)
    theta = np.arccos(x)
    phi = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    weight = np.outer(w, np.full(len(phi), 2 * np.pi / len(phi)))
    lm = [(l, m) for l in range(4) for m in range(-l, l + 1)]
    Y = np.array([paw._ylm(l, m, T, P) for l, m in lm])
    gram = np.einsum("aij,bij,ij->ab", Y.conj(), Y, weight)
    np.testing.assert_allclose(gram, np.eye(len(lm)), atol=1e-12)


@pytest.mark.parametrize("l,m", [(1, 1), (2, 1), (2, 2), (3, 3)])
def test_negative_m_is_the_conjugate_up_to_sign(l, m):
    theta, phi = 0.7, 2.1
    np.testing.assert_allclose(paw._ylm(l, -m, theta, phi),
                               (-1) ** m * np.conj(paw._ylm(l, m, theta, phi)))


def test_angles_handle_the_poles_and_the_origin():
    v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [0.0, -1.0, 0.0]])
    theta, phi = paw._angles(v, np.linalg.norm(v, axis=1))
    np.testing.assert_allclose(theta, [0, 0, np.pi, np.pi / 2])
    np.testing.assert_allclose(phi, [0, 0, 0, 3 * np.pi / 2])


def test_spline_interpolates_a_smooth_function():
    """Away from the outer end, where the spline's zero-curvature boundary
    condition (VASP's) is wrong for this f. The partial-wave differences it
    is used on vanish there, so that end does not matter in practice."""
    f = np.exp(-LOG_GRID) * LOG_GRID ** 2
    s = paw.spline_coeff(LOG_GRID, f)
    r = np.sqrt(LOG_GRID[1:-11] * LOG_GRID[2:-10])  # between the knots
    np.testing.assert_allclose(paw.wave_interpolate(r, LOG_GRID, f, s),
                               np.exp(-r) * r ** 2, atol=1e-7)


def test_interpolators_are_zero_past_the_grid_and_flat_before_it():
    f = np.cos(LOG_GRID)
    s = paw.spline_coeff(LOG_GRID, f)
    out = paw.wave_interpolate(np.array([LOG_GRID[0] / 2, LOG_GRID[-1] * 2]), LOG_GRID, f, s)
    np.testing.assert_array_equal(out, [f[0], 0.0])

    lin = np.linspace(0, 1.5, 100, endpoint=False)
    g = np.cos(lin)
    out = paw.proj_interpolate(np.array([2.0]), 1.5, lin, g, paw.spline_coeff(lin, g))
    np.testing.assert_array_equal(out, [0.0])


def test_cround_rounds_half_away_from_zero():
    """C round(), which the projector sphere centres use, not numpy's half-to-even."""
    np.testing.assert_array_equal(paw._cround(np.array([0.5, 1.5, 2.5, -0.5, -2.5, 1.4])),
                                  [1, 2, 3, -1, -3, 1])


def test_g_sphere_is_complete_and_inside_the_cutoff():
    lattice = np.diag([4.0, 4.0, 4.2])
    encut = 400.0
    b, nbmax = paw._setup(encut, lattice)
    G = paw._g_sphere(np.zeros(3), nbmax, b, encut)
    energy = np.sum((G @ b) ** 2, axis=1) / paw.C
    assert energy.max() <= encut

    # brute force over a box that certainly contains the sphere
    n = int(max(nbmax)) + 2
    box = np.array(np.meshgrid(*[np.arange(-n, n + 1)] * 3, indexing="ij")).reshape(3, -1).T
    inside = box[np.sum((box @ b) ** 2, axis=1) / paw.C <= encut]
    assert {tuple(g) for g in G} == {tuple(g) for g in inside}
    assert len(G) == len(inside)


def test_reciprocal_lattice():
    lattice = np.array([[4.0, 0.0, 0.0], [0.3, 4.1, 0.0], [0.2, -0.1, 4.3]])
    b, _ = paw._setup(100.0, lattice)
    np.testing.assert_allclose(lattice @ b.T, 2 * np.pi * np.eye(3), atol=1e-12)
