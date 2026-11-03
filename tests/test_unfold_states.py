"""Unit tests for unfolding wavefunctions from the IBZ to the full mesh.

The exact test uses states whose plane-wave coefficients are known at every
k: Bloch sums of Gaussians centred on each atom,

    B_k^a(x) = sum_L exp(i k.(L + x_a)) g(x - L - x_a),
    c_k^a(G) = g^(|k + G|) exp(-2 pi i G.x_a).

A symmetry operation maps the set {B_k^a} onto the set {B_Rk^a'}, each
state up to a phase. So unfolding the IBZ states must give, at every mesh
point, states spanning exactly what the analytic ones span - which a wrong
G mapping, phase, translation or time reversal breaks. The structures have
nonzero translations: BaTiO3 shifted off the origin, and hcp Zn, whose
screw axis has a genuine half-cell translation.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from berry_flux_diag import symmetry as sym
from test_symmetry import SPECIES, VASP_POL

SHIFTED = Structure(VASP_POL.lattice, SPECIES, VASP_POL.frac_coords + [0.11, 0.23, 0.07])
HCP = Structure(Lattice.hexagonal(2.66, 4.95), ["Zn", "Zn"], [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]])
WIDTHS = (0.6, 1.1)        # two Gaussians per atom: more bands than atoms
QCUT = 6.0                 # 1/Angstrom


def g_sphere(structure, k):
    """Integer G with |k + G| < QCUT, sorted."""
    recip = structure.lattice.reciprocal_lattice.matrix
    n = int(np.ceil(QCUT / np.min(np.linalg.norm(recip, axis=1)))) + 2
    box = np.array(list(product(range(-n, n + 1), repeat=3)))
    q = np.linalg.norm((k + box) @ recip, axis=1)
    return box[q < QCUT]


def bloch_sums(structure, k, gvecs):
    """(nbands, ng) coefficients of B_k^a for every atom a and width."""
    recip = structure.lattice.reciprocal_lattice.matrix
    q = np.linalg.norm((k + gvecs) @ recip, axis=1)
    rows = []
    for x in structure.frac_coords:
        for w in WIDTHS:
            rows.append(np.exp(-(q * w) ** 2 / 2) * np.exp(-2j * np.pi * (gvecs @ x)))
    return np.array(rows)


def same_span(a, ga, b, gb):
    """1 - (smallest cosine of the principal angles) between the spans of
    a and b, after lining their plane waves up. 0 for the same subspace."""
    cols = {tuple(g): i for i, g in enumerate(ga)}
    assert len(ga) == len(gb) and all(tuple(g) in cols for g in gb), "plane-wave sets differ"
    a = a[:, [cols[tuple(g)] for g in gb]]
    qa, _ = np.linalg.qr(a.T)
    qb, _ = np.linalg.qr(b.T)
    return 1 - np.linalg.svd(qa.conj().T @ qb, compute_uv=False).min()


def unfold_everywhere(structure, n, shift, translations=None):
    """Worst subspace mismatch over the mesh, unfolding analytic IBZ states."""
    s = sym.shared_symmetry(structure, structure)
    t = s.translations["pol"] if translations is None else translations
    mesh = sym.monkhorst_pack(n, shift)
    reps, _ = sym.reduce_mesh(mesh, s.k_rotations)
    ibz = mesh[reps]
    kmap = sym.unfold_kpoints(ibz, s.k_rotations)
    worst = 0.0
    for j in range(len(kmap.full_kpoints)):
        i, o, tr, g0 = kmap.ibz_index[j], kmap.op_index[j], kmap.time_reversal[j], kmap.gshift[j]
        k = ibz[i]
        target = (-1 if tr else 1) * s.k_rotations[o] @ k - g0
        g = g_sphere(structure, k)
        c, gu = sym.unfold_state(bloch_sums(structure, k, g), g, k, s.k_rotations[o], t[o], tr, g0)
        gt = g_sphere(structure, target)
        worst = max(worst, same_span(c, gu, bloch_sums(structure, target, gt), gt))
    return worst, s


@pytest.mark.parametrize("structure, n, shift", [
    (SHIFTED, (4, 4, 4), (1, 1, 1)),
    (SHIFTED, (4, 4, 3), (0, 0, 0)),
    (HCP, (6, 6, 4), (0, 0, 0)),
])
def test_unfolded_states_span_the_states_at_every_mesh_point(structure, n, shift):
    worst, s = unfold_everywhere(structure, n, shift)
    assert np.abs(s.translations["pol"]).max() > 0.05      # translations really in play
    assert worst < 1e-12


@pytest.mark.parametrize("structure, n, shift", [
    (SHIFTED, (4, 4, 4), (1, 1, 1)),
    (HCP, (6, 6, 4), (0, 0, 0)),
])
def test_without_the_translations_they_do_not(structure, n, shift):
    """The test above can tell: dropping t gives the wrong states."""
    s = sym.shared_symmetry(structure, structure)
    worst, _ = unfold_everywhere(structure, n, shift, translations=np.zeros_like(s.translations["pol"]))
    assert worst > 1e-3


def test_the_identity_returns_the_state_unchanged():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(3, 20)) + 1j * rng.normal(size=(3, 20))
    g = rng.integers(-3, 4, size=(20, 3))
    out, gout = sym.unfold_state(c, g, np.array([0.1, 0.2, 0.3]), np.eye(3, dtype=int),
                                 np.zeros(3), False, np.zeros(3, dtype=int))
    assert out.tobytes() == c.tobytes() and np.array_equal(gout, g)


def test_time_reversal_twice_is_the_identity():
    rng = np.random.default_rng(1)
    c = rng.normal(size=(3, 20)) + 1j * rng.normal(size=(3, 20))
    g = rng.integers(-3, 4, size=(20, 3))
    args = (np.eye(3, dtype=int), np.zeros(3), True, np.zeros(3, dtype=int))
    once, g1 = sym.unfold_state(c, g, np.array([0.1, 0.2, 0.3]), *args)
    twice, g2 = sym.unfold_state(once, g1, np.array([-0.1, -0.2, -0.3]), *args)
    assert np.array_equal(twice, c) and np.array_equal(g2, g)
    np.testing.assert_array_equal(once, c.conj())
    np.testing.assert_array_equal(g1, -g)


def test_the_dictionary_unfolds_each_spin_channel_and_keys_by_full_kpoint():
    s = sym.shared_symmetry(SHIFTED, SHIFTED)
    mesh = sym.monkhorst_pack((4, 4, 4), (1, 1, 1))
    reps, _ = sym.reduce_mesh(mesh, s.k_rotations)
    ibz = mesh[reps]
    kmap = sym.unfold_kpoints(ibz, s.k_rotations)
    rng = np.random.default_rng(2)
    d = {}
    for k in ibz:
        g = g_sphere(SHIFTED, k)
        up = rng.normal(size=(2, len(g))) + 1j * rng.normal(size=(2, len(g)))
        down = rng.normal(size=(2, len(g))) + 1j * rng.normal(size=(2, len(g)))
        d[tuple(np.around(k, 6))] = {"wfcn": [up, down], "gvecs": g}
    full = sym.unfold_wfcn_dict(d, ibz, kmap, s.k_rotations, s.translations["pol"])
    assert set(full) == {tuple(k) for k in kmap.full_kpoints} and len(full) == len(mesh)
    for j, kfull in enumerate(kmap.full_kpoints):
        i, o = kmap.ibz_index[j], kmap.op_index[j]
        src = d[tuple(np.around(ibz[i], 6))]
        for channel in (0, 1):
            expected, g = sym.unfold_state(src["wfcn"][channel], src["gvecs"], ibz[i],
                                           s.k_rotations[o], s.translations["pol"][o],
                                           kmap.time_reversal[j], kmap.gshift[j])
            np.testing.assert_array_equal(full[tuple(kfull)]["wfcn"][channel], expected)
            np.testing.assert_array_equal(full[tuple(kfull)]["gvecs"], g)


def test_a_non_spin_dictionary_stays_an_array():
    s = sym.shared_symmetry(SHIFTED, SHIFTED)
    mesh = sym.monkhorst_pack((2, 2, 2), (1, 1, 1))
    reps, _ = sym.reduce_mesh(mesh, s.k_rotations)
    kmap = sym.unfold_kpoints(mesh[reps], s.k_rotations)
    d = {tuple(np.around(k, 6)): {"wfcn": np.ones((1, 4), complex), "gvecs": np.zeros((4, 3), int)}
         for k in mesh[reps]}
    full = sym.unfold_wfcn_dict(d, mesh[reps], kmap, s.k_rotations, s.translations["pol"])
    assert all(isinstance(v["wfcn"], np.ndarray) for v in full.values())
