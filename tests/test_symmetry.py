"""Unit tests for berry_flux_diag.symmetry: the operations the polar and
non-polar structures share, and unfolding an IBZ k-point list to the full
mesh. The BaTiO3 structures are those of the VASP and QE test runs."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from berry_flux_diag import symmetry as sym

SPECIES = ["Ba", "Ti", "O", "O", "O"]
VASP_POL = Structure(Lattice.tetragonal(4.002006, 4.215507), SPECIES,
                     [[0, 0, 0.020229], [0.5, 0.5, 0.538921], [0, 0.5, 0.492023],
                      [0.5, 0, 0.492023], [0.5, 0.5, 0.970803]])
VASP_NP = Structure(Lattice.cubic(4.035583), SPECIES,
                    [[0, 0, 0.0028], [0.5, 0.5, 0.5028], [0, 0.5, 0.5028],
                     [0.5, 0, 0.5028], [0.5, 0.5, 0.0028]])
QE_LATTICE = Lattice.tetragonal(4.001192, 4.215097)
QE_POL = Structure(QE_LATTICE, SPECIES,
                   [[0, 0, 0.020310], [0.5, 0.5, 0.538678], [0, 0.5, 0.491942],
                    [0.5, 0, 0.491942], [0.5, 0.5, 0.971127]])
QE_NP = Structure(QE_LATTICE, SPECIES,
                  [[1, 1, 1], [0.499999, 0.499999, 0.499999], [1, 0.499999, 0.499999],
                   [0.499999, 1, 0.499999], [0.499999, 0.499999, 1]])


def maps_atoms_onto_atoms(structure, rotation, translation, magmoms=None):
    """{W|t} sends every atom onto an atom of the same species (and moment)."""
    x = structure.frac_coords
    labels = [(s.specie.symbol, None if magmoms is None else magmoms[i])
              for i, s in enumerate(structure)]
    for i, xi in enumerate(x):
        image = rotation @ xi + translation
        d = (x - image + 0.5) % 1 - 0.5
        hits = [j for j in np.where(np.all(np.abs(d) < 1e-3, axis=1))[0] if labels[j] == labels[i]]
        if not hits:
            return False
    return True


# --- the shared operations -------------------------------------------------

@pytest.mark.parametrize("pol, np_, groups", [
    (VASP_POL, VASP_NP, {"pol": "P4mm (99)", "np": "Pm-3m (221)"}),
    (QE_POL, QE_NP, {"pol": "P4mm (99)", "np": "P4/mmm (123)"}),
])
def test_batio3_shares_the_eight_operations_of_p4mm(pol, np_, groups):
    s = sym.shared_symmetry(pol, np_)
    assert s.space_groups == groups
    assert len(s) == 8
    assert any(np.array_equal(r, np.eye(3, dtype=int)) for r in s.rotations)
    for label, struct in (("pol", pol), ("np", np_)):
        for r, t in zip(s.rotations, s.translations[label]):
            assert maps_atoms_onto_atoms(struct, r, t)


def test_k_rotations_are_the_inverse_transposes():
    s = sym.shared_symmetry(VASP_POL, VASP_NP)
    for r, kr in zip(s.rotations, s.k_rotations):
        np.testing.assert_array_equal(kr.T @ r, np.eye(3, dtype=int))


def test_each_structure_keeps_its_own_translations():
    """The same structure shifted off the origin has the same rotations, but
    they need nonzero translations, t = (1 - W) c."""
    shift = np.array([0.11, 0.23, 0.07])
    shifted = Structure(VASP_POL.lattice, SPECIES, VASP_POL.frac_coords + shift)
    s = sym.shared_symmetry(VASP_POL, shifted)
    assert len(s) == 8
    assert np.abs(s.translations["np"]).max() > 0.05
    for r, t_pol, t_np in zip(s.rotations, s.translations["pol"], s.translations["np"]):
        assert maps_atoms_onto_atoms(shifted, r, t_np)
        assert not maps_atoms_onto_atoms(shifted, r, t_pol) or np.allclose(r, np.eye(3))


def test_only_moment_preserving_operations_are_kept():
    """Two Fe atoms at x = 0.2 and 0.8: unevenly spaced, so every x -> -x
    mirror exchanges them. With equal moments that is a symmetry; with
    opposite moments it would swap the spin channels, so it is dropped."""
    fe2 = Structure(Lattice.tetragonal(4.0, 3.0), ["Fe", "Fe"], [[0.2, 0, 0], [0.8, 0, 0]])
    mirror = np.diag([-1, 1, 1])
    ferro, _, _ = sym.symmetry_operations(fe2, magmoms=[2.0, 2.0])
    anti, _, _ = sym.symmetry_operations(fe2, magmoms=[2.0, -2.0])
    assert any(np.array_equal(r, mirror) for r in ferro)
    assert not any(np.array_equal(r, mirror) for r in anti)
    s = sym.shared_symmetry(fe2, fe2, [2.0, -2.0], [2.0, -2.0])
    for r, t in zip(s.rotations, s.translations["pol"]):
        assert maps_atoms_onto_atoms(fe2, r, t, magmoms=[2.0, -2.0])


def test_zero_moments_change_nothing():
    plain, _, _ = sym.symmetry_operations(VASP_POL)
    zero, _, _ = sym.symmetry_operations(VASP_POL, magmoms=[0.0] * 5)
    assert len(plain) == len(zero) == 8


def test_moments_must_be_one_per_atom():
    with pytest.raises(ValueError, match="one per atom"):
        sym.symmetry_operations(VASP_POL, magmoms=[1.0, 1.0])


# --- unfolding and reducing ----------------------------------------------------

@pytest.mark.parametrize("pol, np_, n, shift", [
    (VASP_POL, VASP_NP, (6, 6, 6), (1, 1, 1)),     # VASP's 6x6x6 Monkhorst
    (QE_POL, QE_NP, (5, 5, 5), (1, 1, 1)),         # QE's 5 5 5 1 1 1
    (VASP_POL, VASP_NP, (4, 4, 6), (0, 0, 0)),     # Gamma-centred, uneven
])
def test_reduce_then_unfold_recovers_the_mesh(pol, np_, n, shift):
    s = sym.shared_symmetry(pol, np_)
    mesh = sym.monkhorst_pack(n, shift)
    reps, mult = sym.reduce_mesh(mesh, s.k_rotations)
    assert mult.sum() == len(mesh) and len(reps) < len(mesh)
    m = sym.unfold_kpoints(mesh[reps], s.k_rotations)
    assert {tuple(k) for k in m.full_kpoints} == {tuple(k) for k in sym.wrap(mesh)}
    np.testing.assert_array_equal(m.multiplicity, mult)
    for k, i, o, tr, g in zip(m.full_kpoints, m.ibz_index, m.op_index, m.time_reversal, m.gshift):
        image = (-1 if tr else 1) * s.k_rotations[o] @ mesh[reps][i]
        np.testing.assert_allclose(image - g, k, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(g, np.rint(g))


def test_the_batio3_ibz_has_eighteen_points():
    """3 values of |k_z| times 6 unordered pairs (|k_x|, |k_y|)."""
    s = sym.shared_symmetry(VASP_POL, VASP_NP)
    assert len(sym.reduce_mesh(sym.monkhorst_pack((6, 6, 6), (1, 1, 1)), s.k_rotations)[0]) == 18


def test_ibz_points_written_out_of_range_still_unfold():
    """A code may write k = 0.5 or 0.9 for points [-0.5, 0.5) holds as -0.5
    and -0.1: the shift lands in gshift, the k-points do not change."""
    s = sym.shared_symmetry(QE_POL, QE_NP)
    mesh = sym.monkhorst_pack((5, 5, 5), (1, 1, 1))
    reps, _ = sym.reduce_mesh(mesh, s.k_rotations)
    shifted = mesh[reps] + np.array([1, 0, -1])
    a, b = sym.unfold_kpoints(mesh[reps], s.k_rotations), sym.unfold_kpoints(shifted, s.k_rotations)
    np.testing.assert_array_equal(a.full_kpoints, b.full_kpoints)
    assert not np.array_equal(a.gshift, b.gshift)


def test_without_time_reversal_the_ibz_is_larger():
    """P4mm has no inversion, so k and -k are only equivalent by time reversal."""
    s = sym.shared_symmetry(VASP_POL, VASP_NP)
    mesh = sym.monkhorst_pack((6, 6, 6), (1, 1, 1))
    with_tr, _ = sym.reduce_mesh(mesh, s.k_rotations)
    without, mult = sym.reduce_mesh(mesh, s.k_rotations, time_reversal=False)
    assert len(without) > len(with_tr)
    m = sym.unfold_kpoints(mesh[without], s.k_rotations, time_reversal=False)
    assert len(m.full_kpoints) == len(mesh) and not m.time_reversal.any()


def test_equivalent_ibz_points_are_refused():
    s = sym.shared_symmetry(VASP_POL, VASP_NP)
    mesh = sym.monkhorst_pack((6, 6, 6), (1, 1, 1))
    reps, _ = sym.reduce_mesh(mesh, s.k_rotations)
    ibz = np.vstack([mesh[reps], -mesh[reps][3]])          # -k of an IBZ point
    with pytest.raises(ValueError, match="are equivalent under the shared symmetry"):
        sym.unfold_kpoints(ibz, s.k_rotations)


def test_an_incomplete_ibz_is_refused():
    s = sym.shared_symmetry(VASP_POL, VASP_NP)
    mesh = sym.monkhorst_pack((6, 6, 6), (1, 1, 1))
    reps, _ = sym.reduce_mesh(mesh, s.k_rotations)
    with pytest.raises(ValueError, match="do not form a full mesh"):
        sym.unfold_kpoints(mesh[reps[1:]], s.k_rotations)


def test_an_ibz_reduced_with_more_symmetry_is_refused():
    """The non-polar structure's own IBZ (Pm-3m) is smaller than the shared
    one, and does not unfold under the shared rotations alone."""
    shared = sym.shared_symmetry(VASP_POL, VASP_NP)
    own = sym.shared_symmetry(VASP_NP, VASP_NP)
    mesh = sym.monkhorst_pack((6, 6, 6), (1, 1, 1))
    reps, _ = sym.reduce_mesh(mesh, own.k_rotations)
    with pytest.raises(ValueError, match="do not form a full mesh"):
        sym.unfold_kpoints(mesh[reps], shared.k_rotations)


def test_a_mesh_a_rotation_does_not_preserve_is_refused():
    """Cubic rotations exchange x and z, which a 6x6x4 mesh cannot follow."""
    cubic = sym.shared_symmetry(VASP_NP, VASP_NP)
    with pytest.raises(ValueError, match="off the mesh"):
        sym.reduce_mesh(sym.monkhorst_pack((6, 6, 4), (1, 1, 1)), cubic.k_rotations)


# --- small pieces ----------------------------------------------------------

def test_wrap_lands_in_minus_half_to_half():
    got = sym.wrap([[0.5, -0.5, 0.4999999999999], [1.25, -0.75, -0.0]])
    np.testing.assert_array_equal(got, [[-0.5, -0.5, -0.5], [0.25, 0.25, 0.0]])
    assert not np.signbit(got).any() or (got[np.signbit(got)] != 0).all()


def test_monkhorst_pack_matches_vasp_and_qe():
    vasp = sym.monkhorst_pack((6, 6, 6), (1, 1, 1))
    assert sorted(set(np.round(vasp[:, 0], 6))) == sorted(np.round([-5/12, -3/12, -1/12, 1/12, 3/12, 5/12], 6))
    qe = sym.monkhorst_pack((5, 5, 5), (1, 1, 1))
    assert sorted(set(np.round(qe[:, 2], 6))) == [-0.5, -0.3, -0.1, 0.1, 0.3]


def test_input_writers():
    k, w = np.array([[0.1, -0.3, 0.5], [0.0, 0.0, 0.0]]), [8, 1]
    vasp = sym.vasp_kpoints_text(k, w).splitlines()
    assert vasp[1:3] == ["2", "Reciprocal"] and vasp[3].split() == ["0.100000000000", "-0.300000000000", "0.500000000000", "8"]
    qe = sym.qe_kpoints_text(k, w).splitlines()
    assert qe[0] == "K_POINTS crystal" and qe[1].strip() == "2" and qe[2].split()[-1] == "8"


def test_k_rotations_preserve_cartesian_lengths_in_a_hexagonal_cell():
    """In a hexagonal basis W^-T differs from W, and only W^-T is a rotation
    of fractional k: it keeps |k| in Cartesian coordinates."""
    hexa = Structure(Lattice.hexagonal(3.0, 5.0), ["Zn", "Zn"],
                     [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5]])
    s = sym.shared_symmetry(hexa, hexa)
    assert any(not np.array_equal(r, kr) for r, kr in zip(s.rotations, s.k_rotations))
    recip = hexa.lattice.reciprocal_lattice.matrix
    k = np.random.default_rng(0).uniform(-0.5, 0.5, size=(10, 3))
    for kr in s.k_rotations:
        np.testing.assert_allclose(np.linalg.norm((k @ kr.T) @ recip, axis=1),
                                   np.linalg.norm(k @ recip, axis=1), rtol=1e-12)
    mesh = sym.monkhorst_pack((6, 6, 4))
    reps, mult = sym.reduce_mesh(mesh, s.k_rotations)
    assert len(sym.unfold_kpoints(mesh[reps], s.k_rotations).full_kpoints) == len(mesh) == mult.sum()
