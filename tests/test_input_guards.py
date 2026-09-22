"""Unit tests for the cross-run consistency guards.

The README states three constraints on the two DFT runs - same k-point
mesh, same atoms in the same order, wavefunctions over the whole
Brillouin zone - and until now nothing enforced them. Each failure below
used to produce a plausible number instead of an error.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from berry_flux_diag import utils


def mesh(nx, ny, nz):
    """A regular Gamma-centred mesh, in the order a DFT code writes it."""
    return np.array([[i / nx, j / ny, k / nz]
                     for i in range(nx) for j in range(ny) for k in range(nz)])


def structure(species):
    coords = [[0.1 * n, 0.0, 0.0] for n in range(len(species))]
    return Structure(Lattice.cubic(6.0), species, coords)


# --- k-point agreement between the two runs ------------------------------

def test_identical_meshes_pass():
    grid = mesh(4, 4, 4)
    utils.check_kpoints_match(grid, grid.copy())


def test_different_kpoint_counts_raise():
    grid = mesh(4, 4, 4)
    with pytest.raises(ValueError, match="different numbers of k-points"):
        utils.check_kpoints_match(grid, grid[:-1])


def test_same_kpoints_in_a_different_order_raise():
    """Order matters, which is the whole reason for this check.

    Both wavefunction dictionaries are built from the polar k-point list
    while each run's coefficients are read positionally, so a reordered
    non-polar run would have its coefficients labelled with the wrong
    k-points - and would produce a number, not an error.
    """
    grid = mesh(4, 4, 4)
    with pytest.raises(ValueError, match="different k-points"):
        utils.check_kpoints_match(grid, grid[::-1])


def test_small_numerical_differences_are_tolerated():
    """The two codes round k-points independently."""
    grid = mesh(4, 4, 4)
    utils.check_kpoints_match(grid, grid + 1e-9)


def test_a_shifted_mesh_raises():
    grid = mesh(4, 4, 4)
    with pytest.raises(ValueError, match="different k-points"):
        utils.check_kpoints_match(grid, grid + 0.01)


# --- species agreement ---------------------------------------------------

def test_matching_species_pass():
    utils.check_species_match(structure(["Ba", "Ti", "O"]),
                              structure(["Ba", "Ti", "O"]))


def test_reordered_species_raise():
    """Sites are paired by index, so a permutation pairs the wrong atoms."""
    with pytest.raises(ValueError, match="site 1"):
        utils.check_species_match(structure(["Ba", "Ti", "O"]),
                                  structure(["Ba", "O", "Ti"]))


def test_different_atom_counts_raise():
    with pytest.raises(ValueError, match="different numbers of atoms"):
        utils.check_species_match(structure(["Ba", "Ti", "O"]),
                                  structure(["Ba", "Ti"]))


def test_a_substituted_element_raises():
    with pytest.raises(ValueError, match="site 1"):
        utils.check_species_match(structure(["Ba", "Ti", "O"]),
                                  structure(["Ba", "Zr", "O"]))


# --- full Brillouin zone coverage ---------------------------------------

@pytest.mark.parametrize("dims", [(2, 2, 2), (4, 4, 4), (6, 6, 6), (2, 3, 4), (1, 1, 4)])
def test_complete_meshes_pass(dims):
    utils.check_full_bz(mesh(*dims))


def test_irreducible_wedge_raises():
    """The case this guard exists for.

    An i <= j <= k wedge of the 6x6x6 mesh is 56 k-points. get_strings
    would build strings that do not close, and the polarization would be
    meaningless rather than approximate.
    """
    wedge = np.array([[i / 6, j / 6, k / 6]
                      for i in range(6) for j in range(i, 6) for k in range(j, 6)])
    assert len(wedge) == 56
    with pytest.raises(ValueError, match="do not span the full Brillouin zone"):
        utils.check_full_bz(wedge)


def test_the_error_names_the_mesh_it_expected():
    """The message has to be actionable: it names the mesh and the fix."""
    wedge = mesh(4, 4, 4)[:-1]
    with pytest.raises(ValueError, match=r"4x4x4 mesh of 64"):
        utils.check_full_bz(wedge)
    with pytest.raises(ValueError, match="ISYM"):
        utils.check_full_bz(wedge)


def test_one_missing_kpoint_raises():
    with pytest.raises(ValueError, match="do not span the full Brillouin zone"):
        utils.check_full_bz(mesh(4, 4, 4)[:-1])


def test_empty_kpoint_list_raises():
    with pytest.raises(ValueError, match="empty"):
        utils.check_full_bz([])


def test_a_single_gamma_point_passes():
    """Degenerate but complete: a 1x1x1 mesh is the whole zone."""
    utils.check_full_bz(np.array([[0.0, 0.0, 0.0]]))
