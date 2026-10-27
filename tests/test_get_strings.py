"""Unit tests for utils.get_strings, the grouping of k-points into strings.

get_strings decides which k-points the Berry flux is summed over, so a
mistake in it gives a plausible wrong polarization rather than an error.
The case that motivated the rewrite - transverse coordinates within the
tolerance of each other but not bit-equal - used to produce duplicate
strings that over-weighted one line of the mesh.
"""

from __future__ import annotations

import numpy as np
import pytest

from berry_flux_diag import utils

AXIS = {"x": 0, "y": 1, "z": 2}


def mesh(nx, ny, nz):
    """A regular Gamma-centred mesh, in the order a DFT code writes it."""
    return np.array([[i / nx, j / ny, k / nz]
                     for i in range(nx) for j in range(ny) for k in range(nz)])


def as_tuples(strings):
    return [[tuple(k) for k in string] for string in strings]


# --- the shape of the result on a regular mesh ---------------------------

@pytest.mark.parametrize("dims", [(4, 4, 4), (2, 3, 5), (1, 1, 4), (6, 6, 6)])
@pytest.mark.parametrize("direction", ["x", "y", "z"])
def test_one_string_per_transverse_point(dims, direction):
    axis = AXIS[direction]
    strings = utils.get_strings(mesh(*dims), direction)
    assert len(strings) == np.prod(dims) // dims[axis]
    assert {len(s) for s in strings} == {dims[axis] + 1}


@pytest.mark.parametrize("direction", ["x", "y", "z"])
def test_each_string_runs_along_its_axis_and_closes(direction):
    axis = AXIS[direction]
    gvec = np.eye(3)[axis]
    for string in utils.get_strings(mesh(3, 4, 5), direction):
        points = np.array(string)
        transverse = np.delete(points, axis, axis=1)
        assert np.all(transverse == transverse[0]), "a string left its line"
        assert np.all(np.diff(points[:-1, axis]) > 0), "not ordered along the axis"
        np.testing.assert_array_equal(points[-1], points[0] + gvec)


@pytest.mark.parametrize("direction", ["x", "y", "z"])
def test_every_kpoint_is_in_exactly_one_string(direction):
    kpts = mesh(3, 4, 5)
    placed = [k for s in utils.get_strings(kpts, direction) for k in as_tuples([s[:-1]])[0]]
    assert sorted(placed) == sorted(map(tuple, kpts))


def test_strings_are_ordered_by_transverse_coordinates():
    strings = utils.get_strings(mesh(2, 3, 4), "z")
    heads = [tuple(s[0][:2]) for s in strings]
    assert heads == sorted(heads)


def test_input_order_does_not_matter():
    kpts = mesh(4, 3, 5)
    shuffled = kpts[np.random.default_rng(0).permutation(len(kpts))]
    for direction in "xyz":
        assert as_tuples(utils.get_strings(shuffled, direction)) == \
            as_tuples(utils.get_strings(kpts, direction))


def test_a_list_of_arrays_works_like_an_array():
    """The parsers pass a list of rounded k-point arrays."""
    kpts = mesh(3, 3, 3)
    assert as_tuples(utils.get_strings(list(kpts), "y")) == \
        as_tuples(utils.get_strings(kpts, "y"))


# --- the tolerance -------------------------------------------------------

def test_near_equal_transverse_coordinates_form_one_string():
    """The bug the rewrite fixes.

    The old code built its set of lines with exact equality and filled them
    with a 1e-5 tolerance, so these four points became two identical
    strings of five, each holding all four.
    """
    line = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.25],
                     [0.0, 0.0, 0.5], [1e-7, 0.0, 0.75]])
    strings = utils.get_strings(line, "z")
    assert len(strings) == 1
    assert len(strings[0]) == 5


def test_transverse_noise_below_the_tolerance_is_absorbed():
    kpts = mesh(4, 4, 4)
    noisy = kpts + np.random.default_rng(1).uniform(-1e-7, 1e-7, kpts.shape)
    for direction in "xyz":
        clean = utils.get_strings(kpts, direction)
        assert [len(s) for s in utils.get_strings(noisy, direction)] == \
            [len(s) for s in clean]


def test_offsets_beyond_the_tolerance_are_separate_lines():
    line = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5],
                     [0.0, 1e-3, 0.0], [0.0, 1e-3, 0.5]])
    strings = utils.get_strings(line, "z")
    assert len(strings) == 2
    assert {len(s) for s in strings} == {3}


def test_the_tolerance_is_a_parameter():
    line = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5],
                     [0.0, 1e-3, 0.0], [0.0, 1e-3, 0.5]])
    assert len(utils.get_strings(line, "z", tol=1e-2)) == 1


# --- errors --------------------------------------------------------------

def test_a_missing_kpoint_raises():
    with pytest.raises(ValueError, match="differing lengths"):
        utils.get_strings(mesh(4, 4, 4)[:-1], "z")


def test_the_error_names_the_direction():
    with pytest.raises(ValueError, match="along y"):
        utils.get_strings(mesh(4, 4, 4)[:-1], "y")


@pytest.mark.parametrize("bad", [np.zeros((4, 2)), np.zeros(3), np.zeros((2, 3, 3))])
def test_a_malformed_kpoint_array_raises(bad):
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        utils.get_strings(bad, "z")


def test_an_unknown_direction_raises():
    with pytest.raises(ValueError, match="Direction"):
        utils.get_strings(mesh(2, 2, 2), "w")


def test_no_kpoints_gives_no_strings():
    """compute_string_sums turns this into an error naming the direction."""
    assert utils.get_strings([], "z") == []


# --- direction_to_vals ---------------------------------------------------

@pytest.mark.parametrize("direction", ["x", "y", "z"])
def test_the_closing_vector_is_an_array(direction):
    """It is added to k-point arrays, so a list would only work by accident."""
    comps, dir_comp, gvec = utils.direction_to_vals(direction)
    assert isinstance(gvec, np.ndarray)
    np.testing.assert_array_equal(gvec, np.eye(3)[dir_comp])
    assert sorted(comps + (dir_comp,)) == [0, 1, 2]
