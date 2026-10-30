"""Unit tests for the overlap cache inside Overlaps.compute_string_sums.

Each plaquette needs four overlap matrices, and many recur: the polar /
non-polar edge at a k-point appears in the x, y and z strings. The cache
computes each distinct one once. It must be invisible in the results -
bit for bit, not to a tolerance - and must not outlive the call.

These use a small synthetic mesh with random coefficients and a different
G set at every k-point, so that the wrapped string ends and the remapping
in map_coeffs are both exercised.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from berry_flux_diag import Overlaps, utils

N = 3        # 3x3x3 mesh
NBANDS = 3


def sphere(radius):
    r = np.arange(-radius, radius + 1)
    g = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    return g[np.sum(g ** 2, axis=1) <= radius ** 2]


def random_wfcn_dict(rng, kpoints, spin_pol=False):
    full = sphere(3)
    out = {}
    for k in kpoints:
        g = full[np.sort(rng.permutation(len(full))[: len(full) - 10])]
        shape = (2, NBANDS, len(g)) if spin_pol else (NBANDS, len(g))
        c = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        out[tuple(k)] = {"wfcn": c if not spin_pol else list(c), "gvecs": g}
    return out


def make_overlaps(seed=0, spin_pol=False):
    rng = np.random.default_rng(seed)
    kpoints = [np.around(np.array(k) / N, 6) for k in itertools.product(range(N), repeat=3)]
    struct = Structure(Lattice.cubic(4.0), ["Ba"], [[0, 0, 0]])
    parse = {
        "pol_struct": struct, "np_struct": struct,
        "pol_wfcn_dict": random_wfcn_dict(rng, kpoints, spin_pol),
        "np_wfcn_dict": random_wfcn_dict(rng, kpoints, spin_pol),
        "kpoint_list": kpoints, "zval_dict": {"Ba": 10.0},
        "ES_code": "test", "spin_pol": spin_pol,
    }
    if spin_pol:
        parse.update(pol_band_fill_up=NBANDS, np_band_fill_up=NBANDS,
                     pol_band_fill_down=NBANDS, np_band_fill_down=NBANDS)
    else:
        parse.update(pol_band_fill=NBANDS, np_band_fill=NBANDS)
    # Random states overlap badly; silence the conditioning warnings.
    return Overlaps.Overlaps(parse, eig_thresh=10.0, sing_val_thresh=0.0)


def count_computations(ov):
    """Count overlaps actually computed: each fetches two states' coefficients."""
    calls = {"n": 0}
    fetch = ov.get_pw_coeffs_from_state

    def counting(l, kpt):
        calls["n"] += 1
        return fetch(l, kpt)

    ov.get_pw_coeffs_from_state = counting
    return calls


# --- invisible in the results -------------------------------------------

def test_results_are_bit_identical_with_and_without_the_cache():
    cached = make_overlaps().compute_string_sums()
    uncached_ov = make_overlaps()
    assert uncached_ov._overlap_cache is None
    uncached = uncached_ov._compute_string_sums()    # the body, with no cache set
    assert cached[0] == uncached[0]                  # string sums, exactly
    assert cached[1] == uncached[1]
    for d in "xyz":
        for a, b in zip(cached[2][d]["eigs"], uncached[2][d]["eigs"]):
            np.testing.assert_array_equal(a, b)


def test_the_wrapped_ends_and_the_remapping_are_both_exercised():
    """Otherwise the test above would not cover the shifted cache keys."""
    ov = make_overlaps()
    shifted, mapped = [], []
    orig_map = utils.map_coeffs

    def spy(*args):
        mapped.append(True)
        return orig_map(*args)

    utils.map_coeffs, orig = spy, utils.map_coeffs
    try:
        ov.compute_string_sums()
    finally:
        utils.map_coeffs = orig
    assert mapped, "map_coeffs never ran"
    for direction in "xyz":
        for string in utils.get_strings(ov.kpoint_list, direction):
            shifted.append(not any(np.allclose(string[-1], k) for k in ov.kpoint_list))
    assert all(shifted), "every string should end on a wrapped k-point"


# --- it saves what it should --------------------------------------------

def test_each_distinct_overlap_is_computed_once():
    ov = make_overlaps()
    calls = count_computations(ov)
    ov.compute_string_sums()

    edges = set()
    for direction in "xyz":
        for string in utils.get_strings(ov.kpoint_list, direction):
            for k0, k1 in zip(string[:-1], string[1:]):
                path = [(0, k0), (1, k0), (1, k1), (0, k1), (0, k0)]
                for (l0, a), (l1, b) in zip(path[:-1], path[1:]):
                    edges.add((l0, tuple(np.around(a, 6)), l1, tuple(np.around(b, 6))))
    total = 3 * N ** 3 * 4
    assert calls["n"] == 2 * len(edges)
    assert len(edges) < total


def test_without_the_cache_every_edge_is_computed():
    ov = make_overlaps()
    calls = count_computations(ov)
    ov._compute_string_sums()
    assert calls["n"] == 2 * 3 * N ** 3 * 4


# --- it does not outlive the call ---------------------------------------

def test_the_cache_is_dropped_after_the_call():
    ov = make_overlaps()
    ov.compute_string_sums()
    assert ov._overlap_cache is None


def test_the_cache_is_dropped_after_an_error():
    ov = make_overlaps()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    ov.check_wilson_loop_phases = boom
    with pytest.raises(RuntimeError):
        ov.compute_string_sums()
    assert ov._overlap_cache is None


def test_a_second_call_starts_empty_and_agrees():
    ov = make_overlaps()
    first = ov.compute_string_sums()
    calls = count_computations(ov)
    second = ov.compute_string_sums()
    assert first[0] == second[0]
    assert calls["n"] > 0, "second call reused matrices from the first"


def test_direct_compute_overlap_calls_are_not_cached():
    ov = make_overlaps()
    k = ov.kpoint_list[0]
    a = ov.compute_overlap(0, k, 1, k, "z")
    b = ov.compute_overlap(0, k, 1, k, "z")
    assert a is not b
    assert a.flags.writeable
    assert ov._overlap_cache is None


def test_cached_matrices_cannot_be_modified():
    ov = make_overlaps()
    ov._overlap_cache = {}
    k = ov.kpoint_list[0]
    m = ov.compute_overlap(0, k, 1, k, "z")
    assert ov.compute_overlap(0, k, 1, k, "z") is m
    with pytest.raises(ValueError):
        m[0, 0] = 0


# --- the key -------------------------------------------------------------

def test_spin_channels_do_not_share_entries():
    ov = make_overlaps(spin_pol=True)
    ov._overlap_cache = {}
    k = ov.kpoint_list[0]
    ov.spin_state = 0
    up = ov.compute_overlap(0, k, 1, k, "z")
    ov.spin_state = 1
    down = ov.compute_overlap(0, k, 1, k, "z")
    assert not np.array_equal(up, down)
    assert len(ov._overlap_cache) == 2


def test_a_wrapped_kpoint_and_its_image_are_different_entries():
    """k + G and k fetch the same coefficients, but the G list is shifted
    for k + G, so the overlap with an unshifted neighbour differs."""
    ov = make_overlaps()
    ov._overlap_cache = {}
    k, gz = ov.kpoint_list[0], np.array([0, 0, 1])
    neighbour = np.around(k + np.array([0, 0, 2 / N]), 6)
    direct = ov.compute_overlap(0, neighbour, 0, k, "z")
    wrapped = ov.compute_overlap(0, neighbour, 0, k + gz, "z")
    assert len(ov._overlap_cache) == 2
    assert not np.array_equal(direct, wrapped)


def test_a_wrapped_kpoint_on_the_left_is_a_different_entry_too():
    """The same, with the wrapped k-point as the bra. A real mesh never
    asks for both of these, but the key must not depend on that."""
    ov = make_overlaps()
    ov._overlap_cache = {}
    k, gz = ov.kpoint_list[0], np.array([0, 0, 1])
    neighbour = np.around(k + np.array([0, 0, 2 / N]), 6)
    direct = ov.compute_overlap(0, k, 0, neighbour, "z")
    wrapped = ov.compute_overlap(0, k + gz, 0, neighbour, "z")
    assert len(ov._overlap_cache) == 2
    assert not np.array_equal(direct, wrapped)
