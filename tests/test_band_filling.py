"""Unit tests for the occupied-band count.

The count used to be read from the occupations: a band counted as filled
where its occupation was above 1e-6, maximized over k-points. With
smearing, the lowest empty band of a small-gap insulator has a small
occupation near the band edges, so that count included a conduction band
at some k-points and not others, and which one it took depended on the
threshold and on the k-point order. An included band is not weighted by its
occupation: it enters the Berry flux as fully occupied.

The count is now the electron count - NELECT / 2, or a spin channel's
share - and the occupations and band energies only check it. The BaTiO3
fixtures cannot cover the cases that matter, being wide-gap insulators
with integer occupations, so these build the arrays directly.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from berry_flux_diag import QEParser, VASPParser, utils

NK, NB = 8, 24


def insulator(n_filled, n_bands=NB, nk=NK, gap=2.0, seed=0):
    """Energies and occupations of an insulator: n_filled bands below a gap."""
    rng = np.random.default_rng(seed)
    energies = np.sort(rng.uniform(-10, 0, size=(nk, n_bands)), axis=1)
    energies[:, n_filled:] += gap + 10
    occupations = np.zeros((nk, n_bands))
    occupations[:, :n_filled] = 1.0
    return energies, occupations


def smeared(n_filled, tail, at_kpoints, **kwargs):
    """As insulator, with smearing moving tail electrons from the top valence
    band into the lowest conduction band at the given k-points."""
    energies, occupations = insulator(n_filled, **kwargs)
    occupations[at_kpoints, n_filled - 1] -= tail
    occupations[at_kpoints, n_filled] += tail
    return energies, occupations


# --- the count is the electron count -------------------------------------

def test_an_insulator_fills_half_nelect_bands():
    assert utils.occupied_band_count(20, *insulator(20)) == 20


@pytest.mark.parametrize("tail", [1e-5, 1e-3, 0.05])
def test_a_smeared_conduction_band_is_not_counted(tail):
    """The case the occupation threshold got wrong: band 21 is above 1e-6 at
    two k-points. The old count was 21 or 20 depending on whether one of
    those came first; the occupied manifold is 20 bands either way."""
    energies, occupations = smeared(20, tail, at_kpoints=[3, 5])
    assert (occupations[:, 20] > 1e-6).sum() == 2
    assert utils.occupied_band_count(20, energies, occupations) == 20


def test_kpoint_order_does_not_matter():
    energies, occupations = smeared(20, 1e-4, at_kpoints=[0])
    order = np.random.default_rng(1).permutation(NK)
    assert utils.occupied_band_count(20, energies, occupations) == \
        utils.occupied_band_count(20, energies[order], occupations[order]) == 20


def test_every_band_occupied_is_fine():
    """QE's default for an insulator computes no empty bands at all, so there
    is no gap to check, and fixed occupations assume one."""
    assert utils.occupied_band_count(20, *insulator(20, n_bands=20)) == 20


# --- what it refuses ------------------------------------------------------

def test_an_odd_electron_count_is_metallic():
    energies, occupations = insulator(20)
    occupations[:, 20] = 0.5
    with pytest.raises(ValueError, match="part filled.*even NELECT"):
        utils.occupied_band_count(20.5, energies, occupations, "polar")


def test_overlapping_bands_are_metallic():
    """Band 21 dips below the top of band 20 at one k-point: no gap."""
    energies, occupations = insulator(20)
    energies[4, 20] = energies[:, 19].max() - 0.1
    with pytest.raises(ValueError, match="band 21 dips below band 20.*polar run"):
        utils.occupied_band_count(20, energies, occupations, "polar")


def test_touching_bands_are_metallic():
    energies, occupations = insulator(20)
    energies[4, 20] = energies[:, 19].max()
    with pytest.raises(ValueError, match="not separated"):
        utils.occupied_band_count(20, energies, occupations)


def test_an_electron_count_from_another_run_is_caught():
    with pytest.raises(ValueError, match="sum to 20.000000.*not the 21"):
        utils.occupied_band_count(21, *insulator(20))


def test_occupations_on_a_zero_to_two_scale_are_caught():
    energies, occupations = insulator(20)
    with pytest.raises(ValueError, match="do not describe the same run"):
        utils.occupied_band_count(20, energies, 2 * occupations)


def test_more_electrons_than_bands_is_caught():
    energies, occupations = insulator(20, n_bands=20)
    with pytest.raises(ValueError, match="20 occupied bands expected, but only 19"):
        utils.occupied_band_count(20, energies[:, :19], occupations[:, :19] * 20 / 19)


def test_mismatched_shapes_are_caught():
    energies, occupations = insulator(20)
    with pytest.raises(ValueError, match="must both be"):
        utils.occupied_band_count(20, energies, occupations[:, :-1])


# --- the smearing warning ------------------------------------------------

def test_large_smearing_tails_warn_but_do_not_change_the_count(caplog):
    with caplog.at_level(logging.WARNING, logger="berry_flux_diag.utils"):
        assert utils.occupied_band_count(
            20, *smeared(20, 0.01, at_kpoints=[2]), label="non-polar") == 20
    assert "0.01 electrons per k-point above the gap in the non-polar run" in caplog.text


def test_negligible_tails_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="berry_flux_diag.utils"):
        utils.occupied_band_count(20, *smeared(20, 1e-5, at_kpoints=[2]))
    assert caplog.text == ""


# --- spin channels --------------------------------------------------------

def test_spin_channels_are_split_by_their_occupations():
    _, up = insulator(21)
    _, down = insulator(19)
    assert utils.spin_channel_electrons(40, up, down) == (21, 19)


def test_spin_channels_tolerate_smearing_tails():
    _, up = smeared(21, 1e-3, at_kpoints=[1])
    _, down = insulator(19)
    assert utils.spin_channel_electrons(40, up, down) == (21, 19)


def test_a_part_filled_spin_channel_is_metallic():
    _, up = insulator(21)
    up[:, 21] = 0.5
    _, down = insulator(19)
    with pytest.raises(ValueError, match="spin-up occupations in the polar run sum to 21.5"):
        utils.spin_channel_electrons(40.5, up, down, "polar")


def test_spin_channels_must_add_up_to_nelect():
    _, up = insulator(21)
    _, down = insulator(19)
    with pytest.raises(ValueError, match="21 \\+ 19 electrons, not the NELECT of 42"):
        utils.spin_channel_electrons(42, up, down)


# --- where the electron count comes from ----------------------------------

def test_nelect_is_read_from_the_parameters_block(tmp_path):
    f = tmp_path / "vasprun.xml"
    f.write_text('<modeling>\n <parameters>\n   <i type="int" name="NBANDS">    32</i>\n'
                 '   <i name="NELECT">     40.00000000</i>\n </parameters>\n</modeling>\n')
    assert VASPParser.read_nelect(f) == 40.0


def test_a_hand_set_nelect_is_read_too(tmp_path):
    """A charged cell: NELECT set in the INCAR appears first, typed."""
    f = tmp_path / "vasprun.xml"
    f.write_text(' <incar>\n  <i type="string" name="NELECT">41</i>\n </incar>\n'
                 '   <i name="NELECT">     41.00000000</i>\n')
    assert VASPParser.read_nelect(f) == 41.0


def test_a_vasprun_without_nelect_is_an_error(tmp_path):
    f = tmp_path / "vasprun.xml"
    f.write_text("<modeling>\n</modeling>\n")
    with pytest.raises(ValueError, match="no NELECT"):
        VASPParser.read_nelect(f)


def qe_dict(energies_by_k, occupations_by_k, nelec, **band_structure):
    ks = [{"eigenvalues": {"$": list(e)}, "occupations": {"$": list(o)}}
          for e, o in zip(energies_by_k, occupations_by_k)]
    return {"qes:espresso": {"output": {"band_structure": {
        "nelec": nelec, "ks_energies": ks, **band_structure}}}}


@pytest.mark.skipif(QEParser is None, reason="QEParser needs qeschema and h5py")
def test_qe_reads_nelec():
    energies, occupations = smeared(20, 1e-4, at_kpoints=[0])
    xml = qe_dict(energies, occupations, 40.0)
    assert QEParser.get_band_filling_from_qeschema_nospin(xml) == 20


@pytest.mark.skipif(QEParser is None, reason="QEParser needs qeschema and h5py")
def test_qe_splits_its_spin_channels_end_to_end():
    """QE writes up then down in one array per k-point."""
    e_up, o_up = insulator(21, n_bands=24)
    e_dw, o_dw = insulator(19, n_bands=24, seed=1)
    xml = qe_dict(np.hstack([e_up, e_dw]), np.hstack([o_up, o_dw]), 40.0,
                  nbnd_up=24, nbnd_dw=24)
    assert QEParser.get_band_filling_from_qeschema_spinpol(xml) == (21, 19)
