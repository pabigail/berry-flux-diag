"""Spin-polarized against non-spin BaTiO3, for QE and VASP.

BaTiO3 is non-magnetic, so a spin-polarized run of it has two identical
channels and must give the non-spin polarization. These run the full-BZ
calculations made for this in batio3_spin_ibz_runs (see its README); they
are too large to commit, so point at that folder:

    export BFD_BATIO3_SPIN_RUNS=/path/to/batio3_spin_ibz_runs

Two kinds of check:

- Exact. The same states fed through the spin-polarized path - the non-spin
  QE wavefunctions as both channels, and a WAVECAR whose spin-down records
  are copies of the non-spin run's - must give the non-spin polarization
  bit for bit. This tests the code alone.
- Against the DFT runs. The spin-polarized runs are separate SCF
  calculations, which converge to very slightly different states (their
  occupied manifolds differ by about 1e-9 in 1 - singular value for QE, and
  their total energies by 3e-6 Ry), so these agree to a tolerance: 9e-4 and
  2.4e-4 uC/cm^2 out of about 45.5 and 45.8 when this was written.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest

import berry_flux_diag as bfd

RUNS_ENV = "BFD_BATIO3_SPIN_RUNS"

# Polarizations (uC/cm^2) of the four full-BZ cases, captured on branch
# spin_polarized_calcs from the runs made 2026-11-03 (QE 7.5, VASP 6.4.2).
PINNED = {
    ("QE", "nospin_fbz"): 45.506662240158576,
    ("QE", "spin_fbz"): 45.50573945872975,
    ("VASP", "nospin_fbz"): 45.78751000647048,
    ("VASP", "spin_fbz"): 45.7872675872014,
}
PINNED_ATOL = 1e-6

# How closely the spin-polarized runs reproduce the non-spin ones, and their
# two channels each other: set by the DFT convergence (QE conv_thr 1e-8 Ry,
# VASP EDIFF 1e-5 eV), not by this code. Observed: QE 9.2e-4 and 1e-14,
# VASP 2.4e-4 and 3.6e-5.
SPIN_VS_NOSPIN_ATOL = {"QE": 2e-3, "VASP": 1e-3}
UP_VS_DOWN_ATOL = {"QE": 1e-10, "VASP": 1e-4}


@pytest.fixture(scope="module")
def runs() -> Path:
    path = os.environ.get(RUNS_ENV)
    if not path:
        pytest.skip(f"set {RUNS_ENV} to the batio3_spin_ibz_runs folder to run these")
    path = Path(path)
    if not (path / "QE" / "spin_fbz").is_dir():
        pytest.skip(f"{path} has no QE/spin_fbz")
    return path


def parse_qe(d: Path) -> dict:
    return bfd.QEParser.qe_parser(
        str(d / "pol/calculations/batio3_pol.xml"), str(d / "np/calculations/batio3_np.xml"),
        str(d / "pol/calculations/batio3_pol.save") + "/",
        str(d / "np/calculations/batio3_np.save") + "/", str(d / "np/batio3_np.out"))


def parse_vasp(d: Path) -> dict:
    p, n = d / "pol", d / "np"
    return bfd.VASPParser.vasp_parser(p / "POSCAR", n / "POSCAR", p / "WAVECAR", n / "WAVECAR",
                                      p / "POTCAR", p, n)


def channel_string_sums(debug) -> list[np.ndarray]:
    """Each channel's string sums along x, y, z, from compute_polarization's
    debug output: the Wilson-loop phases over 2 pi, summed."""
    channels = debug if isinstance(debug, list) else [debug]
    return [np.array([sum(np.sum(w) for w in ch[d]["eigs"]) / (2 * np.pi) for d in "xyz"])
            for ch in channels]


@pytest.fixture(scope="module")
def results(runs):
    """Parse and polarization of each case, computed once per module."""
    out = {}

    def get(code, case):
        if (code, case) not in out:
            if bfd.QEParser is None and code == "QE":
                pytest.skip("QEParser needs qeschema and h5py")
            parse = (parse_qe if code == "QE" else parse_vasp)(runs / code / case)
            pol, debug = bfd.Overlaps.Overlaps(parse).compute_polarization()
            out[code, case] = (parse, float(pol), channel_string_sums(debug))
        return out[code, case]
    return get


@pytest.mark.parametrize("code", ["QE", "VASP"])
def test_spin_polarization_is_detected(results, code):
    assert results(code, "nospin_fbz")[0]["spin_pol"] is False
    parse = results(code, "spin_fbz")[0]
    assert parse["spin_pol"] is True
    assert (parse["pol_band_fill_up"], parse["pol_band_fill_down"]) == (20, 20)
    k = next(iter(parse["pol_wfcn_dict"]))
    wfcn = parse["pol_wfcn_dict"][k]["wfcn"]
    assert isinstance(wfcn, list) and len(wfcn) == 2
    assert all(c.dtype == np.complex128 for c in wfcn)


@pytest.mark.parametrize("code", ["QE", "VASP"])
def test_the_two_channels_agree(results, code):
    up, down = results(code, "spin_fbz")[2]
    np.testing.assert_allclose(up, down, rtol=0, atol=UP_VS_DOWN_ATOL[code])


@pytest.mark.parametrize("code", ["QE", "VASP"])
def test_spin_polarized_matches_non_spin(results, code):
    spin, nospin = results(code, "spin_fbz")[1], results(code, "nospin_fbz")[1]
    assert abs(spin - nospin) < SPIN_VS_NOSPIN_ATOL[code]


@pytest.mark.parametrize("code, case", sorted(PINNED))
def test_pinned_polarization(results, code, case):
    np.testing.assert_allclose(results(code, case)[1], PINNED[code, case],
                               rtol=0, atol=PINNED_ATOL)


def test_qe_same_states_as_both_channels_is_exact(results):
    parse, nospin, _ = results("QE", "nospin_fbz")
    doubled = copy.copy(parse)
    doubled["spin_pol"] = True
    for w in ("pol", "np"):
        doubled[f"{w}_wfcn_dict"] = {k: {"wfcn": [v["wfcn"], v["wfcn"]], "gvecs": v["gvecs"]}
                                     for k, v in parse[f"{w}_wfcn_dict"].items()}
        doubled[f"{w}_band_fill_up"] = doubled[f"{w}_band_fill_down"] = parse[f"{w}_band_fill"]
    assert bfd.Overlaps.Overlaps(doubled).compute_polarization()[0] == nospin


def double_wavecar(src: Path, dst: Path) -> None:
    """A spin-polarized WAVECAR whose spin-down records copy the spin-up ones:
    the header's spin count 1 -> 2, then every k-point block written twice."""
    raw = np.fromfile(src, dtype=np.float64)
    rec = int(round(raw[0])) // 8
    assert int(round(raw[1])) == 1
    nwk, nband = int(round(raw[rec])), int(round(raw[rec + 1]))
    head, body = raw[:2 * rec].copy(), raw[2 * rec:(2 + nwk * (1 + nband)) * rec]
    head[1] = 2
    np.concatenate([head, body, body]).tofile(dst)


def test_vasp_same_states_as_both_channels_is_exact(results, runs, tmp_path):
    """The whole VASP spin path - reading the spin-down records, projecting
    and reconstructing each channel, the dictionary layout, Overlaps - on a
    WAVECAR whose two channels are the non-spin run's states."""
    parse, nospin, _ = results("VASP", "nospin_fbz")
    src = runs / "VASP" / "nospin_fbz"
    for run in ("pol", "np"):
        (tmp_path / run).mkdir()
        for name in ("POSCAR", "POTCAR", "vasprun.xml"):
            (tmp_path / run / name).symlink_to(src / run / name)
        double_wavecar(src / run / "WAVECAR", tmp_path / run / "WAVECAR")
    doubled = parse_vasp(tmp_path)
    assert doubled["spin_pol"] is True
    for w in ("pol_wfcn_dict", "np_wfcn_dict"):
        for k, v in parse[w].items():
            for channel in doubled[w][k]["wfcn"]:
                assert channel.tobytes() == v["wfcn"].tobytes()
    assert bfd.Overlaps.Overlaps(doubled).compute_polarization()[0] == nospin
