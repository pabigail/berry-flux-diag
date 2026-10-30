"""Unit tests for vasp_parser running the two PAW reconstructions in threads.

The polar and non-polar reconstructions are independent, so vasp_parser
runs them side by side. That must not cross the runs over, drop the cutoff,
swallow an error from either one, or quietly fall back to running them one
after the other. These drive vasp_parser on synthetic WAVECARs with
paw_wfcn_dict replaced, so they need no VASP data; that the real
reconstruction is unchanged bit for bit is checked by test_regression_batio3
when the data is available.
"""

from __future__ import annotations

import itertools
import threading

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from berry_flux_diag import VASPParser
from test_wavecar_header import LATTICE, write_wavecar

MESH = [list(np.array(k) / 2) for k in itertools.product(range(2), repeat=3)]


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """Polar and non-polar run directories, with occupations filling a
    different number of bands in each so the band fillings tell them apart."""
    struct = Structure(Lattice(LATTICE), ["Ba"], [[0, 0, 0]])
    dirs = {}
    for label, nband, seed in (("pol", 4, 1), ("np", 6, 2)):
        d = tmp_path / label
        d.mkdir()
        struct.to(filename=str(d / "POSCAR"), fmt="poscar")
        write_wavecar(d / "WAVECAR", MESH, nband=nband, seed=seed)
        dirs[label] = d
    # A real POTCAR cannot be committed; only its valences are read here.
    monkeypatch.setattr(VASPParser.Potcar, "from_file", staticmethod(lambda f: None))
    monkeypatch.setattr(VASPParser, "zval_dict_from_potcar", lambda p: {"Ba": 10.0})
    return dirs


def parse(dirs, **kwargs):
    p, n = dirs["pol"], dirs["np"]
    return VASPParser.vasp_parser(p / "POSCAR", n / "POSCAR", p / "WAVECAR", n / "WAVECAR",
                                  p / "POTCAR", p, n, **kwargs)


def recording(monkeypatch, body=None):
    calls = []

    def fake(wavecar, potcar, vasprun, band_fill, kpoints, momentum_encut=None):
        calls.append({"wavecar": wavecar, "vasprun": vasprun, "band_fill": band_fill,
                      "encut": momentum_encut, "thread": threading.get_ident()})
        if body is not None:
            body(wavecar)
        return {"from": str(wavecar), "band_fill": band_fill}

    monkeypatch.setattr(VASPParser, "paw_wfcn_dict", fake)
    return calls


def test_each_run_gets_its_own_inputs_and_result(runs, monkeypatch):
    calls = recording(monkeypatch)
    d = parse(runs)
    assert len(calls) == 2
    by_run = {c["wavecar"]: c for c in calls}
    pol, np_ = by_run[runs["pol"] / "WAVECAR"], by_run[runs["np"] / "WAVECAR"]
    assert pol["vasprun"] == runs["pol"] / "vasprun.xml"
    assert np_["vasprun"] == runs["np"] / "vasprun.xml"
    assert pol["band_fill"] == d["pol_band_fill"]
    assert np_["band_fill"] == d["np_band_fill"]
    assert d["pol_band_fill"] != d["np_band_fill"]
    assert d["pol_wfcn_dict"]["from"] == str(runs["pol"] / "WAVECAR")
    assert d["np_wfcn_dict"]["from"] == str(runs["np"] / "WAVECAR")


def test_the_cutoff_reaches_both_runs(runs, monkeypatch):
    calls = recording(monkeypatch)
    parse(runs, momentum_encut=1234.0)
    assert [c["encut"] for c in calls] == [1234.0, 1234.0]


def test_the_two_runs_overlap_in_time(runs, monkeypatch):
    """Each waits for the other to start; run one after the other, the
    first would wait out the timeout and fail."""
    barrier = threading.Barrier(2, timeout=10)
    calls = recording(monkeypatch, lambda wavecar: barrier.wait())
    parse(runs)
    assert len({c["thread"] for c in calls}) == 2


@pytest.mark.parametrize("failing", ["pol", "np"])
def test_an_error_in_either_run_is_raised(runs, monkeypatch, failing):
    def body(wavecar):
        if wavecar == runs[failing] / "WAVECAR":
            raise ValueError(f"{failing} failed")

    recording(monkeypatch, body)
    with pytest.raises(ValueError, match=f"^{failing} failed$"):
        parse(runs)
