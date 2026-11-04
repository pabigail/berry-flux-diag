"""IBZ runs against full-BZ runs of BaTiO3, QE and VASP.

batio3_spin_ibz_runs holds each case twice: on the full mesh, and as an
SCF with symmetry followed by an NSCF on the 18-point IBZ that the polar and
non-polar structures share. Point at it as for test_spin_batio3:

    export BFD_BATIO3_SPIN_RUNS=/path/to/batio3_spin_ibz_runs

Three things are checked:

- Unfolding itself. The full-BZ run's own 18 IBZ states, unfolded, must be
  the states the DFT computed at every other mesh point.
- The IBZ runs end to end. Parsed, they unfold to the full mesh, and their
  states are the full-BZ run's states.
- Their polarization. It agrees with the full-BZ run's to 2.4e-3 uC/cm^2,
  and is pinned.

Agreement between states is measured as 1 - the smallest singular value of
the overlap of the two occupied subspaces, worst over the mesh. Measured
when this was written:

                                      QE       VASP
    FBZ's own IBZ states, unfolded    5e-9     1e-6
    IBZ run against FBZ run           2e-9     1.2e-6

These are set by the DFT convergence (QE conv_thr 1e-8 Ry, VASP EDIFF
1e-5 eV), not by the unfolding: a wrong operation gives a value near 1.
The polarizations differ by more than that would suggest because the
small differences add up coherently over the 125 or 216 plaquettes;
unfolding the full-BZ run's own IBZ states moves QE's by only 6e-5.
"""

from __future__ import annotations

import numpy as np
import pytest

import berry_flux_diag as bfd
from berry_flux_diag import symmetry as sym
from test_spin_batio3 import PINNED as PINNED_FBZ
from test_spin_batio3 import parse_qe, parse_vasp, runs  # noqa: F401  (runs is a fixture)

WORST_GAP = {"QE": 1e-8, "VASP": 1e-5}

# Polarizations (uC/cm^2) of the IBZ runs, unfolded, captured on branch
# ibz_support from the runs made 2026-11-03 (QE 7.5, VASP 6.4.2).
PINNED_IBZ = {
    ("QE", "nospin_ibz"): 45.50894806570832,
    ("QE", "spin_ibz"): 45.50774355639538,
    ("VASP", "nospin_ibz"): 45.789864876456214,
    ("VASP", "spin_ibz"): 45.78895010693397,
}
PINNED_ATOL = 1e-6

# The IBZ runs against the full-BZ runs: observed 1.7e-3 to 2.4e-3.
IBZ_VS_FBZ_ATOL = 5e-3

CASES = [(code, spin) for code in ("QE", "VASP") for spin in ("nospin", "spin")]


def parse_ibz(runs, code, spin):
    """An IBZ run, as the parsers read it: the NSCF's output."""
    d = runs / code / f"{spin}_ibz"
    if code == "QE":
        if bfd.QEParser is None:
            pytest.skip("QEParser needs qeschema and h5py")
        return bfd.QEParser.qe_parser(
            str(d / "pol/calculations/batio3_pol.xml"), str(d / "np/calculations/batio3_np.xml"),
            str(d / "pol/calculations/batio3_pol.save") + "/",
            str(d / "np/calculations/batio3_np.save") + "/", str(d / "np/batio3_np_nscf.out"))
    p, n = d / "pol" / "nscf", d / "np" / "nscf"
    return bfd.VASPParser.vasp_parser(p / "POSCAR", n / "POSCAR", p / "WAVECAR", n / "WAVECAR",
                                      p / "POTCAR", p, n)


@pytest.fixture(scope="module")
def ibz_results(runs):
    """For each IBZ case: spin flag, k-points and polarization. Only these
    are kept - a VASP case's unfolded wavefunctions run to gigabytes."""
    out = {}

    def get(code, spin):
        if (code, spin) not in out:
            parse = parse_ibz(runs, code, spin)
            pol = float(bfd.Overlaps.Overlaps(parse).compute_polarization()[0])
            out[code, spin] = (parse["spin_pol"], {tuple(k) for k in parse["kpoint_list"]}, pol)
        return out[code, spin]
    return get


def subspace_gap(a, ga, b, gb):
    """1 - cosine of the largest principal angle between the spans of a and b."""
    cols = {tuple(g): i for i, g in enumerate(ga)}
    assert len(ga) == len(gb) and all(tuple(g) in cols for g in gb), "plane-wave sets differ"
    a = a[:, [cols[tuple(g)] for g in gb]]
    qa, _ = np.linalg.qr(a.T)
    qb, _ = np.linalg.qr(b.T)
    return 1 - np.linalg.svd(qa.conj().T @ qb, compute_uv=False).min()


@pytest.mark.parametrize("code, case", [("QE", "nospin_fbz"), ("QE", "spin_fbz"),
                                        ("VASP", "nospin_fbz"), ("VASP", "spin_fbz")])
def test_unfolded_ibz_states_are_the_dft_states(runs, code, case):
    if code == "QE" and bfd.QEParser is None:
        pytest.skip("QEParser needs qeschema and h5py")
    parse = (parse_qe if code == "QE" else parse_vasp)(runs / code / case)
    s = sym.shared_symmetry(parse["pol_struct"], parse["np_struct"])
    mesh = np.array(parse["kpoint_list"])
    reps, _ = sym.reduce_mesh(mesh, s.k_rotations)
    assert len(reps) == 18
    kmap = sym.unfold_kpoints(mesh[reps], s.k_rotations)
    assert len(kmap.full_kpoints) == len(mesh)
    unfolded_somewhere = (kmap.op_index != 0) | kmap.time_reversal
    assert unfolded_somewhere.sum() == len(mesh) - 18

    channels = [0, 1] if parse["spin_pol"] else [None]
    for label in ("pol", "np"):
        dft = parse[f"{label}_wfcn_dict"]
        full = sym.unfold_wfcn_dict(dft, mesh[reps], kmap, s.k_rotations, s.translations[label])
        worst = 0.0
        for k in kmap.full_kpoints:
            u, ref = full[tuple(k)], dft[tuple(np.around(k, 6))]
            for c in channels:
                pick = (lambda w: w) if c is None else (lambda w: w[c])
                worst = max(worst, subspace_gap(pick(u["wfcn"]), u["gvecs"],
                                                pick(ref["wfcn"]), ref["gvecs"]))
        assert worst < WORST_GAP[code], f"{label}: {worst:.2e}"


@pytest.mark.parametrize("code, spin", CASES)
def test_an_ibz_run_unfolds_to_the_full_mesh(ibz_results, code, spin):
    spin_pol, kpoints, _ = ibz_results(code, spin)
    assert spin_pol is (spin == "spin")
    mesh = sym.monkhorst_pack((5, 5, 5) if code == "QE" else (6, 6, 6), (1, 1, 1))
    assert kpoints == {tuple(k) for k in sym.wrap(mesh)}


@pytest.mark.parametrize("code, spin", CASES)
def test_an_ibz_run_gives_the_full_bz_polarization(ibz_results, code, spin):
    ibz = ibz_results(code, spin)[2]
    assert abs(ibz - PINNED_FBZ[code, f"{spin}_fbz"]) < IBZ_VS_FBZ_ATOL


@pytest.mark.parametrize("code, spin", CASES)
def test_pinned_ibz_polarization(ibz_results, code, spin):
    np.testing.assert_allclose(ibz_results(code, spin)[2], PINNED_IBZ[code, f"{spin}_ibz"],
                               rtol=0, atol=PINNED_ATOL)


@pytest.mark.parametrize("code, spin", CASES)
def test_an_ibz_runs_states_are_the_full_bz_runs_states(runs, code, spin):
    ibz = parse_ibz(runs, code, spin)
    fbz = (parse_qe if code == "QE" else parse_vasp)(runs / code / f"{spin}_fbz")
    channels = [0, 1] if fbz["spin_pol"] else [None]
    for label in ("pol", "np"):
        worst = 0.0
        for k in fbz["kpoint_list"]:
            a = ibz[f"{label}_wfcn_dict"][tuple(sym.wrap(k))]
            b = fbz[f"{label}_wfcn_dict"][tuple(np.around(k, 6))]
            for c in channels:
                pick = (lambda w: w) if c is None else (lambda w: w[c])
                worst = max(worst, subspace_gap(pick(a["wfcn"]), a["gvecs"], pick(b["wfcn"]), b["gvecs"]))
        assert worst < WORST_GAP[code], f"{label}: {worst:.2e}"
