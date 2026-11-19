"""G-type AFM FeBiO3 in VASP: spin-polarized DFT+U, full BZ and IBZ.

The BaTiO3 spin and IBZ tests use a non-magnetic material, where both spin
channels are the same states. FeBiO3 is the magnetic case: the two Fe
sublattices carry opposite moments (Fe1 down, Fe2 up), so the channels hold
different states, the moments cut the shared symmetry down, and U acts on
the Fe 3d states.

The runs are in febio3_spin_ibz_runs/VASP_AFM (see its README), made with
make_inputs.py there: the 10-atom R3c / R-3c pair of the QE AFM runs,
MAGMOM -5 5 0..., NUPDOWN = 0, Gamma 5x5x5, EDIFF 1e-6, VASP 6.4.2, in
four cases

    u0_fbz    U = 0 (LDAU on),  125 k-points, ISYM = -1
    u0_ibz    U = 0,            SCF with symmetry, NSCF on the 23-point IBZ
    u5p3_fbz  U = 5.3 eV,       125 k-points
    u5p3_ibz  U = 5.3 eV,       IBZ

They are too large to commit (and the POTCAR cannot be redistributed), so
point at that folder:

    export BFD_FEBIO3_AFM_VASP=/path/to/febio3_spin_ibz_runs/VASP_AFM

The moments leave the polar and non-polar structures three shared
rotations (R3 and R-3: the R3c glides exchange Fe1 and Fe2), which with
time reversal reduce the 125 points to 23. Measured when this was written
(1 - smallest singular value of the overlap of two occupied subspaces,
worst over the mesh, both channels):

                                          U = 0     U = 5.3 eV
    FBZ's own IBZ states, unfolded        3.7e-10   5.3e-10
    IBZ run against FBZ run               3.6e-8    3.5e-8
    polarization, IBZ - FBZ (uC/cm^2)     -2.3e-3   -1.1e-3

The shared rotations have zero fractional translation in both structures
here, so these runs do not exercise the translation phase of the
unfolding; test_unfold_states does, on synthetic states.

Each case is parsed once per module, and only scalars are kept: a parsed
pair is several gigabytes. With the data present this file takes about
8 minutes and peaks near 25 GB.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest

import berry_flux_diag as bfd
from berry_flux_diag import symmetry as sym
from test_ibz_batio3 import subspace_gap
from test_spin_batio3 import channel_string_sums

RUNS_ENV = "BFD_FEBIO3_AFM_VASP"

U_CASES = ["u0", "u5p3"]

# Fe1 down, Fe2 up, as in the runs' INCAR.
MAGMOMS = [-5.0, 5.0] + [0.0] * 8
N_FULL, N_IBZ, N_SHARED_ROTATIONS = 125, 23, 3
OCCUPIED = (39, 39)          # 78 valence electrons, zero net moment

# Polarizations (uC/cm^2) captured on branch ibz_support from the runs made
# 2026-11-18 (VASP 6.4.2).
PINNED = {
    "u0_fbz": 95.57746933621665,
    "u0_ibz": 95.57518427898265,
    "u5p3_fbz": 93.06893841387966,
    "u5p3_ibz": 93.06781728254647,
}
PINNED_ATOL = 1e-6

# Set by the DFT convergence, not the unfolding: a wrong operation gives a
# value near 1. Observed values are in the module docstring.
SELF_UNFOLD_MAX_GAP = 1e-8
IBZ_VS_FBZ_MAX_GAP = 1e-6
IBZ_VS_FBZ_ATOL = 5e-3

# Polarization along [111] of the rhombohedral cell: the string sums along
# the three lattice directions agree (observed to 7e-5), and the two spin
# channels of the antiferromagnet agree (observed to 1.1e-4).
DIRECTIONS_ATOL = 5e-4
UP_VS_DOWN_ATOL = 1e-3


@pytest.fixture(scope="module")
def runs() -> Path:
    path = os.environ.get(RUNS_ENV)
    if not path:
        pytest.skip(f"set {RUNS_ENV} to the febio3_spin_ibz_runs/VASP_AFM folder to run these")
    path = Path(path)
    if not (path / "u0_fbz" / "pol" / "WAVECAR").is_file():
        pytest.skip(f"{path} has no u0_fbz/pol/WAVECAR")
    return path


def parse(runs: Path, case: str) -> dict:
    p, n = runs / case / "pol", runs / case / "np"
    if case.endswith("_ibz"):
        p, n = p / "nscf", n / "nscf"
    return bfd.VASPParser.vasp_parser(p / "POSCAR", n / "POSCAR", p / "WAVECAR", n / "WAVECAR",
                                      p / "POTCAR", p, n)


def summary(parse_dict: dict) -> dict:
    pol, debug = bfd.Overlaps.Overlaps(parse_dict).compute_polarization()
    return {
        "spin_pol": parse_dict["spin_pol"],
        "fills": {label: (parse_dict[f"{label}_band_fill_up"], parse_dict[f"{label}_band_fill_down"])
                  for label in ("pol", "np")},
        "kpoints": {tuple(sym.wrap(k)) for k in parse_dict["kpoint_list"]},
        "polarization": float(pol),
        "string_sums": channel_string_sums(debug),
    }


def worst_gap(a: dict, b: dict, kpoints, key_a, key_b) -> float:
    return max(subspace_gap(a[key_a(k)]["wfcn"][c], a[key_a(k)]["gvecs"],
                            b[key_b(k)]["wfcn"][c], b[key_b(k)]["gvecs"])
               for k in kpoints for c in (0, 1))


@pytest.fixture(scope="module")
def results(runs):
    """Everything the tests check for one U, computed once, scalars only."""
    out = {}

    def get(u):
        if u not in out:
            fbz = parse(runs, f"{u}_fbz")
            r = {"fbz": summary(fbz)}

            # The FBZ run's own IBZ states, unfolded, against its states everywhere.
            shared = sym.shared_symmetry(fbz["pol_struct"], fbz["np_struct"], MAGMOMS, MAGMOMS)
            mesh = np.array(fbz["kpoint_list"])
            reps, _ = sym.reduce_mesh(mesh, shared.k_rotations)
            kmap = sym.unfold_kpoints(mesh[reps], shared.k_rotations)
            r["n_shared_rotations"], r["n_ibz"] = len(shared.k_rotations), len(reps)
            r["n_unfolded"] = int(((kmap.op_index != 0) | kmap.time_reversal).sum())
            r["self_unfold"] = {}
            for label in ("pol", "np"):
                dft = fbz[f"{label}_wfcn_dict"]
                full = sym.unfold_wfcn_dict(dft, mesh[reps], kmap, shared.k_rotations,
                                            shared.translations[label])
                r["self_unfold"][label] = worst_gap(full, dft, kmap.full_kpoints, tuple,
                                                    lambda k: tuple(np.around(k, 6)))
                del full

            ibz = parse(runs, f"{u}_ibz")
            r["ibz"] = summary(ibz)
            r["ibz_vs_fbz"] = {
                label: worst_gap(ibz[f"{label}_wfcn_dict"], fbz[f"{label}_wfcn_dict"],
                                 fbz["kpoint_list"], lambda k: tuple(sym.wrap(k)),
                                 lambda k: tuple(np.around(k, 6)))
                for label in ("pol", "np")}
            del fbz, ibz
            gc.collect()
            out[u] = r
        return out[u]
    return get


# --- the runs are read as spin-polarized insulators ----------------------

@pytest.mark.parametrize("u", U_CASES)
@pytest.mark.parametrize("bz", ["fbz", "ibz"])
def test_read_as_spin_polarized_with_39_bands_per_channel(results, u, bz):
    r = results(u)[bz]
    assert r["spin_pol"] is True
    assert r["fills"] == {"pol": OCCUPIED, "np": OCCUPIED}


@pytest.mark.parametrize("u", U_CASES)
def test_moments_are_read_from_the_runs(runs, u):
    """The parser reads MAGMOM from vasprun.xml to pick the shared symmetry."""
    for label in ("pol", "np"):
        assert bfd.VASPParser.read_magmoms(runs / f"{u}_ibz" / label / "nscf" / "vasprun.xml",
                                           10) == MAGMOMS


# --- unfolding -----------------------------------------------------------

@pytest.mark.parametrize("u", U_CASES)
def test_the_moments_leave_three_shared_rotations(results, u):
    r = results(u)
    assert (r["n_shared_rotations"], r["n_ibz"]) == (N_SHARED_ROTATIONS, N_IBZ)
    assert r["n_unfolded"] == N_FULL - N_IBZ


@pytest.mark.parametrize("u", U_CASES)
@pytest.mark.parametrize("label", ["pol", "np"])
def test_unfolded_ibz_states_are_the_dft_states(results, u, label):
    gap = results(u)["self_unfold"][label]
    assert gap < SELF_UNFOLD_MAX_GAP, f"{gap:.2e}"


@pytest.mark.parametrize("u", U_CASES)
def test_an_ibz_run_unfolds_to_the_full_mesh(results, u):
    assert results(u)["ibz"]["kpoints"] == results(u)["fbz"]["kpoints"]
    assert len(results(u)["ibz"]["kpoints"]) == N_FULL


@pytest.mark.parametrize("u", U_CASES)
@pytest.mark.parametrize("label", ["pol", "np"])
def test_an_ibz_runs_states_are_the_full_bz_runs_states(results, u, label):
    gap = results(u)["ibz_vs_fbz"][label]
    assert gap < IBZ_VS_FBZ_MAX_GAP, f"{gap:.2e}"


# --- polarization --------------------------------------------------------

@pytest.mark.parametrize("u", U_CASES)
def test_an_ibz_run_gives_the_full_bz_polarization(results, u):
    r = results(u)
    assert abs(r["ibz"]["polarization"] - r["fbz"]["polarization"]) < IBZ_VS_FBZ_ATOL


@pytest.mark.parametrize("u", U_CASES)
@pytest.mark.parametrize("bz", ["fbz", "ibz"])
def test_pinned_polarization(results, u, bz):
    np.testing.assert_allclose(results(u)[bz]["polarization"], PINNED[f"{u}_{bz}"],
                               rtol=0, atol=PINNED_ATOL)


@pytest.mark.parametrize("u", U_CASES)
@pytest.mark.parametrize("bz", ["fbz", "ibz"])
def test_rhombohedral_directions_and_spin_channels_agree(results, u, bz):
    up, down = results(u)[bz]["string_sums"]
    for channel in (up, down):
        assert np.ptp(channel) < DIRECTIONS_ATOL, channel
    np.testing.assert_allclose(up, down, rtol=0, atol=UP_VS_DOWN_ATOL)


def test_u_lowers_the_polarization(results):
    """U = 5.3 eV moves the electronic term only; the structures, and so the
    ionic term, are the same. Observed: 95.58 -> 93.07 uC/cm^2."""
    assert results("u5p3")["fbz"]["polarization"] < results("u0")["fbz"]["polarization"] - 1
