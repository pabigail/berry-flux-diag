"""Unit tests for collinear spin-polarized runs, VASP and QE.

A spin-polarized run carries two sets of occupied states, up and down, on
the same k-points. Each piece below must keep them apart, pick the right
one, and reduce to the non-spin answer when the two are the same. These
use synthetic data; tests/test_spin_batio3.py runs the real BaTiO3 runs.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.outputs import Wavecar

from berry_flux_diag import Overlaps, QEParser, VASPParser, paw
from test_aug_basis import fake_element
from test_overlap_cache import random_wfcn_dict
from test_wavecar_header import KPTS, LATTICE, write_wavecar

# --- reading a spin-polarized WAVECAR -------------------------------------

def test_both_channels_are_read_as_pymatgen_reads_them(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS, nband=5, spin=2)
    wav, ref = paw.read_wavecar(f), Wavecar(str(f))
    assert wav["nspin"] == 2 and len(wav["kpts"]) == len(KPTS) == len(wav["Gs"])
    for spin in (0, 1):
        assert len(wav["Cs"][spin]) == len(KPTS)
        for k in range(len(KPTS)):
            np.testing.assert_array_equal(wav["Cs"][spin][k], ref.coeffs[spin][k])
    assert not np.array_equal(wav["Cs"][0][0], wav["Cs"][1][0])


def test_a_non_spin_wavecar_has_one_channel(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS, nband=5)
    wav = paw.read_wavecar(f)
    assert wav["nspin"] == 1 and len(wav["Cs"]) == 1 and len(wav["Cs"][0]) == len(KPTS)


def test_nbands_limits_both_channels(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS, nband=5, spin=2)
    wav = paw.read_wavecar(f, nbands=3)
    assert {c.shape[0] for channel in wav["Cs"] for c in channel} == {3}


def test_spin_down_kpoints_must_match_spin_up(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS, nband=4, spin=2)
    raw = np.fromfile(f, dtype=np.float64)
    rec = int(raw[0]) // 8
    first_down = (2 + len(KPTS) * (1 + 4)) * rec      # header record of spin down, k 0
    raw[first_down + 3] += 0.25                         # move its k_z
    raw.tofile(f)
    with pytest.raises(ValueError, match="spin-down k-point 0"):
        paw.read_wavecar(f)


# --- reconstructing either channel ------------------------------------------

def stub_momentum_matrix(same_channels=False, seed=0):
    """A MomentumMatrix over a stand-in Wavefunction: random coefficients and
    projections per channel, real _aug_basis machinery underneath."""
    rng = np.random.default_rng(seed)
    a, c = fake_element([0, 0, 1, 1], width=0.6), fake_element([1, 1], width=0.4)
    site_elems, coords = [a, c], np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    nproj = sum(len(el["lm"]) for el in site_elems)
    b, nbmax = paw._setup(150.0, LATTICE)
    kpts = [np.array(k) for k in KPTS]
    Gs = [paw._g_sphere(k, nbmax, b, 150.0) for k in kpts]
    G_bounds = np.zeros(6, dtype=np.int64)
    for G in Gs:
        G_bounds[0::2] = np.minimum(G_bounds[0::2], G.min(axis=0))
        G_bounds[1::2] = np.maximum(G_bounds[1::2], G.max(axis=0))
    nb = 4

    def channel():
        Cs = [(rng.normal(size=(nb, len(G))) + 1j * rng.normal(size=(nb, len(G)))).astype(np.complex64)
              for G in Gs]
        proj = [rng.normal(size=(nb, nproj)) + 1j * rng.normal(size=(nb, nproj)) for _ in Gs]
        return Cs, proj

    up = channel()
    down = up if same_channels else channel()
    wav = {"lattice": LATTICE, "reclattice": b, "kpts": kpts, "Gs": Gs, "G_bounds": G_bounds,
           "Cs": [up[0], down[0]]}
    wf = SimpleNamespace(wav=wav, encut=150.0, coords=coords, site_elems=site_elems, nspin=2,
                         nbands_loaded=nb, projections=[up[1], down[1]])
    return paw.MomentumMatrix(wf, encut=150.0)


def test_each_channel_uses_its_own_coefficients_and_projections():
    mm = stub_momentum_matrix()
    for spin in (0, 1):
        expected = (mm._pseudo(1, np.arange(4), spin)
                    + mm.wf.projections[spin][1] @ mm._aug_basis(1).T)
        np.testing.assert_array_equal(mm.get_reciprocal_fullfw_kpoint(1, 4, spin), expected)
    assert not np.array_equal(mm.get_reciprocal_fullfw_kpoint(1, 4, 0),
                              mm.get_reciprocal_fullfw_kpoint(1, 4, 1))


def test_each_channels_pseudo_coefficients_land_on_the_grid():
    """Independently of _pseudo itself: every WAVECAR G vector that is on the
    momentum grid carries that channel's coefficient there."""
    mm = stub_momentum_matrix()
    k = 1
    where = {tuple(g): i for i, g in enumerate(mm._grid)}
    pairs = [(j, where[tuple(g)]) for j, g in enumerate(mm.wf.wav["Gs"][k]) if tuple(g) in where]
    assert len(pairs) > 10
    cols, grid_cols = zip(*pairs)
    for spin in (0, 1):
        out = mm._pseudo(k, np.arange(4), spin)
        np.testing.assert_array_equal(out[:, list(grid_cols)],
                                      mm.wf.wav["Cs"][spin][k][:, list(cols)])


def test_identical_channels_reconstruct_identically():
    mm = stub_momentum_matrix(same_channels=True)
    assert mm.get_reciprocal_fullfw_kpoint(2, 4, 0).tobytes() == \
        mm.get_reciprocal_fullfw_kpoint(2, 4, 1).tobytes()


def test_the_augmentation_basis_is_built_once_per_kpoint(monkeypatch):
    mm = stub_momentum_matrix()
    calls = []
    orig = mm._aug_basis
    monkeypatch.setattr(mm, "_aug_basis", lambda k: calls.append(k) or orig(k))
    for k in range(len(KPTS)):
        for spin in (0, 1):
            mm.get_reciprocal_fullfw_kpoint(k, 4, spin)
    assert calls == list(range(len(KPTS)))


def test_the_band_by_band_accessor_keeps_channels_apart():
    mm = stub_momentum_matrix()
    for spin in (0, 1):
        full = mm.get_reciprocal_fullfw_kpoint(0, 4, spin)
        np.testing.assert_array_equal(mm.get_reciprocal_fullfw(2, 0, spin), full[2])
    np.testing.assert_array_equal(mm.get_reciprocal_fullfw(2, 0, 0),
                                  mm.get_reciprocal_fullfw_kpoint(0, 4, 0)[2])


def test_a_channel_the_wavecar_lacks_is_an_error():
    mm = stub_momentum_matrix()
    with pytest.raises(ValueError, match="spin 2"):
        mm.get_reciprocal_fullfw_kpoint(0, 4, 2)
    with pytest.raises(ValueError, match="spin 2 not loaded"):
        mm.get_reciprocal_fullfw(0, 0, 2)


# --- VASPParser -----------------------------------------------------------

def test_band_counts_are_one_number_or_an_up_down_pair():
    assert VASPParser._band_fills_per_channel(20) == [20]
    assert VASPParser._band_fills_per_channel(np.int64(20)) == [20]
    assert VASPParser._band_fills_per_channel((21, 19)) == [21, 19]
    with pytest.raises(ValueError, match="up, down"):
        VASPParser._band_fills_per_channel((20, 20, 20))


@pytest.mark.parametrize("nspin, fill", [(1, (20, 20)), (2, 20)])
def test_band_counts_must_match_the_wavecar(monkeypatch, nspin, fill):
    monkeypatch.setattr(paw.Wavefunction, "from_files",
                        classmethod(lambda cls, *a, **k: SimpleNamespace(nspin=nspin, encut=100.0)))
    with pytest.raises(ValueError, match=f"has {nspin} spin channel"):
        VASPParser.get_paw_wfcn_data("WAVECAR", "POTCAR", "vasprun.xml", fill)


def test_the_dictionary_holds_up_and_down_per_kpoint(monkeypatch):
    nk, ng = 3, 7
    up, down = np.arange(2 * nk * ng).reshape(2, nk, ng), -np.arange(nk * ng).reshape(1, nk, ng)
    monkeypatch.setattr(VASPParser, "get_paw_wfcn_data",
                        lambda *a, **k: (np.zeros((ng, 3)), ng, [up, down]))
    kpts = [np.array([0, 0, i / nk]) for i in range(nk)]
    d = VASPParser.paw_wfcn_dict("W", "P", "v", (2, 1), kpts)
    for i, k in enumerate(kpts):
        wfcn = d[tuple(k)]["wfcn"]
        assert isinstance(wfcn, list) and len(wfcn) == 2
        np.testing.assert_array_equal(wfcn[0], up[:, i, :])
        np.testing.assert_array_equal(wfcn[1], down[:, i, :])


MESH = [list(np.array(k) / 2) for k in itertools.product(range(2), repeat=3)]


@pytest.fixture
def spin_runs(tmp_path, monkeypatch):
    """Spin-polarized runs: polar 3 up / 2 down, non-polar 3 up / 1 down."""
    struct = Structure(Lattice(LATTICE), ["Ba"], [[0, 0, 0]])
    dirs = {}
    for label, filled, seed in (("pol", [3, 2], 1), ("np", [3, 1], 2)):
        d = tmp_path / label
        d.mkdir()
        struct.to(filename=str(d / "POSCAR"), fmt="poscar")
        write_wavecar(d / "WAVECAR", MESH, nband=5, spin=2, seed=seed, filled=filled)
        (d / "vasprun.xml").write_text(f'<i name="NELECT">  {sum(filled)}.0</i>\n')
        dirs[label] = d
    monkeypatch.setattr(VASPParser.Potcar, "from_file", staticmethod(lambda f: None))
    monkeypatch.setattr(VASPParser, "zval_dict_from_potcar", lambda p: {"Ba": 10.0})
    return dirs


def test_vasp_parser_detects_spin_and_passes_each_channels_shared_count(spin_runs, monkeypatch):
    calls = []
    monkeypatch.setattr(VASPParser, "paw_wfcn_dict",
                        lambda w, p, v, fill, k, momentum_encut=None: calls.append(fill) or {})
    p, n = spin_runs["pol"], spin_runs["np"]
    d = VASPParser.vasp_parser(p / "POSCAR", n / "POSCAR", p / "WAVECAR", n / "WAVECAR",
                               p / "POTCAR", p, n)
    assert d["spin_pol"] is True
    assert (d["pol_band_fill_up"], d["pol_band_fill_down"]) == (3, 2)
    assert (d["np_band_fill_up"], d["np_band_fill_down"]) == (3, 1)
    assert calls == [(3, 1), (3, 1)]


# --- QEParser -------------------------------------------------------------

def write_qe_wfc(path, coeffs, miller):
    """A wfc*.hdf5 as QE writes it: evc as (nbnd, 2 igwx) re/im pairs."""
    nbnd, igwx = coeffs.shape
    with h5py.File(path, "w") as f:
        f.attrs["igwx"], f.attrs["nbnd"] = igwx, nbnd
        f["evc"] = np.stack([coeffs.real, coeffs.imag], axis=-1).reshape(nbnd, 2 * igwx)
        f["MillerIndices"] = miller


def test_qe_spin_channels_are_plain_complex_arrays(tmp_path):
    """np.array([up, down], dtype=object) with equal band counts made one
    object array, which np.linalg.svd refuses."""
    rng = np.random.default_rng(0)
    miller = rng.integers(-3, 4, size=(6, 3))
    kpts = [np.zeros(3), np.array([0, 0, 0.5])]
    channels = {}
    for i in (1, 2):
        for tag in ("up", "dw"):
            c = rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6))
            channels[tag, i] = c
            write_qe_wfc(tmp_path / f"wfc{tag}{i}.hdf5", c, miller)
    d = QEParser.get_wfcn_gvecs_from_hdf5_spinpol(str(tmp_path) + "/", kpts, 3, 3)
    for i, k in enumerate(kpts, start=1):
        wfcn = d[tuple(k)]["wfcn"]
        assert isinstance(wfcn, list) and len(wfcn) == 2
        for arr, tag in zip(wfcn, ("up", "dw")):
            assert arr.dtype == np.complex128 and arr.shape == (3, 6)
            np.testing.assert_allclose(arr, channels[tag, i][:3])
        np.linalg.svd(wfcn[0].conj() @ wfcn[1].T)


def test_qe_spin_channels_must_share_g_vectors(tmp_path):
    c = np.ones((2, 4), dtype=complex)
    write_qe_wfc(tmp_path / "wfcup1.hdf5", c, np.zeros((4, 3), dtype=int))
    write_qe_wfc(tmp_path / "wfcdw1.hdf5", c, np.ones((4, 3), dtype=int))
    with pytest.raises(ValueError, match="gvecs not same"):
        QEParser.get_wfcn_gvecs_from_hdf5_spinpol(str(tmp_path) + "/", [np.zeros(3)], 2, 2)


# --- Overlaps -------------------------------------------------------------

N, NBANDS = 3, 3


def parse_dicts(seed=0):
    """A non-spin parse dict, and the spin-polarized one with its states as
    both channels."""
    rng = np.random.default_rng(seed)
    kpoints = [np.around(np.array(k) / N, 6) for k in itertools.product(range(N), repeat=3)]
    struct = Structure(Lattice.cubic(4.0), ["Ba"], [[0, 0, 0]])
    pol, np_ = random_wfcn_dict(rng, kpoints), random_wfcn_dict(rng, kpoints)
    base = {"pol_struct": struct, "np_struct": struct, "kpoint_list": kpoints,
            "zval_dict": {"Ba": 10.0}, "ES_code": "test"}
    nospin = {**base, "spin_pol": False, "pol_wfcn_dict": pol, "np_wfcn_dict": np_,
              "pol_band_fill": NBANDS, "np_band_fill": NBANDS}
    doubled = {**base, "spin_pol": True,
               "pol_wfcn_dict": {k: {"wfcn": [v["wfcn"], v["wfcn"]], "gvecs": v["gvecs"]}
                                 for k, v in pol.items()},
               "np_wfcn_dict": {k: {"wfcn": [v["wfcn"], v["wfcn"]], "gvecs": v["gvecs"]}
                                for k, v in np_.items()},
               "pol_band_fill_up": NBANDS, "pol_band_fill_down": NBANDS,
               "np_band_fill_up": NBANDS, "np_band_fill_down": NBANDS}
    return nospin, doubled


def overlaps(parse):
    return Overlaps.Overlaps(parse, eig_thresh=10.0, sing_val_thresh=0.0)


def test_two_identical_channels_give_the_non_spin_polarization_exactly():
    """Up + down with weight 1 against one channel doubled: x + x == 2x in
    floating point, so this is equality, not a tolerance."""
    nospin, doubled = parse_dicts()
    assert overlaps(doubled).compute_polarization()[0] == overlaps(nospin).compute_polarization()[0]


def with_new_coefficients(parse, seed):
    """The same parse dict on the same G vectors, with other random states:
    a second spin channel shares the k-points and G vectors of the first."""
    rng = np.random.default_rng(seed)
    out = dict(parse)
    for d in ("pol_wfcn_dict", "np_wfcn_dict"):
        out[d] = {k: {"wfcn": rng.normal(size=v["wfcn"].shape) + 1j * rng.normal(size=v["wfcn"].shape),
                      "gvecs": v["gvecs"]} for k, v in parse[d].items()}
    return out


def two_channels(up, down):
    """The spin-polarized parse dict with up's states as spin up and down's
    as spin down."""
    _, doubled = parse_dicts()
    for d in ("pol_wfcn_dict", "np_wfcn_dict"):
        for k, v in doubled[d].items():
            v["wfcn"] = [up[d][k]["wfcn"], down[d][k]["wfcn"]]
    return doubled


def test_each_spin_state_reads_its_own_channel():
    up, _ = parse_dicts()
    down = with_new_coefficients(up, seed=1)
    ov = overlaps(two_channels(up, down))
    ov.band_fill = NBANDS
    ov.spin_state = 1
    assert ov.compute_string_sums()[0] == overlaps(down).compute_string_sums()[0]
    ov.spin_state = 0
    assert ov.compute_string_sums()[0] == overlaps(up).compute_string_sums()[0]
    assert overlaps(up).compute_string_sums()[0] != overlaps(down).compute_string_sums()[0]


def test_the_electronic_term_adds_the_channels_with_weight_one():
    """With different channels the electronic term is (S_up + S_down) / n:
    the mean of the two channels' non-spin terms, 2 S / n."""
    up, _ = parse_dicts()
    down = with_new_coefficients(up, seed=1)
    s_up, s_down = overlaps(up).compute_string_sums(), overlaps(down).compute_string_sums()
    ov = overlaps(two_channels(up, down))
    elec = []
    ov.get_spont_pol = lambda e: elec.append(e) or np.zeros(3)
    ov.compute_polarization()
    expected = [(s_up[0][i] + s_down[0][i]) / s_up[1][i] for i in range(3)]
    assert elec[0] == expected
