"""Unit tests for reading IBZ runs: the parsers detect a k-point list that is
not a full mesh, unfold both runs' wavefunctions with the shared symmetry,
and count bands with each k-point weighted by its multiplicity.

The end-to-end check on the BaTiO3 IBZ runs is in test_ibz_batio3.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from berry_flux_diag import QEParser, VASPParser, symmetry as sym, utils
from test_symmetry import SPECIES, VASP_NP, VASP_POL
from test_wavecar_header import LATTICE, write_wavecar

# --- weighted band counting ------------------------------------------------

def smeared_ibz():
    """Three IBZ points with shares 1/2, 1/4, 1/4. With smearing each holds
    a slightly different number of electrons - the heavy one 0.01 short,
    the light ones 0.01 over - and only the weighted average is the
    electron count. Unweighted, the average is 0.0033 too high."""
    weights = np.array([0.5, 0.25, 0.25])
    energies = np.sort(np.random.default_rng(0).uniform(-10, 0, size=(3, 24)), axis=1)
    energies[:, 20:] += 12
    occ = np.zeros((3, 24))
    occ[:, :20] = 1.0
    occ[0, 19] -= 0.01            # a valence-band tail missing
    occ[1:, 20] += 0.01           # conduction-band tails
    return weights, energies, occ


def test_occupation_checks_weight_each_kpoint_by_its_share():
    weights, energies, occ = smeared_ibz()
    assert utils.occupied_band_count(20, energies, occ, weights=weights) == 20
    with pytest.raises(ValueError, match="do not describe the same run"):
        utils.occupied_band_count(20, energies, occ)


def test_spin_channels_are_split_with_weights():
    """Down: 2.98 electrons at the heavy point, 3.06 at the light one -
    3 on the weighted average, 3.02 unweighted."""
    weights = np.array([0.75, 0.25])
    up = np.zeros((2, 6)); up[:, :4] = 1.0
    down = np.zeros((2, 6)); down[:, :3] = 1.0
    down[0, 2] -= 0.02
    down[1, 3] += 0.06
    assert utils.spin_channel_electrons(7, up, down, weights=weights) == (4, 3)
    with pytest.raises(ValueError, match="spin-down occupations"):
        utils.spin_channel_electrons(7, up, down)


# --- reading weights and moments -------------------------------------------

def test_vasp_kpoint_weights(tmp_path):
    f = tmp_path / "vasprun.xml"
    f.write_text('<kpoints>\n  <varray name="kpointlist" >\n   <v> 0 0 0 </v>\n  </varray>\n'
                 '  <varray name="weights" >\n   <v>       0.03703704 </v>\n'
                 '   <v>       0.11111111 </v>\n  </varray>\n</kpoints>\n')
    assert VASPParser.read_kpoint_weights(f) == [0.03703704, 0.11111111]
    (tmp_path / "empty.xml").write_text("<modeling/>\n")
    assert VASPParser.read_kpoint_weights(tmp_path / "empty.xml") is None


def test_vasp_magmoms(tmp_path):
    f = tmp_path / "vasprun.xml"
    f.write_text(' <incar>\n  <v name="MAGMOM">      1.00000000     -1.00000000</v>\n </incar>\n')
    assert VASPParser.read_magmoms(f, 2) == [1.0, -1.0]
    with pytest.raises(ValueError, match="collinear"):
        VASPParser.read_magmoms(f, 3)
    (tmp_path / "none.xml").write_text("<modeling/>\n")
    assert VASPParser.read_magmoms(tmp_path / "none.xml", 2) is None


def qe_xml(species, atoms, weights=(1.0,)):
    return {"qes:espresso": {
        "input": {"atomic_species": {"species": species}},
        "output": {"atomic_structure": {"atomic_positions": {"atom": [{"@name": a} for a in atoms]}},
                   "band_structure": {"ks_energies": [{"k_point": {"@weight": w}} for w in weights]}}}}


@pytest.mark.skipif(QEParser is None, reason="QEParser needs qeschema and h5py")
def test_qe_weights_and_moments():
    xml = qe_xml([{"@name": "Fe", "starting_magnetization": 0.5}, {"@name": "O"}],
                 ["Fe", "O", "Fe"], weights=(0.25, 0.75))
    assert QEParser.get_kpoint_weights_from_qeschema(xml) == [0.25, 0.75]
    assert QEParser.get_magmoms_from_qeschema(xml) == [0.5, 0.0, 0.5]
    single = qe_xml({"@name": "Ba", "starting_magnetization": None}, ["Ba"])
    assert QEParser.get_magmoms_from_qeschema(single) == [0.0]


# --- the IBZ helper -------------------------------------------------------

def ibz_of(pol, np_, n=(6, 6, 6), shift=(1, 1, 1)):
    s = sym.shared_symmetry(pol, np_)
    mesh = sym.monkhorst_pack(n, shift)
    reps, mult = sym.reduce_mesh(mesh, s.k_rotations)
    return mesh[reps], mult


def test_full_meshes_are_told_from_ibz_lists():
    ibz, _ = ibz_of(VASP_POL, VASP_NP)
    assert sym.is_full_mesh(sym.monkhorst_pack((6, 6, 6), (1, 1, 1)))
    assert not sym.is_full_mesh(ibz)


def test_the_dft_weights_must_match_the_multiplicities():
    ibz, mult = ibz_of(VASP_POL, VASP_NP)
    u = sym.IBZUnfolding(VASP_POL, VASP_NP, ibz, weights=mult / 216 * 2)   # any normalisation
    np.testing.assert_allclose(u.weights, mult / mult.sum())
    assert len(u.full_kpoints) == 216
    with pytest.raises(ValueError, match="weights of the DFT run do not match"):
        sym.IBZUnfolding(VASP_POL, VASP_NP, ibz, weights=np.ones(len(ibz)))


def test_each_run_is_unfolded_with_its_own_translations():
    shifted = Structure(VASP_POL.lattice, SPECIES, VASP_POL.frac_coords + [0.11, 0.23, 0.07])
    ibz, _ = ibz_of(VASP_POL, shifted, n=(2, 2, 2))
    u = sym.IBZUnfolding(VASP_POL, shifted, ibz)
    g = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    d = {tuple(np.around(k, 6)): {"wfcn": np.ones((1, len(g)), complex), "gvecs": g} for k in ibz}
    a, b = u.unfold(d, "pol"), u.unfold(d, "np")
    assert any(not np.allclose(a[k]["wfcn"], b[k]["wfcn"]) for k in a)


def test_a_shared_grid_stays_shared():
    """VASP's momentum grid is closed under the rotations: where no G0 shift
    is needed the unfolded coefficients go back in the grid's order, so
    every k-point holds the very same G list and Overlaps skips remapping."""
    ibz, _ = ibz_of(VASP_POL, VASP_NP)
    u = sym.IBZUnfolding(VASP_POL, VASP_NP, ibz)
    grid = np.array([g for g in np.ndindex(5, 5, 5)]) - 2
    grid = grid[np.sum(grid ** 2, axis=1) <= 4]
    rng = np.random.default_rng(3)
    d = {tuple(np.around(k, 6)): {"wfcn": rng.normal(size=(2, len(grid))) + 0j, "gvecs": grid}
         for k in ibz}
    full = u.unfold(d, "pol")
    assert not u.kmap.gshift.any()                     # this mesh needs no G0
    assert all(v["gvecs"] is grid for v in full.values())
    kmap, s = u.kmap, u.symmetry
    for j, k in enumerate(kmap.full_kpoints):
        src = d[tuple(np.around(ibz[kmap.ibz_index[j]], 6))]
        c, g = sym.unfold_state(src["wfcn"], grid, ibz[kmap.ibz_index[j]], s.k_rotations[kmap.op_index[j]],
                                s.translations["pol"][kmap.op_index[j]], kmap.time_reversal[j], kmap.gshift[j])
        where = {tuple(x): i for i, x in enumerate(g)}
        np.testing.assert_array_equal(full[tuple(k)]["wfcn"], c[:, [where[tuple(x)] for x in grid]])


# --- vasp_parser on an IBZ ------------------------------------------------

MESH = sym.monkhorst_pack((3, 3, 3))

# Polar / non-polar pairs whose shared operations the parsers unfold with.
# "triclinic": one atom, P-1, whose inversion reduces k as time reversal
# does - so unfolding never needs a translation. "tetragonal": BaTiO3 as
# the polar run and a copy shifted off the origin as the non-polar one, so
# the same rotations come with different translations in each.
PAIRS = {
    "triclinic": (Structure(Lattice(LATTICE), ["Ba"], [[0, 0, 0]]),
                  Structure(Lattice(LATTICE), ["Ba"], [[0.1, 0.2, 0.3]])),
    "tetragonal": (VASP_POL, Structure(VASP_POL.lattice, SPECIES,
                                       VASP_POL.frac_coords + [0.11, 0.23, 0.07])),
}


def pair_setup(name):
    """Structures, the IBZ of MESH, multiplicities, and occupations whose
    smearing tails balance only with the multiplicities as weights: 0.002
    above the gap at every point but Gamma, Gamma short by the rest."""
    pol, np_ = PAIRS[name]
    s = sym.shared_symmetry(pol, np_)
    reps, mult = sym.reduce_mesh(MESH, s.k_rotations)
    gamma = int(np.where(mult == 1)[0][0])
    tails = np.zeros((len(reps), 5))
    tails[:, :2] = 1.0
    tails[:, 2] = 0.002
    tails[gamma, 2] = 0.0
    tails[gamma, 1] -= 0.002 * (mult.sum() - 1)
    return pol, np_, MESH[reps], mult, tails


@pytest.mark.parametrize("name", PAIRS)
def test_the_pairs_exercise_what_they_should(name):
    pol, np_, ibz, mult, tails = pair_setup(name)
    u = sym.IBZUnfolding(pol, np_, ibz)
    used = set(u.kmap.op_index)
    with_t = [o for o in used if np.abs(u.symmetry.translations["np"][o]).max() > 1e-3]
    assert bool(with_t) == (name == "tetragonal")
    assert abs(np.average(tails.sum(axis=1), weights=mult) - 2) < 1e-12
    assert abs(tails.sum(axis=1).mean() - 2) > utils.ELECTRON_COUNT_ATOL


@pytest.fixture
def vasp_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(VASPParser.Potcar, "from_file", staticmethod(lambda f: None))
    monkeypatch.setattr(VASPParser, "zval_dict_from_potcar", lambda p: {e: 10.0 for e in ("Ba", "Ti", "O")})

    def make(name, spin=1, weights=None, drop_first=False):
        pol, np_, ibz, mult, tails = pair_setup(name)
        if drop_first:
            ibz, tails, mult = ibz[1:], tails[1:], mult[1:]
        dirs = {}
        for label, struct, seed in (("pol", pol, 1), ("np", np_, 2)):
            d = tmp_path / f"{label}_{name}_{spin}"
            d.mkdir(exist_ok=True)
            struct.to(filename=str(d / "POSCAR"), fmt="poscar")
            write_wavecar(d / "WAVECAR", ibz, nband=5, spin=spin, seed=seed, lattice=struct.lattice.matrix,
                          filled=2 if spin == 1 else [2, 2], occupations=[tails] * spin)
            w = mult / mult.sum() if weights is None else weights
            xml = '<i name="NELECT">  4.0</i>\n'
            if spin == 2:
                xml += f'<v name="MAGMOM">  {"  0.0" * len(struct)}</v>\n'
            if not drop_first:
                xml += '<varray name="weights" >\n' + "".join(f"<v> {x:.8f} </v>\n" for x in w) + "</varray>\n"
            (d / "vasprun.xml").write_text(xml)
            dirs[label] = d
        return dirs, ibz, mult
    return make


def fake_paw(monkeypatch):
    """Stands in for the PAW reconstruction: random states on one grid.
    Returns what it produced, keyed by "pol" / "np"."""
    grid = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    made = {}

    def fake(wavecar, potcar, vasprun, fill, kpoint_list, momentum_encut=None):
        label = wavecar.parent.name.split("_")[0]
        rng = np.random.default_rng(1 if label == "pol" else 2)
        out = {}
        for k in kpoint_list:
            chans = [rng.normal(size=(n, len(grid))) + 1j * rng.normal(size=(n, len(grid)))
                     for n in np.atleast_1d(fill)]
            out[tuple(k)] = {"wfcn": chans if np.ndim(fill) else chans[0], "gvecs": grid}
        made[label] = out
        return out
    monkeypatch.setattr(VASPParser, "paw_wfcn_dict", fake)
    return made


def run_vasp(dirs):
    p, n = dirs["pol"], dirs["np"]
    return VASPParser.vasp_parser(p / "POSCAR", n / "POSCAR", p / "WAVECAR", n / "WAVECAR",
                                  p / "POTCAR", p, n)


def channels(wfcn):
    return wfcn if isinstance(wfcn, list) else [wfcn]


def assert_unfolded_exactly(d, ibz, sources):
    """Each run's dictionary is its IBZ states unfolded with its own
    structure's translations."""
    u = sym.IBZUnfolding(d["pol_struct"], d["np_struct"], ibz)
    for label in ("pol", "np"):
        for k, v in u.unfold(sources[label], label).items():
            got = channels(d[f"{label}_wfcn_dict"][k]["wfcn"])
            for a, b in zip(got, channels(v["wfcn"])):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", PAIRS)
@pytest.mark.parametrize("spin", [1, 2])
def test_vasp_parser_unfolds_an_ibz_run(vasp_runs, monkeypatch, name, spin, caplog):
    made = fake_paw(monkeypatch)
    dirs, ibz, mult = vasp_runs(name, spin)
    with caplog.at_level(logging.INFO, logger="berry_flux_diag.symmetry"):
        d = run_vasp(dirs)
    assert f"IBZ: {len(ibz)} k-points unfold to a 27-point mesh" in caplog.text
    assert {tuple(k) for k in d["kpoint_list"]} == {tuple(k) for k in sym.wrap(MESH)}
    assert all(set(d[w]) == {tuple(k) for k in d["kpoint_list"]} for w in ("pol_wfcn_dict", "np_wfcn_dict"))
    assert isinstance(next(iter(d["pol_wfcn_dict"].values()))["wfcn"], list) == (spin == 2)
    assert_unfolded_exactly(d, ibz, made)


def test_the_fixture_needs_the_weights(vasp_runs):
    """So that a parser not passing them on would be caught."""
    from berry_flux_diag import paw
    dirs, ibz, mult = vasp_runs("tetragonal")
    h = paw.WavecarHeader(dirs["pol"] / "WAVECAR")
    with pytest.raises(ValueError, match="do not describe the same run"):
        VASPParser.get_band_filling_from_wavecar_nospin(h, 4.0)
    assert VASPParser.get_band_filling_from_wavecar_nospin(h, 4.0, weights=mult / mult.sum()) == 2


def test_vasp_parser_refuses_weights_from_another_reduction(vasp_runs, monkeypatch):
    fake_paw(monkeypatch)
    dirs, ibz, _ = vasp_runs("tetragonal", weights=np.ones(len(pair_setup("tetragonal")[2])))
    with pytest.raises(ValueError, match="weights of the DFT run do not match"):
        run_vasp(dirs)


def test_vasp_parser_refuses_a_list_that_does_not_unfold(vasp_runs, monkeypatch):
    fake_paw(monkeypatch)
    dirs, _, _ = vasp_runs("tetragonal", drop_first=True)
    with pytest.raises(ValueError, match="do not form a full mesh"):
        run_vasp(dirs)


# --- qe_parser on an IBZ --------------------------------------------------

class FakePwDocument:
    """Stands in for qeschema.PwDocument: hands back prepared XML dicts."""
    docs = {}

    def read(self, filename, validation=None):
        self.filename = filename

    def to_dict(self, validation=None):
        return self.docs[self.filename]


def qe_run_dict(structure, kpoints, weights, spin, tails, nbnd=5):
    """The parts of a QE XML dict qe_parser reads, for an insulator with the
    given occupations (per k-point, the same in each channel)."""
    cell_bohr = structure.lattice.matrix / 0.529177210544
    alat = float(np.linalg.norm(cell_bohr[0]))
    recip = 2 * np.pi * np.linalg.inv(cell_bohr).T
    cart = np.asarray(kpoints) @ recip / (2 * np.pi / alat)
    ks = [{"k_point": {"$": k.tolist(), "@weight": float(w)},
           "eigenvalues": {"$": list(np.arange(nbnd) * 0.1) * spin},
           "occupations": {"$": list(tails[i]) * spin}}
          for i, (k, w) in enumerate(zip(cart, weights))]
    # 4 electrons: two full bands, counted twice non-spin or once per channel
    band = {"nelec": 4.0, "ks_energies": ks}
    if spin == 2:
        band.update(nbnd_up=nbnd, nbnd_dw=nbnd)
    atoms = [{"@name": site.specie.symbol, "$": (site.coords / 0.529177210544).tolist()}
             for site in structure]
    species = [{"@name": e} for e in dict.fromkeys(site.specie.symbol for site in structure)]
    return {"qes:espresso": {
        "input": {"spin": {"lsda": spin == 2, "noncolin": False, "spinorbit": False},
                  "atomic_species": {"species": species}},
        "output": {"atomic_structure": {"@alat": alat,
                                        "cell": dict(zip(("a1", "a2", "a3"), cell_bohr.tolist())),
                                        "atomic_positions": {"atom": atoms}},
                   "band_structure": band}}}


@pytest.mark.skipif(QEParser is None, reason="QEParser needs qeschema and h5py")
@pytest.mark.parametrize("name", PAIRS)
@pytest.mark.parametrize("spin", [1, 2])
def test_qe_parser_unfolds_an_ibz_run(tmp_path, monkeypatch, name, spin):
    from test_spin_polarized import write_qe_wfc

    pol, np_, ibz, mult, tails = pair_setup(name)
    rng = np.random.default_rng(0)
    miller = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]])
    tags = ["wfcup", "wfcdw"] if spin == 2 else ["wfc"]
    sources = {}
    for label, struct in (("pol", pol), ("np", np_)):
        save = tmp_path / f"{label}.save"
        save.mkdir()
        sources[label] = {}
        for i, k in enumerate(ibz):
            chans = []
            for tag in tags:
                c = rng.normal(size=(5, len(miller))) + 1j * rng.normal(size=(5, len(miller)))
                write_qe_wfc(save / f"{tag}{i + 1}.hdf5", c, miller)
                chans.append(c[:2])
            sources[label][tuple(np.around(k, 6))] = {"wfcn": chans if spin == 2 else chans[0],
                                                      "gvecs": miller}
        # QE normalises the weights to 2 non-spin, 1 spin-polarized
        FakePwDocument.docs[str(tmp_path / f"{label}.xml")] = qe_run_dict(
            struct, ibz, mult / mult.sum() * (2 if spin == 1 else 1), spin, tails)
    monkeypatch.setattr(QEParser.qeschema, "PwDocument", FakePwDocument)
    monkeypatch.setattr(QEParser, "PWOutput", lambda path: None)
    monkeypatch.setattr(QEParser, "get_zval_dict_from_PWOutput", lambda out: {})

    d = QEParser.qe_parser(str(tmp_path / "pol.xml"), str(tmp_path / "np.xml"),
                           str(tmp_path / "pol.save") + "/", str(tmp_path / "np.save") + "/", "out")
    assert d["spin_pol"] is (spin == 2)
    assert {tuple(k) for k in d["kpoint_list"]} == {tuple(k) for k in sym.wrap(MESH)}
    keys = ("pol_band_fill_up", "pol_band_fill_down") if spin == 2 else ("pol_band_fill",)
    assert [d[k] for k in keys] == [2] * len(keys)
    assert_unfolded_exactly(d, ibz, sources)
    # the occupations balance only with the multiplicities as weights
    with pytest.raises(ValueError, match="describe the same run|whole number"):
        (QEParser.get_band_filling_from_qeschema_spinpol if spin == 2
         else QEParser.get_band_filling_from_qeschema_nospin)(FakePwDocument.docs[str(tmp_path / "pol.xml")])
