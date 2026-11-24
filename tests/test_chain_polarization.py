"""Overlaps.chain_polarization: the polarization along a chain of images.

When the polar and non-polar structures are too far apart for one pair,
preprocess interpolates between them, and the polarization is summed along
the chain pol -> image 1 -> ... -> np. As in BFDJobs, only the electronic
part is summed over the pairs; the ionic part is computed once, from the
ends.

Two groups of tests:

- Synthetic, always run. Each pair's Berry-phase step is replaced by given
  electronic changes, so the bookkeeping is checked exactly: what is summed,
  what is taken from the ends, the conversion to a Cartesian vector, and the
  checks on the chain. The expected values are built independently, as
  P = (e / volume) (sum_i elec_i a_i + sum_sites Z dr).
- LiNbO3 in VASP, run when the data is present. The existing runs pair one
  polar structure with five rigid translations of the non-polar one (0.38 to
  1.74 A away), so a chain can be made through them. Point at the folder:

      export BFD_LINBO3_RUNS=/path/to/linbo3_ibz_interp_runs

  The direct pair pol -> np_trans_3 (0.79 A) lands one polarization quantum
  off; the chain pol -> np_trans_1 -> np_trans_3 (0.38 and 0.52 A) does not.
  About 2 minutes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

import berry_flux_diag as bfd
from berry_flux_diag import Overlaps as overlaps_module
from berry_flux_diag.constants import E_PER_ANG2_TO_MUC_PER_CM2
from berry_flux_diag.Overlaps import (chain_polarization, electronic_contrib, ionic_pol_change,
                                      polarization_cartesian, polarization_norm)

ZVAL = {"Ba": 10.0, "Ti": 12.0, "O": 6.0}


# --- synthetic ---------------------------------------------------------------

def skewed_lattice():
    """Non-orthogonal on purpose: the components along the unit lattice
    vectors are then not Cartesian, which is where a wrong conversion shows."""
    return Lattice.from_parameters(4.1, 4.3, 4.6, 78.0, 84.0, 66.0)


def chain_structures(n_images=3):
    lattice = skewed_lattice()
    species = ["Ba", "Ti", "O", "O", "O"]
    np_coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    # Ba sits next to the boundary, so its shortest displacement crosses it.
    shift = np.array([[0.98, 0, 0.01], [0.04, -0.03, 0.06], [-0.02, 0.01, -0.05],
                      [0.01, -0.04, -0.03], [-0.05, 0.02, 0.01]])
    pol = Structure(lattice, species, np.array(np_coords) + shift)
    npl = Structure(lattice, species, np_coords)
    return pol.interpolate(npl, n_images + 1)


def fake_parse(pol, npl, elec, spin_pol=False, zval=None):
    return {"pol_struct": pol, "np_struct": npl, "zval_dict": dict(zval or ZVAL),
            "spin_pol": spin_pol, "elec": list(elec)}


class StubOverlaps:
    """Stands in for Overlaps: returns the electronic change given in the
    parse dict instead of computing Berry phases."""

    def __init__(self, parse_dict, **kwargs):
        self.parse_dict = parse_dict

    def electronic_change(self):
        return list(self.parse_dict["elec"]), {"stub": True}


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(overlaps_module, "Overlaps", StubOverlaps)


def pairs(structs, elecs, **kwargs):
    return [fake_parse(a, b, e, **kwargs) for a, b, e in zip(structs[:-1], structs[1:], elecs)]


def expected_cartesian(pol, npl, elec_total):
    """(e / volume) (sum_i elec_i a_i + sum_sites Z dr), dr the shortest
    vector from each np site to its pol site."""
    lattice = pol.lattice
    dipole = sum(e * a for e, a in zip(elec_total, lattice.matrix))
    for p_site, n_site in zip(pol, npl):
        d = p_site.frac_coords - n_site.frac_coords
        d -= np.round(d)
        dipole = dipole + ZVAL[p_site.specie.symbol] * lattice.get_cartesian_coords(d)
    return E_PER_ANG2_TO_MUC_PER_CM2 / lattice.volume * dipole


def test_the_chain_is_the_electronic_sum_plus_the_ends_ionic_term(stubbed):
    structs = chain_structures()
    elecs = [[0.11, -0.05, 0.27], [0.09, -0.02, 0.31], [0.13, -0.07, 0.22], [0.10, -0.04, 0.25]]
    r = chain_polarization(pairs(structs, elecs))
    total = np.sum(elecs, axis=0)

    np.testing.assert_allclose(r["polarization_cartesian"],
                               expected_cartesian(structs[0], structs[-1], total),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(r["elec_contrib"], electronic_contrib(total, structs[0].lattice),
                               rtol=1e-14)
    np.testing.assert_array_equal(r["ion_contrib"],
                                  ionic_pol_change(structs[0], structs[-1], ZVAL))
    assert r["polarization_norm"] == pytest.approx(np.linalg.norm(r["polarization_cartesian"]),
                                                   rel=1e-15)
    assert [list(e) for e in r["elec_change_per_pair"]] == elecs
    assert r["debug_per_pair"] == [{"stub": True}] * len(elecs)


def test_on_a_straight_chain_the_ends_ionic_term_is_the_sum_of_the_pairs():
    """Why the ionic part can be taken from the ends: with one lattice and
    straight-line images, every site's displacements add up to its total."""
    structs = chain_structures(n_images=5)
    per_pair = sum(ionic_pol_change(a, b, ZVAL) for a, b in zip(structs[:-1], structs[1:]))
    np.testing.assert_allclose(per_pair, ionic_pol_change(structs[0], structs[-1], ZVAL),
                               rtol=1e-12, atol=1e-12)


def test_a_one_pair_chain_is_one_pair(stubbed):
    structs = chain_structures(n_images=0)
    elec = [0.3, -0.1, 0.2]
    r = chain_polarization(pairs(structs, [elec]))
    components = (ionic_pol_change(structs[0], structs[1], ZVAL)
                  + electronic_contrib(elec, structs[0].lattice))
    np.testing.assert_array_equal(r["polarization_components"], components)
    assert r["polarization_norm"] == polarization_norm(components, structs[0].lattice)


def test_a_generator_of_pairs_is_accepted(stubbed):
    structs = chain_structures()
    elecs = [[0.1, 0.2, 0.3]] * 4
    from_list = chain_polarization(pairs(structs, elecs))
    from_gen = chain_polarization(p for p in pairs(structs, elecs))
    assert from_gen["polarization_norm"] == from_list["polarization_norm"]


def test_the_cartesian_vector_sums_along_the_unit_lattice_vectors():
    lattice = skewed_lattice()
    components = np.array([3.0, -1.0, 2.0])
    unit = lattice.matrix / np.linalg.norm(lattice.matrix, axis=1)[:, None]
    np.testing.assert_allclose(polarization_cartesian(components, lattice), components @ unit,
                               rtol=1e-15)
    # The components' own norm is not |P| in a skewed cell (BFDJobs used it).
    assert abs(np.linalg.norm(components) - polarization_norm(components, lattice)) > 0.1


def test_in_an_orthogonal_cell_the_components_are_cartesian():
    lattice = Lattice.orthorhombic(4.0, 4.1, 4.4)
    components = np.array([3.0, -1.0, 2.0])
    np.testing.assert_allclose(polarization_cartesian(components, lattice), components,
                               rtol=0, atol=1e-15)


def test_an_empty_chain_raises(stubbed):
    with pytest.raises(ValueError, match="no pairs"):
        chain_polarization([])


def test_pairs_that_do_not_share_an_image_raise(stubbed):
    structs = chain_structures()
    ps = pairs(structs, [[0.1, 0.1, 0.1]] * 4)
    ps[1], ps[2] = ps[2], ps[1]
    with pytest.raises(ValueError, match="pair 0's second structure is not pair 1's first"):
        chain_polarization(ps)


def test_a_pair_parsed_backwards_raises(stubbed):
    structs = chain_structures()
    ps = pairs(structs, [[0.1, 0.1, 0.1]] * 4)
    ps[1] = fake_parse(ps[1]["np_struct"], ps[1]["pol_struct"], [0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="pair 0's second structure"):
        chain_polarization(ps)


def test_mixed_spin_polarization_raises(stubbed):
    structs = chain_structures()
    ps = pairs(structs, [[0.1, 0.1, 0.1]] * 4)
    ps[2]["spin_pol"] = True
    with pytest.raises(ValueError, match="spin polarization"):
        chain_polarization(ps)


def test_different_valence_charges_raise(stubbed):
    structs = chain_structures()
    ps = pairs(structs, [[0.1, 0.1, 0.1]] * 4)
    ps[3]["zval_dict"] = {**ZVAL, "Ti": 4.0}
    with pytest.raises(ValueError, match="valence charges"):
        chain_polarization(ps)


# --- LiNbO3 in VASP ----------------------------------------------------------

RUNS_ENV = "BFD_LINBO3_RUNS"

# Polarizations (uC/cm^2) from the existing VASP runs (6.3.2, PBE, 5x5x5),
# captured on branch ibz_support. Direct pairs, then chains.
PINNED = {
    ("pol", "np1"): 84.28625569574459,
    ("pol", "np2"): 84.30668995967271,
    ("pol", "np3"): 181.1100326947478,        # one quantum off: see below
    ("pol", "np1", "np2"): 84.30668995967284,
    ("pol", "np1", "np3"): 84.30627505520002,
}
PINNED_ATOL = 1e-6


@pytest.fixture(scope="module")
def linbo3() -> Path:
    path = os.environ.get(RUNS_ENV)
    if not path:
        pytest.skip(f"set {RUNS_ENV} to the linbo3_ibz_interp_runs folder to run these")
    path = Path(path) / "VASP_io"
    if not (path / "pol_orig" / "WAVECAR").is_file():
        pytest.skip(f"{path} has no pol_orig/WAVECAR")
    return path


@pytest.fixture(scope="module")
def chain(linbo3):
    """chain(names) -> chain_polarization's output, computed once per chain.
    A two-name chain is a direct pair."""
    out = {}

    def run_dir(name):
        return linbo3 / ("pol_orig" if name == "pol" else f"np_trans_{name[2:]}")

    def pair(a, b):
        p, n = run_dir(a), run_dir(b)
        return bfd.VASPParser.vasp_parser(p / "POSCAR", n / "POSCAR", p / "WAVECAR",
                                          n / "WAVECAR", p / "POTCAR", p, n)

    def get(*names):
        if names not in out:
            r = chain_polarization(pair(a, b) for a, b in zip(names[:-1], names[1:]))
            r.pop("debug_per_pair")
            out[names] = r
        return out[names]
    get.pair = pair
    return get


def test_a_one_pair_chain_is_compute_polarization(linbo3, chain):
    d = chain.pair("pol", "np1")
    direct, _ = bfd.Overlaps.Overlaps(d).compute_polarization()
    assert chain_polarization([d])["polarization_norm"] == float(direct)


@pytest.mark.parametrize("names", list(PINNED))
def test_pinned(chain, names):
    np.testing.assert_allclose(chain(*names)["polarization_norm"], PINNED[names],
                               rtol=0, atol=PINNED_ATOL)


def test_a_chain_through_a_translation_is_the_direct_pair(chain):
    """np_trans_1 -> np_trans_2 is a rigid translation, which multiplies each
    overlap by phases and so cannot change the Berry phases: the chain and
    the direct pair (0.64 A, on the right branch) agree exactly but for
    rounding. Observed: 1.3e-13."""
    a = chain("pol", "np1", "np2")["polarization_cartesian"]
    b = chain("pol", "np2")["polarization_cartesian"]
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)


def test_a_chain_recovers_the_branch_a_long_pair_misses(linbo3, chain):
    """pol -> np_trans_3 is 0.79 A: the direct pair is one polarization
    quantum (2 e |a_i| / volume) off along each lattice direction. Through
    np_trans_1 the steps are 0.38 and 0.52 A, and the chain agrees with the
    right-branch pairs. Observed: 4e-4 from pol -> np_trans_2."""
    good = chain("pol", "np1", "np3")
    assert abs(good["polarization_norm"] - chain("pol", "np2")["polarization_norm"]) < 1e-3

    direct = chain("pol", "np3")
    lattice = Structure.from_file(linbo3 / "pol_orig" / "POSCAR").lattice
    quantum = 2 * E_PER_ANG2_TO_MUC_PER_CM2 * np.array(lattice.abc) / lattice.volume
    off_by = (direct["polarization_components"] - good["polarization_components"]) / quantum
    np.testing.assert_allclose(off_by, np.round(off_by), rtol=0, atol=5e-3)
    assert np.abs(np.round(off_by)).sum() > 0
