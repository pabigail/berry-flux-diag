"""Unit tests for paw.WavecarHeader, which replaced pymatgen's Wavecar in
VASPParser.

VASPParser used to read each WAVECAR in full with pymatgen - 4.6 s per file
for BaTiO3 - only for its k-points, occupations, spin count and type,
discarding the coefficients. WavecarHeader reads just the header records.
It must give the same answers from the same bytes, so these write small
synthetic WAVECARs in VASP's binary layout and read each one both ways.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.io.vasp.outputs import Wavecar

from berry_flux_diag import paw, utils

LATTICE = np.array([[3.9, 0.0, 0.0], [0.1, 4.0, 0.0], [0.0, 0.2, 4.2]])
ENCUT = 120.0


def full_sphere_count(k):
    b, nbmax = paw._setup(ENCUT, LATTICE)
    return len(paw._g_sphere(np.asarray(k, float), nbmax, b, ENCUT))


def write_wavecar(path, kpoints, nband=4, spin=1, kind="std", tag=45200, seed=0):
    """A WAVECAR in VASP's layout: a header record, a record with the
    counts, cutoff and lattice, then per spin and k-point a record of
    plane-wave count, k, and (Re E, Im E, occupation) per band, followed by
    one coefficient record per band."""
    rng = np.random.default_rng(seed)
    counts = []
    for k in kpoints:
        full = full_sphere_count(k)
        counts.append({"std": full, "ncl": 2 * full, "gam": (full + 1) // 2}[kind])
    recl8 = max(13, 4 + 3 * nband, max(counts))
    records = []

    def record(values, dtype=np.float64):
        buf = np.zeros(recl8, dtype=np.float64)
        arr = np.asarray(values, dtype=dtype)
        buf.view(dtype)[: len(arr)] = arr
        records.append(buf)

    record([recl8 * 8, spin, tag])
    record([len(kpoints), nband, ENCUT, *LATTICE.ravel(), 0.0])
    band_energy = []
    for s in range(spin):
        for k, nplane in zip(kpoints, counts):
            # Occupations fall with band index from a full lowest band, as
            # in a real run, so the band filling is well defined.
            occ = np.sort(rng.uniform(0, 1, size=nband))[::-1]
            occ[0] = 1.0
            e = np.column_stack([np.sort(rng.normal(size=nband)), np.zeros(nband), occ])
            band_energy.append(e)
            record([nplane, *k, *e.ravel()])
            for _ in range(nband):
                c = (rng.normal(size=nplane) + 1j * rng.normal(size=nplane)) * 1e-3
                record(c.astype(np.complex64), dtype=np.complex64)
    np.concatenate(records).tofile(path)
    return band_energy


KPTS = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.25]]


def assert_same_as_pymatgen(path, h):
    w = Wavecar(str(path))
    assert h.vasp_type == w.vasp_type
    assert (h.spin, h.nk, h.nb) == (w.spin, w.nk, w.nb)
    assert h.encut == w.encut
    np.testing.assert_array_equal(h.a, w.a)
    assert len(h.kpoints) == len(w.kpoints)
    for a, b in zip(h.kpoints, w.kpoints):
        np.testing.assert_array_equal(a, b)
    if h.spin == 1:
        energies = [(h.band_energy, w.band_energy)]
    else:
        energies = list(zip(h.band_energy, w.band_energy))
    for mine, theirs in energies:
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            np.testing.assert_array_equal(a, b)


# --- the same answers as pymatgen, from the same bytes --------------------

def test_standard_wavecar_matches_pymatgen(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS)
    assert_same_as_pymatgen(f, paw.WavecarHeader(f))


def test_spin_polarized_wavecar_matches_pymatgen(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS, spin=2)
    h = paw.WavecarHeader(f)
    assert h.spin == 2 and len(h.band_energy) == 2 and len(h.band_energy[1]) == len(KPTS)
    assert_same_as_pymatgen(f, h)


def test_vasp6_single_precision_tag_matches_pymatgen(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS, tag=53300)
    assert_same_as_pymatgen(f, paw.WavecarHeader(f))


def test_noncollinear_is_detected_as_pymatgen_does(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS, kind="ncl")
    h = paw.WavecarHeader(f)
    assert h.vasp_type == "ncl"
    assert h.vasp_type == Wavecar(str(f)).vasp_type


def test_gamma_only_is_detected_as_pymatgen_does(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, [[0.0, 0.0, 0.0]], kind="gam")
    h = paw.WavecarHeader(f)
    assert h.vasp_type == "gam"
    assert h.vasp_type == Wavecar(str(f)).vasp_type


def test_header_values_are_the_ones_written(tmp_path):
    f = tmp_path / "WAVECAR"
    written = write_wavecar(f, KPTS, nband=5)
    h = paw.WavecarHeader(f)
    assert (h.spin, h.nk, h.nb, h.encut) == (1, 3, 5, ENCUT)
    np.testing.assert_array_equal(h.a, LATTICE)
    np.testing.assert_array_equal(np.array(h.kpoints), np.array(KPTS))
    for a, b in zip(h.band_energy, written):
        np.testing.assert_array_equal(a, b)


# --- what VASPParser does with it ------------------------------------------

@pytest.mark.parametrize("kind", ["ncl", "gam"])
def test_unsupported_types_are_rejected_by_the_existing_guard(tmp_path, kind):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, [[0.0, 0.0, 0.0]], kind=kind)
    with pytest.raises(NotImplementedError):
        utils.check_wavecar_type(paw.WavecarHeader(f).vasp_type, "polar")


def test_a_plane_wave_count_matching_no_type_is_rejected(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS)
    raw = np.fromfile(f, dtype=np.float64)
    recl8 = int(raw[0]) // 8
    raw[2 * recl8] += 1          # first k-point: one plane wave too many
    raw.tofile(f)
    h = paw.WavecarHeader(f)
    assert h.vasp_type.startswith("unrecognised")
    with pytest.raises(ValueError, match="unrecognised WAVECAR type"):
        utils.check_wavecar_type(h.vasp_type, "polar")


def test_occupations_feed_the_band_filling_unchanged(tmp_path):
    """VASPParser reads band_energy[k][:, 2]; the layout must match."""
    from berry_flux_diag import VASPParser

    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS, nband=6)
    h, w = paw.WavecarHeader(f), Wavecar(str(f))
    assert VASPParser.get_band_filling_from_wavecar_nospin(h, 0.3) == \
        VASPParser.get_band_filling_from_wavecar_nospin(w, 0.3)


# --- precision tags --------------------------------------------------------

@pytest.mark.parametrize("tag", [45210, 53310])
def test_double_precision_is_rejected(tmp_path, tag):
    """pawpyseed read these as single precision, silently."""
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS, tag=tag)
    with pytest.raises(ValueError, match="double-precision"):
        paw.WavecarHeader(f)
    with pytest.raises(ValueError, match="double-precision"):
        paw.read_wavecar(f)


def test_an_unknown_precision_tag_is_rejected(tmp_path):
    f = tmp_path / "WAVECAR"
    write_wavecar(f, KPTS, tag=12345)
    with pytest.raises(ValueError, match="unrecognised WAVECAR precision tag"):
        paw.WavecarHeader(f)
