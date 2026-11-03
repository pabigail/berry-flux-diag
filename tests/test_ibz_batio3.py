"""IBZ unfolding against the BaTiO3 DFT runs, QE and VASP.

The full-BZ runs in batio3_spin_ibz_runs hold the DFT's own states at every
mesh point. Unfolding just their 18 IBZ states with the shared symmetry
must reproduce the states at all the others: the occupied subspaces agree
to the DFT's convergence, where a wrong operation would put them an order
of magnitude apart. Point at the runs as for test_spin_batio3:

    export BFD_BATIO3_SPIN_RUNS=/path/to/batio3_spin_ibz_runs

Measured when this was written, as 1 - the smallest singular value of the
overlap between the two occupied subspaces, worst over the mesh: QE 5e-9,
VASP 1e-6 (VASP's EDIFF is 1e-5; QE's conv_thr is 1e-8 Ry).
"""

from __future__ import annotations

import numpy as np
import pytest

import berry_flux_diag as bfd
from berry_flux_diag import symmetry as sym
from test_spin_batio3 import parse_qe, parse_vasp, runs  # noqa: F401  (runs is a fixture)

WORST_GAP = {"QE": 1e-8, "VASP": 1e-5}


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
