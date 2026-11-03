"""Symmetry shared by the polar and non-polar structures, and k-point unfolding.

A run on the irreducible Brillouin zone (IBZ) has wavefunctions at one
k-point of each symmetry orbit of the mesh. Every other mesh point k' is
k' = s R k - G0 for an IBZ point k, a rotation R, s = +1 or -1 (time
reversal) and a reciprocal lattice vector G0 that brings it back into the
mesh's range. This module finds those operations and that mapping; the
wavefunctions themselves are unfolded elsewhere.

Berry flux diagonalization compares the polar and non-polar states at the
same k-points, so both runs must use one IBZ list, reduced with the
rotations the two structures share - in general fewer than either has. A
shared rotation can come with a different fractional translation in each
structure, so the translations are kept per structure.

Conventions, all fractional:
- A real-space operation {W|t} maps x to W x + t (spglib's).
- It maps a wavefunction at k to one at W^-T k; k_rotations holds W^-T.
- Time reversal (k -> -k) holds because spin-orbit coupling is not
  supported, and within each channel of a collinear spin-polarized run.

Magnetic runs: only operations that map every atom onto an atom with the
same moment are used. One that exchanged up and down moments would map
each spin channel onto the other, not onto itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib

from berry_flux_diag import utils

# spglib's tolerance, in Angstrom. Coordinates like 0.499999 in a 4 A cell
# are 4e-6 A off the symmetric position, so this has room to spare, and is
# still well below any physical distortion.
SYMPREC = 1e-3

# k-points are compared after rounding to this many decimals, as the
# parsers key their wavefunction dictionaries. Rounding is only for
# comparing: rotations are applied to unrounded k-points. In a hexagonal
# basis a rotation adds and subtracts coordinates, and the rounding errors
# of 1/6 and 1/3 at this precision then add up to a different last digit.
KPOINT_DECIMALS = 6


@dataclass
class SharedSymmetry:
    """Rotations both structures have, with each structure's translations.

    rotations[i] (3x3 int) is the real-space rotation W; k_rotations[i] is
    W^-T, its action on k; translations[label][i] is that structure's t.
    """
    rotations: np.ndarray
    k_rotations: np.ndarray
    translations: dict
    space_groups: dict

    def __len__(self):
        return len(self.rotations)


def _cell(structure, magmoms=None):
    """spglib cell whose atom types distinguish species and, when given,
    magnetic moments, so only moment-preserving operations survive."""
    labels = [site.specie.Z for site in structure]
    if magmoms is not None:
        magmoms = np.asarray(magmoms, dtype=float)
        if magmoms.shape != (len(structure),):
            raise ValueError(f"{len(magmoms)} magnetic moments for {len(structure)} atoms; "
                             "collinear moments only, one per atom")
        labels = list(zip(labels, np.round(magmoms, 3)))
    types = {label: i + 1 for i, label in enumerate(dict.fromkeys(labels))}
    return (structure.lattice.matrix, structure.frac_coords, [types[l] for l in labels])


def symmetry_operations(structure, magmoms=None, symprec=SYMPREC):
    """(rotations, translations, space group symbol) of one structure,
    restricted to operations that preserve every atom's magnetic moment."""
    cell = _cell(structure, magmoms)
    sym = spglib.get_symmetry(cell, symprec=symprec)
    if sym is None:
        raise ValueError("spglib could not determine the symmetry of the structure")
    return sym["rotations"], sym["translations"], spglib.get_spacegroup(cell, symprec=symprec)


def shared_symmetry(pol_struct, np_struct, pol_magmoms=None, np_magmoms=None,
                    symprec=SYMPREC):
    """The rotations the polar and non-polar structures share.

    Rotations are compared as integer matrices in each structure's own
    fractional basis, so the two cells must describe the same lattice
    vectors in the same order - as the parsers already require of the
    k-points. Where a structure has several translations for one rotation
    (a cell that is not primitive), the first is kept; any of them is a
    symmetry.
    """
    ops = {"pol": symmetry_operations(pol_struct, pol_magmoms, symprec),
           "np": symmetry_operations(np_struct, np_magmoms, symprec)}
    first = {}
    for label, (rots, trans, _) in ops.items():
        first[label] = {}
        for r, t in zip(rots, trans):
            first[label].setdefault(r.tobytes(), (r, t))
    keys = [k for k in first["pol"] if k in first["np"]]
    if not keys:
        raise ValueError("the polar and non-polar structures share no symmetry "
                         "operation, not even the identity: are their cells the same?")
    rotations = np.array([first["pol"][k][0] for k in keys])
    k_rotations = np.array([np.rint(np.linalg.inv(r).T).astype(int) for r in rotations])
    translations = {label: np.array([first[label][k][1] for k in keys]) for label in first}
    return SharedSymmetry(rotations, k_rotations, translations,
                          {label: op[2] for label, op in ops.items()})


def _wrapped(k):
    """k with every component brought into [-0.5, 0.5), not rounded. The
    floor is taken of k rounded to 10 decimals, so that 0.4999999999999998
    from a rotation lands at -0.5 with 0.5."""
    k = np.asarray(k, dtype=float)
    return k - np.floor(np.round(k, 10) + 0.5)


def wrap(k):
    """k brought into [-0.5, 0.5) and rounded to the precision the parsers
    key k-points with: for comparing and returning k-points, not for
    further arithmetic."""
    return np.round(_wrapped(k), KPOINT_DECIMALS) + 0.0   # + 0.0: no -0.0


@dataclass
class KpointMap:
    """How every full-mesh k-point is reached from the IBZ.

    full_kpoints[j] = s R ibz_kpoints[ibz_index[j]] - gshift[j], with
    R = k_rotations[op_index[j]] and s = -1 where time_reversal[j].
    multiplicity[i] is the number of full-mesh points IBZ point i stands for.
    """
    full_kpoints: np.ndarray
    ibz_index: np.ndarray
    op_index: np.ndarray
    time_reversal: np.ndarray
    gshift: np.ndarray
    multiplicity: np.ndarray


def unfold_kpoints(ibz_kpoints, k_rotations, time_reversal=True):
    """Unfold an IBZ list to the full mesh it reduces.

    ibz_kpoints should be unrounded, as the DFT code wrote them; the
    returned full_kpoints are rounded to KPOINT_DECIMALS.

    Each IBZ point's orbit under the rotations (and k -> -k) is generated,
    every point wrapped into [-0.5, 0.5). The result must be a complete
    regular mesh with every point reached from exactly one IBZ point;
    anything else means the list was not reduced with these operations.

    Raises
    ------
    ValueError
        If two IBZ points are equivalent, or if the orbits do not fill a
        regular mesh over the whole zone.
    """
    ibz = np.asarray(ibz_kpoints, dtype=float)
    signs = (1, -1) if time_reversal else (1,)
    owner, rows, mult = {}, [], []
    for i, k in enumerate(ibz):
        orbit = set()
        for o, rot in enumerate(k_rotations):
            for s in signs:
                image = s * (rot @ k)
                target = wrap(image)
                key = tuple(target)
                if key in owner and owner[key] != i:
                    raise ValueError(
                        f"IBZ k-points {owner[key]} ({ibz[owner[key]]}) and {i} ({k}) "
                        f"are equivalent under the shared symmetry: the list was not "
                        f"reduced with these operations")
                if key not in owner:
                    owner[key] = i
                    rows.append((target, i, o, s < 0, np.rint(image - target).astype(int)))
                orbit.add(key)
        mult.append(len(orbit))

    full = np.array([r[0] for r in rows])
    try:
        utils.check_full_bz(full)
    except ValueError as err:
        raise ValueError(
            f"the {len(ibz)} IBZ k-points unfold to {len(full)} points that do not "
            f"form a full mesh, so they were not reduced with the symmetry the polar "
            f"and non-polar structures share: generate the list with "
            f"symmetry.reduce_mesh. ({err})") from None
    return KpointMap(full_kpoints=full,
                     ibz_index=np.array([r[1] for r in rows]),
                     op_index=np.array([r[2] for r in rows]),
                     time_reversal=np.array([r[3] for r in rows]),
                     gshift=np.array([r[4] for r in rows]),
                     multiplicity=np.array(mult))


def reduce_mesh(mesh_kpoints, k_rotations, time_reversal=True):
    """The IBZ of a mesh: one point of each orbit, in mesh order.

    Returns (indices into mesh_kpoints, multiplicities). This is the list to
    give both the polar and non-polar NSCF runs.

    Raises
    ------
    ValueError
        If a rotation maps a mesh point off the mesh.
    """
    mesh = np.asarray(mesh_kpoints, dtype=float)
    index = {tuple(wrap(k)): j for j, k in enumerate(mesh)}
    if len(index) != len(mesh):
        raise ValueError("the mesh has repeated k-points")
    signs = (1, -1) if time_reversal else (1,)
    seen, reps, mult = set(), [], []
    for j, k in enumerate(mesh):
        if j in seen:
            continue
        orbit = set()
        for rot in k_rotations:
            for s in signs:
                key = tuple(wrap(s * (rot @ k)))
                if key not in index:
                    raise ValueError(f"a shared rotation maps mesh point {k} off the mesh")
                orbit.add(index[key])
        seen |= orbit
        reps.append(j)
        mult.append(len(orbit))
    return np.array(reps), np.array(mult)


def monkhorst_pack(n, shift=(0, 0, 0)):
    """A regular n1 x n2 x n3 mesh, k_i = (j + shift_i / 2) / n_i, wrapped
    into [-0.5, 0.5) but not rounded. shift 1 on an axis is a half step, as
    in QE's K_POINTS automatic; VASP's Monkhorst is a half step on even axes."""
    axes = [[(j + s / 2) / m for j in range(m)] for m, s in zip(n, shift)]
    return _wrapped(np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T)


def vasp_kpoints_text(kpoints, weights, comment="IBZ shared by the polar and non-polar structures"):
    """An explicit VASP KPOINTS file for an NSCF run (ICHARG = 11, ISYM = 0)."""
    lines = [comment, str(len(kpoints)), "Reciprocal"]
    lines += [f"{k[0]:.12f} {k[1]:.12f} {k[2]:.12f} {w}" for k, w in zip(kpoints, weights)]
    return "\n".join(lines) + "\n"


def qe_kpoints_text(kpoints, weights):
    """A QE K_POINTS crystal card for an NSCF run (nosym and noinv = .true.)."""
    lines = ["K_POINTS crystal", f"  {len(kpoints)}"]
    lines += [f"  {k[0]:.10f} {k[1]:.10f} {k[2]:.10f} {w}" for k, w in zip(kpoints, weights)]
    return "\n".join(lines) + "\n"
