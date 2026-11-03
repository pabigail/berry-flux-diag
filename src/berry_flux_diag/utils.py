import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


def direction_to_vals(direction):
    # THIS CODE DIRECTLY FROM BONINI
    if direction == 'z':
        comps = (0, 1)
        dir_comp = 2
        gvec = np.array([0, 0, 1])
    elif direction == 'x':
        comps = (1, 2)
        dir_comp = 0
        gvec = np.array([1, 0, 0])
    elif direction == 'y':
        comps = (0, 2)
        dir_comp = 1
        gvec = np.array([0, 1, 0])
    else:
        raise ValueError('Direction must be x, y, or z')
    return comps, dir_comp, gvec


def _tolerance_labels(values, tol):
    """Label values so that any two within tol share a label.

    Values are clustered by sorting and cutting wherever the gap to the
    next value exceeds tol, so labels increase with value. Returns one
    label per input position.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")

    labels = np.empty(len(values), dtype=np.intp)
    labels[order[0]] = 0
    label = 0
    for previous, current in zip(order[:-1], order[1:]):
        if values[current] - values[previous] > tol:
            label += 1
        labels[current] = label

    return labels


def get_strings(kpoints, direction, tol=1e-5):
    """Group k-points into strings running along one reciprocal axis.

    Each string is the list of k-points sharing the two transverse
    coordinates, ordered along `direction`, with a final point wrapped by
    one reciprocal lattice vector so the string closes.

    Parameters
    ----------
    tol : float
        Two transverse coordinates closer than this count as the same.

    Raises
    ------
    ValueError
        If kpoints is not an (n, 3) array of coordinates.
        If the strings are not all the same length, which means the
        k-points do not form a regular mesh and the Berry flux summed
        over them would be meaningless.

    An empty kpoints returns an empty list; compute_string_sums turns
    that into its own error naming the direction.

    Notes
    -----
    This used to build the set of transverse points with exact float
    equality and then collect members with a 1e-5 tolerance. The two
    disagreed: coordinates differing by less than the tolerance but not
    bit-equal produced *two* set entries, and each of them then matched
    the k-points of both. The result was duplicate strings that each
    double-counted their k-points, which compute_string_sums summed and
    divided by the inflated string count - a wrong number, not an error.
    Verified on four k-points along one line, one of them offset 1e-7
    transversely, where the old code returned two identical strings of
    length five instead of one (four k-points plus the wrapped copy).

    On every regular mesh, where the two never disagreed, the result is
    unchanged: same strings, same order, same arrays. That was checked
    against the old implementation for 35 meshes in all three directions,
    including the BaTiO3 VASP k-point list.

    Clustering once per axis makes the grouping and the membership test
    the same operation, so they cannot disagree. It also replaces the
    rescan of every k-point per transverse point - O(n^5) for an n^3 mesh
    - with a single sort.
    """
    comps, dir_comp, gvec = direction_to_vals(direction)

    kpts = np.asarray(kpoints, dtype=float)
    if kpts.size == 0:
        return []
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(
            f"expected k-points with shape (n, 3), got {kpts.shape}")

    # Label along each transverse axis, then group by the pair. Labels
    # increase with coordinate, so sorting on them orders the strings the
    # same way sorting on the coordinates themselves would.
    labels = list(zip(_tolerance_labels(kpts[:, comps[0]], tol),
                      _tolerance_labels(kpts[:, comps[1]], tol)))

    members_by_label = {}
    for index, label in enumerate(labels):
        members_by_label.setdefault(label, []).append(index)

    strings = []
    for label in sorted(members_by_label):
        members = members_by_label[label]
        members.sort(key=lambda index: kpts[index, dir_comp])
        string = [kpts[index] for index in members]
        string.append(string[0] + gvec)
        strings.append(string)

    lengths = {len(string) for string in strings}
    if len(lengths) > 1:
        raise ValueError(
            f"k-point strings along {direction} have differing lengths "
            f"{sorted(lengths)}; the k-points do not form a regular mesh, so "
            f"the Berry flux summed over them would be meaningless"
        )

    # Every k-point must appear in exactly one string. The last point of
    # each string is the wrapped copy, so it does not count. This is the
    # check that would have caught the duplication described above: it
    # produced one extra string per affected line, each a copy of another,
    # which over-weighted that line in the average compute_polarization
    # takes. All the strings were the same length, so the test above would
    # not have noticed.
    #
    # With the grouping above it cannot fire - each k-point is placed in
    # exactly one label group by construction - so it is an invariant
    # check, kept so that a later change to the grouping cannot quietly
    # bring the duplication back. It is not a duplicate-input check:
    # repeated k-points are each placed, and are rejected upstream by
    # check_full_bz.
    placed = sum(len(string) - 1 for string in strings)
    if placed != len(kpts):
        raise ValueError(
            f"the {len(strings)} k-point strings along {direction} hold "
            f"{placed} k-points between them, but {len(kpts)} were given; "
            f"every k-point must belong to exactly one string"
        )

    return strings


def empty_parse_dict():
    # this is the common dictionary that interfaces with VASP & QE
    
    keys = ['pol_struct',
            'np_struct',
            'kpoint_list',
            'spin_pol',
            'pol_band_fill',
            'np_band_fill',
            'pol_wfcn_dict',
            'np_wfcn_dict',
            'zval_dict',
            'ES_code']
    
    empty_parse_dict = dict([(key, None) for key in keys])
    
    return empty_parse_dict
    


def check_gvecs_same(gvecs0, gvecs1):
    ''' 
    check if two sets of gvectors are the same
    '''
    TOL = 1e-6
    if len(gvecs0) != len(gvecs1):
        return False
    else:
        return np.allclose(gvecs0, gvecs1, atol=TOL)


def gvec_extrema(gvecs):
    # code adapted from Stephen Gant WF_unfold.py
    #
    # This used to also return gvec_range and a gvec_base_exp of
    # max(gvec_range) + 1. Neither was ever read, and the second was a
    # trap: it is the right base only for an encoding shifted by -min,
    # whereas gvec_to_index shifts by +max_rad, so adopting it would have
    # produced index collisions rather than an error.
    gvec_min = np.min(gvecs, axis=0)
    gvec_max = np.max(gvecs, axis=0)
    gvec_len = len(gvecs)
    return gvec_min, gvec_max, gvec_len


def gvec_to_index(gvec, kpt, max_rad, base):
    # gvec: np.array [x, y, z] of one gvec, or an (n, 3) array of them
    # kpt: accepted but NOT applied, see below
    # max_rad: maximum radius of all gvecs considered
    # base: base of expansion for x*base**2 + y*base + z
    #
    # Returns an int for one gvec and an int64 array for many. np.rint
    # rounds half to even, as the builtin round() this used to call does,
    # so the indices are unchanged; for integer gvecs it is exact either way.

    # The convention in force is that the index labels G alone, not G + k.
    # Whether it should instead be centred on the k-point is unresolved; the
    # shift is left commented out rather than removed so the question stays
    # visible. Both structures are indexed the same way, so a consistent
    # choice cancels in the overlap - which is why this has not shown up as
    # a wrong answer. Pin the convention with a test before changing it.
    kpt_centered_gvec = np.asarray(gvec) # + kpt
    shifted_gvec = np.rint(kpt_centered_gvec + max_rad).astype(np.int64)
    index = (shifted_gvec[..., 0]*base**2 + shifted_gvec[..., 1]*base
             + shifted_gvec[..., 2])
    return int(index) if index.ndim == 0 else index


def _scatter_columns(coeffs, indices, width):
    """Place column i of coeffs at column indices[i] of a zero matrix.

    Where an index repeats, the last column wins, as it did when this was a
    Python loop assigning one column at a time. A G list should not repeat
    a vector, but if one does the result is unchanged rather than whatever
    order numpy's fancy assignment happens to use.
    """
    coeffs = np.asarray(coeffs)
    mapped = np.zeros((len(coeffs), width), dtype=complex)
    reversed_indices = indices[::-1]
    unique, first_in_reversed = np.unique(reversed_indices, return_index=True)
    last = len(indices) - 1 - first_in_reversed
    mapped[:, unique] = coeffs[:, last]
    return mapped


def map_coeffs(coeffs0, gvecs0, kpt0, coeffs1, gvecs1, kpt1):
    """Lay two coefficient sets out on one shared G-vector index.

    Each state's coefficients are scattered into a zero matrix whose
    columns are labelled by gvec_to_index, so column c means the same G
    vector in both and a plain inner product pairs them correctly. A G
    vector present in only one state meets a zero in the other.
    """
    gvec_min0, gvec_max0, _ = gvec_extrema(gvecs0)
    gvec_min1, gvec_max1, _ = gvec_extrema(gvecs1)

    max_g_rad = np.max([np.max(gvec_max0), np.max(gvec_max1)])
    min_g_rad = np.abs(np.min([np.min(gvec_min0), np.min(gvec_min1)]))
    max_rad = int(np.max([max_g_rad, min_g_rad]))
    # Every shifted component lands in [0, 2*max_rad], so this base makes
    # gvec_to_index's x*base**2 + y*base + z encoding collision-free.
    base = (2*max_rad + 1)
    
    width = base**3 + base**2 + base + 1

    coeffs0_mapped = _scatter_columns(
        coeffs0, gvec_to_index(np.asarray(gvecs0), kpt0, max_rad, base), width)
    coeffs1_mapped = _scatter_columns(
        coeffs1, gvec_to_index(np.asarray(gvecs1), kpt1, max_rad, base), width)

    return coeffs0_mapped, coeffs1_mapped

def extract_letters(input_string):
    # Use a regular expression to match only the letters at the beginning of the string
    match = re.match(r'^[A-Za-z]+', input_string)
    if match:
        return match.group(0)
    return ''


def check_kpoints_match(pol_kpoints, np_kpoints, tol=1e-5):
    """Require both runs to sample the same k-points in the same order.

    The README states that the two calculations must use the same k-point
    mesh, but nothing enforced it. The parsers build both wavefunction
    dictionaries from a single k-point list - the polar one - while reading
    each run's coefficients positionally, so a difference in ordering would
    quietly label one run's coefficients with the other run's k-points and
    produce a plausible, wrong answer rather than an error.

    Raises
    ------
    ValueError
        If the two lists differ in length, or if any pair differs by more
        than tol in any component.
    """
    if len(pol_kpoints) != len(np_kpoints):
        raise ValueError(
            f"the two runs have different numbers of k-points: "
            f"{len(pol_kpoints)} in the polar run, {len(np_kpoints)} in the "
            f"non-polar run; both must use the same mesh"
        )

    pol = np.asarray(pol_kpoints, dtype=float)
    nonpol = np.asarray(np_kpoints, dtype=float)

    mismatched = np.flatnonzero(np.abs(pol - nonpol).max(axis=1) > tol)
    if mismatched.size:
        first = mismatched[0]
        raise ValueError(
            f"the two runs sample different k-points: at index {first} the "
            f"polar run has {pol[first]} and the non-polar run has "
            f"{nonpol[first]} ({mismatched.size} of {len(pol)} differ). Both "
            f"runs must use the same mesh, in the same order."
        )


def check_species_match(pol_struct, np_struct):
    """Require the same atoms in the same order in both structures.

    Every site is paired with the site at the same index when computing
    displacements and interpolating, so a different ordering pairs the
    wrong atoms and a different count pairs some atom with nothing.

    Raises
    ------
    ValueError
        If the structures differ in atom count or in species order.
    """
    pol_species = [str(site.specie) for site in pol_struct]
    np_species = [str(site.specie) for site in np_struct]

    if len(pol_species) != len(np_species):
        raise ValueError(
            f"the two structures have different numbers of atoms: "
            f"{len(pol_species)} in the polar structure, {len(np_species)} in "
            f"the non-polar structure"
        )

    for index, (pol_specie, np_specie) in enumerate(zip(pol_species, np_species)):
        if pol_specie != np_specie:
            raise ValueError(
                f"the two structures list different species at site {index}: "
                f"{pol_specie} in the polar structure, {np_specie} in the "
                f"non-polar structure. Both must list the same atoms in the "
                f"same order."
            )


def check_full_bz(kpoint_list, tol=1e-5):
    """Require a k-point set covering the whole Brillouin zone.

    The Berry flux is summed over closed strings that wrap the zone, so an
    irreducible wedge does not merely lose accuracy - get_strings builds
    strings that do not close, and the result is meaningless rather than
    approximate.

    The test is that the number of k-points equals the product of the
    distinct values found along each axis, which holds for a regular mesh
    over the full zone and fails for a wedge. That is a necessary
    condition rather than a sufficient one: it is meant to catch a
    symmetry-reduced run, not to validate an arbitrary k-point set.

    Raises
    ------
    ValueError
        If the k-points do not form a complete regular mesh.
    """
    if len(kpoint_list) == 0:
        raise ValueError("the k-point list is empty")

    kpoints = np.asarray(kpoint_list, dtype=float)
    decimals = max(0, int(round(-np.log10(tol))))

    mesh = [len(np.unique(np.round(kpoints[:, axis], decimals))) for axis in range(3)]
    expected = int(np.prod(mesh))

    if expected != len(kpoints):
        raise ValueError(
            f"the k-points do not span the full Brillouin zone: found "
            f"{len(kpoints)} k-points, but the distinct values along each axis "
            f"imply a {mesh[0]}x{mesh[1]}x{mesh[2]} mesh of {expected}. This is "
            f"what a symmetry-reduced run looks like. Rerun with ISYM = -1 "
            f"(VASP) or nosym = .true. and noinv = .true. (Quantum ESPRESSO)."
        )


def check_wavecar_type(vasp_type, label=""):
    """Require a WAVECAR from the standard VASP build.

    Parameters
    ----------
    vasp_type : str
        The type pymatgen determined, one of "std", "gam" or "ncl" - that
        is, ``Wavecar.vasp_type`` after construction, not the argument
        passed in. Constructing with ``vasp_type=None`` lets pymatgen
        detect it from the plane-wave count.
    label : str, optional
        Which run this is, for the error message.

    Raises
    ------
    NotImplementedError
        For a gamma-only or a noncollinear WAVECAR.

    Notes
    -----
    This used to be moot, because the type was pinned to "std" at the call
    site rather than detected. Pinning it does not make a WAVECAR standard;
    for a noncollinear one it makes the file parse *incorrectly and
    silently*. pymatgen accepts the pinned type whenever the plane-wave
    count is either the G-point count or twice it, and a noncollinear
    WAVECAR satisfies the second, so the check passes. It then skips the
    reshape to (2, nplane // 2) that a "ncl" file needs, and the two spinor
    components are handed downstream as one flat array of scalar
    coefficients, twice as long as the G-vector list it is indexed against.
    Nothing raises. The polarization is simply wrong.
    """
    where = f" for the {label} run" if label else ""

    # Tested before the first-letter match below, which "None" would
    # otherwise satisfy as a noncollinear run.
    if not isinstance(vasp_type, str):
        raise ValueError(
            f"unrecognised WAVECAR type {vasp_type!r}{where}; expected one of "
            f"'std', 'gam' or 'ncl'. pymatgen sets Wavecar.vasp_type during "
            f"construction, so None here means the WAVECAR was never read."
        )

    kind = vasp_type.lower()[:1]

    if kind == "g":
        raise NotImplementedError(
            f"the WAVECAR{where} is from the gamma-only VASP build (vasp_gam), "
            f"which samples the single k-point at Gamma. A Berry flux is a sum "
            f"over strings of k-points that wrap the Brillouin zone, so one "
            f"k-point cannot give a polarization. Rerun with the standard "
            f"build on a full k-point mesh, with ISYM = -1."
        )

    if kind == "n":
        raise NotImplementedError(
            f"the WAVECAR{where} is noncollinear (vasp_ncl), which "
            f"berry_flux_diag does not support: it stores a two-component "
            f"spinor per plane wave, and the overlaps computed here assume a "
            f"scalar wavefunction. Use a collinear calculation."
        )

    if kind != "s":
        raise ValueError(
            f"unrecognised WAVECAR type {vasp_type!r}{where}; expected one of "
            f"'std', 'gam' or 'ncl'"
        )


# How far the occupations may disagree with the electron count before the
# two are treated as describing different runs. Electron number is conserved
# by every smearing scheme, so a real run agrees to the precision VASP and QE
# write occupations with; this catches an electron count read from the wrong
# run or occupations on the wrong scale, not physics.
ELECTRON_COUNT_ATOL = 1e-3

# A spin channel's electron count is not stated in the output, so it is read
# off the channel's occupations, which must sum to a whole number. In an
# insulator they do, up to the smearing tails; a count further than this from
# an integer means a band is part filled.
SPIN_CHANNEL_ATOL = 1e-2

# Above this many electrons per k-point sitting in bands above the occupied
# manifold, the smearing is large enough to matter for the density, which is
# worth a warning even though it does not change the band count.
SMEARING_LEAK_WARN = 1e-3


def occupied_band_count(n_electrons, energies_by_kpoint, occupations_by_kpoint,
                        label=""):
    """Number of bands in the occupied manifold of one spin channel.

    For an insulator this is the channel's electron count, and that is what
    is returned. The occupations are only checked against it: with smearing,
    the lowest empty band has a small occupation near the band edges, so a
    count read from the occupations depends on where the threshold sits and
    on which k-points are looked at.

    Parameters
    ----------
    n_electrons : float
        Electrons in this channel: NELECT / 2 for a non-spin-polarized run,
        or the channel's count from spin_channel_electrons.
    energies_by_kpoint, occupations_by_kpoint : array_like, (nk, nbands)
        Band energies, ascending at each k-point, and occupations on a 0 to 1
        scale. The energies are only compared with each other, so any unit
        will do.
    label : str
        Names the run in error messages, e.g. "polar" or "non-polar, spin up".

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If the electron count is not a whole number, if the occupations do not
        sum to it, if the run has fewer bands than that, or if the highest
        occupied band overlaps the lowest empty one somewhere in the zone.
        Each of these means there is no isolated occupied manifold, and the
        Berry phase of one is not defined.
    """
    where = f" in the {label} run" if label else ""
    energies = np.asarray(energies_by_kpoint, dtype=float)
    occupations = np.asarray(occupations_by_kpoint, dtype=float)
    if energies.ndim != 2 or energies.shape != occupations.shape:
        raise ValueError(
            f"energies and occupations must both be (nk, nbands); got "
            f"{energies.shape} and {occupations.shape}{where}")

    n_bands = int(round(n_electrons))
    if abs(n_electrons - n_bands) > ELECTRON_COUNT_ATOL or n_bands < 1:
        raise ValueError(
            f"{n_electrons:g} electrons{where} do not fill a whole number of "
            f"bands, so the highest band is part filled: the system is "
            f"metallic and its polarization is not defined. A non-spin-"
            f"polarized run needs an even NELECT.")

    total = occupations.sum(axis=1).mean()
    if abs(total - n_electrons) > ELECTRON_COUNT_ATOL:
        raise ValueError(
            f"the occupations{where} sum to {total:.6f} electrons per k-point, "
            f"not the {n_electrons:g} expected: the electron count and the "
            f"occupations do not describe the same run")

    if n_bands > energies.shape[1]:
        raise ValueError(
            f"{n_bands} occupied bands expected{where}, but only "
            f"{energies.shape[1]} were computed")

    if n_bands < energies.shape[1]:
        # Without an empty band - QE's default for an insulator - there is
        # nothing to check the gap against, and fixed occupations assume it.
        gap = energies[:, n_bands].min() - energies[:, n_bands - 1].max()
        if gap <= 0:
            raise ValueError(
                f"band {n_bands + 1} dips below band {n_bands} by "
                f"{-gap:.4g}{where}: the occupied bands are not separated "
                f"from the empty ones by a gap, so the system is metallic "
                f"and its polarization is not defined")

        leak = occupations[:, n_bands:].sum(axis=1).max()
        if leak > SMEARING_LEAK_WARN:
            logger.warning(
                "smearing puts up to %.3g electrons per k-point above the "
                "gap%s; the band count is unaffected, but the density is not "
                "quite that of the insulator. A smaller smearing or the "
                "tetrahedron method avoids this.", leak, where)

    return n_bands


def spin_channel_electrons(nelect, occupations_up, occupations_down, label=""):
    """Electrons in each spin channel of a collinear spin-polarized run.

    NELECT gives only the total. The split is read off each channel's
    occupations, which must sum to whole numbers adding up to NELECT.

    Returns
    -------
    (int, int)
        Electrons in the up and down channels.
    """
    where = f" in the {label} run" if label else ""
    counts = []
    for name, occupations in (("up", occupations_up), ("down", occupations_down)):
        total = np.asarray(occupations, dtype=float).sum(axis=1).mean()
        count = int(round(total))
        if abs(total - count) > SPIN_CHANNEL_ATOL:
            raise ValueError(
                f"the spin-{name} occupations{where} sum to {total:.4f} "
                f"electrons, not a whole number: a band in that channel is "
                f"part filled, so the system is metallic and its "
                f"polarization is not defined")
        counts.append(count)

    if abs(sum(counts) - nelect) > ELECTRON_COUNT_ATOL:
        raise ValueError(
            f"the spin channels{where} hold {counts[0]} + {counts[1]} "
            f"electrons, not the NELECT of {nelect:g}")
    return counts[0], counts[1]
