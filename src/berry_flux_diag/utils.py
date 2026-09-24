import numpy as np
import re


def direction_to_vals(direction):
    # THIS CODE DIRECTLY FROM BONINI
    if direction == 'z':
        comps = (0, 1)
        dir_comp = 2
        gvec = [0, 0, 1]
    elif direction == 'x':
        comps = (1, 2)
        dir_comp = 0
        gvec = [1, 0, 0]
    elif direction == 'y':
        comps = (0, 2)
        dir_comp = 1
        gvec = [0, 1, 0]
    else:
        raise ValueError('Direction must be x, y, or z')
    return comps, dir_comp, gvec


def get_strings(kpoints, direction):
    """ given a list of kpoints (in reciprocal space) and a direction x, y, z
        return a list of strings of kpoints along the corresponding direction """
    # includes the extra kpoint shifted by a gvector
    # adapted from some very old code, super inefficient, should clean up
    # but it's also pretty much never a bottleneck
    
    # re-wrote directly from Bonini's code, removed dependency on abinit Kpoint structure

    comps, dir_comp, gvec = direction_to_vals(direction)
    bz_2d_set = sorted(set([tuple((kpt[i] for i in comps))
                            for kpt in kpoints]))
    strings = []
    for bz_2d_pt in bz_2d_set:
        this_string = []
        for kpt in kpoints:
            in_this_string = ((abs(kpt[comps[0]] - bz_2d_pt[0]) < 1.e-5)
                              and (abs(kpt[comps[1]] - bz_2d_pt[1]) < 1.e-5))
            if in_this_string:
                this_string.append(kpt)
        this_string.sort(key=lambda k: k[dir_comp])
        this_string.append(this_string[0] + gvec)
        
        strings.append(this_string)
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
    gvec_min = np.min(gvecs, axis=0)
    gvec_max = np.max(gvecs, axis=0)
    gvec_range = gvec_max - gvec_min
    gvec_len = len(gvecs)
    gvec_base_exp = np.max(gvec_range) + 1
    return gvec_min, gvec_max, gvec_range, gvec_len, gvec_base_exp


def gvec_to_index(gvec, kpt, max_rad, base):
    # gvec: np.array [x, y, z] of current gvec
    # kpt: accepted but NOT applied, see below
    # max_rad: maximum radius of all gvecs considered
    # base: base of expansion for x*base**2 + y*base + z

    # The convention in force is that the index labels G alone, not G + k.
    # Whether it should instead be centred on the k-point is unresolved; the
    # shift is left commented out rather than removed so the question stays
    # visible. Both structures are indexed the same way, so a consistent
    # choice cancels in the overlap - which is why this has not shown up as
    # a wrong answer. Pin the convention with a test before changing it.
    kpt_centered_gvec = gvec # + kpt
    shift = np.full(3, max_rad)
    shifted_gvec = kpt_centered_gvec + shift
    index = round(shifted_gvec[0])*base**2 + round(shifted_gvec[1])*base + round(shifted_gvec[2])
    return index


def map_coeffs(coeffs0, gvecs0, kpt0, coeffs1, gvecs1, kpt1):
    
    gvec_min0, gvec_max0, gvec_range0, gvec_len0, gvec_base_exp0 = gvec_extrema(gvecs0)
    gvec_min1, gvec_max1, gvec_range1, gvec_len1, gvec_base_exp1 = gvec_extrema(gvecs1)
    
    max_g_rad = np.max([np.max(gvec_max0), np.max(gvec_max1)])
    min_g_rad = np.abs(np.min([np.min(gvec_min0), np.min(gvec_min1)]))
    max_rad = int(np.max([max_g_rad, min_g_rad]))
    # Every shifted component lands in [0, 2*max_rad], so this base makes
    # gvec_to_index's x*base**2 + y*base + z encoding collision-free.
    base = (2*max_rad + 1)
    
    coeffs0_mapped = np.zeros((len(coeffs0), base**3 + base**2 + base + 1), dtype=complex)
    coeffs1_mapped = np.zeros((len(coeffs1), base**3 + base**2 + base + 1), dtype=complex)

    for i in range(gvec_len0):
        mapped_i = gvec_to_index(gvecs0[i], kpt0, max_rad, base)
        coeffs0_mapped[:, mapped_i] = coeffs0[:, i]
    for i in range(gvec_len1):
        mapped_i = gvec_to_index(gvecs1[i], kpt1, max_rad, base)
        coeffs1_mapped[:, mapped_i] = coeffs1[:, i]
        
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


def max_filled_bands(occupations_by_kpoint, tol):
    """Number of occupied bands, maximized over k-points.

    Parameters
    ----------
    occupations_by_kpoint : iterable of sequences of float
        One occupation sequence per k-point, each ordered by ascending band
        energy. For a spin-polarized run, pass one channel at a time.
    tol : float
        A band counts as occupied when its occupation is at least this.

    Returns
    -------
    int
        The largest number of occupied bands found at any single k-point.

    Raises
    ------
    ValueError
        If any k-point has no occupied band at all, which means the tolerance
        or the parsed occupations are wrong rather than that the system is
        unusual.

    Notes
    -----
    The count is taken from the index of the first band below tol: band
    indices start at zero, so that index *is* the number of bands below it.

    Every occupied band must be included in the Berry flux, so the maximum
    over k-points is the right reduction. For an insulator the count is the
    same at every k-point; a count that varies with k means bands cross the
    Fermi level, and the polarization is not well defined. That case is not
    diagnosed here yet.
    """
    max_fill = 0

    for kpt_index, occupations in enumerate(occupations_by_kpoint):
        # No band below tol means every band is occupied, so the count is the
        # number of bands rather than an error.
        fill = next((index for index, occ in enumerate(occupations) if occ < tol),
                    len(occupations))

        if fill == 0:
            raise ValueError(
                f"no occupied bands at k-point {kpt_index}: the lowest band has "
                f"occupation {occupations[0]}, below the tolerance {tol}"
            )

        max_fill = max(max_fill, fill)

    return max_fill
