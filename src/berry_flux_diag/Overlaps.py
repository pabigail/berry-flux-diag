#!/usr/bin/env python

import logging

import numpy as np
import berry_flux_diag.utils as utils
from berry_flux_diag.constants import E_PER_ANG2_TO_MUC_PER_CM2
from pymatgen.core.structure import Structure
from pymatgen.util.coord_cython import pbc_shortest_vectors

logger = logging.getLogger(__name__)


# Diagnostic thresholds. Neither is a physical constant: both flag a k-mesh
# too coarse for the phase to be tracked from one k-point to the next, and
# both are warnings rather than errors.

# A Wilson loop phase lives in (-pi, pi]. One approaching the branch cut at
# +/- pi cannot be assigned to a branch with confidence, so the loop sum
# becomes unreliable. 2.8 rad is about 0.891 pi, i.e. within 11% of the cut.
EIG_THRESH = 2.8

# Singular values of the overlap matrix between the occupied manifolds at
# two adjacent k-points: 1 is perfect overlap, 0 means a band has rotated
# entirely out of the manifold. When the smallest approaches 0 the unitary
# recovered as u @ v is reconstructing a nearly singular rotation, and the
# parallel transport through that plaquette cannot be trusted.
SING_VAL_THRESH = 0.2


class Overlaps:
    
    def __init__(self, parse_dict, eig_thresh=EIG_THRESH,
                 sing_val_thresh=SING_VAL_THRESH):
        ''' intialize using vasp_parse_dict or qe_parse_dict

            pol_struct: pymatgen structure
            np_struct: pymatgen structure
            kpoint_list: list of np arrays
            pol_wave_coeffs: numpy array of coefficients in the format
            np_wave_coeffs: numpy array of coefficients in the format

            eig_thresh: warn above this Wilson loop phase, in radians
            sing_val_thresh: warn below this overlap singular value

            Collinear spin-polarized runs are detected from parse_dict['spin_pol']:
            each 'wfcn' is then [up, down], and the two channels' Berry phases
            are added with weight 1 rather than one channel's doubled.
            Must use the full k-grid (not IBZ).
         '''
        self.pol_struct = parse_dict['pol_struct']
        self.np_struct = parse_dict['np_struct']
        self.pol_wfcn_dict = parse_dict['pol_wfcn_dict']
        self.np_wfcn_dict = parse_dict['np_wfcn_dict']
        self.kpoint_list = parse_dict['kpoint_list']
        self.zval_dict = parse_dict['zval_dict']
        self.eig_thresh = eig_thresh
        self.sing_val_thresh = sing_val_thresh
        self.ES_code = parse_dict['ES_code']
        self.spin_pol = parse_dict['spin_pol']
        self.spin_state = 0 # this only matters for spin-polarized calculations

        # Overlap matrices already computed, while compute_string_sums is
        # running; None otherwise, so a direct compute_overlap call always
        # computes. See compute_overlap for what a key holds.
        self._overlap_cache = None

        if self.spin_pol:
            self.band_fill_up = np.min([parse_dict['pol_band_fill_up'],
                                        parse_dict['np_band_fill_up']])
            self.band_fill_down = np.min([parse_dict['pol_band_fill_down'],
                                        parse_dict['np_band_fill_down']])
        else:
            self.band_fill = np.min([parse_dict['pol_band_fill'],
                                     parse_dict['np_band_fill']])
    
    # calculate overlaps
    def compute_overlap(self, l0, kpt0, l1, kpt1, direction):
        '''
        returns overlap matrix between states (l0, kpt0) and (l1, kpt1)
        '''
        _, _, gvec_shift = utils.direction_to_vals(direction)
        
        # shift kpt0, kpt1 appropriately to extract pw coeffs
        shift0_bool = False
        shift1_bool = False
        if not np.any(np.all(np.isclose(kpt0, self.kpoint_list), axis=1)):
            kpt0 = kpt0 - gvec_shift
            shift0_bool = True
        if not np.any(np.all(np.isclose(kpt1, self.kpoint_list), axis=1)):
            kpt1 = kpt1 - gvec_shift
            shift1_bool = True

        # The result is a function of exactly these: the spin channel, each
        # side's structure and k-point as looked up, and the G shift applied
        # to each side. Keyed on them, a cached matrix is the one this call
        # would compute, bit for bit. A plaquette's polar/non-polar edge at
        # one k-point recurs in the x, y and z strings, which is most of the
        # reuse: 2592 overlaps become 1835 for a 6x6x6 mesh. The adjoint of
        # a cached matrix is deliberately not reused for the reversed edge:
        # it differs in the last bit, and would end bit-identity with the
        # uncached computation for another 8%.
        key = None
        if self._overlap_cache is not None:
            key = (self.spin_state,
                   l0, tuple(np.around(kpt0, 6)),
                   tuple(gvec_shift) if shift0_bool else None,
                   l1, tuple(np.around(kpt1, 6)),
                   tuple(gvec_shift) if shift1_bool else None)
            cached = self._overlap_cache.get(key)
            if cached is not None:
                return cached

        pw_coeffs0, gvecs0 = self.get_pw_coeffs_from_state(l0, kpt0)
        pw_coeffs1, gvecs1 = self.get_pw_coeffs_from_state(l1, kpt1)
        
        # shift gvecs if shifted kpoint
        if shift0_bool:
            gvecs0 = np.asarray(gvecs0) - gvec_shift
        if shift1_bool:
            gvecs1 = np.asarray(gvecs1) - gvec_shift
        
        # check if gvecs0 and gvecs1 are the same
        if utils.check_gvecs_same(gvecs0, gvecs1):
            mapped_pw_coeffs0 = pw_coeffs0
            mapped_pw_coeffs1 = pw_coeffs1
        
        # if not, need to do mapping trick before taking inner product
        else:
            mapped_pw_coeffs0, mapped_pw_coeffs1 = utils.map_coeffs(pw_coeffs0, gvecs0, kpt0, pw_coeffs1, gvecs1, kpt1)
            
        # overlap[i, j] = <band i at state 0 | band j at state 1>: one
        # matrix product rather than a vdot per pair of bands. vdot
        # conjugates its first argument, so the first factor is conjugated
        # here. Rows are bands, so the second factor is transposed.
        overlap = np.conj(mapped_pw_coeffs0) @ np.transpose(mapped_pw_coeffs1)

        if key is not None:
            # Shared between callers from here on, so it must not change.
            overlap.flags.writeable = False
            self._overlap_cache[key] = overlap

        return overlap
    
                                
    def get_pw_coeffs_from_state(self, l, kpt):
        '''
        returns pww coefficients and gvecs corresponding to state (l, kpt)
        l = 0 is non-polar
        l = 1 is polar
        
        use self.state = 0 or 1 to extract proper spin-polarized unit
        '''
        # round k-point so matches indexing in kpoint dict
        kpt = np.around(kpt, 6)

        if self.spin_pol:
            if l == 0:
                pw_coeffs = self.np_wfcn_dict[tuple(kpt)]['wfcn'][self.spin_state]
                gvecs = self.np_wfcn_dict[tuple(kpt)]['gvecs']
            elif l == 1:
                pw_coeffs = self.pol_wfcn_dict[tuple(kpt)]['wfcn'][self.spin_state]
                gvecs = self.pol_wfcn_dict[tuple(kpt)]['gvecs']

        else:
            if l == 0:
                pw_coeffs = self.np_wfcn_dict[tuple(kpt)]['wfcn']
                gvecs = self.np_wfcn_dict[tuple(kpt)]['gvecs']
            elif l == 1:
                pw_coeffs = self.pol_wfcn_dict[tuple(kpt)]['wfcn']
                gvecs = self.pol_wfcn_dict[tuple(kpt)]['gvecs']

        return pw_coeffs, gvecs                         
    
    
    def check_overlap_conditioning(self, s):
        """Warn when the occupied manifolds at adjacent k-points barely overlap.

        Returns the smallest singular value so a caller can record it
        without walking the debug payload afterwards.
        """
        smallest_sing_val = min(s)
        if smallest_sing_val < self.sing_val_thresh:
            logger.warning('min singular value %s is below %s; occupied bands '
                           'barely overlap between adjacent k-points, so this '
                           'plaquette is unreliable',
                           smallest_sing_val, self.sing_val_thresh)
        return smallest_sing_val

    def check_wilson_loop_phases(self, wlevs):
        """Warn when a Wilson loop phase approaches the branch cut at +/- pi.

        Returns the largest phase magnitude found.
        """
        largest = 0.0
        for eig in wlevs:
            if np.abs(eig) > self.eig_thresh:
                logger.warning('found eigenvalue %s; k-sampling is underconverged', eig)
            largest = max(largest, np.abs(eig))
        return largest

    # get unitary along path
    def get_unitary_along_path(self, loop_path, direction):
        path_pairs = zip(loop_path[:-1], loop_path[1:])
        # Complex from the start: every curly_M is complex, so a real
        # identity is upcast on the first dot anyway.
        curly_U = np.identity(self.band_fill, dtype=complex)
        
        # save for debugging
        dict_svd = {}
        dict_svd['curly_M'] = []
        dict_svd['states'] = []
        dict_svd['s'] = []
        dict_svd['U'] = []
        
        for states in path_pairs:
            ((l0, kpt0), (l1, kpt1)) = states
            # compute each overlap on-the-fly
            M = self.compute_overlap(l0, kpt0, l1, kpt1, direction)
            u, s, v = np.linalg.svd(M)
            self.check_overlap_conditioning(s)
            curly_M = np.dot(u, v)
            curly_U = np.dot(curly_U, curly_M)
            
            # save for debugging
            dict_svd['curly_M'].append(curly_M)
            dict_svd['states'].append(states)
            dict_svd['s'].append(s)
            dict_svd['U'].append(curly_U)
            
        return curly_U, dict_svd
    
    
    # calculate electronic contribution
    def compute_string_sums(self):
        ''' Modified directly from Bonini code '''
        # Overlaps are cached for the length of one call and dropped after,
        # so the memory is returned and a later call - after the spin
        # channel or band filling has changed - starts from nothing.
        self._overlap_cache = {}
        try:
            return self._compute_string_sums()
        finally:
            self._overlap_cache = None

    def _compute_string_sums(self):
        strings_sums = []
        strings_len = []
        
        # for debugging
        dict_debug = {}
        
        for direction in ["x", "y", "z"]:
            logger.debug('computing strings along %s', direction)
            strings = utils.get_strings(self.kpoint_list, direction)
            if not strings:
                raise ValueError(
                    f"no k-point strings found along {direction} in a list of "
                    f"{len(self.kpoint_list)} k-points; the mesh does not span "
                    f"the Brillouin zone in that direction"
                )
            string_phases = []
            
            # save for debugging
            dict_eigs = {}
            dict_eigs['loops'] = []
            dict_eigs['svd_dict'] = []
            dict_eigs['eigs'] = []
            
            for string in strings:
                inner_loop_sum = 0.
                for kpt0, kpt1 in zip(string[:-1], string[1:]):
                    loop_path = [(0, kpt0), 
                                 (1, kpt0), 
                                 (1, kpt1), 
                                 (0, kpt1), 
                                 (0, kpt0)]
                    curly_U, dict_svd = self.get_unitary_along_path(loop_path, direction)
                    # np.angle is the imaginary part of np.log for a nonzero
                    # argument, without computing the log magnitude that was
                    # then thrown away. Eigenvalues of a unitary matrix have
                    # modulus 1, so the zero case cannot arise.
                    wlevs = np.angle(np.linalg.eigvals(curly_U))
                    self.check_wilson_loop_phases(wlevs)
                    inner_loop_sum += sum(wlevs) / (2 * np.pi)
                    
                    # save for debugging
                    dict_eigs['loops'].append(loop_path)
                    dict_eigs['svd_dict'].append(dict_svd)
                    dict_eigs['eigs'].append(wlevs)
                    
                string_phases.append(inner_loop_sum)

            dict_debug[direction] = dict_eigs
            strings_sums.append(sum(string_phases))
            strings_len.append(len(strings))
        return strings_sums, strings_len, dict_debug
    
    
    # modified from pymatgen to use frac_coord as input, otherwise does not properly 
    # take into account periodic boundary conditions when taking ionic differences
    def calc_ionic(self, frac_coord, structure: Structure, zval: float) -> np.ndarray:
        """
        Calculate the ionic dipole moment using ZVAL from pseudopotential.

        frac_coord: fractional coordinate of a single site
        structure: Structure
        zval: Charge value for ion (ZVAL for VASP pseudopotential)

        Returns polarization in electron Angstroms.
        """
        return _calc_ionic(frac_coord, structure, zval)


    def get_ionic_pol_change(self):
        """ ionic part of polarization """
        return ionic_pol_change(self.pol_struct, self.np_struct, self.zval_dict)


    def get_spont_pol(self, elec_change):
        # The three components are coefficients along the unit lattice
        # vectors a/|a|, b/|b|, c/|c|, not Cartesian ones, which is why
        # polarization_norm sums them along those vectors.
        ion_contrib = self.get_ionic_pol_change() # muC / cm^2
        elec_contrib = electronic_contrib(elec_change, self.pol_struct.lattice)

        logger.info('electronic contribution: %s', elec_contrib)
        logger.info('ionic contribution: %s', ion_contrib)

        return ion_contrib + elec_contrib


    def electronic_change(self):
        """The electronic change from the non-polar to the polar structure.

        Returns
        -------
        elec_change : list of 3 floats
            Along a, b and c: the Berry phases of each direction's strings
            over 2 pi, averaged over the strings and weighted by occupation
            (2 per band without spin, 1 per band per channel with it), in
            units of e times a lattice vector.
        dict_debug : dict, or [up, down] of them
            compute_string_sums' debug output.
        """
        if self.spin_pol:

            self.spin_state = 0
            self.band_fill = self.band_fill_up
            strings_sum_up, strings_len_up, dict_debug_up = self.compute_string_sums()
            logger.info('string_sums_up: %s', strings_sum_up)

            self.spin_state = 1
            self.band_fill = self.band_fill_down
            strings_sum_down, strings_len_down, dict_debug_down = self.compute_string_sums()
            logger.info('string_sums_down: %s', strings_sum_down)

            occ_fact = 1
            elec_change = [occ_fact*(strings_sum_up[0] + strings_sum_down[0])/strings_len_up[0],
                           occ_fact*(strings_sum_up[1] + strings_sum_down[1])/strings_len_up[1],
                           occ_fact*(strings_sum_up[2] + strings_sum_down[2])/strings_len_up[2]]

            dict_debug = [dict_debug_up, dict_debug_down]

        else:

            strings_sum, strings_len, dict_debug = self.compute_string_sums()
            logger.info('string_sums: %s', strings_sum)

            occ_fact = 2
            elec_change = [occ_fact * strings_sum[0] / strings_len[0],
                           occ_fact * strings_sum[1] / strings_len[1],
                           occ_fact * strings_sum[2] / strings_len[2]]

        return elec_change, dict_debug


    def compute_polarization(self):

        elec_change, dict_debug = self.electronic_change()

        # components along the unit lattice vectors
        final_pol = self.get_spont_pol(elec_change)
        P_norm = polarization_norm(final_pol, self.pol_struct.lattice)

        logger.info('final_pol frac: %s', final_pol)
        logger.info('polarization: %s muC / cm^2', P_norm)
        return P_norm, dict_debug


def _calc_ionic(frac_coord, structure: Structure, zval: float) -> np.ndarray:
    """The ionic dipole of one site in electron angstroms, as -zval times
    its fractional displacement scaled by the lattice lengths."""
    norms = structure.lattice.lengths
    return np.multiply(norms, -np.array(frac_coord) * zval)


def ionic_pol_change(pol_struct, np_struct, zval_dict):
    """Ionic change in polarization from np_struct to pol_struct, in muC/cm^2,
    as components along the unit lattice vectors of pol_struct.

    Each site moves along the shortest vector between its two positions
    across the periodic boundaries.
    """
    lattice = pol_struct.lattice
    fcoords_pol = pol_struct.frac_coords
    fcoords_np = np_struct.frac_coords
    pbc_shortest_vecs = pbc_shortest_vectors(lattice, fcoords_np, fcoords_pol)
    cart_coords_for_ion_calc = []

    # calculate differences between ions across pbc
    for i in range(len(pbc_shortest_vecs)):
        cart_coords_for_ion_calc.append(pbc_shortest_vecs[i][i])
    frac_coords_for_ion_calc = lattice.get_fractional_coords(cart_coords_for_ion_calc)

    # extract charge of ions from zval dict
    tot_ionic = []
    for site, frac_coord in zip(pol_struct, frac_coords_for_ion_calc):
        zval = zval_dict[utils.extract_letters(str(site.specie))]
        tot_ionic.append(_calc_ionic(frac_coord, pol_struct, zval))
    ion_diff = np.sum(tot_ionic, axis=0) # in electron Angstroms:

    # convert to muC/cm^2. The sign is negative because calc_ionic already
    # returns -z*dr, so the two cancel to the usual +sum(z*dr).
    pol_units = -E_PER_ANG2_TO_MUC_PER_CM2 / pol_struct.lattice.volume
    return pol_units * ion_diff # in muC/cm^2


def electronic_contrib(elec_change, lattice):
    """Electronic change in muC/cm^2 from Overlaps.electronic_change's
    output, as components along the unit lattice vectors of lattice."""
    scale = E_PER_ANG2_TO_MUC_PER_CM2 / lattice.volume

    elec_contrib_x = scale * np.array(elec_change[0]) * lattice.a
    elec_contrib_y = scale * np.array(elec_change[1]) * lattice.b
    elec_contrib_z = scale * np.array(elec_change[2]) * lattice.c

    return np.array([elec_contrib_x, elec_contrib_y, elec_contrib_z]) # muC / cm^2


def polarization_cartesian(components, lattice):
    """The Cartesian polarization vector from components along the unit
    lattice vectors a/|a|, b/|b|, c/|c|.

    The components are not Cartesian unless the cell is orthogonal, so
    their own norm is not |P|: for rhombohedral LiNbO3 it is 0.69 |P|.
    """
    a, b, c = lattice.matrix
    a, b, c = a / np.linalg.norm(a), b / np.linalg.norm(b), c / np.linalg.norm(c)
    return a * components[0] + b * components[1] + c * components[2]


def polarization_norm(components, lattice):
    """|P| from components along the unit lattice vectors."""
    return np.linalg.norm(polarization_cartesian(components, lattice))


# Two structures count as the same image of a chain within this many
# angstroms per site, across the periodic boundaries, and per lattice-vector
# component. They come from the same DFT run, read by two parses, so they
# agree far more closely than this.
CHAIN_STRUCT_ATOL = 1e-4


def _same_structure(a, b, atol=CHAIN_STRUCT_ATOL):
    if len(a) != len(b) or [str(s.specie) for s in a] != [str(s.specie) for s in b]:
        return False
    if not np.allclose(a.lattice.matrix, b.lattice.matrix, rtol=0, atol=atol):
        return False
    vecs = pbc_shortest_vectors(a.lattice, a.frac_coords, b.frac_coords)
    return max(np.linalg.norm(vecs[i][i]) for i in range(len(a))) <= atol


def chain_polarization(parse_dicts, eig_thresh=EIG_THRESH, sing_val_thresh=SING_VAL_THRESH):
    """Polarization change along a chain of interpolated images.

    When the polar and non-polar structures are too far apart for one pair
    (see preprocess.preprocess_structs), the change is computed along the
    chain pol -> image 1 -> ... -> np, one adjacent pair at a time. Only the
    electronic part is summed over the pairs, as BFDJobs does: each pair's
    Berry phases are small and on a definite branch, and their sum follows
    the path. The ionic part is computed once, from the two ends.

    Parameters
    ----------
    parse_dicts : iterable of dict
        The parser output (VASPParser.vasp_parser or QEParser.qe_parser) of
        each adjacent pair in order: (pol, image 1), (image 1, image 2), ...,
        (image n, np), each pair parsed with its first structure as the
        polar one. A generator is fine, and for VASP better: only one pair
        is held at a time, and a parsed pair can be gigabytes.
    eig_thresh, sing_val_thresh
        Passed to each pair's Overlaps.

    Returns
    -------
    dict
        polarization_norm : |P| in muC/cm^2.
        polarization_cartesian : P as a Cartesian vector in muC/cm^2.
        polarization_components, elec_contrib, ion_contrib : components
            along the unit lattice vectors of the polar structure, in
            muC/cm^2, as Overlaps logs them.
        elec_change_per_pair : each pair's Overlaps.electronic_change.
        debug_per_pair : each pair's compute_string_sums debug output.

    Raises
    ------
    ValueError
        If the chain is empty, if one pair's second structure is not the
        next pair's first, or if the pairs differ in spin polarization or
        valence charges.

    Notes
    -----
    The summed electronic change is converted with the polar structure's
    lattice, as BFDJobs does. That is exact when every image has the same
    lattice; with interpolated lattices it is an approximation.
    """
    first_pol = last_np = zval_dict = spin_pol = None
    elec_total = np.zeros(3)
    elec_per_pair, debug_per_pair = [], []

    for index, parse_dict in enumerate(parse_dicts):
        if index == 0:
            first_pol = parse_dict["pol_struct"]
            zval_dict, spin_pol = parse_dict["zval_dict"], parse_dict["spin_pol"]
        else:
            if not _same_structure(last_np, parse_dict["pol_struct"]):
                raise ValueError(
                    f"pair {index - 1}'s second structure is not pair {index}'s first: "
                    f"the pairs must run pol -> image 1 -> ... -> np, each sharing an "
                    f"image with the next")
            if parse_dict["spin_pol"] != spin_pol:
                raise ValueError(f"pair {index} differs from pair 0 in spin polarization")
            if parse_dict["zval_dict"] != zval_dict:
                raise ValueError(f"pair {index} has valence charges {parse_dict['zval_dict']}, "
                                 f"pair 0 {zval_dict}")
        last_np = parse_dict["np_struct"]

        elec_change, dict_debug = Overlaps(parse_dict, eig_thresh=eig_thresh,
                                           sing_val_thresh=sing_val_thresh).electronic_change()
        logger.info("chain pair %d: electronic change %s", index, elec_change)
        elec_per_pair.append(np.array(elec_change))
        debug_per_pair.append(dict_debug)
        elec_total += np.array(elec_change)
        del parse_dict

    if first_pol is None:
        raise ValueError("the chain has no pairs")

    lattice = first_pol.lattice
    elec = electronic_contrib(elec_total, lattice)
    ion = ionic_pol_change(first_pol, last_np, zval_dict)
    components = ion + elec
    cartesian = polarization_cartesian(components, lattice)

    logger.info("chain of %d pairs: electronic contribution %s, ionic contribution %s",
                len(elec_per_pair), elec, ion)
    logger.info("chain polarization: %s muC / cm^2", np.linalg.norm(cartesian))
    return {
        "polarization_norm": float(np.linalg.norm(cartesian)),
        "polarization_cartesian": cartesian,
        "polarization_components": components,
        "elec_contrib": elec,
        "ion_contrib": ion,
        "elec_change_per_pair": elec_per_pair,
        "debug_per_pair": debug_per_pair,
    }
