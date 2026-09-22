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

            in this version, no spin polarization, must use full k-grid (not IBZ)
         '''
        self.pol_struct = parse_dict['pol_struct']
        self.np_struct = parse_dict['np_struct']
        self.pol_wfcn_dict = parse_dict['pol_wfcn_dict']
        self.np_wfcn_dict = parse_dict['np_wfcn_dict']
        self.kpoint_list = parse_dict['kpoint_list']
#         self.band_fill = np.min([parse_dict['pol_band_fill'], 
#                                  parse_dict['np_band_fill']]) # changed this from np.max to np.min
        self.zval_dict = parse_dict['zval_dict']
        self.eig_thresh = eig_thresh
        self.sing_val_thresh = sing_val_thresh
        self.ES_code = parse_dict['ES_code']
        self.spin_pol = parse_dict['spin_pol']
        self.spin_state = 0 # this only matters for spin-polarized calculations

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
        
        pw_coeffs0, gvecs0 = self.get_pw_coeffs_from_state(l0, kpt0)
        pw_coeffs1, gvecs1 = self.get_pw_coeffs_from_state(l1, kpt1)
        
        # shift gvecs if shifted kpoint
        if shift0_bool:
            gvecs0 = [g - gvec_shift for g in gvecs0]
        if shift1_bool:
            gvecs1 = [g - gvec_shift for g in gvecs1]
        
        overlap = np.zeros((len(pw_coeffs0), len(pw_coeffs1)), dtype=complex)
        
        # check if gvecs0 and gvecs1 are the same
        if utils.check_gvecs_same(gvecs0, gvecs1):
            mapped_pw_coeffs0 = pw_coeffs0
            mapped_pw_coeffs1 = pw_coeffs1
        
        # if not, need to do mapping trick before taking inner product
        else:
            mapped_pw_coeffs0, mapped_pw_coeffs1 = utils.map_coeffs(pw_coeffs0, gvecs0, kpt0, pw_coeffs1, gvecs1, kpt1)
            
        for i, wv0 in enumerate(mapped_pw_coeffs0):
            for j, wv1 in enumerate(mapped_pw_coeffs1):
                overlap[i,j] = np.vdot(wv0, wv1)
                           
        # return overlap
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
        curly_U = np.identity(self.band_fill)
        
        # save for debugging
        dict_svd = {}
        dict_svd['M'] = []
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
            dict_svd['M'].append(curly_M)
            dict_svd['states'].append(states)
            dict_svd['s'].append(s)
            dict_svd['U'].append(curly_U)
            
        return curly_U, dict_svd
    
    
    # calculate electronic contribution 
    def compute_string_sums(self):
        ''' Modified directly from Bonini code '''
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
                    wlevs = np.log(np.linalg.eigvals(curly_U)).imag
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
        norms = structure.lattice.lengths
        return np.multiply(norms, -np.array(frac_coord) * zval)
    
    
    def get_ionic_pol_change(self):
        """ ionic part of polarization """
        
        lattice = self.pol_struct.lattice
        fcoords_pol = self.pol_struct.frac_coords
        fcoords_np = self.np_struct.frac_coords
        pbc_shortest_vecs = pbc_shortest_vectors(lattice, fcoords_np, fcoords_pol)
        cart_coords_for_ion_calc = []
        
        # calculate differences between ions across pbc
        for i in range(len(pbc_shortest_vecs)):
            cart_coords_for_ion_calc.append(pbc_shortest_vecs[i][i])
        frac_coords_for_ion_calc = lattice.get_fractional_coords(cart_coords_for_ion_calc) 
        
        # extract charge of ions from zval dict
        tot_ionic = []
        for site, frac_coord in zip(self.pol_struct, frac_coords_for_ion_calc):
            zval = self.zval_dict[utils.extract_letters(str(site.specie))]
            tot_ionic.append(self.calc_ionic(frac_coord, self.pol_struct, zval))
        ion_diff =  np.sum(tot_ionic, axis=0) # in electron Angstroms:
        
        # convert to muC/cm^2. The sign is negative because calc_ionic already
        # returns -z*dr, so the two cancel to the usual +sum(z*dr).
        pol_units = -E_PER_ANG2_TO_MUC_PER_CM2 / self.pol_struct.lattice.volume
        ionic_contrib = pol_units * ion_diff # in muC/cm^2
        
        return ionic_contrib
    
    
    
    def get_spont_pol(self, elec_change):
        # Returns all three components. Valid only for an orthogonal cell:
        # the fractional electronic change is scaled by a, b and c
        # separately, which is not the general lattice transformation.
        ion_contrib = self.get_ionic_pol_change() # muC / cm^2

        lattice = self.pol_struct.lattice
        scale = E_PER_ANG2_TO_MUC_PER_CM2 / lattice.volume

        elec_contrib_x = scale * np.array(elec_change[0]) * lattice.a
        elec_contrib_y = scale * np.array(elec_change[1]) * lattice.b
        elec_contrib_z = scale * np.array(elec_change[2]) * lattice.c

        elec_contrib = np.array([elec_contrib_x, elec_contrib_y, elec_contrib_z]) # muC / cm^2
        
        
        logger.info('electronic contribution: %s', elec_contrib)
        logger.info('ionic contribution: %s', ion_contrib)

        return ion_contrib + elec_contrib
    
    
    def compute_polarization(self):

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


        # fractional coordinates
        final_pol = self.get_spont_pol(elec_change)

        # normalization
        a, b, c = self.pol_struct.lattice.matrix
        a, b, c = a / np.linalg.norm(a), b / np.linalg.norm(b), c / np.linalg.norm(c)

        P_norm = np.linalg.norm(a * final_pol[0] +
                                b * final_pol[1] +
                                c * final_pol[2])

        logger.info('final_pol frac: %s', final_pol)
        logger.info('polarization: %s muC / cm^2', P_norm)
        return P_norm, dict_debug
