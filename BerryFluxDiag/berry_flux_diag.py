#!/usr/bin/env python

import numpy as np
import qeschema
import h5py
import itertools
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Wavecar, Kpoints, Potcar
from pymatgen.util.coord_cython import pbc_shortest_vectors
from pymatgen.analysis.ferroelectricity.polarization import zval_dict_from_potcar
from pymatgen.io.pwscf import PWOutput
from pymatgen.optimization.linear_assignment import LinearAssignment
# from pymatgen.analysis.ferroelectricity.polarization import get_total_ionic_dipole

ECHARGE = 1.60217733 * 10**-19

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
    
    
def get_band_filling_from_wavecar(wavecar, tol):
    
    max_band_fill = 0
    num_kpoints = len(wavecar.band_energy)
    
    for kpt in range(0, num_kpoints):
        band_filling = wavecar.band_energy[kpt][:, 2]
        temp_band_index = next(index for index, i in enumerate(band_filling) if i < tol)
        if temp_band_index != 0:
            if (temp_band_index - 1) > max_band_fill:
                max_band_fill = temp_band_index 
        else:
            raise ValueError("band filling is zero")
    
    return max_band_fill


def get_wfcn_dict_from_vasp(wavecar, kpoint_list, max_band_fill):
    # key of wfcn dict is a k-point
    # each k-point has associated dict with coeffs and gvecs
    
    wfcn_dict = {}
    all_coeffs = wavecar.coeffs
    all_gvecs = wavecar.Gpoints
    
    num_kpts = len(kpoint_list)
    for index in range(0, num_kpts):
        wfcn = np.array(all_coeffs[index][0:max_band_fill])
        gvecs = all_gvecs[index]

        coeff_gvec_dict = {}
        coeff_gvec_dict['wfcn'] = wfcn
        coeff_gvec_dict['gvecs'] = gvecs
        wfcn_dict[tuple(kpoint_list[index])] = coeff_gvec_dict
    
    return wfcn_dict
    
    
def vasp_parser(pol_POSCAR, np_POSCAR, pol_WAVECAR, np_WAVECAR, POTCAR):
    
    pol_struct = Structure.from_file(pol_POSCAR)
    np_struct = Structure.from_file(np_POSCAR)
    
    pol_wavecar = Wavecar(pol_WAVECAR)
    np_wavecar = Wavecar(np_WAVECAR)
    
    potcar = Potcar.from_file(POTCAR)
    zval_dict = zval_dict_from_potcar(potcar)
    
    # only works for full Brillouin zone
    kpoint_list = pol_wavecar.kpoints
    # round k-point list so can find matching k-points
    kpoint_list = [np.around(kpt, 6) for kpt in kpoint_list]
    
    FILLING_TOL = 1e-6
    pol_max_band_fill = get_band_filling_from_wavecar(pol_wavecar, FILLING_TOL)
    np_max_band_fill = get_band_filling_from_wavecar(np_wavecar, FILLING_TOL)
    
    if pol_max_band_fill != np_max_band_fill:
        print("CAUTION: max band filling for polar and non-polar structures are not the same")
        
    common_max_band_fill = np.min([pol_max_band_fill, np_max_band_fill]) # any rationale to change to np.max?
    
    pol_wfcn_dict = get_wfcn_dict_from_vasp(pol_wavecar, kpoint_list, common_max_band_fill)
    np_wfcn_dict = get_wfcn_dict_from_vasp(np_wavecar, kpoint_list, common_max_band_fill)
    
    vasp_parse_dict = empty_parse_dict()
    vasp_parse_dict['pol_struct'] = pol_struct
    vasp_parse_dict['np_struct'] = np_struct
    vasp_parse_dict['kpoint_list'] = kpoint_list
    vasp_parse_dict['pol_band_fill'] = pol_max_band_fill
    vasp_parse_dict['np_band_fill'] = np_max_band_fill
    vasp_parse_dict['pol_wfcn_dict'] = pol_wfcn_dict
    vasp_parse_dict['np_wfcn_dict'] = np_wfcn_dict
    vasp_parse_dict['zval_dict'] = zval_dict
    vasp_parse_dict['ES_code'] = 'VASP'
    
    # Hard-coded now, will need to change for spin-polarized calcs
    vasp_parse_dict['spin_pol'] = False
    
    return vasp_parse_dict


def get_struct_from_qeschema(xml_data):
    
    # convert from Bohr to Angstrom
    BOHR_TO_ANGSTROM = 0.5291772
    
    atomic_struct = xml_data['qes:espresso']['output']['atomic_structure']

    lattice = [list(np.array(atomic_struct['cell']['a1'])*BOHR_TO_ANGSTROM),
               list(np.array(atomic_struct['cell']['a2'])*BOHR_TO_ANGSTROM),
               list(np.array(atomic_struct['cell']['a3'])*BOHR_TO_ANGSTROM)]
    
    species = []
    coords = []
    for atom in atomic_struct['atomic_positions']['atom']:
        species.append(atom['@name'])
        coords.append(np.array(atom['$'])*BOHR_TO_ANGSTROM) # convert from Bohr to angstrom
        
    structure = Structure(lattice, species, coords, coords_are_cartesian=True)
    
    return structure


def get_kpt_list_from_qeschema(xml_data):
    
    ks_energies = xml_data['qes:espresso']['output']['band_structure']['ks_energies']
    
    kpoint_list = []
    for kpt in ks_energies:
        kpoint_list.append(np.array(kpt['k_point']['$']))
        
    return kpoint_list


def get_band_filling_from_qeschema(xml_data, tol):
    
    max_band_fill = 0
    ks_energies = xml_data['qes:espresso']['output']['band_structure']['ks_energies']
    for kpt in ks_energies:
        band_filling = kpt['occupations']['$']
        temp_band_index = next(index for index, i in enumerate(band_filling) if i < tol)
        
        if temp_band_index != 0:
            if (temp_band_index - 1) > max_band_fill:
                max_band_fill = temp_band_index
        else:
            raise ValueError("band filling is zero")
    
    return max_band_fill


def get_wavefunctions_hdf5(filename, start_band=None, stop_band=None):
    """
    Returns a numpy array with the wave functions for bands from start_band to
    stop_band. If not specified starts from 1st band and ends with last one.
    Band numbering is Python style starts from 0.abs

    :param filename: path to the wfc file
    :param start_band: first band to read, default first band in the file
    :param stop_band:  last band to read, default last band in the file
    :return: a numpy array with shape [nbnd,npw]
    
    # REWROTE SLIGHTLY FROM QESCHEMA
    """
    with h5py.File(filename, "r") as f:
        igwx = f.attrs.get('igwx')
        if start_band is None:
            start_band = 0
        if stop_band is None:
            stop_band = f.attrs.get('nbnd')
        if stop_band == start_band:
            stop_band = start_band + 1
        res = f.get('evc')[start_band:stop_band, :]

    # returns an array that is n bands by igwx for a single k-point
    coeffs = np.asarray([x.reshape([igwx, 2]).dot([1.0, 1.0j]) for x in res[:]])
    return coeffs
        

def get_wfcn_gvecs_from_hdf5(wfc_folder_path, kpoint_list, max_band_fill):
    
    wfcn_dict = {}
    
    num_kpts = len(kpoint_list)
    for index in range(0, num_kpts):
        filename = wfc_folder_path+'wfc'+str(index+1)+'.hdf5'
        wfcn = get_wavefunctions_hdf5(filename, stop_band=max_band_fill)
        gvecs = get_wfc_miller_indices(filename)
        coeff_gvec_dict = {}
        coeff_gvec_dict['wfcn'] = wfcn
        coeff_gvec_dict['gvecs'] = gvecs
        wfcn_dict[tuple(kpoint_list[index])] = coeff_gvec_dict
    
    return wfcn_dict


def get_zval_dict_from_PWOutput(pw_out):
    
    zval_pattern = {'element': 'for\\s+(\\w+)\\s+read\\sfrom\\sfile',
                'zval': 'Zval\\s+=\\s+([\\d+\\.]+)\\s'}
    
    pw_out.read_pattern(zval_pattern)
    pw_out_data = pw_out.data
    
    zval_dict = {}
    for entry in pw_out_data['element']:
        element = entry[0][0]
        line_num = entry[1]
        for zvals in pw_out_data['zval']:
            if zvals[1] == line_num+3:
                zval = float(zvals[0][0])
        zval_dict[element] = zval
        
    return zval_dict


def get_wfc_miller_indices(filename):
    """
    Reads miller indices from the wfc file

    :param filename: path to the wfc HDF5 file
    :return: a np.array of integers with shape [igwx,3]
    
    # DIRECTLY FROM QESCHEMA
    """
    with h5py.File(filename, "r") as f:
        res = f.get("MillerIndices")[:, :]
    return res


def qe_parser(pol_xml_file, np_xml_file, pol_wfcn_path, np_wfcn_path, pw_out_path):
    
    pol_pw_doc = qeschema.PwDocument()
    np_pw_doc = qeschema.PwDocument()
    
    pol_pw_doc.read(pol_xml_file, validation='lax')
    np_pw_doc.read(np_xml_file, validation='lax')
    
    pol_xml_data = pol_pw_doc.to_dict(validation='lax')
    np_xml_data = np_pw_doc.to_dict(validation='lax')
    
    pol_struct = get_struct_from_qeschema(pol_xml_data)
    np_struct = get_struct_from_qeschema(np_xml_data)
    
    kpoint_list = get_kpt_list_from_qeschema(pol_xml_data)
    # round k-point list so can find matching k-points
    kpoint_list = [np.around(kpt, 6) for kpt in kpoint_list]
    
    pw_out = PWOutput(pw_out_path)
    zval_dict = get_zval_dict_from_PWOutput(pw_out)
    
    FILLING_TOL = 1e-6
    pol_max_band_fill = get_band_filling_from_qeschema(pol_xml_data, FILLING_TOL)
    np_max_band_fill = get_band_filling_from_qeschema(np_xml_data, FILLING_TOL)
    
    if pol_max_band_fill != np_max_band_fill:
        print("CAUTION: max band filling for polar and non-polar structures are not the same")
    
    common_max_band_fill = np.min([pol_max_band_fill, np_max_band_fill]) # any rationale to change to np.max?
    
    pol_wfcn_dict = get_wfcn_gvecs_from_hdf5(pol_wfcn_path, kpoint_list, common_max_band_fill)
    np_wfcn_dict = get_wfcn_gvecs_from_hdf5(np_wfcn_path, kpoint_list, common_max_band_fill)
    
    qe_parse_dict = empty_parse_dict()
    qe_parse_dict['pol_struct'] = pol_struct
    qe_parse_dict['np_struct'] = np_struct
    qe_parse_dict['kpoint_list'] = kpoint_list 
    qe_parse_dict['pol_band_fill'] = pol_max_band_fill
    qe_parse_dict['np_band_fill'] = np_max_band_fill
    qe_parse_dict['pol_wfcn_dict'] = pol_wfcn_dict
    qe_parse_dict['np_wfcn_dict'] = np_wfcn_dict
    qe_parse_dict['zval_dict'] = zval_dict
    qe_parse_dict['ES_code'] = 'QE'
    
    # setting manually, need to change for future
    qe_parse_dict['spin_pol'] = False

    return qe_parse_dict


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
    # kpt: g-vectors are shifted first by the kpoint of where the kpoint is located
    # max_rad: maximum radius of all gvecs considered
    # base: base of expansion for x*base**2 + y*base + z
    
    # not really sure about this kpt shift that may be rounded away
    kpt_centered_gvec = gvec # + kpt
    shift = np.full(3, max_rad)
    shifted_gvec = kpt_centered_gvec + shift
    index = round(shifted_gvec[0])*base**2 + round(shifted_gvec[1])*base + round(shifted_gvec[2])
    return index


def index_to_gvec(index, kpt, max_rad, base):
    # index: shifted index correpsonding to a gvec
    # kpt: kpoint corresponding to wavefunction coefficients of planewaves at give kpt
    # max_rad: maximum radius of all gvecs considered
    # base: base of expansion for x*base**2 + y*base + z
    
    # return: gvec = [x, y, z]
    
    x_shifted_gvec = np.floor(index / base**2)
    y_shifted_gvec = np.floor((index - x_shifted_gvec*base**2)/ base)
    z_shifted_gvec = index - x_shifted_gvec*base**2 - y_shifted_gvec*base
    
    shifted_gvec = np.array((x_shifted_gvec,
                             y_shifted_gvec, 
                             z_shifted_gvec))
    
    shift = np.full(3, max_rad)
    kpt_centered_gvec = shifted_gvec - shift
    # not totally sure about this rounding kpt value
    gvec = np.around(kpt_centered_gvec - kpt)
    
    return gvec


def map_coeffs(coeffs0, gvecs0, kpt0, coeffs1, gvecs1, kpt1):
    
    gvec_min0, gvec_max0, gvec_range0, gvec_len0, gvec_base_exp0 = gvec_extrema(gvecs0)
    gvec_min1, gvec_max1, gvec_range1, gvec_len1, gvec_base_exp1 = gvec_extrema(gvecs1)
    
    max_g_rad = np.max([np.max(gvec_max0), np.max(gvec_max1)])
    min_g_rad = np.abs(np.min([np.min(gvec_min0), np.min(gvec_min1)]))
    max_rad = int(np.max([max_g_rad, min_g_rad]))
    base = int(np.max([gvec_base_exp0, gvec_base_exp1]))
    
    coeffs0_mapped = np.zeros((len(coeffs0), base**3 + base**2 + base + 1), dtype=complex)
    coeffs1_mapped = np.zeros((len(coeffs1), base**3 + base**2 + base + 1), dtype=complex)
    
    # print(f'coeffs_0 len: {len(coeffs0_mapped[0])}, coeffs_1 len: {len(coeffs1_mapped[0])}, gvec0: {gvec_len0}, gvec1: {gvec_len1}')
    
    for i in range(gvec_len0):
        mapped_i = gvec_to_index(gvecs0[i], kpt0, max_rad, base)
        # print(mapped_i, i)
        coeffs0_mapped[:, mapped_i] = coeffs0[:, i]
    for i in range(gvec_len1):
        mapped_i = gvec_to_index(gvecs1[i], kpt1, max_rad, base)
        coeffs1_mapped[:, mapped_i] = coeffs1[:, i]
        
    return coeffs0_mapped, coeffs1_mapped


# def get_mask(struct2, struct1):
#     ''' directly from Bonini'''
#     mask = np.zeros((len(struct2), len(struct1)), dtype=np.bool)

#     inner = []
#     for sp2, i in itertools.groupby(enumerate(struct2.species_and_occu),
#                                     key=lambda x: x[1]):
#         i = list(i)
#         inner.append((sp2, slice(i[0][0], i[-1][0]+1)))

#     for sp1, j in itertools.groupby(enumerate(struct1.species_and_occu),
#                                     key=lambda x: x[1]):
#         j = list(j)
#         j = slice(j[0][0], j[-1][0]+1)
#         for sp2, i in inner:
#             mask[i, j,] = str(sp1) != str(sp2)
#     return mask


class Overlaps:
    
    def __init__(self, parse_dict):
        ''' intialize using vasp_parse_dict or qe_parse_dict
        
            pol_struct: pymatgen structure 
            np_struct: pymatgen structure
            kpoint_list: list of np arrays
            pol_wave_coeffs: numpy array of coefficients in the format
            np_wave_coeffs: numpy array of coefficients in the format
            
            in this version, no spin polarization, must use full k-grid (not IBZ)
         '''
        self.pol_struct = parse_dict['pol_struct']
        self.np_struct = parse_dict['np_struct']
        self.pol_wfcn_dict = parse_dict['pol_wfcn_dict']
        self.np_wfcn_dict = parse_dict['np_wfcn_dict']
        self.kpoint_list = parse_dict['kpoint_list']
        self.band_fill = np.min([parse_dict['pol_band_fill'], 
                                 parse_dict['np_band_fill']]) # changed this from np.max to np.min
        self.zval_dict = parse_dict['zval_dict']
        self.eig_thresh = 2.8
        self.ES_code = parse_dict['ES_code']
    
    # calculate overlaps
    # NEED TO CHECK BEFORE WHETHER TO COMPUTE OVERLAP
    def compute_overlap(self, l0, kpt0, l1, kpt1, direction, space = 'g'):
        '''
        returns overlap matrix between states (l0, kpt0) and (l1, kpt1)
        '''
        _, _, gvec_shift = direction_to_vals(direction)
        
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
        if check_gvecs_same(gvecs0, gvecs1):
            mapped_pw_coeffs0 = pw_coeffs0
            mapped_pw_coeffs1 = pw_coeffs1
        
        # if not, need to do mapping trick before taking inner product
        else:
            mapped_pw_coeffs0, mapped_pw_coeffs1 = map_coeffs(pw_coeffs0, gvecs0, kpt0, pw_coeffs1, gvecs1, kpt1)
            
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
        '''
        # round k-point so matches indexing in kpoint dict
        kpt = np.around(kpt, 6)
        
        if l == 0:
            pw_coeffs = self.np_wfcn_dict[tuple(kpt)]['wfcn']
            gvecs = self.np_wfcn_dict[tuple(kpt)]['gvecs']
        elif l == 1:
            pw_coeffs = self.pol_wfcn_dict[tuple(kpt)]['wfcn']
            gvecs = self.pol_wfcn_dict[tuple(kpt)]['gvecs']
        
        return pw_coeffs, gvecs                          
    
    
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
            smallest_sing_val = min(s)
            if smallest_sing_val < 0.2:
                print(f'min singular value: {min(s)}')
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
            print(direction)
            strings = get_strings(self.kpoint_list, direction)
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
                    for eig in wlevs:
                        if np.abs(eig) > 2.8:
                            print(f'found eigenvalue {eig}; k-sampling is underconverged')
                    inner_loop_sum += sum(wlevs) / (2 * np.pi)
                    
                    # save for debugging
                    dict_eigs['loops'].append(loop_path)
                    dict_eigs['svd_dict'].append(dict_svd)
                    dict_eigs['eigs'].append(wlevs)
                    
                string_phases.append(inner_loop_sum)
                string_sum = sum(string_phases)
                
            dict_debug[direction] = dict_eigs
            strings_sums.append(string_sum)
            strings_len.append(len(strings))
        return strings_sums, strings_len, dict_debug
    
    
#     def get_ionic_pol_change_old(self):
#         """ ionic part of polarization """
        
#         struct2 = self.np_struct
#         struct1 = self.pol_struct
        
#         mask = get_mask(struct2, struct1)
#         vecs, d_2 = pbc_shortest_vectors(struct2.lattice, 
#                                      struct2.frac_coords, 
#                                      struct1.frac_coords,
#                                      mask, 
#                                      return_d2=True, 
#                                      lll_frac_tol=[0.4, 0.4, 0.4])
#         lin = LinearAssignment(d_2)
#         s = lin.solution
#         species = [struct1[i].species_string for i in s]
#         short_vecs = vecs[np.arange(len(s)), s]
#         print(short_vecs)
#         pol_change = np.array([0., 0., 0.])
#         for v, sp in zip(short_vecs, species):
#             pol_change += (v) * self.zval_dict[str(sp)]
            
#         print(pol_change)
#         print((ECHARGE * 10**20) / struct2.lattice.volume)
#         return (ECHARGE * 10**20 ) * pol_change / struct2.lattice.volume 


#     def std_frac_coords_for_ionic_calc(self, struct):
        
#         # get_total_ionic_dipole() from pymatgen does not 
#         # properly treat sites 0.0 = 1.0 in fractional coordinates, so 
#         # use this to transform fractional ionic cooridates
        
#         frac_coords_for_ionic_calc = []
#         for site in struct.frac_coords:
#             new_site = []
#             for coord in np.array(site):
#                 if coord == 1.0:
#                     new_site.append(0.0)
#                 else:
#                     new_site.append(coord)
#             frac_coords_for_ionic_calc.append(new_site)
#         return frac_coords_for_ionic_calc
    
    
    
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
            zval = self.zval_dict[str(site.specie)]
            tot_ionic.append(self.calc_ionic(frac_coord, self.pol_struct, zval))
        ion_diff =  np.sum(tot_ionic, axis=0) # in electron Angstroms:
        
        # convert to muC/cm^2
        e_to_muC = -1.6021766e-13
        cm2_to_A2 = 1e16
        pol_volume = [self.pol_struct.lattice.volume]
        pol_units = 1.0 / np.array(pol_volume)
        pol_units *= e_to_muC * cm2_to_A2
        ionic_contrib = pol_units * ion_diff # in muC/cm^2                      
        
        return ionic_contrib
    
    
    
    def get_spont_pol(self, elec_change):
        # still hard coded for c direction polarization
    
        ion_contrib = self.get_ionic_pol_change() # muC / cm^2

        elec_contrib_x = ((ECHARGE * 10 ** 20) * np.array(elec_change[0]) * 
                          self.pol_struct.lattice.a / self.pol_struct.lattice.volume)
        elec_contrib_y = ((ECHARGE * 10 ** 20) * np.array(elec_change[1]) * 
                          self.pol_struct.lattice.b / self.pol_struct.lattice.volume)
        elec_contrib_z = ((ECHARGE * 10 ** 20) * np.array(elec_change[2]) * 
                          self.pol_struct.lattice.c / self.pol_struct.lattice.volume)
        
        elec_contrib = 100 * np.array([elec_contrib_x, elec_contrib_y, elec_contrib_z]) # muC / cm^2
        
        
        print(f'electronic contribution: {elec_contrib}')
        print(f'ionic contribution: {ion_contrib}')

        return ion_contrib + elec_contrib
    
    
    def compute_polarization(self):
        
        occ_fact = 2
        strings_sum, strings_len, dict_debug = self.compute_string_sums()
        print(f'string_sums: {strings_sum}')
        
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
        
        print(f'final_pol frac: {final_pol}')
        print(f'polarization: {P_norm} muC / cm^2')
        return P_norm, dict_debug
