from pymatgen.core.structure import Structure
from pymatgen.io.pwscf import PWOutput
import qeschema
import h5py
import numpy as np
import BerryFluxDiag.utils as utils

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
    
    qe_parse_dict = utils.empty_parse_dict()
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