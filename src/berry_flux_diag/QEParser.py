import logging

from pymatgen.core.structure import Structure
from pymatgen.io.pwscf import PWOutput
import qeschema
import h5py
import numpy as np
import berry_flux_diag.utils as utils
from berry_flux_diag.constants import BOHR_TO_ANGSTROM

logger = logging.getLogger(__name__)

def get_struct_from_qeschema(xml_data):

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


def get_spin_pol_from_qeschema(xml_data):
    '''returns False if calc is non-spin polarized
    returns True if calc is spin-polarized'''
    spin_data = xml_data['qes:espresso']['input']['spin'] 
    lsda_bool = spin_data['lsda']
    noncolin_bool = spin_data['noncolin']
    spinorbit_bool = spin_data['spinorbit'] 
    return lsda_bool or noncolin_bool or spinorbit_bool


def get_kpt_list_from_qeschema(xml_data):
    
    ks_energies = xml_data['qes:espresso']['output']['band_structure']['ks_energies']
    
    kpoint_list = []
    for kpt in ks_energies:
        kpoint_list.append(np.array(kpt['k_point']['$']))
        
    return kpoint_list


def get_band_filling_from_qeschema_nospin(xml_data, tol):

    band_struct_dict = xml_data['qes:espresso']['output']['band_structure']
    ks_energies = band_struct_dict['ks_energies']

    occupations = (kpt['occupations']['$'] for kpt in ks_energies)

    return utils.max_filled_bands(occupations, tol)

def get_band_filling_from_qeschema_spinpol(xml_data, tol):

    band_struct_dict = xml_data['qes:espresso']['output']['band_structure']
    ks_energies = band_struct_dict['ks_energies']
    nband_up = band_struct_dict['nbnd_up']
    nband_dw = band_struct_dict['nbnd_dw']

    # QE writes both spin channels end to end in one occupations array.
    occupations_up = (kpt['occupations']['$'][0:nband_up] for kpt in ks_energies)
    occupations_dw = (kpt['occupations']['$'][nband_up:nband_up + nband_dw]
                      for kpt in ks_energies)

    max_band_fill_up = utils.max_filled_bands(occupations_up, tol)
    max_band_fill_dw = utils.max_filled_bands(occupations_dw, tol)

    return max_band_fill_up, max_band_fill_dw


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


def get_wfcn_gvecs_from_hdf5_nospin(wfc_folder_path, kpoint_list, max_band_fill):

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


def get_wfcn_gvecs_from_hdf5_spinpol(wfc_folder_path, kpoint_list, max_band_fill_up, max_band_fill_dw):

    wfcn_dict = {}

    num_kpts = len(kpoint_list)
    for index in range(0, num_kpts):
        filename_up = wfc_folder_path+'wfcup'+str(index+1)+'.hdf5'
        wfcn_up = get_wavefunctions_hdf5(filename_up, stop_band=max_band_fill_up)
        gvecs_up = get_wfc_miller_indices(filename_up)

        filename_dw = wfc_folder_path+'wfcdw'+str(index+1)+'.hdf5'
        wfcn_dw = get_wavefunctions_hdf5(filename_dw, stop_band=max_band_fill_dw)
        gvecs_dw = get_wfc_miller_indices(filename_dw)

        wfcn = []
        wfcn.append(wfcn_up)
        wfcn.append(wfcn_dw)
        wfcn = np.array(wfcn, dtype=object)

        if not utils.check_gvecs_same(gvecs_up, gvecs_dw):
            raise ValueError("gvecs not same for up and down spin channels")

        gvecs = gvecs_up

        coeff_gvec_dict = {}
        coeff_gvec_dict['wfcn'] = wfcn
        coeff_gvec_dict['gvecs'] = gvecs
        wfcn_dict[tuple(kpoint_list[index])] = coeff_gvec_dict

    return wfcn_dict


def get_zval_dict_from_PWOutput(pw_out):
    """Valence charge per element, read from the pseudopotential banner.

    Quantum ESPRESSO prints one block per species, for example::

        PseudoPot. # 1 for Ba read from file:
        /path/to/Ba_ONCV_PBE-1.0.upf
        MD5 check sum: e597b880c13c16463245d39d15e4590c
        Pseudo is Norm-conserving, Zval = 10.0

    Each Zval is matched to its element by position - it is the Zval line
    lying between this element's line and the next element's - rather than
    by a fixed offset from the element line. The block is not a fixed
    height: the MD5 line is absent in older QE versions, which shifts Zval
    up by one.
    """
    zval_pattern = {'element': 'for\\s+(\\w+)\\s+read\\sfrom\\sfile',
                'zval': 'Zval\\s+=\\s+([\\d+\\.]+)\\s'}

    pw_out.read_pattern(zval_pattern)
    pw_out_data = pw_out.data

    elements = pw_out_data.get('element') or []
    zvals = pw_out_data.get('zval') or []

    if not elements:
        raise ValueError(
            f"no pseudopotential blocks found in {pw_out.filename}; it may be "
            f"truncated, or from a run that did not reach the banner"
        )

    if len(elements) != len(zvals):
        raise ValueError(
            f"{pw_out.filename} has {len(elements)} pseudopotential blocks but "
            f"{len(zvals)} Zval lines; every species must report a valence charge"
        )

    zval_dict = {}
    for index, (element_entry, zval_entry) in enumerate(zip(elements, zvals)):
        element, element_line = element_entry[0][0], element_entry[1]
        zval, zval_line = float(zval_entry[0][0]), zval_entry[1]

        # The pairing is only trustworthy if this Zval falls inside this
        # element's block, so check it rather than assume the order holds.
        next_element_line = (elements[index + 1][1]
                             if index + 1 < len(elements) else float('inf'))

        if not element_line < zval_line < next_element_line:
            raise ValueError(
                f"in {pw_out.filename}, the Zval on line {zval_line + 1} does not "
                f"fall inside the block for {element} starting on line "
                f"{element_line + 1}; the pseudopotential output could not be "
                f"read reliably"
            )

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
    utils.check_species_match(pol_struct, np_struct)

    # round k-point lists so can find matching k-points
    kpoint_list = [np.around(kpt, 6)
                   for kpt in get_kpt_list_from_qeschema(pol_xml_data)]
    np_kpoint_list = [np.around(kpt, 6)
                      for kpt in get_kpt_list_from_qeschema(np_xml_data)]

    # Only the polar list is carried forward, so the two runs must agree.
    utils.check_kpoints_match(kpoint_list, np_kpoint_list)
    utils.check_full_bz(kpoint_list)


    pw_out = PWOutput(pw_out_path)
    zval_dict = get_zval_dict_from_PWOutput(pw_out)
    spin_pol_pol = get_spin_pol_from_qeschema(pol_xml_data)
    spin_pol_np = get_spin_pol_from_qeschema(np_xml_data)

    if spin_pol_pol != spin_pol_np:
        raise ValueError("two reference calculations have different spin-polarizations")

    spin_pol = spin_pol_pol
    FILLING_TOL = 1e-6

    if spin_pol:
        pol_max_band_fill_up, pol_max_band_fill_dw = get_band_filling_from_qeschema_spinpol(pol_xml_data, FILLING_TOL)
        np_max_band_fill_up, np_max_band_fill_dw = get_band_filling_from_qeschema_spinpol(np_xml_data, FILLING_TOL)
        if (pol_max_band_fill_up != np_max_band_fill_up or
            pol_max_band_fill_dw != np_max_band_fill_dw):
            logger.warning("max band filling differs between the polar (up %s, down %s) and non-polar (up %s, down %s) runs; using the smaller of each",
                           pol_max_band_fill_up, pol_max_band_fill_dw,
                           np_max_band_fill_up, np_max_band_fill_dw)
        common_max_band_fill_up = np.min([pol_max_band_fill_up,
                                          np_max_band_fill_up])
        common_max_band_fill_dw = np.min([pol_max_band_fill_dw,
                                          np_max_band_fill_dw])
    else:
        np_max_band_fill = get_band_filling_from_qeschema_nospin(np_xml_data,
                                                                 FILLING_TOL)
        pol_max_band_fill = get_band_filling_from_qeschema_nospin(pol_xml_data,
                                                                  FILLING_TOL)
        if pol_max_band_fill != np_max_band_fill:
            logger.warning("max band filling differs between the polar (%s) and non-polar (%s) runs; using the smaller",
                           pol_max_band_fill, np_max_band_fill)
        common_max_band_fill = np.min([pol_max_band_fill, np_max_band_fill])

    if spin_pol:
        pol_wfcn_dict = get_wfcn_gvecs_from_hdf5_spinpol(pol_wfcn_path,
                                                         kpoint_list,
                                                         common_max_band_fill_up,
                                                         common_max_band_fill_dw)
        np_wfcn_dict = get_wfcn_gvecs_from_hdf5_spinpol(np_wfcn_path,
                                                        kpoint_list,
                                                        common_max_band_fill_up,
                                                        common_max_band_fill_dw)
    else:
        pol_wfcn_dict = get_wfcn_gvecs_from_hdf5_nospin(pol_wfcn_path,
                                                        kpoint_list,
                                                        common_max_band_fill)
        np_wfcn_dict = get_wfcn_gvecs_from_hdf5_nospin(np_wfcn_path,
                                                       kpoint_list,
                                                       common_max_band_fill)
    
    qe_parse_dict = utils.empty_parse_dict()
    qe_parse_dict['pol_struct'] = pol_struct
    qe_parse_dict['np_struct'] = np_struct
    qe_parse_dict['kpoint_list'] = kpoint_list
    qe_parse_dict['zval_dict'] = zval_dict
    qe_parse_dict['ES_code'] = 'QE'
    qe_parse_dict['spin_pol'] = spin_pol
    qe_parse_dict['pol_wfcn_dict'] = pol_wfcn_dict
    qe_parse_dict['np_wfcn_dict'] = np_wfcn_dict
    if spin_pol:
        qe_parse_dict['pol_band_fill_up']= pol_max_band_fill_up
        qe_parse_dict['pol_band_fill_down']= pol_max_band_fill_dw
        qe_parse_dict['np_band_fill_up']= np_max_band_fill_up
        qe_parse_dict['np_band_fill_down']= np_max_band_fill_dw
    else:
        qe_parse_dict['pol_band_fill'] = pol_max_band_fill
        qe_parse_dict['np_band_fill'] = np_max_band_fill

    return qe_parse_dict
