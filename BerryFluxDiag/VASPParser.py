from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Wavecar, Kpoints, Potcar
from pymatgen.analysis.ferroelectricity.polarization import zval_dict_from_potcar
import numpy as np
import BerryFluxDiag.utils as utils


def get_band_filling_from_wavecar_nospin(wavecar, tol):

    max_band_fill = 0
    num_kpoints = np.array(wavecar.band_energy).shape[0]

    for kpt in range(0, num_kpoints):

        band_filling = wavecar.band_energy[kpt][:, 2]
        temp_band_index = next(index for index, i in enumerate(band_filling) if i < tol)
        if temp_band_index != 0:
            if (temp_band_index - 1) > max_band_fill:
                max_band_fill = temp_band_index
        else:
            raise ValueError("band filling is zero")

    return max_band_fill


def get_band_filling_from_wavecar_spinpol(wavecar, tol, spin_channel):

    max_band_fill = 0
    num_kpoints = np.array(wavecar.band_energy).shape[1]

    for kpt in range(0, num_kpoints):
        band_filling = wavecar.band_energy[spin_channel][kpt][:, 2]
        temp_band_index = next(index for index, i in enumerate(band_filling) if i < tol)
        if temp_band_index != 0:
            if (temp_band_index - 1) > max_band_fill:
                max_band_fill = temp_band_index
        else:
            raise ValueError("band filling is zero")

    return max_band_fill


def get_wfcn_dict_from_vasp(wavecar, kpoint_list, spin_pol):
    # key of wfcn dict is a k-point
    # each k-point has associated dict with coeffs and gvecs

    wfcn_dict = {}
    all_coeffs = wavecar.coeffs
    all_gvecs = wavecar.Gpoints

    TOL = 1e-6

    if spin_pol:
        spin_up = 0
        spin_down = 1
        max_band_fill_up = get_band_filling_from_wavecar_spinpol(wavecar, TOL, spin_up)
        max_band_fill_down = get_band_filling_from_wavecar_spinpol(wavecar, TOL, spin_down)
        print(f'max_band_fill_up: {max_band_fill_up}')
        print(f'max_band_fill_down: {max_band_fill_down}')
    else:
        max_band_fill = get_band_filling_from_wavecar_nospin(wavecar, TOL)
        print(f'max_band_fill: {max_band_fill}')

    num_kpts = len(kpoint_list)
    for index in range(0, num_kpts):
        if spin_pol:
            spin_up = 0
            spin_down = 1
            wfcn_up = np.array(all_coeffs[spin_up][index][0:max_band_fill_up])
            wfcn_down = np.array(all_coeffs[spin_down][index][0:max_band_fill_down])
            wfcn = []
            wfcn.append(wfcn_up)
            wfcn.append(wfcn_down)
            wfcn = np.array(wfcn, dtype=object)
        else:
            wfcn = np.array(all_coeffs[index][0:max_band_fill])
        gvecs = all_gvecs[index]

        coeff_gvec_dict = {}
        coeff_gvec_dict['wfcn'] = wfcn
        coeff_gvec_dict['gvecs'] = gvecs
        wfcn_dict[tuple(kpoint_list[index])] = coeff_gvec_dict

    if spin_pol:
        return wfcn_dict, max_band_fill_up, max_band_fill_down
    else:
        return wfcn_dict, max_band_fill


def vasp_parser(pol_POSCAR, np_POSCAR, pol_WAVECAR, np_WAVECAR, POTCAR):
    
    pol_struct = Structure.from_file(pol_POSCAR)
    np_struct = Structure.from_file(np_POSCAR)

    pol_wavecar = Wavecar(pol_WAVECAR)
    np_wavecar = Wavecar(np_WAVECAR)

    potcar = Potcar.from_file(POTCAR)
    zval_dict = zval_dict_from_potcar(potcar)

    # determine whether spin-polarized from shape of coefficient array
    if len(np.shape(np.array(pol_wavecar.coeffs, dtype=object))) == 3:
        pol_spin_pol = True
    elif len(np.shape(np.array(pol_wavecar.coeffs, dtype=object))) == 2:
        pol_spin_pol = False
    else:
        raise ValueError("dimensions of polar WAVECAR coefficients are inconsistent")

    if len(np.shape(np.array(np_wavecar.coeffs, dtype=object))) == 3:
        np_spin_pol = True
    elif len(np.shape(np.array(np_wavecar.coeffs, dtype=object))) == 2:
        np_spin_pol = False
    else:
        raise ValueError("dimensions of non-polar WAVECAR coefficients are inconsistent")

    if pol_spin_pol != np_spin_pol:
        raise ValueError("polar and non-polar spin polarizations are inconsistent")

    spin_pol = pol_spin_pol # Boolean if calculation is spin polarized or not

    # only works for full Brillouin zone
    kpoint_list = pol_wavecar.kpoints
    # round k-point list so can find matching k-points
    kpoint_list = [np.around(kpt, 6) for kpt in kpoint_list]


    if spin_pol:
        pol_wfcn_dict, pol_band_fill_up, pol_band_fill_down = get_wfcn_dict_from_vasp(pol_wavecar, kpoint_list, spin_pol)
        np_wfcn_dict, np_band_fill_up, np_band_fill_down = get_wfcn_dict_from_vasp(np_wavecar, kpoint_list, spin_pol)
    else:
        pol_wfcn_dict, pol_band_fill = get_wfcn_dict_from_vasp(pol_wavecar, kpoint_list, spin_pol)
        np_wfcn_dict, np_band_fill = get_wfcn_dict_from_vasp(np_wavecar, kpoint_list, spin_pol)
    
    vasp_parse_dict = utils.empty_parse_dict()
    vasp_parse_dict['pol_struct'] = pol_struct
    vasp_parse_dict['np_struct'] = np_struct
    vasp_parse_dict['pol_wfcn_dict'] = pol_wfcn_dict
    vasp_parse_dict['np_wfcn_dict'] = np_wfcn_dict
    vasp_parse_dict['zval_dict'] = zval_dict
    vasp_parse_dict['ES_code'] = 'VASP'
    vasp_parse_dict['spin_pol'] = spin_pol

    if spin_pol:
        vasp_parse_dict['pol_band_fill_up']= pol_band_fill_up
        vasp_parse_dict['pol_band_fill_down']= pol_band_fill_down
        vasp_parse_dict['np_band_fill_up']= np_band_fill_up
        vasp_parse_dict['np_band_fill_down']= np_band_fill_down
    else:
        vasp_parse_dict['pol_band_fill'] = pol_band_fill
        vasp_parse_dict['np_band_fill'] = np_band_fill

    return vasp_parse_dict