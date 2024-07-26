from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Wavecar, Kpoints, Potcar
from pymatgen.analysis.ferroelectricity.polarization import zval_dict_from_potcar
import numpy as np
import BerryFluxDiag.utils as utils


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
    
    vasp_parse_dict = utils.empty_parse_dict()
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