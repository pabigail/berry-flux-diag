from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.outputs import Wavecar, Kpoints, Potcar, Outcar # clean later
from pymatgen.analysis.ferroelectricity.polarization import zval_dict_from_potcar
from pawpyseed.core.wavefunction import Wavefunction, CoreRegion
from pawpyseed.core.momentum import MomentumMatrix
from pawpyseed.core import pawpyc
import numpy as np
import BerryFluxDiag.utils as utils
from pathlib import Path

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



# remove following
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

# only works for non-spin-polarized wavefunctions...
def get_wfcn_data_from_vasp_pawpy(wavecar_file, potcar_file, vasprun_file, max_band_fill, cutoff=1000):
    
    # wavefunctions and momentum matrix objects
    vr = Vasprun(vasprun_file)
    structure = vr.final_structure
    dim = np.array([vr.parameters["NGX"], vr.parameters["NGY"], vr.parameters["NGZ"]])
    symprec = vr.parameters["SYMPREC"]
    potcar = Potcar.from_file(potcar_file)
    pwf = pawpyc.PWFPointer(wavecar_file, vr)
    wf = Wavefunction(structure, pwf, CoreRegion(potcar), dim, symprec, True)
    mm = MomentumMatrix(wf, cutoff)

    # g-point grid and number of k-points
    gpoints = mm.momentum_grid
    nkpts = wf.nwk
    # nbands = wf.nband
    nbands = max_band_fill
    TOL = 1e-6
    ngpoints = len(gpoints)

    wfc_data = np.zeros((nbands, nkpts, ngpoints), dtype=complex)

    for kpt_idx in range(nkpts):
        for band_idx in range(nbands):
            wfc_data[band_idx, kpt_idx, :] = mm.get_reciprocal_fullfw(band_idx, kpt_idx, 0)
    
    return gpoints, ngpoints, wfc_data

def wfcn_dict_from_pawpy(wavecar_file, potcar_file, vasprun_file, max_band_fill, kpoint_list):
    """
    Generate wfcn_dict from PAWPyseed wavefunction data.
    The dictionary maps each k-point to its corresponding wavefunction coefficients and g-vectors.
    
    Inputs:
        wavecar_file (str): Path to WAVECAR
        potcar_file (str): Path to POTCAR
        vasprun_file (str): Path to vasprun.xml
        kpoint_list (List[np.ndarray]): List of k-points (fractional coordinates)

    Returns:
        dict: {kpt (tuple): {'wfcn': array(nbands, ngpoints), 'gvecs': array(ngpoints, 3)}}
    """
    # Call the original function to get PAW-corrected data
    gvecs, ngpoints, wfc_data = get_wfcn_data_from_vasp_pawpy(
        wavecar_file, potcar_file, vasprun_file, max_band_fill
    )

    nkpts = len(kpoint_list)
    nbands = wfc_data.shape[0]

    wfcn_dict = {}

    for kpt_idx in range(nkpts):
        # Get wavefunction coefficients for this k-point (shape: nbands, ngpoints)
        wfcn = wfc_data[:, kpt_idx, :]  # shape: (nbands, ngpoints)
        coeff_gvec_dict = {
            'wfcn': wfcn,
            'gvecs': gvecs  # same for all k-points in pawpyseed's MomentumMatrix
        }
        wfcn_dict[tuple(kpoint_list[kpt_idx])] = coeff_gvec_dict

    return wfcn_dict


def get_band_filling_from_outcar(outcar, spin_pol, occ_tol=1e-6):
    # outcar is a pymatgen.io.vasp.outputs Outcar object
    outcar.read_eigenval()
    
    if spin_pol:
        occ_up = np.array(outcar.occupancies[0])
        occ_down = np.array(outcar.occupancies[1])
        fill_up = np.sum(occ_up[0] > occ_tol)
        fill_down = np.sum(occ_down[0] > occ_tol)
        return int(fill_up), int(fill_down)
    else:
        occ = np.array(outcar.occupancies)
        fill = np.sum(occ[0] > occ_tol)
        return int(fill)

def vasp_parser(pol_POSCAR, np_POSCAR, pol_WAVECAR, np_WAVECAR, POTCAR, pol_directory, np_directory):
   
    
     
    pol_struct = Structure.from_file(pol_POSCAR)
    np_struct = Structure.from_file(np_POSCAR)

    pol_wavecar = Wavecar(pol_WAVECAR, vasp_type="std")
    np_wavecar = Wavecar(np_WAVECAR, vasp_type="std")

    potcar = Potcar.from_file(POTCAR)
    zval_dict = zval_dict_from_potcar(potcar)

    pol_wfcn = Wavefunction.from_directory(pol_directory)
    np_wfcn = Wavefunction.from_directory(np_directory)
    
    pol_vasprun = Path(pol_directory) / 'vasprun.xml'
    np_vasprun = Path(np_directory) / 'vasprun.xml'

    # pol_wfcn_pawpy = get_wfcn_dict_from_vasp_pawpy(pol_WAVECAR, POTCAR,  pol_vasprun)

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
        pol_wfcn_dict, _, _ = get_wfcn_dict_from_vasp(pol_wavecar, kpoint_list, spin_pol)
        np_wfcn_dict, _, _ = get_wfcn_dict_from_vasp(np_wavecar, kpoint_list, spin_pol)
    else:
        pol_wfcn_dict, _ = get_wfcn_dict_from_vasp(pol_wavecar, kpoint_list, spin_pol)
        np_wfcn_dict, _ = get_wfcn_dict_from_vasp(np_wavecar, kpoint_list, spin_pol)

    TOL = 1e-6    

    if spin_pol:
        spin_up = 0
        spin_down = 1
        pol_band_fill_up = get_band_filling_from_wavecar_spinpol(pol_wavecar, TOL, spin_up)
        pol_band_fill_down = get_band_filling_from_wavecar_spinpol(pol_wavecar, TOL, spin_down)
        np_band_fill_up = get_band_filling_from_wavecar_spinpol(np_wavecar, TOL, spin_up)
        np_band_fill_down = get_band_filling_from_wavecar_spinpol(np_wavecar, TOL, spin_down)
    else:
        pol_band_fill = get_band_filling_from_wavecar_nospin(pol_wavecar, TOL)
        np_band_fill = get_band_filling_from_wavecar_nospin(np_wavecar, TOL)    
    
    pol_wfcn_dict_pawpy = wfcn_dict_from_pawpy(pol_WAVECAR, POTCAR, pol_vasprun, pol_band_fill, kpoint_list)
    np_wfcn_dict_pawpy = wfcn_dict_from_pawpy(np_WAVECAR, POTCAR, np_vasprun, np_band_fill, kpoint_list)

    vasp_parse_dict = utils.empty_parse_dict()
    vasp_parse_dict['pol_struct'] = pol_struct
    vasp_parse_dict['np_struct'] = np_struct
    vasp_parse_dict['pol_wfcn_dict'] = pol_wfcn_dict_pawpy
    vasp_parse_dict['np_wfcn_dict'] = np_wfcn_dict_pawpy
    vasp_parse_dict['kpoint_list'] = kpoint_list
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
