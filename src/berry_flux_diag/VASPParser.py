import logging

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.outputs import Wavecar, Potcar
from pymatgen.analysis.ferroelectricity.polarization import zval_dict_from_potcar
from pawpyseed.core.wavefunction import Wavefunction, CoreRegion
from pawpyseed.core.momentum import MomentumMatrix
from pawpyseed.core import pawpyc
import numpy as np
import berry_flux_diag.utils as utils
from pathlib import Path

logger = logging.getLogger(__name__)

# Cutoff in eV for pawpyseed's momentum grid. PINNED, NOT JUSTIFIED.
#
# pawpyseed's own default is 4 x the run's ENCUT, and that is the physically
# motivated choice: the matrix elements are <psi|exp(iG.r)|psi>, which needs G
# out to twice the wavefunction's G_max, and energy goes as G squared, so twice
# the radius is four times the energy. The grid holds every G at or below the
# cutoff, and get_reciprocal_fullfw projects each state onto it, so the cutoff
# is the size of the basis the wavefunction is re-expressed in.
#
# 1000 eV is a fixed number standing in for a quantity that scales with ENCUT.
# For a run at ENCUT = 520 eV the cutoff should be 2080 eV, so this grid holds
# roughly (1000/2080)^(3/2) - about a third - of the plane waves pawpyseed
# considers necessary, and every state is projected onto a basis too small to
# hold it. The shortfall is material-dependent: softer potentials are affected
# less, harder ones more.
#
# It is kept only because moving it moves the VASP polarization, and there is
# nothing yet to move it against - tests/reference/batio3_vasp_nospin.json does
# not exist, because capturing it needs MKL and so has to happen on NERSC.
# Capture that fixture at this value first, then set this to None (pawpyseed's
# default), recapture, and record both numbers.
MOMENTUM_ENCUT = 1000

def get_band_filling_from_wavecar_nospin(wavecar, tol):

    num_kpoints = np.array(wavecar.band_energy).shape[0]
    occupations = (wavecar.band_energy[kpt][:, 2] for kpt in range(num_kpoints))

    return utils.max_filled_bands(occupations, tol)

def get_band_filling_from_wavecar_spinpol(wavecar, tol, spin_channel):

    num_kpoints = np.array(wavecar.band_energy).shape[1]
    occupations = (wavecar.band_energy[spin_channel][kpt][:, 2]
                   for kpt in range(num_kpoints))

    return utils.max_filled_bands(occupations, tol)



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
        logger.info('max_band_fill_up: %s', max_band_fill_up)
        logger.info('max_band_fill_down: %s', max_band_fill_down)
    else:
        max_band_fill = get_band_filling_from_wavecar_nospin(wavecar, TOL)
        logger.info('max_band_fill: %s', max_band_fill)

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
def get_wfcn_data_from_vasp_pawpy(wavecar_file, potcar_file, vasprun_file,
                                  max_band_fill, momentum_encut=MOMENTUM_ENCUT):
    """PAW-reconstructed coefficients on pawpyseed's momentum grid.

    Parameters
    ----------
    momentum_encut : float or None
        Plane-wave cutoff in eV defining pawpyseed's momentum grid.
        Defaults to MOMENTUM_ENCUT, which is pinned at the historical
        value rather than chosen - read the comment on it. Pass None for
        pawpyseed's own default of four times the run's ENCUT, which is
        the physically motivated choice, at the cost of changing the
        answer relative to every VASP number this code has produced so
        far.
    """
    # wavefunctions and momentum matrix objects
    vr = Vasprun(vasprun_file)
    structure = vr.final_structure
    dim = np.array([vr.parameters["NGX"], vr.parameters["NGY"], vr.parameters["NGZ"]])
    symprec = vr.parameters["SYMPREC"]
    potcar = Potcar.from_file(potcar_file)
    pwf = pawpyc.PWFPointer(wavecar_file, vr)
    wf = Wavefunction(structure, pwf, CoreRegion(potcar), dim, symprec, True)

    # Report the cutoff against pawpyseed's own, so that a pinned value too
    # small for the run says so at the time it is used rather than only in a
    # comment. The plane-wave count goes as the cutoff to the 3/2.
    pawpy_default = 4 * wf.encut

    if momentum_encut is None:
        logger.info("momentum grid cutoff: pawpyseed default, 4 x ENCUT = %s eV",
                    pawpy_default)
    elif momentum_encut < pawpy_default:
        logger.warning(
            "momentum grid cutoff is %s eV, below pawpyseed's default of "
            "4 x ENCUT = %s eV for this run: the grid holds roughly %.0f%% of "
            "the plane waves it would otherwise, so each state is projected "
            "onto a basis too small to hold it. See VASPParser.MOMENTUM_ENCUT.",
            momentum_encut, pawpy_default,
            100 * (momentum_encut / pawpy_default) ** 1.5)
    else:
        logger.info("momentum grid cutoff: %s eV (pawpyseed default is "
                    "4 x ENCUT = %s eV)", momentum_encut, pawpy_default)

    mm = MomentumMatrix(wf, momentum_encut)

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

def wfcn_dict_from_pawpy(wavecar_file, potcar_file, vasprun_file, max_band_fill,
                         kpoint_list, momentum_encut=MOMENTUM_ENCUT):
    """
    Generate wfcn_dict from PAWPyseed wavefunction data.
    The dictionary maps each k-point to its corresponding wavefunction coefficients and g-vectors.

    Inputs:
        wavecar_file (str): Path to WAVECAR
        potcar_file (str): Path to POTCAR
        vasprun_file (str): Path to vasprun.xml
        kpoint_list (List[np.ndarray]): List of k-points (fractional coordinates)
        momentum_encut (float, None): momentum grid cutoff in eV; defaults
            to the pinned MOMENTUM_ENCUT, None uses pawpyseed's 4 x ENCUT.
            See get_wfcn_data_from_vasp_pawpy.

    Returns:
        dict: {kpt (tuple): {'wfcn': array(nbands, ngpoints), 'gvecs': array(ngpoints, 3)}}
    """
    # Call the original function to get PAW-corrected data
    gvecs, ngpoints, wfc_data = get_wfcn_data_from_vasp_pawpy(
        wavecar_file, potcar_file, vasprun_file, max_band_fill,
        momentum_encut=momentum_encut
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

def vasp_parser(pol_POSCAR, np_POSCAR, pol_WAVECAR, np_WAVECAR, POTCAR,
                pol_directory, np_directory, momentum_encut=MOMENTUM_ENCUT):
   
    
     
    pol_struct = Structure.from_file(pol_POSCAR)
    np_struct = Structure.from_file(np_POSCAR)
    utils.check_species_match(pol_struct, np_struct)

    # vasp_type is left for pymatgen to detect from the plane-wave count,
    # rather than asserted to be "std". Asserting it does not make it true:
    # see utils.check_wavecar_type for what a pinned "std" does to a
    # noncollinear WAVECAR.
    pol_wavecar = Wavecar(pol_WAVECAR)
    np_wavecar = Wavecar(np_WAVECAR)

    utils.check_wavecar_type(pol_wavecar.vasp_type, "polar")
    utils.check_wavecar_type(np_wavecar.vasp_type, "non-polar")

    potcar = Potcar.from_file(POTCAR)
    zval_dict = zval_dict_from_potcar(potcar)

    pol_wfcn = Wavefunction.from_directory(pol_directory)
    np_wfcn = Wavefunction.from_directory(np_directory)
    
    pol_vasprun = Path(pol_directory) / 'vasprun.xml'
    np_vasprun = Path(np_directory) / 'vasprun.xml'

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

    # round k-point lists so can find matching k-points
    kpoint_list = [np.around(kpt, 6) for kpt in pol_wavecar.kpoints]
    np_kpoint_list = [np.around(kpt, 6) for kpt in np_wavecar.kpoints]

    # Only the polar list is carried forward, and each run's coefficients are
    # read positionally against it, so the two runs must agree exactly.
    utils.check_kpoints_match(kpoint_list, np_kpoint_list)
    utils.check_full_bz(kpoint_list)


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
    
    pol_wfcn_dict_pawpy = wfcn_dict_from_pawpy(pol_WAVECAR, POTCAR, pol_vasprun,
                                               pol_band_fill, kpoint_list,
                                               momentum_encut=momentum_encut)
    np_wfcn_dict_pawpy = wfcn_dict_from_pawpy(np_WAVECAR, POTCAR, np_vasprun,
                                              np_band_fill, kpoint_list,
                                              momentum_encut=momentum_encut)

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
