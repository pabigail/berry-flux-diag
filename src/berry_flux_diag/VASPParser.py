import logging
from concurrent.futures import ThreadPoolExecutor

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Potcar
from pymatgen.analysis.ferroelectricity.polarization import zval_dict_from_potcar
import numpy as np
import berry_flux_diag.paw as paw
import berry_flux_diag.utils as utils
from pathlib import Path

logger = logging.getLogger(__name__)

# Cutoff in eV for the PAW momentum grid. None means 4 x the run's ENCUT.
#
# 4 x ENCUT is the physically motivated choice: the matrix elements are
# <psi|exp(iG.r)|psi>, which needs G out to twice the wavefunction's G_max,
# and energy goes as G squared, so twice the radius is four times the
# energy. The grid holds every G at or below the cutoff, and
# get_reciprocal_fullfw projects each state onto it, so the cutoff is the
# size of the basis the wavefunction is re-expressed in. It scales with the
# run, where a fixed number cannot.
#
# This used to be pinned at 1000 eV, a historical value rather than a
# chosen one. For BaTiO3 (ENCUT = 680 eV) the convergence with cutoff is
#
#                         G vectors   electronic    ionic      total   (uC/cm^2)
#     1000 eV (old)          4833      17.1839    28.6538    45.8377
#     3 x ENCUT, 2040 eV    14111      17.1446    28.6538    45.7984
#     4 x ENCUT, 2720 eV    21671      17.1337    28.6538    45.7875   <- default
#     6 x ENCUT, 4080 eV    39955      17.1297    28.6538    45.7835
#
# The steps shrink about as cutoff^-4. On that rough three-point fit, 4 x
# ENCUT is within about 0.005 uC/cm^2 (0.03% of the electronic term) of the
# converged value, against 0.054 for the old 1000 eV. 6 x ENCUT would roughly
# double the cost for a 0.004 change, so 4 x is kept.
#
# tests/reference/batio3_vasp_nospin.json pins the default row, and
# momentum_encut=1000 reproduces the old one. How fast this converges is
# material-dependent: harder potentials put more weight at large G, so a
# 6 x ENCUT spot-check is worth doing on a new material.
#
# The cost is in the grid size. The PAW step itself only roughly doubles
# (5.4 s -> 11.7 s per WAVECAR here), but everything downstream that loops
# over G vectors scales with their number, 4.5x here; end to end the BaTiO3
# capture went from about 2 to about 8 minutes, peaking at 5 GB.
MOMENTUM_ENCUT = None

def get_band_filling_from_wavecar_nospin(wavecar, tol):

    num_kpoints = np.array(wavecar.band_energy).shape[0]
    occupations = (wavecar.band_energy[kpt][:, 2] for kpt in range(num_kpoints))

    return utils.max_filled_bands(occupations, tol)

def get_band_filling_from_wavecar_spinpol(wavecar, tol, spin_channel):

    num_kpoints = np.array(wavecar.band_energy).shape[1]
    occupations = (wavecar.band_energy[spin_channel][kpt][:, 2]
                   for kpt in range(num_kpoints))

    return utils.max_filled_bands(occupations, tol)



# only works for non-spin-polarized wavefunctions...
def get_paw_wfcn_data(wavecar_file, potcar_file, vasprun_file,
                      max_band_fill, momentum_encut=MOMENTUM_ENCUT):
    """PAW-reconstructed coefficients on a k-independent momentum grid.

    See berry_flux_diag.paw for what is computed and how it was checked
    against the pawpyseed implementation it replaces.

    Parameters
    ----------
    momentum_encut : float or None
        Plane-wave cutoff in eV defining the momentum grid. Defaults to
        MOMENTUM_ENCUT, which is None: four times the run's ENCUT, the
        physically motivated choice - read the comment on it. Pass 1000
        to reproduce VASP numbers this code produced before that became
        the default.

    Returns
    -------
    gpoints : (ngpoints, 3) int array
    ngpoints : int
    wfc_data : (max_band_fill, nkpts, ngpoints) complex array
    """
    wf = paw.Wavefunction.from_files(wavecar_file, potcar_file, vasprun_file,
                                     nbands=max_band_fill)

    # Report the cutoff against the 4 x ENCUT default, so that a cutoff passed
    # in too small for the run says so at the time it is used rather than only
    # in a comment. The plane-wave count goes as the cutoff to the 3/2.
    default_encut = 4 * wf.encut

    if momentum_encut is None:
        logger.info("momentum grid cutoff: default, 4 x ENCUT = %s eV",
                    default_encut)
    elif momentum_encut < default_encut:
        logger.warning(
            "momentum grid cutoff is %s eV, below the default of "
            "4 x ENCUT = %s eV for this run: the grid holds roughly %.0f%% of "
            "the plane waves it would otherwise, so each state is projected "
            "onto a basis too small to hold it. See VASPParser.MOMENTUM_ENCUT.",
            momentum_encut, default_encut,
            100 * (momentum_encut / default_encut) ** 1.5)
    else:
        logger.info("momentum grid cutoff: %s eV (default is "
                    "4 x ENCUT = %s eV)", momentum_encut, default_encut)

    mm = paw.MomentumMatrix(wf, momentum_encut)

    gpoints = mm.momentum_grid
    ngpoints = len(gpoints)
    wfc_data = np.zeros((max_band_fill, wf.nwk, ngpoints), dtype=complex)

    for kpt_idx in range(wf.nwk):
        wfc_data[:, kpt_idx, :] = mm.get_reciprocal_fullfw_kpoint(kpt_idx, max_band_fill)

    return gpoints, ngpoints, wfc_data

def paw_wfcn_dict(wavecar_file, potcar_file, vasprun_file, max_band_fill,
                  kpoint_list, momentum_encut=MOMENTUM_ENCUT):
    """
    Generate wfcn_dict from PAW-reconstructed wavefunction data.
    The dictionary maps each k-point to its corresponding wavefunction coefficients and g-vectors.

    Inputs:
        wavecar_file (str): Path to WAVECAR
        potcar_file (str): Path to POTCAR
        vasprun_file (str): Path to vasprun.xml
        kpoint_list (List[np.ndarray]): List of k-points (fractional coordinates)
        momentum_encut (float, None): momentum grid cutoff in eV; defaults
            to MOMENTUM_ENCUT, and None means 4 x ENCUT.
            See get_paw_wfcn_data.

    Returns:
        dict: {kpt (tuple): {'wfcn': array(nbands, ngpoints), 'gvecs': array(ngpoints, 3)}}
    """
    gvecs, ngpoints, wfc_data = get_paw_wfcn_data(
        wavecar_file, potcar_file, vasprun_file, max_band_fill,
        momentum_encut=momentum_encut
    )

    nkpts = len(kpoint_list)

    wfcn_dict = {}

    for kpt_idx in range(nkpts):
        # Get wavefunction coefficients for this k-point (shape: nbands, ngpoints)
        wfcn = wfc_data[:, kpt_idx, :]  # shape: (nbands, ngpoints)
        coeff_gvec_dict = {
            'wfcn': wfcn,
            'gvecs': gvecs  # the momentum grid is the same for every k-point
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

    # Only the header records: k-points, occupations, spin count and type.
    # The coefficients are read once, by the PAW reconstruction below.
    # Reading the files with pymatgen's Wavecar as well used to cost about
    # 4.6 s per WAVECAR for BaTiO3 at 6x6x6, for coefficients nothing used.
    # The type is detected from the plane-wave count, as Wavecar did, rather
    # than asserted to be "std". Asserting it does not make it true: see
    # utils.check_wavecar_type for what a pinned "std" does to a noncollinear
    # WAVECAR.
    pol_wavecar = paw.WavecarHeader(pol_WAVECAR)
    np_wavecar = paw.WavecarHeader(np_WAVECAR)

    utils.check_wavecar_type(pol_wavecar.vasp_type, "polar")
    utils.check_wavecar_type(np_wavecar.vasp_type, "non-polar")

    potcar = Potcar.from_file(POTCAR)
    zval_dict = zval_dict_from_potcar(potcar)

    pol_vasprun = Path(pol_directory) / 'vasprun.xml'
    np_vasprun = Path(np_directory) / 'vasprun.xml'

    # determine whether spin-polarized from the header's spin count
    for wavecar, label in ((pol_wavecar, "polar"), (np_wavecar, "non-polar")):
        if wavecar.spin not in (1, 2):
            raise ValueError(f"{label} WAVECAR reports {wavecar.spin} spin "
                             "channels; expected 1 or 2")
    pol_spin_pol = pol_wavecar.spin == 2
    np_spin_pol = np_wavecar.spin == 2

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
        logger.info('max_band_fill: polar %s, non-polar %s', pol_band_fill, np_band_fill)
    
    # The two reconstructions are independent, so run them side by side.
    # Threads rather than processes: most of the time is in NumPy and BLAS,
    # which release the GIL, and a process would have to send its ~1.5 GB of
    # coefficients back. For BaTiO3 this took the pair from 14.7 s to 9.4 s
    # (processes: 17.8 s). Each run is computed exactly as it was serially;
    # result() re-raises anything either one raised.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pol_future = pool.submit(paw_wfcn_dict, pol_WAVECAR, POTCAR, pol_vasprun,
                                 pol_band_fill, kpoint_list,
                                 momentum_encut=momentum_encut)
        np_future = pool.submit(paw_wfcn_dict, np_WAVECAR, POTCAR, np_vasprun,
                                np_band_fill, kpoint_list,
                                momentum_encut=momentum_encut)
        pol_wfcn_dict_paw = pol_future.result()
        np_wfcn_dict_paw = np_future.result()

    vasp_parse_dict = utils.empty_parse_dict()
    vasp_parse_dict['pol_struct'] = pol_struct
    vasp_parse_dict['np_struct'] = np_struct
    vasp_parse_dict['pol_wfcn_dict'] = pol_wfcn_dict_paw
    vasp_parse_dict['np_wfcn_dict'] = np_wfcn_dict_paw
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
