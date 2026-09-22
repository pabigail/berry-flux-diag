"""VASP parsing without the PAW augmentation terms.

WHAT THIS IS FOR
----------------
This module is a deliberate parallel to VASPParser that reads a WAVECAR
using pymatgen alone. It imports no pawpyseed, and so it is the only VASP
path that runs on a machine without the Intel MKL - a Mac, most obviously.
Use it to exercise the workflow end to end: preprocessing, string
construction, the overlap and diagonalization machinery, the jobflow
plumbing, and the shape of the parse dictionary.

THE POLARIZATION IT PRODUCES IS NOT PHYSICALLY CORRECT
------------------------------------------------------
A WAVECAR stores pseudo-wavefunction coefficients. In the PAW formalism
these are not the all-electron wavefunctions, and they are orthonormal
only under the PAW overlap operator, not under the plain plane-wave inner
product used here. Reconstructing the missing augmentation contribution
inside each PAW sphere is exactly what pawpyseed does for VASPParser.

Without it the overlap matrices are not unitary and the Berry phases are
wrong. Only the electronic term is affected; the ionic term comes from
ZVAL and site positions, so it is unchanged. For the BaTiO3 case in
tests/ (6x6x6 mesh, VASP PAW potentials):

                      electronic    ionic      total
    pawpyseed            17.18      28.65      45.84
    this module          16.13      28.65      44.78
                        -6.1%         --      -2.3%

The 2.3% figure on the total is flattering: it is diluted by an ionic
term that was never in question. The error lives entirely in the 6%, and
that number is an uncontrolled, species-dependent artifact rather than a
bounded approximation - it is not a constant offset, and there is no
reason to expect a comparable size for a different material. Never quote,
plot or publish a number from this module. For a correct VASP result use
VASPParser on a machine with MKL.

WHY IT IS NOT IMPORTED BY __init__.py
--------------------------------------
Reaching it requires naming it, so that no one gets these numbers by
accident when pawpyseed happens to be missing:

    from berry_flux_diag import VASPParser_unnormalized as vasp_unnorm
    parse_dict = vasp_unnorm.vasp_parser(pol_POSCAR, np_POSCAR,
                                         pol_WAVECAR, np_WAVECAR, POTCAR)

Note the signature differs from VASPParser.vasp_parser, which also takes
the two run directories that pawpyseed needs. The two are not drop-in
replacements for one another, which is intentional.

STATUS
------
Temporary. pawpyseed was last released in 2021 and its MKL requirement is
what forces this split in the first place; replacing it with a
maintainable PAW overlap implementation would remove the need for this
module. Until then it stays, and it stays documented.
"""

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Wavecar, Kpoints, Potcar
from pymatgen.analysis.ferroelectricity.polarization import zval_dict_from_potcar
import numpy as np
import berry_flux_diag.utils as utils


def get_band_filling_from_wavecar_nospin(wavecar, tol):

    num_kpoints = np.array(wavecar.band_energy).shape[0]
    occupations = (wavecar.band_energy[kpt][:, 2] for kpt in range(num_kpoints))

    return utils.max_filled_bands(occupations, tol)


def get_band_filling_from_wavecar_spinpol(wavecar, tol, spin_channel):

    num_kpoints = np.array(wavecar.band_energy).shape[1]
    occupations = (wavecar.band_energy[spin_channel][kpt][:, 2]
                   for kpt in range(num_kpoints))

    return utils.max_filled_bands(occupations, tol)


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
    """Build a parse dictionary from pseudo-wavefunctions alone.

    Takes five paths, where VASPParser.vasp_parser takes seven: the two run
    directories are omitted because only pawpyseed needs them.

    The result drives the rest of the workflow unchanged, but the
    polarization computed from it is not physically correct - the PAW
    augmentation terms are missing. See the module docstring.
    """
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
