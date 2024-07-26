import numpy as np


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