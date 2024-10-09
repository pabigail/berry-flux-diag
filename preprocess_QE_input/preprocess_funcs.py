from itertools import product
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar

def calc_max_disp(struct_a, struct_b, trans):
    st_a_copy = struct_a.copy()
    st_a_copy.translate_sites(range(len(struct_a)),trans)
    max_disp = max(np.diag(st_a_copy.lattice.get_all_distances(st_a_copy.frac_coords,
                                                               struct_b.frac_coords)))
    
    return max_disp


# from Francesco
def get_refined_oshift(st_a,st_b,grid_range=0.1):
    max_disps = []
    ogrid = product(np.arange(-grid_range,grid_range,0.01),
                    np.arange(-grid_range,grid_range,0.01),
                    np.arange(-grid_range,grid_range,0.01))
    for oshift in ogrid:
        st_a_copy = st_a.copy()
        st_a_copy.translate_sites(range(len(st_a)),oshift)
        max_disps.append([oshift,max(np.diag(st_a_copy.lattice.get_all_distances(st_a_copy.frac_coords,
                                                                                 st_b.frac_coords)))])
    return max_disps

def make_one_translation(folder_path):
    pol_POSCAR_file = folder_path+'POSCAR_pol_orig'
    np_POSCAR_file = folder_path+'POSCAR_np_orig'
    # load in a structures
    pol_struct_orig = Structure.from_file(pol_POSCAR_file)
    np_struct_orig = Structure.from_file(np_POSCAR_file)
    # make copies so don't change original files
    pol_struct = pol_struct_orig.copy()
    np_struct = np_struct_orig.copy()
    # get displacements
    max_disps = get_refined_oshift(np_struct, pol_struct)
    sorted_max_disps = sorted(max_disps,key=lambda x: x[1])
    # pick 5 points along which to save translated POSCARS
    # translate NP wrt POL
    index_1 = 0
    trans_1 = sorted_max_disps[index_1][0]
    disp_1 = sorted_max_disps[index_1][1]

    print(f'min_disp: {sorted_max_disps[0][-1]}')

    # save translated NP poscars in respective folders
    base_save_path = folder_path
    save_file_1 = base_save_path+'POSCAR_np_trans_1' # this is the "best" translation

    np_trans_1 = np_struct_orig.copy()
    np_trans_1.translate_sites(range(len(np_trans_1)), trans_1)
    np_trans_1.to(filename=save_file_1, fmt='POSCAR')
    
    return np_trans_1, pol_struct, disp_1