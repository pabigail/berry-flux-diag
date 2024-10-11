from preprocess_funcs import *

def main():
    CrO3_path = 'CrO3_fe_db_cells/'
    CrO3_np_trans_struct, CrO3_pol_struct, CrO3_max_disp = make_one_translation(CrO3_path)

    file_path = 'CrO3_fe_db_cells/'
    out_path = file_path
    material = 'CrO3'
    pseudo_dir = '/global/homes/p/pabigail/qe_pseudos/' # change to path to pseudo directory
    kshift = (0, 0, 0)
    sym = False 
    kpoints_grid = (7, 7, 7)
    species_dict = {"species": ["Cr", "O"], # update with appropriate information
                "masses": [51.9961, 15.9994],
                "psuedos": ["Cr_ONCV_PBE-1.0.upf", # these need to be in the pseudo_dir
                            "O_ONCV_PBE-1.0.upf"]}

    # np_trans_1
    poscar_file = file_path+'POSCAR_np_trans_1'
    tag = "np_trans_1"
    poscar_to_qe_io_scf_nomag(file_path, poscar_file, material, tag, sym, pseudo_dir,
                          kpoints_grid, kshift, species_dict, out_path)

    # pol_orig
    poscar_file = file_path+'POSCAR_pol_orig'
    tag = "pol_orig"
    poscar_to_qe_io_scf_nomag(file_path, poscar_file, material, tag, sym, pseudo_dir,
                          kpoints_grid, kshift, species_dict, out_path)


    return 0

if __name__ == '__main__':
    main()