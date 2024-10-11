from preprocess_funcs import *

def main():

    # in the original files, need same number of atoms in the same order
    # tested with magnetic, spin-polarized material
    base_path = 'FeBiO3_AFM_cells/'
    orig_pol_qe_in_filename = base_path+'FeBiO3_AFM_k555_pol_orig_scf_u_3.in' # QE file alread prepared
    orig_np_qe_in_filename = base_path+'FeBiO3_AFM_k555_np_orig_scf_u_3.in' # QE file already prepared
    bfd_pol_qe_in_filename = base_path+'FeBiO3_AFM_k555_pol_orig_scf_u_3_nosym.in' # QE file that will write out
    bfd_np_qe_in_filename = base_path+'FeBiO3_AFM_k555_np_trans_scf_u_3_nosym.in' # QE file that will write out

    generate_qe_in_from_qe_in(orig_pol_qe_in_filename, orig_np_qe_in_filename, 
                          bfd_pol_qe_in_filename, bfd_np_qe_in_filename)
    
    return 0

if __name__ == '__main__':
    main()