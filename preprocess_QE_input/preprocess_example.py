from preprocess_funcs import *

def main():
    CrO3_path = 'CrO3_fe_db_cells/'
    CrO3_np_trans_struct, CrO3_pol_struct, CrO3_max_disp = make_one_translation(CrO3_path)
    return 0

if __name__ == '__main__':
    main()