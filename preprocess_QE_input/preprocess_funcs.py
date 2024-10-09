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



def poscar_to_qe_io_scf_nomag(file_path, poscar_file, material, tag, sym,
                          pseudo_dir, kpoints_grid, kshift, species_dict, out_path):
    '''
    file_path: file where poscar is located and where QE.in file will be written to
    poscar_file: name of poscar file
    material: name of material in string form, ex "BaTiO3"
    tag: for naming prefix and QE input file ex "np_orig" or "trans_1" or "interp_1"
    sym: boolean (true or false)
    pseudo_dir: location of psuedopotential files
    kpoints_grid: ex (5, 5, 5)
    kshift: either (0, 0, 0) or (1, 1, 1)
    species_dict: contains atoms and materials
    out_path: where to write file out to 
    '''
    
    poscar = Poscar.from_file(poscar_file)
    structure = poscar.structure
    k1, k2, k3 = kpoints_grid
    name = material+'_'+tag

    control_dict = {
    "calculation": "scf",
    "prefix": name,
    "pseudo_dir": pseudo_dir,
    "outdir": 'calculations_'+str(k1)+str(k2)+str(k3)+'/',
    "verbosity": "high"
    }

    system_dict = {
    "ibrav": 0,
    "ecutwfc": 100,
    "input_dft": "pbe",
    "nat": structure.num_sites,
    "ntyp": len(structure.composition.elements),
    }
    
    if not sym:
        system_dict["nosym"] = True
        system_dict["noinv"] = True

    
    electrons_dict = {
    "conv_thr": 1e-8,
    "mixing_beta": 0.7,
    "mixing_mode": 'plain',
    "mixing_ndim": 8,
    "diagonalization": 'david'
    }
    
    if sym:
        sym_str = "sym"
    else:
        sym_str = "nosym"

    filename = out_path+name+'_k'+str(k1)+str(k2)+str(k3)+'_scf_'+sym_str+'.in'

    with open(filename, "w") as f:
    
        f.write("&CONTROL\n")
        for key, value in control_dict.items():
            if isinstance(value, str):
                f.write(f"  {key} = '{value}'\n")
            else:
                f.write(f"  {key} = {value}\n") 
        f.write("  /\n")
    
        f.write(" &SYSTEM\n")
        for key, value in system_dict.items():
            if isinstance(value, bool):
                value_str = ".TRUE." if value else ".FALSE."
                f.write(f"  {key} = {value_str}\n")
            elif isinstance(value, str):
                f.write(f"  {key} = '{value}'\n")
            else:
                f.write(f"  {key} = {value}\n")
        f.write("  /\n")
    
        f.write("  &ELECTRONS\n")
        for key, value in electrons_dict.items():
            if key == "conv_thr":
                # Special case for scientific notation in Quantum Espresso style
                f.write(f"  {key} = {value:.1e}".replace('e', 'd') + "\n")
            elif isinstance(value, bool):
                value_str = ".TRUE." if value else ".FALSE."
                f.write(f"  {key} = {value_str}\n")
            elif isinstance(value, str):
                f.write(f"  {key} = '{value}'\n")
            else:
                f.write(f"  {key} = {value}\n")
        # Add custom adaptive_thr parameter
        f.write("  adaptive_thr = .TRUE.\n")
        f.write("  /\n")

        f.write("ATOMIC_SPECIES\n")
        for i in range(len(species_dict["species"])):
            species = species_dict["species"][i]
            mass = species_dict["masses"][i]
            pseudo = species_dict["psuedos"][i]
            f.write(f"  {species}  {mass:.4f} {pseudo}\n")
    
        f.write("ATOMIC_POSITIONS crystal\n")
        for site in structure.sites:
            species = site.specie.symbol
            frac_coords = site.frac_coords
            f.write(f"{species} {frac_coords[0]:.6f} {frac_coords[1]:.6f} {frac_coords[2]:.6f}\n")

        f.write("K_POINTS automatic\n")
        f.write(f"  {kpoints_grid[0]} {kpoints_grid[1]} {kpoints_grid[2]} {kshift[0]} {kshift[1]} {kshift[2]}\n")
    
        f.write("CELL_PARAMETERS angstrom\n")
        for vec in structure.lattice.matrix:
            f.write(f"  {vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}\n")
        
    return 0