#!/bin/bash/python

from itertools import product
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar
import re
import os

def preprocess_structs(pol_orig_struct, np_orig_struct, translate=True,
                       num_interps = 'auto', MAX_DISP=0.3):

    # find translation that minimizes max atomic displacement between pol and np structs
    if translate:
        np_trans_struct = translate_structs(pol_orig_struct, np_orig_struct)
    else:
        np_trans_struct = np_orig_struct
    
    orig_max_disp = calc_max_disp(pol_orig_struct, np_orig_struct)

    # compute number of interpolated structures to ensure max atomic distance is below max_disp in num_interps is not specified
    if num_interps == 'auto':
        num_interps = int(np.ceil(orig_max_disp/MAX_DISP))
    elif not isinstance(num_interps, int) or num_interps < 0:
        raise TypeError("num_interps must be a non-negative integer or the string 'auto'")

    if orig_max_disp > MAX_DISP:
        structs = pol_orig_struct.interpolate(np_trans_struct, num_interps, interpolate_lattice=True)
    else:
        structs = [pol_orig_struct, np_trans_struct]

    adj_max_disp = max_atomic_displacement_between_adjacent_structs(structs)

    return {
         "pol_orig_struct": pol_orig_struct,
         "np_orig_struct": np_orig_struct,
         "np_trans_struct": np_trans_struct,
         "structs": structs,
         "orig_max_disp": orig_max_disp,
         "adj_max_disp": adj_max_disp,
     }


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


def calc_max_disp(struct_a, struct_b, trans=None):

    if trans is None:
        trans = np.array([0.0, 0.0, 0.0])
    
    st_a_copy = struct_a.copy()
    st_a_copy.translate_sites(range(len(struct_a)),trans)
    max_disp = max(np.diag(st_a_copy.lattice.get_all_distances(st_a_copy.frac_coords,
                                                               struct_b.frac_coords)))
    return max_disp


def translate_structs(pol_struct, np_struct, translation=None):

    if translation is None:
        max_disps = get_refined_oshift(np_struct, pol_struct)
        sorted_max_disps = sorted(max_disps, key=lambda x: x[1])
        translation = sorted_max_disps[0][0]

    np_trans_struct = np_struct.copy()
    np_trans_struct.translate_sites(range(len(np_trans_struct)), translation)

    return np_trans_struct


def max_atomic_displacement_between_adjacent_structs(structs):
    max_disp = 0.0
    for i in range(len(structs) - 1):
        st_a = structs[i+1] # should re-write this so it is clearer, but max(diag(dist(A, B))) ≠ max(diag(dist(B, A)))
        st_b = structs[i] # so made it consistent with pol vs np that computing distances between

        if len(st_a) != len(st_b):
            raise ValueError(f"Structures at index {i} and {i+1} have different number of atoms.")

        trans = np.array([0.0, 0.0, 0.0])
        pair_disp = calc_max_disp(st_a, st_b, trans)
        max_disp = max(max_disp, pair_disp)

    return max_disp


def write_qe_in_scf_files(file_path, structs, material, pseudo_dir,
                          kpoints_grid, kshift, species_dict):

    if len(structs) == 2:
        tags = ["pol_orig", "np_trans"]
    elif len(structs) > 2:
        tags = []
        for i in range(len(structs)):
            if i == 0:
                tags.append("pol_orig")
            elif i == len(structs) - 1:
                tags.append("np_trans")
            else:
                tags.append(f"interp_{i}")
    

    sym = False
    for struct, tag in zip(structs, tags):
        poscar_to_qe_io_scf_nomag(struct, material, tag, sym,
                                  pseudo_dir, kpoints_grid, kshift, species_dict, file_path)

    return 0

def poscar_to_qe_io_scf_nomag(structure, material, tag, sym,
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
            pseudo = species_dict["pseudos"][i]
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



def parse_qe_input(input_file):
    control_dict = {}
    system_dict = {}
    electrons_dict = {}
    species_dict = {"species": [], "masses": [], "psuedos": []}
    k_points_dict = {"setting": "", "k_points": (), "k_shift": ()}
    atomic_positions = []
    cell_parameters = []
    hubbard_dict = {"setting": "", "U": []}

    with open(input_file, 'r') as f:
        lines = f.readlines()

    current_section = None

    for line in lines:
        line = line.strip()

        # Ignore lines starting with !, or remove everything after !
        if line.startswith("!") or line == "/":
            continue
        if "!" in line:
            line = line.split("!", 1)[0].strip()

        # Section headers and their settings
        if line.startswith("&CONTROL"):
            current_section = "control"
        elif line.startswith("&SYSTEM"):
            current_section = "system"
        elif line.startswith("&ELECTRONS"):
            current_section = "electrons"
        elif line.startswith("ATOMIC_SPECIES"):
            current_section = "atomic_species"
        elif line.startswith("ATOMIC_POSITIONS"):
            current_section = "atomic_positions"
            atomic_positions = []
        elif line.startswith("K_POINTS"):
            current_section = "k_points"
            k_points_dict["setting"] = line.split()[1]  # Get the setting like "automatic"
        elif line.startswith("CELL_PARAMETERS"):
            current_section = "cell_parameters"
            cell_parameters = []
        elif line.startswith("HUBBARD"):
            current_section = "hubbard"
            hubbard_dict["setting"] = line.split()[1]  # Get the setting like "ortho-atomic"
        elif current_section == "control" and line:
            key, value = [item.strip() for item in line.split("=")]
            control_dict[key] = value.strip("'")
        elif current_section == "system" and line:
            if "=" in line:
                key, value = [item.strip() for item in line.split("=")]
                system_dict[key] = value
        elif current_section == "electrons" and line:
            if "=" in line:
                key, value = [item.strip() for item in line.split("=")]
                electrons_dict[key] = value
        elif current_section == "atomic_species" and line:
            species, mass, pseudo = line.split()
            species_dict["species"].append(species)
            species_dict["masses"].append(float(mass))
            species_dict["psuedos"].append(pseudo)
        elif current_section == "atomic_positions" and line:
            atomic_positions.append(line.split())
        elif current_section == "k_points" and line:
            k_points = [int(x) for x in line.split()[:3]]
            k_shift = [int(x) for x in line.split()[3:]]
            k_points_dict["k_points"] = tuple(k_points)
            k_points_dict["k_shift"] = tuple(k_shift)
        elif current_section == "cell_parameters" and line:
            cell_parameters.append([float(x) for x in line.split()])
        elif current_section == "hubbard" and line:
            if "U" in line:
                hubbard_dict["U"].append(line.strip())

    # Convert atomic_positions to pymatgen structure
    species_list = [item[0] for item in atomic_positions]
    coords = [[float(x) for x in item[1:]] for item in atomic_positions]
    lattice = Lattice(cell_parameters)
    structure = Structure(lattice, species_list, coords)

    return {
        'control': control_dict,
        'system': system_dict,
        'electrons': electrons_dict,
        'species': species_dict,
        'kpoints': k_points_dict,
        'structure': structure,
        'hubbard': hubbard_dict
    }


def write_qe_input(control_dict, system_dict, electron_dict, species_dict, kpoint_dict, hubbard_dict, structure, filename, hubbard_true=True):
    
    def map_species_name(species_string):
        if species_string.endswith('+'):
            # Remove the trailing '+' sign
            base_name = species_string.rstrip('+')
            # Check if the last character is a digit
            if base_name[-1].isdigit():
                return base_name  # Return the number without the '+'
            else:
                return base_name + '1'  # Append '1' if no number precedes '+'
        return species_string  # Return as-is if no trailing '+'
    
    with open(filename, 'w') as f:
        # Write CONTROL section
        f.write("&CONTROL\n")
        for key, value in control_dict.items():
            if isinstance(value, bool):
                value_str = '.TRUE.' if value else '.FALSE.'
            else:
                value_str = f"'{value}'" if isinstance(value, str) else f"{value}"
            f.write(f"  {key} = {value_str}\n")
        f.write("/\n")

        # Write SYSTEM section
        f.write("&SYSTEM\n")
        for key, value in system_dict.items():
            if isinstance(value, bool):
                value_str = '.TRUE.' if value else '.FALSE.'
            else:
                value_str = f"{value}" if isinstance(value, str) and key != "tot_magnetization" else f"{value}"
            f.write(f"  {key} = {value_str}\n")
        f.write("/\n")

        # Write ELECTRONS section
        f.write("&ELECTRONS\n")
        for key, value in electron_dict.items():
            f.write(f"  {key} = {value}\n")
        f.write("/\n")

        # Write ATOMIC_SPECIES section
        f.write("ATOMIC_SPECIES\n")
        for species, mass, pseudo in zip(species_dict['species'], species_dict['masses'], species_dict['psuedos']):
            f.write(f"  {species} {mass} {pseudo}\n")

        # Write ATOMIC_POSITIONS section
        f.write("ATOMIC_POSITIONS crystal\n")
        for site in structure.sites:
            species_name = map_species_name(site.species_string)
            f.write(f"  {species_name} {site.frac_coords[0]:.16f} {site.frac_coords[1]:.16f} {site.frac_coords[2]:.16f}\n")

        # Write K_POINTS section
        f.write(f"K_POINTS {kpoint_dict['setting']}\n")
        f.write(f"  {' '.join(map(str, kpoint_dict['k_points']))} {' '.join(map(str, kpoint_dict['k_shift']))}\n")

        # Write CELL_PARAMETERS section
        f.write("CELL_PARAMETERS angstrom\n")
        for vector in structure.lattice.matrix:
            f.write(f"  {' '.join(f'{x:.16f}' for x in vector)}\n")
        
        if hubbard_true:
            # Write HUBBARD section
            f.write(f"HUBBARD {hubbard_dict['setting']}\n")
            for u_value in hubbard_dict['U']:
                f.write(f"  {u_value}\n")
    
    
def generate_qe_in_from_qe_in(orig_pol_qe_in_filename, orig_np_qe_in_filename,
                              bfd_pol_qe_in_filename, bfd_np_qe_in_filename):
    
    pol_input_dict = parse_qe_input(orig_pol_qe_in_filename)
    np_input_dict = parse_qe_input(orig_np_qe_in_filename)
    
    pol_struct = pol_input_dict["structure"]
    np_struct = np_input_dict["structure"]
    
    np_struct_trans = translate_structs(pol_struct, np_struct)
    
    pol_input_dict["system"]["nosym"] = True
    pol_input_dict["system"]["noinv"] = True
    np_input_dict["system"]["nosym"] = True
    np_input_dict["system"]["noinv"] = True
    
    write_qe_input(pol_input_dict['control'],
               pol_input_dict['system'],
               pol_input_dict['electrons'],
               pol_input_dict['species'],
               pol_input_dict['kpoints'],
               pol_input_dict['hubbard'],
               pol_struct,
               bfd_pol_qe_in_filename)
    
    write_qe_input(np_input_dict['control'],
               np_input_dict['system'],
               np_input_dict['electrons'],
               np_input_dict['species'],
               np_input_dict['kpoints'],
               np_input_dict['hubbard'],
               np_struct_trans,
               bfd_np_qe_in_filename)
    return 0


def write_vasp_in_poscar_files(save_path, structs, material):
    """
    Write a list of pymatgen Structures to POSCAR files with a specific naming convention.

    Parameters
    ----------
    save_path : str
        Path to directory where POSCAR files will be saved.
    structs : list of pymatgen.Structure
        List of Structure objects to write.
    material : str
        Material name to include in the POSCAR filename.
    """
    os.makedirs(save_path, exist_ok=True)

    n_structs = len(structs)
    if n_structs < 2:
        raise ValueError("Structs list must contain at least two structures.")

    for i, struct in enumerate(structs):
        if i == 0:
            tag = "pol_orig"
        elif i == n_structs - 1:
            tag = "np_trans"
        else:
            tag = f"interp_{i-1}"

        filename = os.path.join(save_path, f"POSCAR_{material}_{tag}")
        poscar = Poscar(struct)
        poscar.write_file(filename)
        print(f"Wrote {filename}")
