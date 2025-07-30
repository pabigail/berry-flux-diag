from jobflow import job, Flow, Response
import preprocess
from context import BerryFluxDiag as bfd
import numpy as np
import os
from monty.serialization import dumpfn
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.powerups import update_user_incar_settings, update_user_kpoints_settings
from typing import List
from dataclasses import field
from pathlib import Path
import gzip
import shutil
from pymatgen.util.coord_cython import pbc_shortest_vectors
import re


def convert_complex(obj):
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    elif isinstance(obj, np.ndarray):
        # Convert ndarray to nested lists, then recursively sanitize elements
        return convert_complex(obj.tolist())
    elif isinstance(obj, dict):
        return {k: convert_complex(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_complex(v) for v in obj]
    else:
        return obj

def clean_path(path):
    path_str = str(path)  # Ensure it's a string
    if ":" in path_str and not Path(path_str).exists():
        parts = path_str.split(":", 1)
        if parts[1].startswith("/"):
            return Path(parts[1])
    return Path(path)


def ensure_unzipped(path):
    
    path = clean_path(path)

    if path.exists():
        return path
    gz_path = path.with_name(path.name + ".gz")
    if gz_path.exists():
        with gzip.open(gz_path, 'rb') as f_in:
            with open(path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return path
    else:
        raise FileNotFoundError(f"Neither {path} nor {gz_path} exist")


def compute_elec_contrib(string_sums, string_lens, pol_struct: Structure, occ_fact=2):
    """
    Compute total electronic polarization contribution in μC/cm².
    
    Args:
        string_sums: List of [x, y, z] vectors from Berry phase strings
        string_lens: List of corresponding normalization lengths
        pol_struct: Pymatgen Structure (polarized configuration)
        occ_fact: Occupation factor (2 for non-spin, 1 for spin-polarized)
    
    Returns:
        Numpy array of shape (3,) giving [Px, Py, Pz] in μC/cm²
    """

    # Sum normalized string contributions
    total_elec_change = np.zeros(3)
    for svec, slen in zip(string_sums, string_lens):
        total_elec_change += np.array(svec) / slen
    
    # Apply occupation factor
    total_elec_change *= occ_fact  # in units of e·fractional coord

    # Lattice constants and volume
    lattice = pol_struct.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    vol = lattice.volume

    # Physical constants
    ECHARGE = 1.6021766e-19  # C
    scaling = 100 * ECHARGE * 1e20 / vol  # convert to μC/cm²

    # Project onto lattice vectors
    elec_contrib = scaling * np.array([
        total_elec_change[0] * a,
        total_elec_change[1] * b,
        total_elec_change[2] * c
    ])

    return elec_contrib

def calc_ionic(frac_coord, structure: Structure, zval: float) -> np.ndarray:
    """
    Calculate the ionic dipole moment using ZVAL from pseudopotential.
 
    frac_coord: fractional coordinate of a single site
    structure: Structure
    zval: Charge value for ion (ZVAL for VASP pseudopotential)
 
    Returns polarization in electron Angstroms.
    """
    norms = structure.lattice.lengths
    return np.multiply(norms, -np.array(frac_coord) * zval)


def extract_letters(input_string):
    # Use a regular expression to match only the letters at the beginning of the string
    match = re.match(r'^[A-Za-z]+', input_string)
    if match:
        return match.group(0)
    return ''


def compute_ionic_contrib(pol_struct: Structure, np_struct: Structure, zval_dict: dict):
    """
    Compute ionic contribution to polarization change between np_struct and pol_struct.
    
    Args:
        pol_struct: Polar structure (final state)
        np_struct: Non-polar structure (initial state)
        zval_dict: Dictionary mapping element symbols to ZVAL charges
    
    Returns:
        Ionic polarization contribution as np.array in μC/cm²
    """

    lattice = pol_struct.lattice
    fcoords_pol = pol_struct.frac_coords
    fcoords_np = np_struct.frac_coords

    # Get shortest PBC displacements for corresponding sites
    pbc_short_vecs = pbc_shortest_vectors(lattice, fcoords_np, fcoords_pol)

    # Extract site-wise displacements (assumes 1-to-1 ordering)
    cart_coords_diff = [pbc_short_vecs[i][i] for i in range(len(pbc_short_vecs))]
    frac_coords_diff = lattice.get_fractional_coords(cart_coords_diff)

    # Compute ionic dipole contribution per site
    tot_ionic = []
    for site, d_frac in zip(pol_struct, frac_coords_diff):
        element = extract_letters(str(site.specie))
        zval = zval_dict[element]
        tot_ionic.append(calc_ionic(d_frac, pol_struct, zval))  # electron·Å

    ion_diff = np.sum(tot_ionic, axis=0)  # electron·Å

    # Convert to μC/cm²
    e_to_muC = -1.6021766e-13  # electron·Å⁻² to μC/cm²
    scale = e_to_muC * 1e16 / pol_struct.lattice.volume  # 1e16 for Å² to cm²
    ionic_contrib = scale * ion_diff  # μC/cm²

    return ionic_contrib


@job
def preprocess_POSCARS(pol_orig_POSCAR_file, np_orig_POSCAR_file, max_disp=0.3):

    structs = []
    
    # find translation that minimizes max atomic displacement between pol and np structs
    pol_orig_struct, np_trans_1_struct, orig_max_disp, translation = preprocess.translate_poscars(pol_orig_POSCAR_file,
                                                                                               np_orig_POSCAR_file)
    # if the max atomic displacement is larger than max_disp, then add interpolations
    if orig_max_disp > max_disp:
        num_interps = int(np.ceil(orig_max_disp/max_disp))
        structs = pol_orig_struct.interpolate(np_trans_1_struct, num_interps, interpolate_lattices=True)
    else:
        structs = [pol_orig_struct, np_trans_1_struct]

    adj_max_disp = preprocess.max_atomic_displacement_between_adjacent_structs(structs)
    
    return {
        "pol_orig_struct": pol_orig_struct,
        "np_orig_struct": Structure.from_file(np_orig_POSCAR_file),
        "np_trans_struct": np_trans_1_struct,
        "structs": structs, 
        "orig_max_disp": orig_max_disp, 
        "adj_max_disp": adj_max_disp,
        "translation": translation
    }

@job
def scf_with_fixed_kpoints(
    structures: List[Structure],
) -> Response:
    # Get k-points for first and last structures
    generator_pol = StaticSetGenerator(structures[0])
    generator_np = StaticSetGenerator(structures[-1])

    kpoints_pol = generator_pol.kpoints.kpts[0]
    kpoints_np = generator_np.kpoints.kpts[0]

    # Total number of k-points (e.g., 6x6x6 → 216)
    total_kpoints_pol = np.prod(kpoints_pol)
    total_kpoints_np = np.prod(kpoints_np)

    # Pick denser mesh
    if total_kpoints_pol >= total_kpoints_np:
        best_kpoints = kpoints_pol
    else:
        best_kpoints = kpoints_np

    # Fix INCAR overrides
    incar_updates = {"ISYM": -1, "ISPIN": 1, "LWAVE": True}

    # Make new VaspMaker with fixed kpoints
    fixed_kpoints = Kpoints(kpts=[best_kpoints], style="Monkhorst")
    
    scf_maker = StaticMaker(input_set_generator=StaticSetGenerator())
    scf_maker = update_user_incar_settings(scf_maker,
                                           incar_updates=incar_updates)
    scf_maker = update_user_kpoints_settings(scf_maker, 
                                            kpoints_updates=fixed_kpoints)

    # Build SCF jobs
    jobs = []
    vasp_dirs = []
    for i, struct in enumerate(structures):
        fw = scf_maker.make(struct)
        if i == 0:
            fw.append_name("_scf_pol")
        elif i == len(structures)-1:
            fw.append_name("scf_np")
        else:
            fw.append_name(f"_scf_{i}")
        jobs.append(fw)
        vasp_dirs.append(fw.output.dir_name)

    return Response(replace=Flow(jobs), output={"kpoints": list(best_kpoints),
                                                "vasp_dirs": vasp_dirs})

def get_min_singular_val(dict_debug):
    
    min_s = 10000
    max_e = 0
    
    for direction in ["x", "y", "z"]:
        dict_eigs = dict_debug[direction]
        svd_dict = dict_eigs['svd_dict']
        eigs = dict_eigs['eigs']
        for i in range(0, len(svd_dict)):
            for j in range(4):
                temp_min = np.min(svd_dict[i]['s'][j])
                if temp_min < min_s:
                    min_s = temp_min
        for i in range(0, len(eigs)):
            temp_max = np.max(np.abs(eigs[i]))
            if temp_max > max_e:
                max_e = temp_max
                    
    return min_s, max_e

@job
def get_string_sums_from_VASP(pol_dir: str, np_dir: str) -> Response:
    
    pol_dir = clean_path(pol_dir)
    np_dir = clean_path(np_dir)
        
    pol_POSCAR_file = ensure_unzipped(pol_dir/"POSCAR")
    np_POSCAR_file = ensure_unzipped(np_dir/"POSCAR")
    pol_wavecar = ensure_unzipped(pol_dir/"WAVECAR")
    np_wavecar = ensure_unzipped(np_dir/"WAVECAR")
    potcar_file = ensure_unzipped(np_dir/"POTCAR")
    potcar_file = ensure_unzipped(pol_dir/"POTCAR")
    pol_OUTCAR = ensure_unzipped(pol_dir/"OUTCAR")
    np_OUTCAR = ensure_unzipped(np_dir/"OUTCAR")

    # just unzip CONTCAR files and vasprun are unzipped
    pol_CONTCAR = ensure_unzipped(pol_dir/"CONTCAR")
    np_CONTCAR = ensure_unzipped(np_dir/"CONTCAR")
    pol_vasprun = ensure_unzipped(pol_dir/"vasprun.xml")
    np_vasprun = ensure_unzipped(np_dir/"vasprun.xml")

    parse_dict = bfd.VASPParser.vasp_parser(pol_POSCAR_file,
                                                np_POSCAR_file,
                                                pol_wavecar,
                                                np_wavecar,
                                                potcar_file,
                                                pol_dir,
                                                np_dir)

    overlaps = bfd.Overlaps.Overlaps(parse_dict)

    string_sums, string_lens, svd_eig_dict = overlaps.compute_string_sums()

    min_s, max_e = get_min_singular_val(svd_eig_dict)
     
    return Response(output=convert_complex({"string_sums": string_sums,
                            "string_lens": string_lens,
                            "zval_dict": parse_dict["zval_dict"],
                            "min_s": min_s,
                            "max_e": max_e,
                            "pair": (Structure.from_file(pol_POSCAR_file), 
                                     Structure.from_file(np_POSCAR_file)),
                            }))



@job
def make_string_sum_jobs(vasp_dirs: List[str]) -> Response:
    
    jobs = []
    for i in range(len(vasp_dirs)-1):
        pol_dir = vasp_dirs[i]
        np_dir = vasp_dirs[i+1]
        job_i = get_string_sums_from_VASP(pol_dir, np_dir)
        job_i.append_name(f"string_sum_{i}")
        jobs.append(job_i)

    return Response(replace=Flow(jobs), output={"string_sum_outputs": [j.output for j in jobs]})


@job
def calc_polarization_no_spin(preprocess_output,
                              string_sums_output):

    pol_struct = preprocess_output["pol_orig_struct"]
    np_trans_struct = preprocess_output["np_trans_struct"]
    zval_dict = string_sums_output[0]["zval_dict"]

    string_sums = []
    string_lens = []
    for output in string_sums_output:
        string_sums.append(output["string_sums"])
        string_lens.append(output["string_lens"])

    # get_ion_contrib
    ion_contrib = compute_ionic_contrib(pol_struct, np_trans_struct, zval_dict)

    # get_elec_contrib
    elec_contrib = compute_elec_contrib(string_sums, string_lens, pol_struct)

    pol_vec = ion_contrib + elec_contrib
    pol_norm = np.linalg.norm(pol_vec)

    return {
        "polarization_norm": pol_norm,
        "polarization_vec": pol_vec,
        "elec_contrib": elec_contrib,
        "ion_contrib": ion_contrib,

    }


@job
def bfd_schema(preprocess_output,
               scf_output,
               string_sum_outputs,
               calc_pol_output,
               save_name,
               save_dir="bfd_output"):
    os.makedirs(save_dir, exist_ok=True)

    save_dict = {
        "pol_orig_struct": preprocess_output["pol_orig_struct"],
        "np_orig_struct": preprocess_output["np_orig_struct"],
        "np_trans_struct": preprocess_output["np_trans_struct"],
        "structs": preprocess_output["structs"],
        "orig_max_disp": preprocess_output["orig_max_disp"],
        "adj_max_disp": preprocess_output["adj_max_disp"],
        "translation": preprocess_output["translation"],
        "kpoints": scf_output["kpoints"],
        "string_sums": [],
        "min_s": [],
        "max_e": [],
        "string_lens": [],
        "pairs": [],
        "zval_dict" : string_sum_outputs[0]["zval_dict"],
        "polarization_norm": calc_pol_output["polarization_norm"],
        "polarization_vec": calc_pol_output["polarization_vec"],
        "elec_contrib": calc_pol_output["elec_contrib"],
        "ion_contrib": calc_pol_output["ion_contrib"],
    }

    for output in string_sum_outputs:
        save_dict["string_sums"].append(output["string_sums"])
        save_dict["string_lens"].append(output["string_lens"])
        save_dict["min_s"].append(output["min_s"])
        save_dict["max_e"].append(output["max_e"])
        save_dict["pairs"].append(output["pair"])

    dumpfn(save_dict, os.path.join(save_dir, f"bfd_schema_{save_name}.json"))
    return save_dict
