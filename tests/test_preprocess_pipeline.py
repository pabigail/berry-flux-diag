"""Unit tests for the preprocessing pipeline and the jobflow layer above it.

Three of these cover paths that could not previously run at all:
preprocess_POSCARS called preprocess.translate_poscars, which does not
exist, and with translate=False left names unbound; BFDMaker imported
BFDJobs as a top-level module. The fourth covers a key spelled two
different ways in one file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from berry_flux_diag import preprocess

SRC = Path(__file__).resolve().parent.parent / "src" / "berry_flux_diag"

SPECIES = ["Ba", "Ti", "O"]
COORDS = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]]


def cell(shift=0.0):
    """A cell whose sites are rigidly displaced by `shift` along x."""
    coords = [[x + shift, y, z] for x, y, z in COORDS]
    return Structure(Lattice.cubic(6.0), SPECIES, coords)


# --- translate=False used to leave names unbound -------------------------

def test_no_translation_runs_at_all():
    out = preprocess.preprocess_structs(cell(), cell(0.05), translate=False)
    assert out["structs"]


def test_no_translation_reports_a_zero_shift():
    out = preprocess.preprocess_structs(cell(), cell(0.05), translate=False)
    assert np.allclose(out["translation"], np.zeros(3))


def test_no_translation_leaves_the_structure_alone():
    out = preprocess.preprocess_structs(cell(), cell(0.05), translate=False)
    assert out["np_trans_struct"] == out["np_orig_struct"]


@pytest.mark.parametrize("translate", [True, False])
def test_both_paths_return_the_same_keys(translate):
    out = preprocess.preprocess_structs(cell(), cell(0.05), translate=translate)
    assert set(out) == {"pol_orig_struct", "np_orig_struct", "np_trans_struct",
                        "structs", "orig_max_disp", "trans_max_disp",
                        "adj_max_disp", "translation"}


# --- the image count comes from the translated displacement --------------

def test_translation_is_measured_as_well_as_applied():
    """A rigid 0.05 shift is exactly undone, so the residual is zero."""
    out = preprocess.preprocess_structs(cell(), cell(0.05))
    assert out["orig_max_disp"] == pytest.approx(0.3)
    assert out["trans_max_disp"] == pytest.approx(0.0, abs=1e-9)


def test_no_images_are_inserted_for_a_pure_translation():
    """The bug: 0.3 Angstrom against max_disp=0.1 asked for three images,
    although the translation had already removed the whole displacement."""
    out = preprocess.preprocess_structs(cell(), cell(0.05), max_disp=0.1)
    assert len(out["structs"]) == 2


def test_the_old_untranslated_count_would_have_been_larger():
    """Pins the size of the waste, so the fix cannot silently regress.

    Each surplus image is a full SCF run, so the difference here between
    three-or-more images and none is the whole point of the fix.
    """
    out = preprocess.preprocess_structs(cell(), cell(0.05), max_disp=0.1)
    # What the old rule would have asked for, from the untranslated number.
    assert int(np.ceil(out["orig_max_disp"] / 0.1)) >= 3
    # What the translated number asks for: nothing, so no interpolation runs.
    assert out["trans_max_disp"] <= 0.1
    assert len(out["structs"]) == 2


def test_images_are_still_inserted_when_translation_cannot_help():
    """Not a rigid shift, so the displacement survives translation."""
    pol = cell()
    far = Structure(Lattice.cubic(6.0), SPECIES,
                    [[0.0, 0.0, 0.0], [0.6, 0.5, 0.5], [0.5, 0.5, 0.0]])
    out = preprocess.preprocess_structs(pol, far, max_disp=0.1)
    assert out["trans_max_disp"] > 0.1
    assert len(out["structs"]) > 2


def test_adjacent_displacement_respects_the_limit():
    """The guarantee the README actually makes."""
    pol = cell()
    far = Structure(Lattice.cubic(6.0), SPECIES,
                    [[0.0, 0.0, 0.0], [0.8, 0.5, 0.5], [0.5, 0.5, 0.0]])
    out = preprocess.preprocess_structs(pol, far, max_disp=0.3)
    assert out["adj_max_disp"] <= 0.3 + 1e-9


def test_translation_never_increases_the_displacement():
    out = preprocess.preprocess_structs(cell(), cell(0.03))
    assert out["trans_max_disp"] <= out["orig_max_disp"] + 1e-9


# --- the displacement does not depend on the order of the structures -----

def batio3_pair():
    """The tutorial's BaTiO3 pair: tetragonal polar, cubic non-polar."""
    species = ["Ba", "Ti", "O", "O", "O"]
    pol = Structure(Lattice([[4.002006, 0, 0], [0, 4.002006, 0], [0, 0, 4.215507]]), species,
                    [[0, 0, 0.020229], [0.5, 0.5, 0.538921], [0, 0.5, 0.492023],
                     [0.5, 0, 0.492023], [0.5, 0.5, 0.970803]])
    npl = Structure(Lattice.cubic(4.035583), species,
                    [[0, 0, 0.0028], [0.5, 0.5, 0.5028], [0, 0.5, 0.5028],
                     [0.5, 0, 0.5028], [0.5, 0.5, 0.0028]])
    return pol, npl


def test_the_displacement_is_symmetric_when_the_lattices_differ():
    """It used to be measured in the first argument's lattice only, so
    swapping the two gave 0.1523 against 0.1458 Angstrom for BaTiO3."""
    pol, npl = batio3_pair()
    assert preprocess.calc_max_disp(pol, npl) == preprocess.calc_max_disp(npl, pol)


def test_the_displacement_is_the_larger_of_the_two_lattices():
    pol, npl = batio3_pair()
    in_each = [max(np.diag(s.lattice.get_all_distances(pol.frac_coords, npl.frac_coords)))
               for s in (pol, npl)]
    assert preprocess.calc_max_disp(pol, npl) == pytest.approx(max(in_each), abs=1e-12)


def test_a_two_structure_pair_reports_one_displacement():
    """With no images, the pair is its own only step: the translated and
    adjacent displacements are one number, not two."""
    pol, npl = batio3_pair()
    out = preprocess.preprocess_structs(pol, npl)
    assert len(out["structs"]) == 2
    assert out["adj_max_disp"] == pytest.approx(out["trans_max_disp"], abs=1e-12)


def test_the_translation_search_minimizes_the_reported_displacement():
    """find_translation and calc_max_disp now use the same measure, so the
    chosen shift is the best one on the grid by the number reported."""
    pol, npl = batio3_pair()
    shift = preprocess.find_translation(pol, npl)
    best = preprocess.calc_max_disp(preprocess.translate_structs(pol, npl, shift), pol)
    for trial in ([0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [0.0, 0.0, -0.02]):
        other = preprocess.translate_structs(pol, npl, np.array(trial))
        assert best <= preprocess.calc_max_disp(other, pol) + 1e-12


@pytest.mark.parametrize("max_disp", [0.05, 0.1, 0.2])
def test_steps_respect_the_limit_when_the_lattices_differ(max_disp):
    pol, npl = batio3_pair()
    out = preprocess.preprocess_structs(pol, npl, max_disp=max_disp)
    n = len(out["structs"]) - 1
    assert out["adj_max_disp"] <= max_disp + 1e-9
    assert out["adj_max_disp"] <= out["trans_max_disp"] / n + 1e-9


def test_the_minimum_image_is_found_in_a_skewed_cell():
    """Agrees with pymatgen's own minimum-image distance when the lattices
    are the same, including across a boundary of a non-orthogonal cell."""
    lattice = Lattice.from_parameters(4.1, 4.3, 4.6, 78.0, 84.0, 66.0)
    a = Structure(lattice, ["Ba", "Ti"], [[0.98, 0.02, 0.5], [0.5, 0.5, 0.97]])
    b = Structure(lattice, ["Ba", "Ti"], [[0.03, 0.96, 0.52], [0.47, 0.55, 0.04]])
    expected = np.diag(lattice.get_all_distances(a.frac_coords, b.frac_coords))
    got = preprocess.site_displacements(a.frac_coords, lattice, b.frac_coords, lattice)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


# --- reading the pair from POSCAR files ----------------------------------

def test_preprocess_poscars_reads_files(tmp_path):
    from pymatgen.io.vasp import Poscar

    pol_file = tmp_path / "POSCAR_pol"
    np_file = tmp_path / "POSCAR_np"
    Poscar(cell()).write_file(str(pol_file))
    Poscar(cell(0.05)).write_file(str(np_file))

    out = preprocess.preprocess_POSCARS(str(pol_file), str(np_file))
    assert out["pol_orig_struct"].composition == cell().composition
    assert out["trans_max_disp"] == pytest.approx(0.0, abs=1e-6)


def test_preprocess_poscars_honours_translate_false(tmp_path):
    from pymatgen.io.vasp import Poscar

    pol_file = tmp_path / "POSCAR_pol"
    np_file = tmp_path / "POSCAR_np"
    Poscar(cell()).write_file(str(pol_file))
    Poscar(cell(0.05)).write_file(str(np_file))

    out = preprocess.preprocess_POSCARS(str(pol_file), str(np_file),
                                        translate=False)
    assert np.allclose(out["translation"], np.zeros(3))


# --- the pseudopotential key was spelled two ways ------------------------

SPECIES_DICT = {"species": ["Ba", "Ti", "O"],
                "masses": [137.327, 47.867, 15.999],
                "pseudos": ["Ba.upf", "Ti.upf", "O.upf"]}


def test_a_written_input_can_be_read_back(tmp_path):
    """The halves of this module could not talk to each other: the writer
    read species_dict['pseudos'], the parser produced 'psuedos'."""
    preprocess.poscar_to_qe_io_scf_nomag(
        cell(), "BaTiO3", "pol_orig", False, "/pseudos/", (6, 6, 6), (0, 0, 0),
        SPECIES_DICT, str(tmp_path) + "/")

    written = next(tmp_path.glob("*.in"))
    parsed = preprocess.parse_qe_input(str(written))
    assert parsed["species"]["pseudos"] == SPECIES_DICT["pseudos"]


def test_a_parsed_input_can_be_written_again(tmp_path):
    """The round trip that used to raise KeyError('pseudos')."""
    preprocess.poscar_to_qe_io_scf_nomag(
        cell(), "BaTiO3", "pol_orig", False, "/pseudos/", (6, 6, 6), (0, 0, 0),
        SPECIES_DICT, str(tmp_path) + "/")
    parsed = preprocess.parse_qe_input(str(next(tmp_path.glob("*.in"))))

    out = tmp_path / "again"
    out.mkdir()
    preprocess.write_qe_in_scf_files(
        str(out) + "/", [cell(), cell(0.05)], "BaTiO3", "/pseudos/",
        (6, 6, 6), (0, 0, 0), parsed["species"])
    assert len(list(out.glob("*.in"))) == 2


def test_no_misspelled_key_remains():
    assert "psuedo" not in (SRC / "preprocess.py").read_text()


# --- the jobflow layer, which cannot be imported without atomate2 --------

def module_ast(name):
    return ast.parse((SRC / name).read_text())


def test_bfdjobs_imports_preprocess_from_this_package():
    """It used to import from a package called 'bfd', which does not exist."""
    imports = [n.module for n in ast.walk(module_ast("BFDJobs.py"))
               if isinstance(n, ast.ImportFrom)]
    assert "bfd" not in imports
    assert "berry_flux_diag" in imports


def test_bfdjobs_does_not_call_a_function_that_does_not_exist():
    """Checked on attribute accesses, not raw text - the docstring names
    the dead function deliberately."""
    accessed = {n.attr for n in ast.walk(module_ast("BFDJobs.py"))
                if isinstance(n, ast.Attribute)}
    assert "translate_poscars" not in accessed
    assert not hasattr(preprocess, "translate_poscars")


def test_bfdmaker_imports_bfdjobs_by_its_full_path():
    imports = [n.module for n in ast.walk(module_ast("BFDMaker.py"))
               if isinstance(n, ast.ImportFrom)]
    assert "BFDJobs" not in imports
    assert "berry_flux_diag.BFDJobs" in imports


def test_bfdmaker_is_a_dataclass():
    """jobflow reconstructs a Maker from its dataclass fields."""
    cls = next(n for n in module_ast("BFDMaker.py").body
               if isinstance(n, ast.ClassDef) and n.name == "BFDMaker")
    assert any(getattr(d, "id", getattr(d, "attr", None)) == "dataclass"
               for d in cls.decorator_list)


def test_bfdmaker_has_a_defaulted_name_field():
    cls = next(n for n in module_ast("BFDMaker.py").body
               if isinstance(n, ast.ClassDef) and n.name == "BFDMaker")
    fields = {n.target.id: n.value for n in cls.body
              if isinstance(n, ast.AnnAssign)}
    assert "name" in fields and isinstance(fields["name"], ast.Constant)


def test_bfdmaker_defines_no_init():
    """A hand-written __init__ is what stopped it having fields at all."""
    cls = next(n for n in module_ast("BFDMaker.py").body
               if isinstance(n, ast.ClassDef) and n.name == "BFDMaker")
    assert not [n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"]


def test_no_absolute_scratch_path_is_baked_in():
    """save_dir defaulted to one user's NERSC CFS directory."""
    source = (SRC / "BFDMaker.py").read_text()
    assert "/global/cfs" not in source
