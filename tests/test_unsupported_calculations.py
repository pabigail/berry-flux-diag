"""Unit tests for the calculation types the code does not support.

Each of these used to be accepted. A noncollinear Quantum ESPRESSO run
was routed down the collinear two-channel path; a noncollinear WAVECAR
was read as a standard one because the type was pinned rather than
detected. Neither produced an error at the point the assumption broke.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from berry_flux_diag import utils

SRC = Path(__file__).resolve().parent.parent / "src" / "berry_flux_diag"


# --- Quantum ESPRESSO: noncollinear and spin-orbit ------------------------

def qe_xml(lsda=False, noncolin=False, spinorbit=False):
    """The only part of the qeschema dict get_spin_pol_from_qeschema reads."""
    return {"qes:espresso": {"input": {"spin": {
        "lsda": lsda, "noncolin": noncolin, "spinorbit": spinorbit}}}}


@pytest.fixture(scope="module")
def qe():
    return pytest.importorskip("berry_flux_diag.QEParser")


def test_unpolarized_run_reports_false(qe):
    assert qe.get_spin_pol_from_qeschema(qe_xml()) is False


def test_lsda_run_reports_true(qe):
    assert qe.get_spin_pol_from_qeschema(qe_xml(lsda=True)) is True


def test_noncollinear_run_raises(qe):
    with pytest.raises(NotImplementedError, match="noncollinear"):
        qe.get_spin_pol_from_qeschema(qe_xml(noncolin=True))


def test_spinorbit_run_raises(qe):
    with pytest.raises(NotImplementedError, match="noncollinear"):
        qe.get_spin_pol_from_qeschema(qe_xml(spinorbit=True))


def test_the_error_says_how_to_fix_it(qe):
    """A user has to be able to act on it without reading the source."""
    with pytest.raises(NotImplementedError, match="noncolin = .false."):
        qe.get_spin_pol_from_qeschema(qe_xml(noncolin=True))


def test_noncollinear_raises_even_when_lsda_is_also_set(qe):
    """The old code OR-ed the three flags, so this combination read as True."""
    with pytest.raises(NotImplementedError):
        qe.get_spin_pol_from_qeschema(qe_xml(lsda=True, noncolin=True))


# --- VASP: WAVECAR type ---------------------------------------------------

def test_standard_wavecar_passes():
    utils.check_wavecar_type("std")


def test_gamma_only_wavecar_raises():
    """One k-point cannot give a Berry flux, whatever the storage format."""
    with pytest.raises(NotImplementedError, match="gamma-only"):
        utils.check_wavecar_type("gam")


def test_noncollinear_wavecar_raises():
    with pytest.raises(NotImplementedError, match="noncollinear"):
        utils.check_wavecar_type("ncl")


def test_unrecognised_wavecar_type_raises():
    with pytest.raises(ValueError, match="unrecognised"):
        utils.check_wavecar_type("banana")


def test_none_type_raises_rather_than_passing():
    """Wavecar.vasp_type is None only if construction never resolved it."""
    with pytest.raises(ValueError, match="unrecognised"):
        utils.check_wavecar_type(None)


@pytest.mark.parametrize("label,expected", [("polar", "polar"),
                                            ("non-polar", "non-polar")])
def test_the_error_names_which_run(label, expected):
    with pytest.raises(NotImplementedError, match=expected):
        utils.check_wavecar_type("ncl", label)


def test_pymatgen_only_checks_the_first_letter_and_so_do_we():
    """pymatgen accepts 'Gamma'/'NCL'; the guard must not be stricter."""
    with pytest.raises(NotImplementedError):
        utils.check_wavecar_type("NCL")
    with pytest.raises(NotImplementedError):
        utils.check_wavecar_type("Gamma")


# --- VASP: the momentum grid cutoff ---------------------------------------
#
# VASPParser imports pawpyseed, which needs MKL and does not build on arm64
# macOS, so these read the source rather than the module.

def vasp_parser_ast():
    return ast.parse((SRC / "VASPParser.py").read_text())


def function_def(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function named {name}")


def test_the_pinned_cutoff_is_defined_once():
    """1000 eV is held deliberately, pending a reference to move it against.

    It lives in one named constant so that switching to pawpyseed's
    4 x ENCUT later is a one-line change, not a hunt for literals.
    """
    tree = vasp_parser_ast()
    assignments = [n for n in tree.body
                   if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "MOMENTUM_ENCUT"
                           for t in n.targets)]
    assert len(assignments) == 1
    assert assignments[0].value.value == 1000


@pytest.mark.parametrize("name", ["get_wfcn_data_from_vasp_pawpy",
                                  "wfcn_dict_from_pawpy",
                                  "vasp_parser"])
def test_the_cutoff_default_is_the_named_constant(name):
    """No function may carry its own copy of the number."""
    fn = function_def(vasp_parser_ast(), name)
    args = [a.arg for a in fn.args.args]
    default = dict(zip(args[len(args) - len(fn.args.defaults):],
                       fn.args.defaults))["momentum_encut"]
    assert isinstance(default, ast.Name), (
        f"{name} defaults momentum_encut to {ast.unparse(default)}, "
        f"not the shared MOMENTUM_ENCUT")
    assert default.id == "MOMENTUM_ENCUT"


def test_no_function_still_takes_a_hardcoded_cutoff():
    """Checked on the parsed signatures, not the text - the docstring
    mentions the old value deliberately."""
    for node in ast.walk(vasp_parser_ast()):
        if not isinstance(node, ast.FunctionDef):
            continue
        args = [a.arg for a in node.args.args]
        defaults = dict(zip(args[len(args) - len(node.args.defaults):],
                            node.args.defaults))
        cutoff = defaults.get("cutoff")
        assert cutoff is None, (
            f"{node.name} still takes cutoff={ast.unparse(cutoff)}")


@pytest.mark.parametrize("name", ["wfcn_dict_from_pawpy", "vasp_parser"])
def test_the_cutoff_is_overridable_end_to_end(name):
    """It has to reach vasp_parser to be usable for a convergence check."""
    fn = function_def(vasp_parser_ast(), name)
    assert "momentum_encut" in [a.arg for a in fn.args.args]


def test_wavecar_type_is_not_pinned():
    """Pinning 'std' makes a noncollinear WAVECAR parse silently wrong."""
    source = (SRC / "VASPParser.py").read_text()
    assert 'vasp_type="std"' not in source
    assert "vasp_type='std'" not in source
