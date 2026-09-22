"""Unit tests for the Overlaps convergence diagnostics.

Both thresholds warn that a k-mesh is too coarse to track the phase from
one k-point to the next. They are exercised here with synthetic singular
values and Wilson loop phases, so the boundary behaviour can be pinned
without a DFT run.
"""

from __future__ import annotations

import numpy as np
import pytest

from berry_flux_diag.Overlaps import EIG_THRESH, SING_VAL_THRESH, Overlaps


def make_overlaps(**kwargs):
    """An Overlaps carrying only what __init__ touches.

    The diagnostics depend on the thresholds alone, so the wavefunction
    dictionaries can stay empty and no DFT output is needed.
    """
    parse_dict = {
        'pol_struct': None,
        'np_struct': None,
        'pol_wfcn_dict': {},
        'np_wfcn_dict': {},
        'kpoint_list': [],
        'zval_dict': {},
        'ES_code': 'QE',
        'spin_pol': False,
        'pol_band_fill': 4,
        'np_band_fill': 4,
    }
    return Overlaps(parse_dict, **kwargs)


def test_defaults_are_the_documented_values():
    """The historical values, unchanged: 2.8 rad and 0.2."""
    overlaps = make_overlaps()
    assert overlaps.eig_thresh == EIG_THRESH == 2.8
    assert overlaps.sing_val_thresh == SING_VAL_THRESH == 0.2


def test_thresholds_are_overridable():
    """The regression case: eig_thresh was assigned but never read.

    Setting it had no effect, because both use sites carried their own
    hardcoded literal.
    """
    overlaps = make_overlaps(eig_thresh=1.0, sing_val_thresh=0.5)
    assert overlaps.eig_thresh == 1.0
    assert overlaps.sing_val_thresh == 0.5


def test_singular_value_below_threshold_warns(capsys):
    overlaps = make_overlaps()
    overlaps.check_overlap_conditioning([1.0, 0.9, 0.05])
    assert "min singular value" in capsys.readouterr().out


def test_singular_value_above_threshold_is_quiet(capsys):
    overlaps = make_overlaps()
    overlaps.check_overlap_conditioning([1.0, 0.9, 0.5])
    assert capsys.readouterr().out == ""


def test_singular_value_threshold_is_honoured(capsys):
    """A run that tolerates poorer conditioning must stay quiet at 0.05."""
    overlaps = make_overlaps(sing_val_thresh=0.01)
    overlaps.check_overlap_conditioning([1.0, 0.05])
    assert capsys.readouterr().out == ""


def test_singular_value_check_returns_the_smallest():
    """Returned so a caller can record it without the debug payload."""
    overlaps = make_overlaps()
    assert overlaps.check_overlap_conditioning([1.0, 0.3, 0.7]) == 0.3


def test_wilson_phase_near_the_branch_cut_warns(capsys):
    """2.9 rad is past 2.8 and close to the cut at pi."""
    overlaps = make_overlaps()
    overlaps.check_wilson_loop_phases(np.array([0.1, 2.9]))
    assert "underconverged" in capsys.readouterr().out


def test_wilson_phase_warns_for_negative_phases(capsys):
    """The cut is at both +pi and -pi, so the check is on the magnitude."""
    overlaps = make_overlaps()
    overlaps.check_wilson_loop_phases(np.array([-2.9]))
    assert "underconverged" in capsys.readouterr().out


def test_wilson_phase_well_inside_the_range_is_quiet(capsys):
    overlaps = make_overlaps()
    overlaps.check_wilson_loop_phases(np.array([0.1, -0.4, 1.2]))
    assert capsys.readouterr().out == ""


def test_wilson_phase_threshold_is_honoured(capsys):
    """A stricter run must warn at a phase the default tolerates."""
    overlaps = make_overlaps(eig_thresh=1.0)
    overlaps.check_wilson_loop_phases(np.array([1.2]))
    assert "underconverged" in capsys.readouterr().out


def test_wilson_phase_check_returns_the_largest_magnitude():
    overlaps = make_overlaps()
    assert overlaps.check_wilson_loop_phases(np.array([0.1, -2.0, 1.5])) == 2.0


def test_eig_thresh_sits_just_inside_pi():
    """The default is meaningful only relative to the branch cut."""
    assert EIG_THRESH < np.pi
    assert EIG_THRESH / np.pi == pytest.approx(0.891, abs=1e-3)
