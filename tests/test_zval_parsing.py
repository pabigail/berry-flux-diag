"""Unit tests for reading valence charges out of a QE output banner.

These build tiny synthetic pw.x outputs rather than using the BaTiO3
fixture, so the layouts that the old fixed-offset lookup silently got wrong
can be exercised without a DFT run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("qeschema")
pytest.importorskip("h5py")

from pymatgen.io.pwscf import PWOutput  # noqa: E402

from berry_flux_diag import QEParser  # noqa: E402

HEADER = """
     Program PWSCF v.7.1 starts on 1Jan2026 at 0: 0: 0

     bravais-lattice index     =            0
"""

# The layout written by QE 6.x and later: path, MD5, then Zval.
BLOCK_WITH_MD5 = """
     PseudoPot. # {n} for {element} read from file:
     /pseudos/{element}_ONCV_PBE-1.0.upf
     MD5 check sum: 0123456789abcdef0123456789abcdef
     Pseudo is Norm-conserving, Zval = {zval}
     Generated using ONCVPSP code by D. R. Hamann
     Using radial grid of  602 points,  4 beta functions with:
                l(1) =   0
"""

# Older QE printed no MD5 line, so Zval sits one line higher.
BLOCK_WITHOUT_MD5 = """
     PseudoPot. # {n} for {element} read from file:
     /pseudos/{element}_ONCV_PBE-1.0.upf
     Pseudo is Norm-conserving, Zval = {zval}
     Generated using ONCVPSP code by D. R. Hamann
"""


def write_output(tmp_path, species, block=BLOCK_WITH_MD5):
    """Write a synthetic pw.x output containing only the banner we parse."""
    text = HEADER + "".join(
        block.format(n=n, element=element, zval=zval)
        for n, (element, zval) in enumerate(species, start=1)
    )
    path = tmp_path / "pw.out"
    path.write_text(text)
    return PWOutput(str(path))


def test_reads_each_species(tmp_path):
    """The ordinary case, matching the BaTiO3 fixture's layout."""
    pw_out = write_output(tmp_path, [("Ba", "10.0"), ("O", "6.0"), ("Ti", "12.0")])
    assert QEParser.get_zval_dict_from_PWOutput(pw_out) == {
        "Ba": 10.0,
        "O": 6.0,
        "Ti": 12.0,
    }


def test_reads_species_without_md5_line(tmp_path):
    """The regression case for the fixed offset.

    Without the MD5 line, Zval is three lines after the element line minus
    one. The old lookup searched for line_num + 3 exactly, found nothing,
    and silently left zval holding the previous species' value - so O and
    Ti would both have come back as 10.0 here.
    """
    pw_out = write_output(
        tmp_path,
        [("Ba", "10.0"), ("O", "6.0"), ("Ti", "12.0")],
        block=BLOCK_WITHOUT_MD5,
    )
    assert QEParser.get_zval_dict_from_PWOutput(pw_out) == {
        "Ba": 10.0,
        "O": 6.0,
        "Ti": 12.0,
    }


def test_single_species(tmp_path):
    """One block: the old code raised NameError if the offset missed here."""
    pw_out = write_output(tmp_path, [("Si", "4.0")], block=BLOCK_WITHOUT_MD5)
    assert QEParser.get_zval_dict_from_PWOutput(pw_out) == {"Si": 4.0}


def test_missing_zval_raises(tmp_path):
    """A block with no Zval must fail loudly, not inherit a neighbour's."""
    text = HEADER + BLOCK_WITH_MD5.format(n=1, element="Ba", zval="10.0")
    text += """
     PseudoPot. # 2 for O  read from file:
     /pseudos/O_ONCV_PBE-1.0.upf
     MD5 check sum: 0123456789abcdef0123456789abcdef
     Generated using ONCVPSP code by D. R. Hamann
"""
    path = tmp_path / "pw.out"
    path.write_text(text)

    with pytest.raises(ValueError, match="2 pseudopotential blocks but 1 Zval"):
        QEParser.get_zval_dict_from_PWOutput(PWOutput(str(path)))


def test_no_pseudopotential_blocks_raises(tmp_path):
    """A truncated output should say so rather than return an empty dict."""
    path = tmp_path / "pw.out"
    path.write_text(HEADER)

    with pytest.raises(ValueError, match="no pseudopotential blocks found"):
        QEParser.get_zval_dict_from_PWOutput(PWOutput(str(path)))


def test_zval_outside_its_block_raises(tmp_path):
    """Right number of Zvals, wrong places.

    Both Zval lines sit in the first block and the second block has none, so
    the counts match but positional pairing would hand O the value 6.0 read
    from Ba's block. The ordering check is what makes pairing by position
    safe rather than merely plausible.
    """
    text = HEADER + """
     PseudoPot. # 1 for Ba read from file:
     /pseudos/Ba_ONCV_PBE-1.0.upf
     MD5 check sum: 0123456789abcdef0123456789abcdef
     Pseudo is Norm-conserving, Zval = 10.0
     Pseudo is Norm-conserving, Zval =  6.0

     PseudoPot. # 2 for O  read from file:
     /pseudos/O_ONCV_PBE-1.0.upf
     MD5 check sum: 0123456789abcdef0123456789abcdef
     Generated using ONCVPSP code by D. R. Hamann
"""
    path = tmp_path / "pw.out"
    path.write_text(text)

    with pytest.raises(ValueError, match="does not fall inside the block"):
        QEParser.get_zval_dict_from_PWOutput(PWOutput(str(path)))


def test_matches_the_batio3_fixture():
    """Guard the real file too, since it is what the regression test parses."""
    fixture = (Path(__file__).resolve().parent
               / "BaTiO3_QE_IO_nospin" / "non_pol" / "batio3_np.out")
    pw_out = PWOutput(str(fixture))
    assert QEParser.get_zval_dict_from_PWOutput(pw_out) == {
        "Ba": 10.0,
        "O": 6.0,
        "Ti": 12.0,
    }
