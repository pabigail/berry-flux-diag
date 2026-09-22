"""Capture a BaTiO3 reference result for regression testing.

Run this with the CURRENT code before refactoring, and store the resulting
JSON under tests/reference/. The regression tests in
test_regression_batio3.py recompute the same quantities and compare against
the stored file, so a refactor that changes a number gets caught.

Usage
-----
    python tests/capture_reference.py qe
    python tests/capture_reference.py vasp --pol-dir <dir> --np-dir <dir>
    python tests/capture_reference.py vasp-unnormalized --pol-dir <dir> --np-dir <dir>

The QE inputs live in the repo (tests/BaTiO3_QE_IO_nospin). VASP inputs do
not: WAVECAR files are hundreds of megabytes and POTCAR files are
copyrighted, so pass their location on the command line or set
BFD_VASP_BATIO3_POL / BFD_VASP_BATIO3_NP.

The vasp-unnormalized mode captures the no-pawpyseed path. Its number is
not a physical polarization, but it is deterministic, and unlike the vasp
mode it can be captured and checked on a machine without MKL.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import berry_flux_diag as bfd  # noqa: E402

QE_DIR = REPO / "tests" / "BaTiO3_QE_IO_nospin"
REFERENCE_DIR = REPO / "tests" / "reference"

# Files pawpyseed and the VASP parser expect to find unzipped in a run directory.
VASP_FILES = ("POSCAR", "CONTCAR", "WAVECAR", "POTCAR", "OUTCAR", "vasprun.xml")

# The unnormalized parser reads the WAVECAR with pymatgen alone, so it needs
# neither the run directory nor the files pawpyseed goes looking for.
VASP_UNNORMALIZED_FILES = ("POSCAR", "WAVECAR", "POTCAR")


def provenance() -> dict:
    """Record enough about this run to reproduce or explain the numbers."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha, dirty = "unknown", False

    from importlib.metadata import PackageNotFoundError, version

    try:
        pymatgen_version = version("pymatgen")
    except PackageNotFoundError:
        pymatgen_version = "unknown"

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pymatgen": pymatgen_version,
    }


def summarize(overlaps, pol_norm: float) -> dict:
    """Pull the quantities worth regressing out of an Overlaps run.

    The string sums are the electronic quantity the Berry flux machinery
    actually produces; the polarization is what a user reads off. Keep both:
    a refactor that breaks one usually breaks the other, but storing both
    says which half went wrong.
    """
    string_sums, string_lens, _debug = overlaps.compute_string_sums()

    summary = {
        "polarization_norm": float(pol_norm),
        "string_sums": [float(np.real(s)) for s in string_sums],
        "string_lens": [int(n) for n in string_lens],
        "num_kpoints": len(overlaps.kpoint_list),
        "spin_polarized": bool(overlaps.spin_pol),
        "zval_dict": {k: float(v) for k, v in overlaps.zval_dict.items()},
        "pol_formula": overlaps.pol_struct.composition.reduced_formula,
        "pol_lattice_abc": list(overlaps.pol_struct.lattice.abc),
        "np_lattice_abc": list(overlaps.np_struct.lattice.abc),
    }
    if overlaps.spin_pol:
        summary["band_fill_up"] = int(overlaps.band_fill_up)
        summary["band_fill_down"] = int(overlaps.band_fill_down)
    else:
        summary["band_fill"] = int(overlaps.band_fill)
    return summary


def capture_qe() -> dict:
    """Reference from the Quantum ESPRESSO inputs committed to the repo."""
    parse_dict = bfd.QEParser.qe_parser(
        str(QE_DIR / "pol" / "batio3_pol.xml"),
        str(QE_DIR / "non_pol" / "batio3_np.xml"),
        str(QE_DIR / "pol" / "batio3_pol.save") + "/",
        str(QE_DIR / "non_pol" / "batio3_np.save") + "/",
        str(QE_DIR / "non_pol" / "batio3_np.out"),
    )
    overlaps = bfd.Overlaps.Overlaps(parse_dict)
    pol_norm, _ = overlaps.compute_polarization()

    return {
        "code": "QE",
        "system": "BaTiO3, non-spin-polarized",
        "inputs": {"source": "tests/BaTiO3_QE_IO_nospin (in repo)"},
        "provenance": provenance(),
        **summarize(overlaps, pol_norm),
    }


def stage_vasp_dir(src: Path, dest: Path, names=VASP_FILES) -> Path:
    """Copy the files the parsers need into dest, gunzipping as required.

    The source run directories are left untouched: they are research data,
    not scratch space, and pawpyseed wants plain files beside each other.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        plain, gz = src / name, src / (name + ".gz")
        if plain.exists():
            shutil.copy2(plain, dest / name)
        elif gz.exists():
            with gzip.open(gz, "rb") as fin, open(dest / name, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            raise FileNotFoundError(f"neither {plain} nor {gz} exists")
    return dest


def capture_vasp(pol_src: Path, np_src: Path) -> dict:
    if bfd.VASPParser is None:
        raise SystemExit(
            "berry_flux_diag.VASPParser did not import - pawpyseed is missing "
            "from this environment, so the VASP reference cannot be captured here."
        )

    with tempfile.TemporaryDirectory(prefix="bfd_vasp_ref_") as tmp:
        pol_dir = stage_vasp_dir(pol_src, Path(tmp) / "pol")
        np_dir = stage_vasp_dir(np_src, Path(tmp) / "np")

        parse_dict = bfd.VASPParser.vasp_parser(
            str(pol_dir / "POSCAR"),
            str(np_dir / "POSCAR"),
            str(pol_dir / "WAVECAR"),
            str(np_dir / "WAVECAR"),
            str(pol_dir / "POTCAR"),
            str(pol_dir),
            str(np_dir),
        )
        overlaps = bfd.Overlaps.Overlaps(parse_dict)
        pol_norm, _ = overlaps.compute_polarization()

        return {
            "code": "VASP",
            "system": "BaTiO3, non-spin-polarized",
            "inputs": {
                "pol_dir": str(pol_src),
                "np_dir": str(np_src),
                "note": "WAVECAR/POTCAR are not committed; point the test at these "
                        "directories with BFD_VASP_BATIO3_POL / BFD_VASP_BATIO3_NP.",
            },
            "provenance": provenance(),
            **summarize(overlaps, pol_norm),
        }


def capture_vasp_unnormalized(pol_src: Path, np_src: Path) -> dict:
    """Reference for the no-pawpyseed VASP path.

    The polarization here is physically wrong - the PAW augmentation terms
    are missing - but it is deterministically wrong, which is all a
    regression fixture needs. Its value is that it runs anywhere pymatgen
    does, so the VASP string construction, k-point handling and parse
    dictionary stay covered on machines that cannot build pawpyseed.
    """
    from berry_flux_diag import VASPParser_unnormalized as vasp_unnorm

    with tempfile.TemporaryDirectory(prefix="bfd_vasp_unnorm_ref_") as tmp:
        stage = VASP_UNNORMALIZED_FILES
        pol_dir = stage_vasp_dir(pol_src, Path(tmp) / "pol", stage)
        np_dir = stage_vasp_dir(np_src, Path(tmp) / "np", stage)

        parse_dict = vasp_unnorm.vasp_parser(
            str(pol_dir / "POSCAR"),
            str(np_dir / "POSCAR"),
            str(pol_dir / "WAVECAR"),
            str(np_dir / "WAVECAR"),
            str(pol_dir / "POTCAR"),
        )
        overlaps = bfd.Overlaps.Overlaps(parse_dict)
        pol_norm, _ = overlaps.compute_polarization()

        return {
            "code": "VASP_unnormalized",
            "system": "BaTiO3, non-spin-polarized",
            "warning": "NOT a physical polarization: the PAW augmentation "
                       "terms are missing. This fixture exists only to detect "
                       "unintended changes in the no-pawpyseed code path.",
            "inputs": {
                "pol_dir": str(pol_src),
                "np_dir": str(np_src),
                "note": "WAVECAR/POTCAR are not committed; point the test at these "
                        "directories with BFD_VASP_BATIO3_POL / BFD_VASP_BATIO3_NP.",
            },
            "provenance": provenance(),
            **summarize(overlaps, pol_norm),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", choices=["qe", "vasp", "vasp-unnormalized"])
    parser.add_argument("--pol-dir", type=Path, help="VASP run directory, polar structure")
    parser.add_argument("--np-dir", type=Path, help="VASP run directory, nonpolar structure")
    parser.add_argument("--out", type=Path, help="where to write the JSON")
    args = parser.parse_args()

    if args.code == "qe":
        result = capture_qe()
        out = args.out or REFERENCE_DIR / "batio3_qe_nospin.json"
    else:
        if not (args.pol_dir and args.np_dir):
            parser.error(f"{args.code} needs --pol-dir and --np-dir")
        if args.code == "vasp":
            result = capture_vasp(args.pol_dir, args.np_dir)
            out = args.out or REFERENCE_DIR / "batio3_vasp_nospin.json"
        else:
            result = capture_vasp_unnormalized(args.pol_dir, args.np_dir)
            out = args.out or REFERENCE_DIR / "batio3_vasp_unnormalized_nospin.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"polarization_norm = {result['polarization_norm']:.10f} uC/cm^2")
    print(f"string_sums       = {result['string_sums']}")


if __name__ == "__main__":
    main()
