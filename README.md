# berry-flux-diag

**berry-flux-diag** is a DFT pre- and post-processing package to compute **differences in formal polarization** using the **Berry Flux diagonalization** approach. For details on the methodology, see:

- [Bonini et al., *Phys. Rev. B* 102, 045141 (2020)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.045141)  
- [Poteshman et al., *npj Comp. Mat.* 12(1), (2026) ](https://www.nature.com/articles/s41524-025-01955-1)

**berry-flux-diag** supports multiple DFT codes, including **Quantum ESPRESSO** and **VASP**, via optional dependency groups.

---

## What's new

This release fixes several bugs that produced a plausible but wrong
polarization rather than an error. It removes the Intel MKL dependency
by reimplementing the part of `pawpyseed` that was used, makes the VASP
path roughly 15× faster, and adds more robust and systematic testing.

### 1. Bug fixes

Several of these changed a polarization without raising an error, so
**results from earlier versions may be affected**, as noted per item.

- **Occupied-band count (off by one).** The filled-band count accepted a
  k-point only if it had *two* more filled bands than the running maximum.
  A single extra band was dropped from the Berry flux, depending on
  k-point order. Materials with uniform filling across k-points, such as
  BaTiO₃, are unaffected, but small-gap systems may be.
- **Duplicate k-point strings.** `get_strings` grouped k-points by exact
  float equality but filled the groups with a 1e-5 tolerance. When the two
  disagreed, a string was counted twice and the polarization averaged over
  an inflated string count. Regular meshes, including BaTiO₃, are unchanged.
- **QE valence charges.** `Zval` was read at a fixed line offset in the QE
  output, which differs between QE builds. When the offset missed, the
  previous species' charge was silently reused, giving a wrong ionic term.
  Charges are now paired to species by position and checked.
- **PAW momentum-grid cutoff.** The grid VASP wavefunctions are expressed
  on was pinned at 1000 eV. It now defaults to 4 × ENCUT, the size the
  matrix elements actually need, and a convergence check backs it (see
  [PAW reconstruction](#paw-reconstruction-for-vasp)). BaTiO₃ moves from
  45.84 to 45.79 µC/cm².
- **Physical constants.** The exact SI value of the elementary charge is
  now pinned, alongside the untruncated Bohr radius. Previous versions used
  different levels of precision at different points of the calculation.
- **Stated constraints are now enforced.** Each of these used to give a
  plausible number instead of an error:
  - the two runs having different k-point meshes or k-point order;
  - the two structures listing atoms in different orders;
  - a symmetry-reduced (irreducible-wedge) k-point set.
- **Unsupported calculations are rejected, not misread.** These now raise
  an error that says what to change:
  - QE noncollinear and spin-orbit runs;
  - noncollinear and Γ-only WAVECARs;
  - VASP 6 double-precision WAVECARs, which `pawpyseed` read silently as
    single precision.
- **Preprocessing.** The number of interpolated images was decided from the
  displacement *before* translation, so it asked for more SCF runs than
  needed (six structures instead of two in one test case).
- **jobflow layer** (`BFDMaker`, `BFDJobs`). This code could not have run:
  - it called a function that does not exist;
  - it imported modules by names that do not exist;
  - a `psuedos`/`pseudos` misspelling made a parsed QE input fail to write
    back out.

  These are fixed, and the preprocessing logic now lives in one tested
  place. BFDJobs and BFDMaker still cannot be imported in an environment
  where atomate2 and pymatgen conflict (`pymatgen.core.entries`), so these
  fixes are covered only by checks on their source.
- **Diagnostics.** The `eig_thresh` setting was stored but never read. Both
  convergence thresholds are now named, explained, and adjustable.
- **Packaging.** `pyproject.toml` had no build system, every install command
  in this README failed under zsh, and `pip install .` did not install the
  `all` extra. These are fixed, and the package is licensed BSD-3-Clause.

**Breaking changes:**
- `preprocess_structs` and `preprocess_POSCARS` take `max_disp`, not
  `MAX_DISP`.
- `VASPParser.get_wfcn_data_from_vasp_pawpy` and `wfcn_dict_from_pawpy` are
  renamed `get_paw_wfcn_data` and `paw_wfcn_dict`. `vasp_parser`'s
  signature is unchanged.
- Progress and results are logged rather than printed (see
  [Logging](#logging)).

### 2. Dependency removal: `pawpyseed` and the Intel MKL

VASP wavefunctions need a PAW reconstruction step, which used to come from
[`pawpyseed`](https://github.com/kylebystrom/pawpyseed). Its C core needs
the Intel MKL, so the VASP path could not run on a Mac, and the package has
not been released since 2021.

`berry_flux_diag.paw` is a NumPy reimplementation of exactly the part of
`pawpyseed` that was used, and nothing else. The VASP extra no longer
installs anything beyond the base dependencies. The reimplementation was
checked against `pawpyseed`'s own C code, and every coefficient agrees to
1.5 × 10⁻¹⁴ relative. Details and attribution are in
[PAW reconstruction](#paw-reconstruction-for-vasp).

### 3. Optimizations

Measured on the BaTiO₃ test case (6×6×6 mesh, 20 occupied bands,
4 × ENCUT grid, M3 Max):

| Stage | Before | After |
|---|---|---|
| PAW reconstruction, per WAVECAR, at the old 1000 eV grid | ~115 s (`pawpyseed`) | ~6 s (first NumPy version) |
| `vasp_parser` (both runs) | 30.3 s (first NumPy version) | **9.3 s** |
| Overlaps and Berry phases (`compute_string_sums`) | 281 s | **11 s** |
| BaTiO₃ VASP regression test | ~8 min | **32 s** |

How:
- **Overlaps.** Each overlap matrix is one matrix product instead of a
  double loop, the plane-wave remapping is vectorized, and an overlap
  matrix that recurs across strings is computed once per call.
- **Parsing.**
  - The WAVECAR headers are read once instead of re-reading each whole file
    with pymatgen.
  - The angular and radial factors of the PAW sum are built once per
    k-point rather than per band and per site.
  - The polar and non-polar runs are reconstructed in parallel threads.
- **String sums.** These used to be re-summed once per string (quadratic in
  the number of strings).

Every optimization step was required to leave the output **bit-identical**
to the previous step, not merely within tolerance.

### 4. Testing

The previous single test file had never run; it failed to import. There
are now **224 tests in 17 files**. Only the two VASP regression tests need
data that is not in the repository. Coverage includes:
- **Regression fixtures** for BaTiO₃ (QE, VASP, and VASP without the PAW
  step), compared with `rtol=0`. The default relative tolerance had let a
  constant change pass unnoticed.
- **The PAW building blocks against closed forms:**
  - the spherical Bessel transform of a Gaussian;
  - Y_lm orthonormality;
  - spline accuracy;
  - the G sphere against brute force.
- **A synthetic WAVECAR writer.** The new header reader is checked against
  pymatgen on every WAVECAR layout, including the ones that are then
  rejected.
- **Tests for the bug fixes above**, covering the inputs the old code got
  wrong. Examples are a band that fills at only one k-point, a QE banner
  layout without the MD5 line, and k-points off a line by 1e-7.
- **Optimizations against the code they replaced.** Each optimization is
  tested against the replaced code, kept verbatim in its test, at the bit
  level. For the later steps the tests were also *mutation-checked*: the
  new code was deliberately broken in several ways to confirm a test
  fails.

See [Testing](#testing) for how to run them.

---

## Workflow Overview

To compute differences in formal polarization:

1. **Pre-process your input DFT files**  
   - **VASP**: `INCAR`, `POSCAR`, `KPOINTS`  
   - **Quantum ESPRESSO**: `qe.in`  
   - See tutorials/preprocess\_DFT.ipynb

2. **Run the DFT calculation**

3. **Post-process the DFT output** to compute differences in formal polarization
   -  (See tutorials/postprocess\_DFT.ipynb)

Unlike the standard approach—where formal polarization is computed separately for two reference structures and then subtracted along the same branch—**Berry Flux diagonalization computes a gauge-invariant difference directly from the wavefunctions of both structures together**.

---

## Constraints

The first three are checked when the output is parsed, and a violation
raises an error naming what to change.

- Both reference structures must have **the same number of atoms in the same order** in the input files.  
- The DFT calculations for the two structures must use **the same k-point mesh discretization**.
- Need to save wavefunctions over the entire BZ
    - VASP: `INCAR` must have `LWAVE = True` and `ISYM = -1`
    - Quantum ESPRESSO: .in file must have `nosym = True` in `&SYSTEM` and `wf_collect = True` in `&CONTROL`
- Apply translation that minimizes maximum atomic displacement between two reference structures; if the translation that minimizes this distance is still > 0.3 Angstroms, add as many interpolations as necessary so that the max atomic displacement between any two adjacent structures is <= 0.3 Angstroms. See the preprocessing tutorial for automated code to do so, and [Poteshman et al., *npj Comp. Mat.* 12(1), (2026) ](https://www.nature.com/articles/s41524-025-01955-1) for more details. 

### Supported calculations

| | Quantum ESPRESSO | VASP |
|---|---|---|
| Non-spin-polarized | ✅ | ✅ |
| Collinear spin-polarized | ❌ not yet supported | ❌ not yet supported |
| Noncollinear / spin-orbit | ❌ | ❌ |
| Γ-only | — | ❌ |

Noncollinear, spin-orbit and Γ-only runs raise an error. Collinear
spin-polarized runs are not yet rejected with a clear error: a QE run
still returns a number, which should not be used.

---

## Logging

The package logs rather than prints, so importing it stays quiet and it
will not interfere with the logging of a program that uses it. Warnings —
an underconverged k-mesh, a band filling that differs between the two runs
— appear by default. Progress and results do not, until you ask:

```python
import berry_flux_diag as bfd

bfd.configure_logging()          # INFO: string sums, contributions, result
bfd.configure_logging("DEBUG")   # adds per-direction progress
```

This touches only the `berry_flux_diag` logger, never the root logger, and
calling it twice replaces the handler rather than duplicating messages.

---

## 🔧 Installation

The recommended workflow is:

1. **Clone this repository**
2. **Install the package (with the dependency options you want)**

### Clone the GitHub repository

To get started, clone the repository and move into the project directory:

```bash
git clone https://github.com/pabigail/berry-flux-diag.git
cd berry-flux-diag
```

### ⚙️ Installation Options

berry-flux-diag offers four optional installation modes:

1. Quantum ESPRESSO — [QE]
    
dependencies:
- `qeschema`
- `h5py`

    ```bash
    pip install '.[QE]'
    ```

2. VASP — [VASP]

No dependencies beyond the base install (`numpy`, `pymatgen`). The PAW
reconstruction `VASPParser` needs is done in NumPy by `berry_flux_diag.paw`,
so it runs anywhere, including Apple Silicon Macs. The extra is kept,
empty, so that existing install commands still work.

    ```bash
    pip install '.[VASP]'
    ```

3. VASP with atomate2 jobflows — [VASP\_atomate2]
    
dependencies:
- `atomate2`
- `jobflow`
- `monty`

    ```bash
    pip install '.[VASP_atomate2]'
    ```

4. All supported workflows — [all]
   ```bash
    pip install '.[all]'
    ```

> [!NOTE]
> **conda users:** take NumPy and its BLAS from the same channel. A
> `defaults` NumPy running against a `conda-forge` OpenBLAS can abort
> with `Fatal Python error: Aborted` inside the overlap matrix products.
> Installing both from conda-forge fixes it.

---

## PAW reconstruction for VASP

A WAVECAR holds pseudo-wavefunctions, which are orthonormal only under the
PAW overlap operator. `VASPParser` adds back the augmentation contribution
inside each PAW sphere before the overlaps are taken. This used to be done
by `pawpyseed`, which needs the Intel MKL and has not been maintained since
2021. It is now done by `berry_flux_diag.paw`, a NumPy port of exactly the
part of pawpyseed that was used.

The port was checked against pawpyseed's own C code, compiled with a small
stand-in for the few MKL calls it makes. On the BaTiO₃ case in `tests/`
(6×6×6 mesh, 20 occupied bands) the momentum grid is identical and every
reconstructed coefficient agrees to 1.5×10⁻¹⁴ relative, which is rounding
from a different summation order. The polarization it gives at pawpyseed's
grid, 45.8377 µC/cm², matches the pawpyseed figure of 45.84.

### Acknowledgement

`berry_flux_diag/paw.py` is derived from
[**pawpyseed**](https://github.com/kylebystrom/pawpyseed) by Kyle Bystrom
and coauthors, © 2017 Kyle Bystrom, distributed under the BSD 3-Clause
License. Its `Pseudopotential` and `CoreRegion` classes are copied
verbatim, and the rest is a translation of pawpyseed's C:
- `read_wavecar` (reader.c);
- `get_projector_list`, `setup_site` and `onto_projector` (projector.c);
- `get_momentum_grid` and `fullwf_reciprocal` (momentum.c);
- the splines, Y_lm and Talman's spherical Bessel transform they use
  (utils.c, sbt.c).

The docstring of each function names the C function it reproduces, and
pawpyseed's full license notice is in the module docstring. If you use the
VASP path, please also cite pawpyseed:

> K. Bystrom, D. Broberg, S. Dwaraknath, K. A. Persson, and M. Asta,
> "Pawpyseed: Perturbation-extrapolation band shifting corrections for
> point defect calculations," arXiv:1904.11572 (2019).

### Momentum grid cutoff

The coefficients are expressed on one G grid shared by every k-point, with
cutoff `VASPParser.MOMENTUM_ENCUT`. It defaults to 4 × the run's ENCUT,
the physically motivated basis size. It used to be pinned at 1000 eV, and
`momentum_encut=1000` reproduces numbers from before the change. For
BaTiO₃ (µC/cm²):

| | electronic | ionic | total |
|---|---|---|---|
| `VASPParser`, 4 × ENCUT = 2720 eV (default) | 17.13 | 28.65 | 45.79 |
| `VASPParser`, 1000 eV (before) | 17.18 | 28.65 | 45.84 |
| `VASPParser_unnormalized` | 16.13 | 28.65 | 44.78 |

A convergence scan supports the default:

| Cutoff | G vectors | Electronic term |
|---|---|---|
| 3 × ENCUT | 14111 | 17.1446 |
| 4 × ENCUT | 21671 | 17.1337 |
| 6 × ENCUT | 39955 | 17.1297 |

The steps between these rows shrink roughly as cutoff⁻⁴, which puts
4 × ENCUT within about 0.005 µC/cm² of converged for BaTiO₃. The shift is
material-dependent, so pass a larger `momentum_encut` to check a new
system.

### `VASPParser_unnormalized`

A parallel module that skips the PAW reconstruction and reads the WAVECAR
with pymatgen alone. It was the only VASP path that ran without MKL, and is
now redundant.

> [!WARNING]
> **The polarization it produces is not physically correct.** Without the
> augmentation terms the overlap matrices are not unitary and the Berry
> phases are wrong — by 5.9% in the electronic term above, and by an
> uncontrolled, species-dependent amount in general. Do not quote, plot or
> publish a number from it.

It is not imported by `berry_flux_diag/__init__.py` and must be named
explicitly (`from berry_flux_diag import VASPParser_unnormalized`). Its
fixture is regenerated with `capture_reference.py vasp-unnormalized`. It is
a candidate for deletion.

---

## Testing

```bash
pip install '.[QE,test]'
pytest
```

Without VASP data this runs 222 tests in about 7 seconds. The two VASP
regression tests skip, and the QE regression test runs on inputs
committed in `tests/`. WAVECAR files are too large to commit and POTCARs
cannot be redistributed, so to run the VASP regression tests too, point
them at a pair of run directories:

```bash
export BFD_VASP_BATIO3_POL=/path/to/polar/run
export BFD_VASP_BATIO3_NP=/path/to/nonpolar/run
pytest            # 224 tests, about a minute
```

`tests/reference/` pins the BaTiO₃ results for each path. To regenerate a
fixture deliberately — never to make a failing test pass:

```bash
python tests/capture_reference.py vasp \
    --pol-dir "$BFD_VASP_BATIO3_POL" --np-dir "$BFD_VASP_BATIO3_NP"
```

---

## License

BSD 3-Clause; see [LICENSE](LICENSE). `berry_flux_diag/paw.py` includes
code derived from pawpyseed, also BSD 3-Clause, whose notice is
reproduced in that file.
