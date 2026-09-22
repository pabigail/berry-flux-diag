# berry-flux-diag

**berry-flux-diag** is a DFT pre- and post-processing package to compute **differences in formal polarization** using the **Berry Flux diagonalization** approach. For details on the methodology, see:

- [Bonini et al., *Phys. Rev. B* 102, 045141 (2020)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.045141)  
- [Poteshman et al., *npj Comp. Mat.* 12(1), (2026) ](https://www.nature.com/articles/s41524-025-01955-1)

**berry-flux-diag** supports multiple DFT codes, including **Quantum ESPRESSO** and **VASP**, via optional dependency groups.

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

- Both reference structures must have **the same number of atoms in the same order** in the input files.  
- The DFT calculations for the two structures must use **the same k-point mesh discretization**.
- Need to save wavefunctions over the entire BZ
    - VASP: `INCAR` must have `LWAVE = True` and `ISYM = -1`
    - Quantum ESPRESSO: .in file must have `nosym = True` in `&SYSTEM` and `wf_collect = True` in `&CONTROL`
- Apply translation that minimizes maximum atomic displacement between two reference structures; if the translation that minimizes this distance is still > 0.3 Angstroms, add as many interpolations as necessary so that the max atomic displacement between any two adjacent structures is <= 0.3 Angstroms. See the preprocessing tutorial for automated code to do so, and [Poteshman et al., *npj Comp. Mat.* 12(1), (2026) ](https://www.nature.com/articles/s41524-025-01955-1) for more details. 

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

BerryFluxDiag offers four optional installation modes:

1. Quantum ESPRESSO — [QE]
    
depedencies:
- `qeschema`
- `h5py`

    ```bash
    pip install .[QE]
    ```

2. VASP — [VASP]
    
dependencies:
- `pawpyseed` (note: this requires the Intel Math Kernel Library (MKL), which is widely available on most computing clusters but is not necessarily available on Macs — see [Running the VASP workflow without pawpyseed](#running-the-vasp-workflow-without-pawpyseed))

    ```bash
    pip install .[VASP]
    ```

3. VASP with atomate2 jobflows — [VASP\_atomate2]
    
dependencies:
- `pawpyseed` (see above note)
- `atomate2`
- `jobflow`
- `monty`

    ```bash
    pip install .[VASP_atomate2]
    ```

4. All supported workflows — [all]
   ```bash
    pip install .
    ```

---

## Running the VASP workflow without pawpyseed

`pawpyseed` requires the Intel MKL and therefore does not build on machines
without it, including Apple Silicon Macs. For development on such a machine
there is a parallel module, `VASPParser_unnormalized`, that reads a WAVECAR
with pymatgen alone and needs no pawpyseed:

```python
from berry_flux_diag import VASPParser_unnormalized as vasp_unnorm

parse_dict = vasp_unnorm.vasp_parser(pol_POSCAR, np_POSCAR,
                                     pol_WAVECAR, np_WAVECAR, POTCAR)
```

> [!WARNING]
> **The polarization it produces is not physically correct.** A WAVECAR
> holds pseudo-wavefunctions, which are orthonormal only under the PAW
> overlap operator. Reconstructing the missing augmentation contribution
> inside each PAW sphere is precisely what pawpyseed does for
> `VASPParser`. Without it the overlap matrices are not unitary and the
> Berry phases are wrong. Do not quote, plot or publish a number from
> this module.

Only the electronic term is affected — the ionic term comes from ZVAL and
site positions, so it is unchanged. For the BaTiO₃ case in `tests/`
(6×6×6 mesh, VASP PAW potentials):

| | electronic | ionic | total |
|---|---|---|---|
| `VASPParser` (pawpyseed) | 17.18 | 28.65 | 45.84 |
| `VASPParser_unnormalized` | 16.13 | 28.65 | 44.78 |
| | **−6.1%** | — | −2.3% |

The 2.3% on the total is flattering: it is diluted by an ionic term that
was never in question. The error lives entirely in the 6%, and it is an
uncontrolled, species-dependent artifact rather than a bounded
approximation — there is no reason to expect a comparable size for a
different material.

What the module *is* good for is exercising everything around the
overlaps: preprocessing, string construction, the diagonalization
machinery, and the jobflow plumbing. For a correct VASP result, use
`VASPParser` on a machine with MKL.

It is regression-tested on exactly that basis. `tests/reference/` pins the
number it produces for BaTiO₃ — not because the number is right, but
because it is deterministic, which makes it the only VASP coverage
available without MKL. WAVECAR files are too large to commit, so point the
test at a pair of run directories:

```bash
export BFD_VASP_BATIO3_POL=/path/to/polar/run
export BFD_VASP_BATIO3_NP=/path/to/nonpolar/run
pytest tests/test_regression_batio3.py
```

Without those variables the VASP tests skip and the QE test still runs.
To regenerate the fixture deliberately — never to make a red test pass:

```bash
python tests/capture_reference.py vasp-unnormalized \
    --pol-dir "$BFD_VASP_BATIO3_POL" --np-dir "$BFD_VASP_BATIO3_NP"
```

Note that it is not imported by `berry_flux_diag/__init__.py` and must be
named explicitly, so that these numbers cannot be reached by accident when
pawpyseed is merely missing. Its `vasp_parser` also takes five arguments
rather than the seven `VASPParser.vasp_parser` takes — the two run
directories are only needed by pawpyseed — so the two are not drop-in
replacements, by design.

This split exists only because of pawpyseed's MKL requirement and its lack
of maintenance since 2021. Replacing pawpyseed with a maintainable PAW
overlap implementation would remove the need for this module.
