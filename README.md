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

---

## PAW reconstruction for VASP

A WAVECAR holds pseudo-wavefunctions, which are orthonormal only under the
PAW overlap operator. `VASPParser` adds back the augmentation contribution
inside each PAW sphere before the overlaps are taken. This used to be done
by `pawpyseed`, which needs the Intel MKL and has not been maintained since
2021. It is now done by `berry_flux_diag.paw`, a NumPy port of exactly the
part of pawpyseed that was used.

The port was checked against pawpyseed's own C code. On the BaTiO₃ case in
`tests/` (6×6×6 mesh, 20 occupied bands) every reconstructed coefficient
agrees to 1.4×10⁻¹⁴ relative, and the polarization is unchanged:

| | electronic | ionic | total |
|---|---|---|---|
| `VASPParser` | 17.18 | 28.65 | 45.84 |
| `VASPParser_unnormalized` | 16.13 | 28.65 | 44.78 |
| | **−6.1%** | — | −2.3% |

It is also about 19× faster: roughly 6 s per WAVECAR against pawpyseed's
115 s for that case.

### Regression tests

`tests/reference/` pins the BaTiO₃ numbers for both VASP paths. WAVECAR
files are too large to commit, so point the tests at a pair of run
directories:

```bash
export BFD_VASP_BATIO3_POL=/path/to/polar/run
export BFD_VASP_BATIO3_NP=/path/to/nonpolar/run
pytest tests/test_regression_batio3.py
```

Without those variables the VASP tests skip and the QE test still runs.
To regenerate a fixture deliberately — never to make a red test pass:

```bash
python tests/capture_reference.py vasp \
    --pol-dir "$BFD_VASP_BATIO3_POL" --np-dir "$BFD_VASP_BATIO3_NP"
```

### `VASPParser_unnormalized`

A parallel module that skips the PAW reconstruction and reads the WAVECAR
with pymatgen alone. It was the only VASP path that ran without MKL, and is
now redundant.

> [!WARNING]
> **The polarization it produces is not physically correct.** Without the
> augmentation terms the overlap matrices are not unitary and the Berry
> phases are wrong — by 6% in the electronic term above, and by an
> uncontrolled, species-dependent amount in general. Do not quote, plot or
> publish a number from it.

It is not imported by `berry_flux_diag/__init__.py` and must be named
explicitly (`from berry_flux_diag import VASPParser_unnormalized`). Its
fixture is regenerated with `capture_reference.py vasp-unnormalized`. It is
a candidate for deletion.
