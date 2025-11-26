# berry-flux-diag

**berry-flux-diag** is a DFT pre- and post-processing package to compute **differences in formal polarization** using the **Berry Flux diagonalization** approach. For details on the methodology, see:

- [Bonini et al., *Phys. Rev. B* 102, 045141 (2020)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.045141)  
- [Poteshman et al., arXiv:2511.18586](https://arxiv.org/abs/2511.18586)

**berry-flux-diag** supports multiple DFT codes, including **Quantum ESPRESSO** and **VASP**, via optional dependency groups.

---

## Workflow Overview

To compute differences in formal polarization:

1. **Pre-process your input DFT files**  
   - **VASP**: `INCAR`, `POSCAR`, `KPOINTS`  
   - **Quantum ESPRESSO**: `qe.in`  

2. **Run the DFT calculation**

3. **Post-process the DFT output** to compute differences in formal polarization

Unlike the standard approach—where formal polarization is computed separately for two reference structures and then subtracted along the same branch—**Berry Flux diagonalization computes a gauge-invariant difference directly from the wavefunctions of both structures together**.

---

## Constraints

- Both reference structures must have **the same number of atoms in the same order** in the input files.  
- The DFT calculations for the two structures must use **the same k-point mesh discretization**.
- Need to save wavefunctions over the entire BZ
    - VASP: `INCAR` must have `LWAVE = True` and `ISYM = -1`
    - Quantum ESPRESSO: .in file must have `nosym = True` in `&SYSTEM` and `wf_collect = True` in `&CONTROL`
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
    ```bash
    pip install .[QE]
    ```
2. VASP — [VASP]
    ```bash
    pip install .[VASP]
    ```
3. VASP with atomate2 jobflows — [VASP\_atomate2]
    ```bash
    pip install .[VASP_atomate]
    ```
4. All supported workflows — [all]
   ```bash
    pip install .
    ```
