# berry-flux-diag

DFT pre- and  post-processing package to compute differences in formal polarization via the Berry Flux diagonalization approach. For details on the methodology, see [Bonini et al., Phys. Rev B 102, 045141 (2020)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.045141) and [Poteshman et al., arXiv:2511.18586](https://arxiv.org/abs/2511.18586).

berry-flux-diag supports multiple DFT codes (Quantum ESPRESSO and VASP) through optional dependency groups.  

The general workflow to compute differences in formal polarization is to (1) pre-process your input DFT files (INCAR, POSCAR, KPOINTS for VASP) and (qe.in file for Quantum ESPRESSO), (2) run the DFT calculation, and (3) post-process the DFT output to compute differences in formal polarization. Unlike the standard approach to compute formal polarization differences within the modern theory of polarization by computing formal polarization for two reference structures separately and then finding the differences in formal polarization along the same branch, Berry flux diagonalization computes a gauge-invariant difference in formal polarization directly from the wavefunctions of the two reference structures together. Practically speaking, the main constraint of this post-processing tool is that the two reference structures (no matter which DFT tool you're using) must have the same number of atoms in the same order in the input files, and you must enforce that the DFT calculations for the two reference structures you're computing differences between must have the same k-point mesh discretization. We provide both automatic tools to pre-process your DFT input and instructions for the constraints you must satisfy if you would like to manually pre-process your DFT input.    

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
2. VASP — [VASP]
3. VASP with atomate2 jobflows — [VASP-atomate2]
4. All supported workflows — [all]

## 1. Quantum ESPRESSO [QE]



1. polar .xml file
2. non-polar .xml file
3. path to folder with polar wavefunctions in .hdf5 format
4. path to folder with non-polar wavefunctions in .hdf5 format
5. non-polar .out file
