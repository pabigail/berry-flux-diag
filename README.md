# berry-flux-diag

# 📘 berry-flux-diag

DFT post-processing package to compute differences in formal polarization via the Berry Flux diagonalization approach. For details on the methodology, see [Bonini et al., Phys. Rev B 102, 045141 (2020)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.045141) and [Poteshman et al., arXiv:2511.18586](https://arxiv.org/abs/2511.18586).

berry-flux-diag supports multiple DFT codes (Quantum ESPRESSO and VASP) through optional dependency groups.  

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


## ⚙️ Installation Options

BerryFluxDiag offers four optional installation modes:

1. Quantum ESPRESSO — [QE]
2. VASP — [VASP]
3. VASP with atomate2 jobflows — [VASP-atomate2]
4. All supported workflows — [all]

## Quantum ESPRESSO [QE]


1. polar .xml file
2. non-polar .xml file
3. path to folder with polar wavefunctions in .hdf5 format
4. path to folder with non-polar wavefunctions in .hdf5 format
5. non-polar .out file
