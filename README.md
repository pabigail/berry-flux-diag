# berry-flux-diag

Main dependencies: qeschema, h5py, and pymatgen

to run example:
- clone and install in an environment that has qeshema, h5py, and pymatgen
- from the main folder run: python tests/test_BaTiO3_QE_no_spin_polarization.py

to preprare QE input:
- from VASP POSCAR -> QE input: see example on CrO3 (this only currently works for non-magnetic materials without DFT+U)
- from QE input -> QE input: see example on FeBiO3 (this currently works only for spin-polarized wavefunctions with DFT+U corrections)
- In general, need: (1) polar and non-polar reference structures need to be translated to minimize the maximal atomic displacement between the two structures, (2) nosym = .TRUE. and noinv = .TRUE. in &SYSTEM, and (3) wavefunctions need to be saved in hdf5 format

to run bfd code need:
- polar .xml file
- non-polar .xml file
- path to folder with polar wavefunctions in .hdf5 format
- path to folder with non-polar wavefunctions in .hdf5 format
- non-polar .out file
