# berry-flux-diag

Main dependencies: qeschema, h5py, and pymatgen

to run example:
- clone and install in an environment that has qeshema, h5py, and pymatgen
- from the main folder run: python tests/test_BaTiO3_QE_no_spin_polarization.py

to preprare QE input:
- polar and non-polar reference structures need to be translated to minimize the maximal atomic displacement between the two structures
- in &SYSTEM, need tags: nosym = .TRUE. and noinv = .TRUE.
- wavefunctions need to be saved in hdf5 format 
