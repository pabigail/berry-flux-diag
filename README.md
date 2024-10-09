# berry-flux-diag

Main dependencies: qeschema, h5py, and pymatgen

to run example:
- clone and install in an environment that has qeshema, h5py, and pymatgen
- from the main folder run: python tests/test_BaTiO3_QE_no_spin_polarization.py

to preprare QE input:
- polar and non-polar reference structures need to be translated to minimize the maximal atomic displacement between the two structures
- in &SYSTEM, need tags: nosym = .TRUE. and noinv = .TRUE.
- wavefunctions need to be saved in hdf5 format 
- see example for how to generate the optimal translation from preprocess_QE_example: (input POSCAR_pol_orig and POSCAR_np_orig -> output POSCAR_np_trans_1), and run scf calculation on POSCAR_pol_orig and POSCAR_np_trans_1
- see example for how to generate QE.in file from POSCAR_np_trans_1 and POSCAR_pol_orig (this only works for non-magnetic materials at the moment, but can manually update .in file for magnetic materials and DFT+U calcs)

to run bfd code need:
- polar .xml file
- non-polar .xml file
- path to folder with polar wavefunctions in .hdf5 format
- path to folder with non-polar wavefunctions in .hdf5 format
- non-polar .out file
