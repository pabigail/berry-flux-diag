from context import BerryFluxDiag as bfd


base_path_qe = 'tests/BaTiO3_QE_IO_nospin/'
np_xml_file = base_path_qe+'non_pol/batio3_np.xml'
pol_xml_file = base_path_qe+'pol/batio3_pol.xml'
np_wfcn_path = base_path_qe+'non_pol/batio3_np.save/'
pol_wfcn_path = base_path_qe+'pol/batio3_pol.save/'
np_out_file = base_path_qe+'non_pol/batio3_np.out'

qe_dict = bfd.QEParser.qe_parser(pol_xml_file,
                        np_xml_file,
                        pol_wfcn_path,
                        np_wfcn_path,
                        np_out_file)

BaTiO3_Overlaps = bfd.Overlaps.Overlaps(qe_dict)

pol, _ = BaTiO3_Overlaps.compute_polarization()
print(pol)
print("test complete")
