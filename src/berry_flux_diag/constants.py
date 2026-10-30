"""Physical constants and unit conversions.

Every module takes its constants from here, so that a polarization computed
through the Overlaps class and one computed through the jobflow jobs differ
only where the physics differs.
"""

# Elementary charge in coulombs. Exact by the 2019 SI redefinition of the
# ampere, so this value is fixed and will not move with future CODATA
# adjustments.
ELEMENTARY_CHARGE = 1.602176634e-19

# Polarization is assembled here as an electron-Angstrom dipole divided by a
# cell volume in Angstrom^3, i.e. in e/Angstrom^2, and reported in uC/cm^2:
#
#   1 e/Ang^2 = ELEMENTARY_CHARGE C / (1e-8 cm)^2 = ELEMENTARY_CHARGE * 1e16 C/cm^2
#             = ELEMENTARY_CHARGE * 1e16 * 1e6 uC/cm^2
#
# which is 1602.176634 uC/cm^2 per e/Ang^2.
E_PER_ANG2_TO_MUC_PER_CM2 = ELEMENTARY_CHARGE * 1e16 * 1e6

# Bohr radius in Angstrom, for the cell and atomic positions Quantum
# ESPRESSO writes in atomic units. Unlike the elementary charge this is a
# measured quantity, CODATA 2022: a0 = 5.29177210544(82)e-11 m. Its relative
# uncertainty is about 1.6e-12, four orders of magnitude below the error
# from truncating it, so carrying the full value costs nothing and removes
# the question.
BOHR_TO_ANGSTROM = 0.529177210544
