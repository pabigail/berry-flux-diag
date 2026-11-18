"""QE species labels map to elements, and the labels themselves are kept.

Magnetic QE runs give each sublattice its own species label (Fe1 and Fe2
in an antiferromagnet). pymatgen used to receive these as they were: it
read Fe1 and Fe2 as the ions Fe+ and Fe2+, and failed on labels such as
Fe_up or FeA, which QE accepts.
"""

from __future__ import annotations

import pytest
from pymatgen.core.periodic_table import Element

from berry_flux_diag import QEParser, utils

pytestmark = pytest.mark.skipif(QEParser is None, reason="QEParser needs qeschema and h5py")


@pytest.mark.parametrize("label, element", [
    ("Fe", "Fe"), ("Fe1", "Fe"), ("Fe2", "Fe"), ("Fe10", "Fe"), ("Fe_up", "Fe"),
    ("Fe-1", "Fe"), ("FeA", "Fe"), ("Feup", "Fe"), ("FE1", "Fe"),
    ("O", "O"), ("O1", "O"), ("Oup", "O"), ("O_2", "O"),
    ("Co", "Co"), ("C", "C"), ("Bi", "Bi"), ("Ba", "Ba"), ("B1", "B"),
])
def test_label_to_element(label, element):
    assert QEParser.element_from_qe_label(label) == element


@pytest.mark.parametrize("label", ["", "1Fe", "_Fe", "Xx"])
def test_unreadable_labels_raise(label):
    with pytest.raises(ValueError, match="cannot read an element"):
        QEParser.element_from_qe_label(label)


def xml_dict(labels, cart_bohr):
    return {"qes:espresso": {"output": {"atomic_structure": {
        "cell": {"a1": [8.0, 0, 0], "a2": [0, 8.0, 0], "a3": [0, 0, 8.0]},
        "atomic_positions": {"atom": [{"@name": l, "$": list(c)}
                                      for l, c in zip(labels, cart_bohr)]},
    }}}}


def test_afm_structure_has_plain_elements_and_keeps_labels():
    labels = ["Fe1", "Fe2", "Bi", "O"]
    s = QEParser.get_struct_from_qeschema(
        xml_dict(labels, [[0, 0, 0], [4, 4, 4], [2, 2, 2], [6, 6, 6]]))
    assert [str(site.specie) for site in s] == ["Fe", "Fe", "Bi", "O"]
    assert all(isinstance(site.specie, Element) for site in s)  # not the ion Fe+
    assert s.site_properties["qe_label"] == labels


def test_labels_pymatgen_cannot_read_now_parse():
    s = QEParser.get_struct_from_qeschema(xml_dict(["Fe_up", "FeA"], [[0, 0, 0], [4, 4, 4]]))
    assert [str(site.specie) for site in s] == ["Fe", "Fe"]


def test_a_single_atom_is_read():
    """xmlschema gives a lone repeated element as a dict, not a list."""
    d = xml_dict(["Fe1"], [[0, 0, 0]])
    atoms = d["qes:espresso"]["output"]["atomic_structure"]["atomic_positions"]
    atoms["atom"] = atoms["atom"][0]
    assert len(QEParser.get_struct_from_qeschema(d)) == 1


def test_swapped_magnetic_sublattices_raise():
    """Same elements, but Fe1 and Fe2 exchanged between the runs: each spin
    channel would be paired with the other's sublattice."""
    cart = [[0, 0, 0], [4, 4, 4]]
    pol = QEParser.get_struct_from_qeschema(xml_dict(["Fe1", "Fe2"], cart))
    non_pol = QEParser.get_struct_from_qeschema(xml_dict(["Fe2", "Fe1"], cart))
    with pytest.raises(ValueError, match="site 0: Fe1 .* Fe2"):
        utils.check_species_match(pol, non_pol)
    utils.check_species_match(pol, pol)
