import pytest
from adp3d import ADP3D
from adp3d import get_elements_from_XCS, minimal_XCS_to_Structure
from chroma import Protein
import gemmi


def test_ADP3D_init():
    adp = ADP3D()
    assert adp is not None


def test_get_elements_from_XCS():
    structure = Protein("confA.cif")
    X, _, S = structure.to_XCS()
    elements = get_elements_from_XCS(X, S)
    assert elements.size() > 0
    assert elements.size() > S.size()
    assert set(elements) == set(
        [
            "N",
            "CA",
            "C",
            "O",
            "OD1",
            "OD2",
            "OE1",
            "OE2",
            "OH",
            "OG",
            "OG1",
            "CB",
            "CG",
            "CG1",
            "CG2",
            "CD",
            "CD1",
            "CD2",
            "CE",
            "CE1",
            "CE2",
            "CE3",
            "CZ",
            "CZ2",
            "CZ3",
            "CH2",
            "NE",
            "NE1",
            "NE2",
            "NH1",
            "NH2",
            "ND1",
            "ND2",
            "NZ",
            "OD1",
            "OE1",
            "OD2",
            "SD",
            "SG",
            "H",
        ]
    )


def test_minimal_XCS_to_Structure():
    structure = Protein("confA.cif")
    X, _, S = structure.to_XCS()
    minimal_structure = minimal_XCS_to_Structure(X, S)
    assert minimal_structure is not None
    for key in ["coor", "b", "q", "e", "active"]:
        assert key in minimal_structure.keys()
    num_atoms = X.size()[1] * X.size()[2]
    assert minimal_structure["coor"].shape == num_atoms
    assert minimal_structure["b"].shape == num_atoms
    assert minimal_structure["q"].shape == num_atoms
    assert minimal_structure["active"].shape == num_atoms



def test_XCS_to_Structure():
    pass


def test_gamma():
    structure = Protein("confA.cif")
    X, C, S = structure.to_XCS()
    y = gemmi.read_structure("confA.cif")
    adp = ADP3D()
