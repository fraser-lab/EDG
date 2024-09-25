import pytest
import adp3d
from adp3d import ADP3D
from adp3d.adp.optimizer import get_elements_from_XCS, minimal_XCS_to_Structure
from chroma import Protein
import gemmi
from pathlib import Path


@pytest.fixture
def cif_file():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "4yuo.cif")
    )
    return file


@pytest.fixture
def density_file():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "4yuo.ccp4")
    )
    return file


# def test_ADP3D_init(cif_file, density_file):
#     density = gemmi.read_ccp4_map(density_file)
#     sequence = Protein(cif_file).to_XCS()[2]

#     with pytest.raises(ValueError):
#         adp = ADP3D()

#     adp = ADP3D(y=density, seq=sequence, structure=cif_file)
#     assert adp is not None


def test_get_elements_from_XCS(cif_file):
    structure = Protein(cif_file)
    X, _, S = structure.to_XCS(all_atom=True)
    elements = get_elements_from_XCS(X, S)
    assert len(elements) > 0
    assert len(elements) > S.size()[0]
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


# def test_minimal_XCS_to_Structure(cif_file):
#     structure = Protein(cif_file)
#     X, _, S = structure.to_XCS()
#     minimal_structure = minimal_XCS_to_Structure(X, S)
#     assert minimal_structure is not None
#     for key in ["coor", "b", "q", "e", "active"]:
#         assert key in minimal_structure.keys()
#     num_atoms = X.size()[1] * X.size()[2]
#     assert minimal_structure["coor"].shape == (num_atoms, 3)
#     assert int(minimal_structure["b"].shape) == num_atoms
#     assert int(minimal_structure["q"].shape) == num_atoms
#     assert int(minimal_structure["active"].shape) == num_atoms


# def test_XCS_to_Structure():
#     pass


# def test_gamma(cif_file):
#     structure = Protein(cif_file)
#     X, C, S = structure.to_XCS()
#     y = gemmi.read_structure(cif_file)
#     adp = ADP3D()
