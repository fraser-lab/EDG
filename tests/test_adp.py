import pytest
import adp3d
import gemmi
import numpy as np

from adp3d import ADP3D
from adp3d.adp.optimizer import get_elements_from_XCS, minimal_XCS_to_Structure
from chroma import Chroma, Protein
from pathlib import Path
from qfit.structure.elements import ELEMENTS


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


# def test_get_elements_from_XCS(cif_file):
#     structure = Protein(cif_file)
#     X, _, S = structure.to_XCS(all_atom=True)
#     elements = get_elements_from_XCS(X, S)
#     assert len(elements) > 0
#     assert len(elements) > S.size()[0]
#     assert set(elements).issubset(
#         set(
#             [
#                 "N",
#                 "C",
#                 "O",
#                 "S",
#                 "H",
#             ]
#         )
#     )


# def test_minimal_XCS_to_Structure(cif_file):
#     structure = Protein(cif_file)
#     X, _, S = structure.to_XCS()
#     minimal_structure = minimal_XCS_to_Structure(X, S)
#     assert minimal_structure is not None
#     for key in ["coor", "b", "q", "e", "active"]:
#         assert key in minimal_structure.keys()
#     num_atoms = X.size()[1] * X.size()[2]
#     assert minimal_structure["coor"].shape == (num_atoms, 3)
#     assert minimal_structure["b"].shape == (num_atoms, )
#     assert minimal_structure["q"].shape == (num_atoms, )
#     assert minimal_structure["active"].shape == (num_atoms, )


# def test_XCS_to_Structure():
#     pass

def test_chroma_cif_with_qfit(cif_file):
    pass

def test_gamma(cif_file, density_file):
    density = gemmi.read_ccp4_map(density_file)
    structure = Protein(cif_file)

    X, C, S = structure.to_XCS()

    y = gemmi.read_structure(cif_file)
    adp = ADP3D(y=density, seq=S, structure=cif_file)
    volume = adp.gamma(X, S)
    assert volume is not None
    assert np.any(volume > 0)
