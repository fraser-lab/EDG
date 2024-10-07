import pytest
import adp3d
import gemmi
import numpy as np
import torch

from adp3d import ADP3D
from adp3d.adp.optimizer import get_elements_from_XCS, minimal_XCS_to_Structure
from chroma import Chroma, Protein
from pathlib import Path
from qfit.structure.elements import ELEMENTS
from einops import rearrange


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@pytest.fixture
def adp_init(density_file, cif_file):
    density = gemmi.read_ccp4_map(density_file)
    structure = Protein(cif_file)

    X, C, S = structure.to_XCS()

    y = gemmi.read_structure(cif_file)
    adp = ADP3D(y=density, seq=S, structure=cif_file)
    return adp


@pytest.fixture
def peptides():
    complete_peptide = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "GGG.cif")
    )
    incomplete_peptide = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "G_G.pdb")
    )
    return complete_peptide, incomplete_peptide


# def test_ADP3D_init(cif_file, density_file):
#     density = gemmi.read_ccp4_map(density_file)
#     sequence = Protein(cif_file).to_XCS()[2]

#     with pytest.raises(ValueError):
#         adp = ADP3D(None, None, None)

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
#     assert np.any(minimal_structure["coor"] != (0, 0, 0))
#     assert minimal_structure["b"].shape == (num_atoms,)
#     assert minimal_structure["q"].shape == (num_atoms,)
#     assert minimal_structure["active"].shape == (num_atoms,)
#     assert minimal_structure["natoms"] == num_atoms


# def test_XCS_to_Structure():
#     pass


# def test_chroma_cif_with_qfit(cif_file):
#     pass


# def test_gamma(adp_init):
#     adp = adp_init
#     X, S = adp.x_bar, adp.seq
#     volume = adp.gamma(X, S)

#     assert volume is not None
#     assert np.any(volume > 0)


# def test_correlation_matrix(peptides, device):
#     # 3 residue protein, each with coordinate at (0, 0, 0)
#     X = torch.zeros(1, 3, 4, 3, device=device)
#     C = torch.ones(1, 3, device=device)
#     adp = ADP3D(y=1, seq=torch.zeros(20, 3), structure=peptides[0])
#     chroma_whitened_test = adp.multiply_inverse_corr(X, C)
#     chroma_unwhitened_test = adp.multiply_corr(chroma_whitened_test, C)
#     assert torch.allclose(X, chroma_unwhitened_test, atol=1e-5)

#     x_rearranged = rearrange(X, "b r a c -> b (r a) c").squeeze()
#     my_whitened_test = adp.R @ x_rearranged
#     assert torch.allclose(
#         rearrange(chroma_whitened_test, "b r a c -> b (r a) c").squeeze(),
#         my_whitened_test,
#         atol=1e-5,
#     )


def test_ll_incomplete_structure(peptides, device):

    complete_peptide, incomplete_peptide = peptides

    # test simple case: both structures are the same, ll should be 0
    complete_peptide_XCS = Protein(complete_peptide).to_XCS(device=device)
    adp = ADP3D(
        y=1, seq=complete_peptide_XCS[2], structure=complete_peptide
    )  # NOTE: density is arbitrary here (just needs to be not None), but if type-checked later in development this might throw an error
    z = adp.multiply_inverse_corr(complete_peptide_XCS[0], complete_peptide_XCS[1]).to(
        device
    )
    ll = adp.ll_incomplete_structure(z)
    assert ll is not None
    assert torch.isclose(ll, torch.zeros(1, device=device), atol=1e-5)

    # test slightly more complex case: one structure is a subset of the other
    adp = ADP3D(
        y=1, seq=complete_peptide_XCS[2], structure=incomplete_peptide
    )  # NOTE: density is arbitrary here (just needs to be not None), but if type-checked later in development this might throw an error
    # z stays the same, ll w.r.t. complete structure
    ll = adp.ll_incomplete_structure(z)
    assert ll is not None
    assert ll < 0