import pytest
import adp3d
import gemmi
import numpy as np
import torch

from adp3d import ADP3D
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
def cif_file_2():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "7pzt.cif")
    )
    return file


@pytest.fixture
def density_file():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "4yuo.ccp4")
    )
    return file


@pytest.fixture
def sf_file():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "7pzt-sf.cif")
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


def test_ADP3D_init(cif_file, density_file):
    sequence = Protein(cif_file).to_XCS()[2]

    with pytest.raises(ValueError):
        adp = ADP3D(None, None, None)

    adp = ADP3D(y=density_file, seq=sequence, structure=cif_file)
    assert adp is not None


# def test_gamma(adp_init):
#     adp = adp_init
#     X, S = adp.x_bar, adp.seq
#     volume = adp.gamma(X, S)

#     assert volume is not None
#     assert np.any(volume > 0)


def test_correlation_matrix(peptides, density_file, device):
    # 3 residue protein, each with coordinate at (0, 0, 0)
    X = torch.zeros(1, 3, 4, 3, device=device)
    C = torch.ones(1, 3, device=device)
    adp = ADP3D(y=density_file, seq=torch.zeros(20, 3), structure=peptides[0])
    chroma_whitened_test = adp.multiply_inverse_corr(X, C)
    chroma_unwhitened_test = adp.multiply_corr(chroma_whitened_test, C)
    assert torch.allclose(X, chroma_unwhitened_test, atol=1e-5)

    x_rearranged = rearrange(X, "b r a c -> b (r a) c").squeeze()
    my_whitened_test = adp.R @ x_rearranged
    assert torch.allclose(
        rearrange(chroma_whitened_test, "b r a c -> b (r a) c").squeeze(),
        my_whitened_test,
        atol=1e-5,
    )


def test_ll_incomplete_structure(peptides, density_file, device):

    complete_peptide, incomplete_peptide = peptides

    # test simple case: both structures are the same, ll should be 0
    complete_peptide_XCS = Protein(complete_peptide).to_XCS(device=device)
    adp = ADP3D(
        y=density_file, seq=complete_peptide_XCS[2], structure=complete_peptide
    )  # NOTE: density is arbitrary here (just needs to be not None), but if type-checked later in development this might throw an error
    z = adp.multiply_inverse_corr(complete_peptide_XCS[0], complete_peptide_XCS[1]).to(
        device
    )
    ll = adp.ll_incomplete_structure(z)
    assert ll is not None
    assert torch.isclose(ll, torch.zeros(1, device=device), atol=1e-5)

    # test slightly more complex case: one structure is a subset of the other
    adp = ADP3D(
        y=density_file, seq=complete_peptide_XCS[2], structure=incomplete_peptide
    )  # NOTE: density is arbitrary here (just needs to be not None), but if type-checked later in development this might throw an error
    # z stays the same, ll w.r.t. complete structure
    ll = adp.ll_incomplete_structure(z)
    assert ll is not None
    assert ll < 0


def test_grad_ll_incomplete_structure(peptides, density_file, device):
    complete_peptide, incomplete_peptide = peptides

    # test simple case: both structures are the same, ll should be 0, grad should be 0
    complete_peptide_XCS = Protein(complete_peptide).to_XCS(device=device)
    adp = ADP3D(
        y=density_file, seq=complete_peptide_XCS[2], structure=complete_peptide
    )  # NOTE: density is arbitrary here (just needs to be not None), but if type-checked later in development this might throw an error
    z = adp.multiply_inverse_corr(complete_peptide_XCS[0], complete_peptide_XCS[1]).to(
        device
    )
    grad_ll = adp.grad_ll_incomplete_structure(z)
    assert grad_ll is not None
    assert torch.allclose(grad_ll, torch.zeros_like(grad_ll), atol=1e-5)

    # test slightly more complex case: one structure is a subset of the other
    adp = ADP3D(
        y=density_file, seq=complete_peptide_XCS[2], structure=incomplete_peptide
    )  # NOTE: density is arbitrary here (just needs to be not None), but if type-checked later in development this might throw an error
    # z stays the same, ll w.r.t. complete structure
    grad_ll = adp.grad_ll_incomplete_structure(z)
    assert grad_ll is not None
    assert not torch.allclose(grad_ll, torch.zeros_like(grad_ll), atol=1e-5)


def test_ll_density_and_grad(density_file, sf_file, cif_file_2, device):
    # using SFcalculator
    prot = Protein(cif_file_2).to_XCS(device=device)
    adp = ADP3D(y=sf_file, seq=prot[2], structure=cif_file_2)
    z = adp.multiply_inverse_corr(prot[0], prot[1]).to(device)
    grad_ll = adp.grad_ll_density(z)  # tests ll_density in this function
    assert grad_ll is not None
    # TODO: compare with gamma calculation


def test_optimizer(density_file, sf_file, cif_file_2, device):
    prot = Protein(cif_file_2).to_XCS(device=device)
    adp = ADP3D(y=sf_file, seq=prot[2], structure=cif_file_2)
    
    # default args, 10 epochs
    output_model = adp.model_refinement_optimizer(output_dir="./tests/output", epochs = 4000)
    assert output_model is not None
    output_model.to_CIF("./tests/output/final.cif")