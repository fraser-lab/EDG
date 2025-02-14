import pytest
import adp3d
import gemmi
import numpy as np
import torch

from adp3d import DensityGuidedDiffusion
from pathlib import Path
from adp3d.utils.utility import try_gpu
from einops import rearrange


@pytest.fixture
def device():
    return try_gpu()


@pytest.fixture
def cif_4yuo():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "4yuo.cif")
    )
    return file


@pytest.fixture
def cif_7pzt():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "7pzt.cif")
    )
    return file


@pytest.fixture
def density_4yuo():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "4yuo.ccp4")
    )
    return file


@pytest.fixture
def sf_cif_7pzt():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "7pzt-sf.cif")
    )
    return file


@pytest.fixture
def sim_data_1az5():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "1az5_sim.cif")
    )
    map = str(
        Path(
            Path(adp3d.__file__).parent.parent,
            "tests",
            "resources",
            "1az5_sim_map.ccp4",
        )
    )
    return file, map


@pytest.fixture
def sim_data_7pzt():
    file = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "7pzt_sim.cif")
    )
    map = str(
        Path(
            Path(adp3d.__file__).parent.parent,
            "tests",
            "resources",
            "7pzt_sim_map.ccp4",
        )
    )
    return file, map


@pytest.fixture
def adp_init(density_4yuo, cif_4yuo, device):
    structure = Protein(cif_4yuo)

    _, _, S = structure.to_XCS()

    adp = ADP3D(y=density_4yuo, seq=S, structure=cif_4yuo, device=device)
    return adp


@pytest.fixture
def peptides():
    complete_peptide = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "GGG.cif")
    )
    incomplete_peptide = str(
        Path(Path(adp3d.__file__).parent.parent, "tests", "resources", "G_G.cif")
    )
    altconf_peptide = str(
        Path(
            Path(adp3d.__file__).parent.parent, "tests", "resources", "GGG_altconf.pdb"
        )
    )
    return complete_peptide, incomplete_peptide, altconf_peptide


def test_ADP3D_init(cif_4yuo, density_4yuo):
    sequence = Protein(cif_4yuo).to_XCS()[2]

    with pytest.raises(ValueError):
        adp = ADP3D(None, None, None)

    adp = ADP3D(y=density_4yuo, seq=sequence, structure=cif_4yuo)
    assert adp is not None


# def test_extract_elements(adp_init): # TODO: UPDATE THIS TEST FOR BOLTZ
#     adp = adp_init
#     elements = adp._extract_elements()
#     assert elements is not None
#     assert np.all([e in ELEMENTS.values() for e in torch.flatten(elements)])
#     assert elements.size() == torch.Size([adp.seq.size(1), 4])

#     elements = adp._extract_elements(all_atom=True)
#     assert elements is not None
#     assert np.all([e in ELEMENTS.values() for e in torch.flatten(elements)])
#     assert elements.size() == torch.Size([adp.seq.size(1), 14])


def test_gamma(sim_data_1az5, device):
    protein = Protein(sim_data_1az5[0])
    X, _, S = protein.to_XCS(device=device)  # backbone coordinates
    flat_X = rearrange(X, "b r a c -> b (r a) c").squeeze()
    mask = flat_X != 0
    values = torch.where(mask, flat_X, torch.tensor(float("nan")))
    center_shift = torch.nanmean(values, dim=0)
    X -= center_shift  # centering
    adp = adp3d.ADP3D(
        y=sim_data_1az5[1], seq=S, structure=sim_data_1az5[0], device=device
    )

    density = gemmi.read_ccp4_map(sim_data_1az5[1])
    size = density.grid.shape

    # test backbone
    volume = adp._gamma(X, resolution=2.0, real=True, all_atom=False)

    assert volume is not None
    assert torch.any(volume > 0)

    volume_np = volume.cpu().numpy()
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = gemmi.FloatGrid(
        volume_np, cell=density.grid.unit_cell, spacegroup=density.grid.spacegroup
    )
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map("tests/output/gamma.ccp4")

    # test all atom
    X_aa, _, S = protein.to_XCS(device=device, all_atom=True)  # all atom coordinates
    flat_X_aa = rearrange(X, "b r a c -> b (r a) c").squeeze()
    mask = flat_X_aa != 0
    values = torch.where(mask, flat_X_aa, torch.tensor(float("nan")))
    center_shift = torch.nanmean(values, dim=0)
    X_aa -= center_shift  # centering
    adp = adp3d.ADP3D(
        y=sim_data_1az5[1],
        seq=S,
        structure=sim_data_1az5[0],
        all_atom=True,
        device=device,
    )
    volume = adp._gamma(X_aa, resolution=2.0, real=True, all_atom=True)

    assert volume is not None
    assert torch.any(volume > 0)

    volume_np = volume.cpu().numpy()
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = gemmi.FloatGrid(
        volume_np, cell=density.grid.unit_cell, spacegroup=density.grid.spacegroup
    )
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map("tests/output/gamma_aa.ccp4")


def test_correlation_matrix(peptides, density_4yuo, device):
    # 3 residue protein, each with coordinate at (0, 0, 0)
    X = torch.zeros(1, 3, 4, 3, device=device)
    C = torch.ones(1, 3, device=device)
    adp = ADP3D(
        y=density_4yuo, seq=torch.zeros(1, 3), structure=peptides[0], device=device
    )
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


def test_ll_incomplete_structure(peptides, density_4yuo, device):

    complete_peptide, incomplete_peptide, altconf_peptide = peptides

    # test simple case: both structures are the same, ll should be 0
    complete_peptide_XCS = Protein(complete_peptide).to_XCS(device=device)
    adp = ADP3D(
        y=density_4yuo,
        seq=complete_peptide_XCS[2],
        structure=complete_peptide,
        device=device,
    )  # NOTE: density is arbitrary here (just needs to be not None), but if type-checked later in development this might throw an error
    z = adp.multiply_inverse_corr(
        complete_peptide_XCS[0] - adp.center_shift,  # centering
        complete_peptide_XCS[1],
    ).to(device)
    ll = adp.ll_incomplete_structure(z)
    assert ll is not None
    assert torch.isclose(ll, torch.zeros(1, device=device), atol=1e-5)

    # test slightly more complex case: one structure is a subset of the other
    adp = ADP3D(
        y=density_4yuo,
        seq=complete_peptide_XCS[2],
        structure=incomplete_peptide,
        device=device,
    )
    # z stays the same, ll w.r.t. complete structure
    ll = adp.ll_incomplete_structure(z)
    assert ll is not None
    assert ll < 0

    # test alternative conformation of peptide
    adp = ADP3D(
        y=density_4yuo,
        seq=complete_peptide_XCS[2],
        structure=altconf_peptide,
        device=device,
    )
    # z stays the same, ll w.r.t. complete structure
    ll = adp.ll_incomplete_structure(z)
    assert ll is not None
    assert ll < 0


def test_grad_ll_incomplete_structure(peptides, density_4yuo, device):
    complete_peptide, incomplete_peptide, _ = peptides

    # test simple case: both structures are the same, ll should be 0, grad should be 0
    complete_peptide_XCS = Protein(complete_peptide).to_XCS(device=device)
    adp = ADP3D(
        y=density_4yuo,
        seq=complete_peptide_XCS[2],
        structure=complete_peptide,
        device=device,
    )  # NOTE: density is arbitrary here (just needs to be not None), but if type-checked later in development this might throw an error
    z = adp.multiply_inverse_corr(
        complete_peptide_XCS[0] - adp.center_shift, complete_peptide_XCS[1]
    ).to(device)
    grad_ll, _ = adp.grad_ll_incomplete_structure(z)
    assert grad_ll is not None
    assert torch.allclose(grad_ll, torch.zeros_like(grad_ll), atol=1e-5)

    # test slightly more complex case: one structure is a subset of the other
    adp = ADP3D(
        y=density_4yuo,
        seq=complete_peptide_XCS[2],
        structure=incomplete_peptide,
        device=device,
    )  # NOTE: density is arbitrary here (just needs to be not None), but if type-checked later in development this might throw an error
    # z stays the same, ll w.r.t. complete structure
    grad_ll, _ = adp.grad_ll_incomplete_structure(z)
    assert grad_ll is not None
    assert not torch.allclose(grad_ll, torch.zeros_like(grad_ll), atol=1e-5)


def test_ll_density_and_grad(sim_data_7pzt, device): 
    protein = Protein(sim_data_7pzt[0])
    X, _, S = protein.to_XCS(all_atom=True, device=device)  # backbone coordinates
    flat_X = rearrange(X, "b r a c -> b (r a) c").squeeze()
    mask = flat_X != 0
    values = torch.where(mask, flat_X, torch.tensor(float("nan")))
    center_shift = torch.nanmean(values, dim=0)
    X -= center_shift  # centering
    adp = adp3d.ADP3D(
        y=sim_data_7pzt[1],
        seq=S,
        structure=sim_data_7pzt[0],
        all_atom=True,
        em=True,
        device=device,
    )

    grad_ll, loss = adp.grad_ll_density(X, all_atom=True, resolution=4.0, real=False)
    assert grad_ll is not None
    assert torch.allclose(grad_ll, torch.zeros_like(grad_ll, device=device), atol=0.3)

    grad_ll, loss = adp.grad_ll_density(X, all_atom=True, resolution=4.0, real=True)
    assert grad_ll is not None
    assert torch.allclose(grad_ll, torch.zeros_like(grad_ll, device=device), atol=0.3)


def test_optimizer(sim_data_1az5, device):
    prot = Protein(sim_data_1az5[0]).to_XCS(device=device)
    adp = ADP3D(
        y=sim_data_1az5[1], seq=prot[2], structure=sim_data_1az5[0], device=device
    )

    # default args, 10 epochs
    output_model = adp.model_refinement_optimizer(
        output_dir="./tests/output", epochs=10
    )
    assert output_model is not None
    output_model.to_PDB("./tests/output/final.pdb")
    assert Path("./tests/output/final.pdb").exists()
