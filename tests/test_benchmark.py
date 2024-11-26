import pytest
import adp3d
import gemmi
import numpy as np
import torch

from adp3d import ADP3D
from chroma import Chroma, Protein
from pathlib import Path
from adp3d.data.sf import ELEMENTS
from adp3d.utils.utility import try_gpu
from adp3d.adp.density import downsample_fft, to_density
from einops import rearrange, repeat

@pytest.fixture(scope="class")
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


class TestOptimizationsWithSetup:

    @pytest.fixture(autouse=True)
    def setup(self, sim_data_7pzt):
        device = try_gpu()
        protein = Protein(sim_data_7pzt[0])
        X, C, S = protein.to_XCS(all_atom=True, device=device)  # backbone coordinates
        flat_X = rearrange(X, "b r a c -> b (r a) c").squeeze()
        mask = flat_X != 0
        values = torch.where(mask, flat_X, torch.tensor(float("nan")))
        center_shift = torch.nanmean(values, dim=0)
        X -= center_shift  # centering
        self.flat_X = rearrange(X, "b r a c -> b (r a) c").squeeze()
        self.C_expand = repeat(C, "b r -> b (r a)", a=14).squeeze()
        self.adp = adp3d.ADP3D(
            y=sim_data_7pzt[1],
            seq=S,
            structure=sim_data_7pzt[0],
            all_atom=True,
            em=True,
            device=device,
        )
        self.resolution = 2.0
        self.adp.density_calculator.set_filter_and_mask(self.resolution)
        self.adp.f_y = self.adp.density_calculator.apply_filter_and_mask(self.adp.f_y, True)
        self.adp.y = torch.abs(to_density(self.adp.f_y))
        self.elements = rearrange(self.adp._extract_elements(all_atom=True), "r a -> (r a)")

    @pytest.mark.benchmark(group="ll_density")
    def test_ll_density_real(self, benchmark):
        flat_X = self.flat_X.clone().detach().requires_grad_(True)
        result = benchmark.pedantic(
            self.adp.density_calculator.forward, args=(self.flat_X, self.elements, self.C_expand, self.resolution, True), iterations=10, rounds=3
        )
        assert result is not None

    @pytest.mark.benchmark(group="ll_density")
    def test_ll_density_fourier(self, benchmark):
        flat_X = self.flat_X.clone().detach().requires_grad_(True)
        result = benchmark.pedantic(
            self.adp.density_calculator.forward, args=(flat_X, self.elements, self.C_expand, self.resolution, False), iterations=10, rounds=3
        )
        assert result is not None

    @pytest.mark.benchmark(group="grad_ll_density")
    def test_grad_ll_density_real(self, benchmark):
        def grad_ll_density_real():
            flat_X = self.flat_X.clone().detach().requires_grad_(True)
            density = self.adp.density_calculator.forward(flat_X, self.elements, self.C_expand, self.resolution, True)
            loss = torch.sum((self.adp.y - density) ** 2)
            return torch.autograd.grad(loss, flat_X)[0]
        
        result = benchmark.pedantic(
            grad_ll_density_real, iterations=3, rounds=3 # more expensive, so fewer iterations
        )

        assert result is not None

    @pytest.mark.benchmark(group="grad_ll_density")
    def test_grad_ll_density_fourier(self, benchmark):
        def grad_ll_density_fourier():
            flat_X = self.flat_X.clone().detach().requires_grad_(True)
            f_density = self.adp.density_calculator.forward(flat_X, self.elements, self.C_expand, self.resolution, False)
            loss = torch.sum(torch.abs((torch.flatten(self.adp.f_y) - f_density)))
            return torch.autograd.grad(loss, flat_X)[0]

        result = benchmark.pedantic(
            grad_ll_density_fourier, iterations=3, rounds=3 # more expensive, so fewer iterations
        )

        assert result is not None