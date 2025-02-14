import pytest
import adp3d
import gemmi
import numpy as np
import torch

from adp3d import DensityGuidedDiffusion
from pathlib import Path
from adp3d.utils.utility import try_gpu
from adp3d.adp.density import downsample_fft, to_density
from adp3d.data.io import structure_to_density_input
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
        coords, elements, resolution = structure_to_density_input(sim_data_7pzt[0])
        self.coords = coords.to(device)
        self.elements = elements.to(device)
        self.adp = adp3d.ADP3D(
            y=sim_data_7pzt[1],
            structure=sim_data_7pzt[0],
            all_atom=True,
            em=True,
            device=device,
        )
        self.resolution = resolution
        self.adp.density_calculator.set_filter_and_mask(self.resolution)
        get_to_y = self.adp.density_calculator.apply_filter_and_mask(self.adp.f_y, True)
        self.adp.f_y = self.adp.density_calculator.apply_filter_and_mask(self.adp.f_y, False)
        self.adp.y = torch.abs(to_density(get_to_y))
        self.elements = rearrange(self.adp._extract_elements(all_atom=True), "r a -> (r a)")

    @pytest.mark.benchmark(group="ll_density")
    def test_ll_density_real(self, benchmark):
        coords = self.coords.clone().detach().requires_grad_(True)
        result = benchmark.pedantic(
            self.adp.density_calculator.forward, args=(coords, self.elements, self.resolution, True), iterations=10, rounds=3
        )
        assert result is not None

    @pytest.mark.benchmark(group="ll_density")
    def test_ll_density_fourier(self, benchmark):
        coords = self.coords.clone().detach().requires_grad_(True)
        result = benchmark.pedantic(
            self.adp.density_calculator.forward, args=(coords, self.elements, self.resolution, False), iterations=10, rounds=3
        )
        assert result is not None

    @pytest.mark.benchmark(group="grad_ll_density")
    def test_grad_ll_density_real(self, benchmark):
        def grad_ll_density_real():
            coords = self.coords.clone().detach().requires_grad_(True)
            density = self.adp.density_calculator.forward(coords, self.elements, self.resolution, True)
            loss = torch.sum((self.adp.y - density) ** 2)
            return torch.autograd.grad(loss, coords)[0]
        
        result = benchmark.pedantic(
            grad_ll_density_real, iterations=3, rounds=3 # more expensive, so fewer iterations
        )

        assert result is not None

    @pytest.mark.benchmark(group="grad_ll_density")
    def test_grad_ll_density_fourier(self, benchmark):
        def grad_ll_density_fourier():
            coords = self.coords.clone().detach().requires_grad_(True)
            f_density = self.adp.density_calculator.forward(coords, self.elements, self.resolution, False)
            loss = torch.sum(torch.abs((torch.flatten(self.adp.f_y) - f_density)))
            return torch.autograd.grad(loss, coords)[0]

        result = benchmark.pedantic(
            grad_ll_density_fourier, iterations=3, rounds=3 # more expensive, so fewer iterations
        )

        assert result is not None