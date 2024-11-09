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
from einops import rearrange

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
        X, _, S = protein.to_XCS(all_atom=True, device=device)  # backbone coordinates
        flat_X = rearrange(X, "b r a c -> b (r a) c").squeeze()
        mask = flat_X != 0
        values = torch.where(mask, flat_X, torch.tensor(float("nan")))
        center_shift = torch.nanmean(values, dim=0)
        X -= center_shift  # centering
        self.X = X
        self.adp = adp3d.ADP3D(
            y=sim_data_7pzt[1],
            seq=S,
            structure=sim_data_7pzt[0],
            all_atom=True,
            em=True,
            device=device,
        )

    @pytest.mark.benchmark(group="ll_density")
    def test_ll_density_real(self, benchmark):
        result = benchmark.pedantic(
            self.adp.ll_density_real, args=(self.X,), iterations=10, rounds=3
        )
        assert result is not None

    @pytest.mark.benchmark(group="ll_density")
    def test_ll_density_fourier(self, benchmark):
        result = benchmark.pedantic(
            self.adp.ll_density, args=(self.X,), iterations=10, rounds=3
        )
        assert result is not None

    @pytest.mark.benchmark(group="grad_ll_density")
    def test_grad_ll_density_real(self, benchmark):
        def grad_ll_density_real():
            X = self.X.clone().detach().requires_grad_(True)
            result = self.adp.ll_density_real(X)
            return torch.autograd.grad(result, X)[0]
        
        result = benchmark.pedantic(
            grad_ll_density_real, iterations=5, rounds=1 # more expensive, so fewer iterations
        )

        assert result is not None

    @pytest.mark.benchmark(group="grad_ll_density")
    def test_grad_ll_density_fourier(self, benchmark):
        def grad_ll_density_fourier():
            X = self.X.clone().detach().requires_grad_(True)
            result = self.adp.ll_density(X)
            return torch.autograd.grad(result, X)[0]

        result = benchmark.pedantic(
            grad_ll_density_fourier, iterations=5, rounds=1 # more expensive, so fewer iterations
        )

        assert result is not None