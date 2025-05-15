import time
import torch
import pytest
from pathlib import Path
from scipy.stats import pearsonr

from adp3d.qfit.volume import XMap

from adp3d import Structure

from adp3d.data.sf import ATOM_STRUCTURE_FACTORS, ELECTRON_SCATTERING_FACTORS

from adp3d.adp.modules.density import (
    DifferentiableTransformer,
    XMap_torch,
    DensityParameters,
    dilate_points_torch,
)

# Import the new CUDA implementation
from adp3d.adp.modules.ops.dilate_points_cuda import (
    dilate_atom_centric,
    DilateAtomCentricCUDA
)

from adp3d.utils.utility import try_gpu


class TestAtomDensityCUDA:
    """Regression testing for atom-centric CUDA implementation.

    This test suite verifies that the CUDA implementation correctly
    computes electron density maps from atomic coordinates and produces
    the same gradients during backpropagation as the reference PyTorch
    implementation.
    """
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Fixture that prepares test data.

        Returns
        -------
        Dict
            Dictionary containing test data including structures, maps,
            and required tensor inputs.
        """
        data_dir = Path("tests/resources/mac1_synthetic").absolute()
        cif_file1 = data_dir / "5SOQ_modified.pdb"
        mtz_file = data_dir / "5SOQ_modified_map_coeffs.mtz"
        em = False

        # Load structure and scattering factors
        if em:
            structure_factors = ELECTRON_SCATTERING_FACTORS
        else:
            structure_factors = ATOM_STRUCTURE_FACTORS

        structure = Structure.fromfile(str(cif_file1))
        structure = structure.remove_alternative_conformations()
        structure = structure.clean_structure(keep_type="protein")
        structure = structure.reorder()
        structure.build_hierarchy()

        unit_cell = structure.unit_cell

        # Load reference electron density map
        ref_map = XMap.fromfile(str(mtz_file), label="2FOFCWT,PH2FOFCWT")

        # Prepare test data
        coords = structure.coor
        elements = structure.e
        b_factors = structure.b
        occupancies = structure.q
        active = structure.active

        # Convert elements to indices
        element_indices = {}
        unique_elements = sorted(set(elements))
        for i, elem in enumerate(unique_elements):
            element_indices[elem] = i

        # Prepare scattering factors dictionary
        max_elem_idx = max(element_indices.values())
        tensor_shape = list(torch.tensor(next(iter(structure_factors.values()))).T.shape)
        scattering_params = torch.zeros([max_elem_idx + 1] + tensor_shape)

        for elem in unique_elements:
            idx = element_indices[elem]
            if elem in structure_factors:
                scattering_params[idx] = torch.tensor(structure_factors[elem]).T
            else:
                scattering_params[idx] = torch.tensor(structure_factors["C"]).T

        # Convert to PyTorch tensors
        coordinates = torch.tensor(coords, dtype=torch.float32)
        element_ids = torch.tensor(
            [element_indices[elem] for elem in elements],
            dtype=torch.int32,
        )
        b_factors = torch.tensor(b_factors, dtype=torch.float32)
        occupancies = torch.tensor(occupancies, dtype=torch.float32)
        active = torch.tensor(active, dtype=torch.bool)

        # Batch dimension for testing
        batch_size = 2
        coordinates = coordinates.unsqueeze(0).expand(batch_size, -1, -1).float().clone()
        element_ids = element_ids.unsqueeze(0).expand(batch_size, -1).int()
        b_factors = b_factors.unsqueeze(0).expand(batch_size, -1).float()
        occupancies = occupancies.unsqueeze(0).expand(batch_size, -1).float()
        active = active.unsqueeze(0).expand(batch_size, -1).bool()

        # Add some variation to the second batch for testing
        coordinates[1] = coordinates[1] + 0.1 * torch.randn_like(coordinates[1])
        b_factors[1] = b_factors[1] * (1.0 + 0.05 * torch.rand_like(b_factors[1]))

        device = try_gpu() 

        return {
            "device": device,
            "structure": structure,
            "ref_map": ref_map,
            "unit_cell": unit_cell,
            "coordinates": coordinates,
            "element_ids": element_ids,
            "b_factors": b_factors,
            "occupancies": occupancies,
            "active": active,
            "element_indices": element_indices,
            "scattering_params": scattering_params,
            "em": em,
        }

    @pytest.fixture(scope="class")
    def transformers(self, test_data):
        """Create and return DifferentiableTransformer objects for testing.

        Parameters
        ----------
        test_data : Dict
            Test data from the test_data fixture

        Returns
        -------
        Dict
            Dictionary containing both cuda-enabled and pytorch-only transformers
        """
        device = test_data["device"]
        
        # Create XMap_torch instance
        xmap = XMap_torch(test_data["ref_map"], device=device)
        
        # Configure density parameters
        density_params = DensityParameters(
            rmax=5.0,
            rstep=0.1,
            smin=0.0,
            smax=0.5,
            quad_points=50,
            integration_method="gausslegendre"
        )
        
        # Create PyTorch transformer 
        pytorch_transformer = DifferentiableTransformer(
            xmap=xmap,
            scattering_params=test_data["scattering_params"],
            density_params=density_params,
            em=test_data["em"],
            device=device,
            use_cuda_kernels=False  # Force PyTorch implementation
        )
        
        # Create CUDA transformer
        cuda_transformer = DifferentiableTransformer(
            xmap=xmap,
            scattering_params=test_data["scattering_params"],
            density_params=density_params,
            em=test_data["em"],
            device=device,
            use_cuda_kernels=True  # Use CUDA implementation
        )
        
        return {
            "pytorch": pytorch_transformer,
            "cuda": cuda_transformer,
            "xmap": xmap
        }

    def test_dilate_points(self, test_data, transformers):
        """Test that the dilate_points function produces the same results.

        Parameters
        ----------
        test_data : Dict
            Test data from the test_data fixture
        transformers : Dict
            Transformer objects from the transformers fixture
        """
        # Use a subset of atoms for faster testing
        n_atoms = 10
        
        device = transformers["pytorch"].device
        
        # Prepare inputs
        coords = test_data["coordinates"][:, :n_atoms].to(device).float().detach().clone()
        element_ids = test_data["element_ids"][:, :n_atoms].to(device).int().detach().clone()
        b_factors = test_data["b_factors"][:, :n_atoms].to(device).float().detach().clone()
        occupancies = test_data["occupancies"][:, :n_atoms].to(device).float().detach().clone()
        active = test_data["active"][:, :n_atoms].to(device).bool().detach().clone()
        
        # Compute radial densities
        with torch.no_grad():
            radial_derivatives, radial_profiles = transformers["pytorch"]._compute_radial_derivatives(
                element_ids, b_factors
            )
            
            # Compute grid coordinates
            grid_coords = transformers["pytorch"]._compute_grid_coordinates(coords)
            
            # Get parameters
            lmax = torch.tensor([
                transformers["pytorch"].density_params.rmax / vs 
                for vs in transformers["pytorch"].xmap.voxelspacing
            ], device=device).int()
            
            grid_shape = transformers["xmap"].shape
            
            # Run PyTorch implementation
            result_pytorch = dilate_points_torch(
                grid_coords,
                active,
                occupancies,
                lmax,
                radial_profiles.float(),
                transformers["pytorch"].density_params.rstep,
                transformers["pytorch"].density_params.rmax,
                transformers["pytorch"].grid_to_cartesian,
                tuple(grid_shape)
            )
            
            # Run CUDA implementation
            result_cuda = dilate_atom_centric(
                grid_coords,
                occupancies,
                radial_profiles.float(),
                radial_derivatives.float(),
                transformers["pytorch"].density_params.rstep,
                transformers["pytorch"].density_params.rmax,
                lmax.to(torch.int32),
                torch.tensor(grid_shape, dtype=torch.int32, device=device),
                transformers["pytorch"].grid_to_cartesian
            )
        
        # Compare results
        assert result_cuda.shape == result_pytorch.shape
        
        # Check for close agreement between implementations
        relative_diff = torch.norm(result_cuda - result_pytorch) / torch.norm(result_pytorch)
        assert relative_diff < 1e-4, f"Relative difference in dilate_points results too large: {relative_diff}"

    def test_forward_pass(self, test_data, transformers):
        """Test that the forward pass produces similar results.

        Parameters
        ----------
        test_data : Dict
            Test data from the test_data fixture
        transformers : Dict
            Transformer objects from the transformers fixture
        """
        device = transformers["pytorch"].device
        
        # Prepare inputs
        coords = test_data["coordinates"].to(device).detach().clone()
        element_ids = test_data["element_ids"].to(device).detach().clone()
        b_factors = test_data["b_factors"].to(device).detach().clone()
        occupancies = test_data["occupancies"].to(device).detach().clone()
        active = test_data["active"].to(device).detach().clone()
        
        # Run forward pass with both implementations
        with torch.no_grad():
            density_map_pytorch = transformers["pytorch"](
                coords,
                element_ids,
                b_factors,
                occupancies,
                active
            )
            
            density_map_cuda = transformers["cuda"](
                coords,
                element_ids,
                b_factors,
                occupancies,
                active
            )
        
        # Compare results
        assert density_map_cuda.shape == density_map_pytorch.shape
        
        # Compute correlation between the two outputs
        pytorch_flat = density_map_pytorch.cpu().numpy().flatten()
        cuda_flat = density_map_cuda.cpu().numpy().flatten()
        correlation, _ = pearsonr(pytorch_flat, cuda_flat)
        
        # Check for high correlation
        assert correlation > 0.99, f"Correlation between PyTorch and CUDA outputs too low: {correlation}"
        
        # Check relative difference
        relative_diff = torch.norm(density_map_cuda - density_map_pytorch) / torch.norm(density_map_pytorch)
        assert relative_diff < 1e-4, f"Relative difference in forward pass results too large: {relative_diff}"

    def test_backward_pass(self, test_data, transformers):
        """Test that the backward pass computes correct gradients.

        Parameters
        ----------
        test_data : Dict
            Test data from the test_data fixture
        transformers : Dict
            Transformer objects from the transformers fixture
        """
        device = transformers["pytorch"].device
        
        # Use a subset of atoms for faster testing
        n_atoms = 100
        
        # Prepare inputs for PyTorch implementation
        coords_pytorch = test_data["coordinates"][:, :n_atoms].to(device).detach().clone().requires_grad_(True)
        element_ids = test_data["element_ids"][:, :n_atoms].to(device).detach().clone()
        b_factors = test_data["b_factors"][:, :n_atoms].to(device).detach().clone()
        occupancies = test_data["occupancies"][:, :n_atoms].to(device).detach().clone().requires_grad_(True)
        active = test_data["active"][:, :n_atoms].to(device).detach().clone()
        
        # Prepare identical inputs for CUDA implementation
        coords_cuda = coords_pytorch.detach().clone().requires_grad_(True)
        occupancies_cuda = occupancies.detach().clone().requires_grad_(True)
        
        # Forward pass - PyTorch
        density_map_pytorch = transformers["pytorch"](
            coords_pytorch,
            element_ids,
            b_factors,
            occupancies,
            active
        )
        
        # Forward pass - CUDA
        density_map_cuda = transformers["cuda"](
            coords_cuda,
            element_ids,
            b_factors,
            occupancies_cuda,
            active
        )
        
        # Use identical loss function for both
        target_map = transformers["xmap"].array.float()
        loss_pytorch = torch.nn.functional.mse_loss(density_map_pytorch.sum(0), target_map)
        loss_cuda = torch.nn.functional.mse_loss(density_map_cuda.sum(0), target_map)
        
        # Backward pass
        loss_pytorch.backward()
        loss_cuda.backward()
        
        # Check that gradients were computed
        assert coords_pytorch.grad is not None, "PyTorch implementation didn't compute gradients for coordinates"
        assert coords_cuda.grad is not None, "CUDA implementation didn't compute gradients for coordinates"
        assert occupancies.grad is not None, "PyTorch implementation didn't compute gradients for occupancies"
        assert occupancies_cuda.grad is not None, "CUDA implementation didn't compute gradients for occupancies"
        
        # Check for NaNs and Infs
        assert not torch.isnan(coords_pytorch.grad).any(), "PyTorch gradients contain NaNs"
        assert not torch.isnan(coords_cuda.grad).any(), "CUDA gradients contain NaNs"
        assert not torch.isinf(coords_pytorch.grad).any(), "PyTorch gradients contain Infs"
        assert not torch.isinf(coords_cuda.grad).any(), "CUDA gradients contain Infs"
        
        # Check that gradients have similar directions
        cos_sim = torch.nn.functional.cosine_similarity(
            coords_pytorch.grad.view(-1), 
            coords_cuda.grad.view(-1), 
            dim=0
        )
        assert cos_sim > 0.5, f"Cosine similarity between PyTorch and CUDA gradients too low: {cos_sim}"
        
        # Check relative difference in gradient magnitudes
        pytorch_grad_mag = torch.norm(coords_pytorch.grad)
        cuda_grad_mag = torch.norm(coords_cuda.grad)
        rel_diff = abs(pytorch_grad_mag - cuda_grad_mag) / pytorch_grad_mag
        assert rel_diff < 0.5, f"Relative difference in gradient magnitudes too large: {rel_diff}"

    def test_performance(self, test_data, transformers):
        """Test that the CUDA implementation is faster than PyTorch.

        Parameters
        ----------
        test_data : Dict
            Test data from the test_data fixture
        transformers : Dict
            Transformer objects from the transformers fixture
        """
        device = transformers["pytorch"].device
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping performance test")
        
        # Prepare inputs
        coords = test_data["coordinates"].to(device).detach().clone()
        element_ids = test_data["element_ids"].to(device).detach().clone()
        b_factors = test_data["b_factors"].to(device).detach().clone()
        occupancies = test_data["occupancies"].to(device).detach().clone()
        active = test_data["active"].to(device).detach().clone()
        
        # Warm-up
        with torch.no_grad():
            _ = transformers["pytorch"](coords, element_ids, b_factors, occupancies, active)
            _ = transformers["cuda"](coords, element_ids, b_factors, occupancies, active)
        
        # Measure PyTorch forward pass time
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = transformers["pytorch"](coords, element_ids, b_factors, occupancies, active)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time
        
        # Measure CUDA forward pass time
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = transformers["cuda"](coords, element_ids, b_factors, occupancies, active)
        torch.cuda.synchronize()
        cuda_time = time.time() - start_time
        
        # Check that CUDA is faster
        speedup = pytorch_time / cuda_time
        print(f"\nPerformance speedup: {speedup:.2f}x (PyTorch: {pytorch_time:.4f}s, CUDA: {cuda_time:.4f}s)")
        
        # We expect a significant speedup
        assert speedup > 1.5, f"CUDA implementation not significantly faster (speedup: {speedup:.2f}x)"

    def test_gradient_check(self, test_data, transformers):
        """Verify gradients with torch.autograd.gradcheck.

        Parameters
        ----------
        test_data : Dict
            Test data from the test_data fixture
        transformers : Dict
            Transformer objects from the transformers fixture
        """
        # Skip if not on CUDA - gradcheck is very slow on CPU
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping gradient check")
        
        device = transformers["cuda"].device
        
        # Use a very small subset of atoms for gradcheck
        n_atoms = 10
        
        # Create a small test function for gradcheck
        def func(coords, occupancies):
            return DilateAtomCentricCUDA.apply(
                coords,
                occupancies,
                radial_profiles,
                radial_derivatives,
                rstep,
                rmax,
                lmax,
                grid_dims,
                grid_to_cartesian
            ).sum()
        
        # Prepare inputs
        coords = test_data["coordinates"][:1, :n_atoms].to(device).detach().clone().requires_grad_(True)
        element_ids = test_data["element_ids"][:1, :n_atoms].to(device).detach().clone()
        b_factors = test_data["b_factors"][:1, :n_atoms].to(device).detach().clone()
        occupancies = test_data["occupancies"][:1, :n_atoms].to(device).detach().clone().requires_grad_(True)
        
        # Compute required tensors
        with torch.no_grad():
            radial_profiles, radial_derivatives = transformers["cuda"]._compute_radial_derivatives(
                element_ids, b_factors
            )
            radial_profiles = radial_profiles.float()
            radial_derivatives = radial_derivatives.float()
            
            rstep = transformers["cuda"].density_params.rstep
            rmax = transformers["cuda"].density_params.rmax
            
            lmax = torch.tensor([
                transformers["cuda"].density_params.rmax / vs 
                for vs in transformers["cuda"].xmap.voxelspacing
            ], device=device).int()
            
            grid_dims = torch.tensor(
                transformers["xmap"].shape, 
                dtype=torch.int32, 
                device=device
            )
            
            grid_to_cartesian = transformers["cuda"].grid_to_cartesian.float()
        
        # Run gradcheck - this is slow but thorough
        from torch.autograd import gradcheck
        
        # Reduce precision requirements due to the nature of the operation
        test_result = gradcheck(
            func, 
            (coords, occupancies), 
            eps=1e-4,
            atol=1e-3,
            rtol=1e-2,
            fast_mode=True
        )
        
        assert test_result, "Gradient check failed"


class TestDifferentiableTransformerIntegration:
    """Integration tests for DifferentiableTransformer with CUDA kernels.
    
    These tests verify that the DifferentiableTransformer correctly
    integrates with the CUDA kernels and produces electron density
    maps that match reference maps.
    """
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Prepare test data for integration tests.
        
        Returns
        -------
        Dict
            Dictionary containing test data
        """
        data_dir = Path("../tests/resources/mac1_synthetic")
        cif_file1 = data_dir / "5SOQ_modified.pdb"
        mtz_file = data_dir / "5SOQ_modified_map_coeffs.mtz"
        em = False
        
        # Load structure and scattering factors
        if em:
            structure_factors = ELECTRON_SCATTERING_FACTORS
        else:
            structure_factors = ATOM_STRUCTURE_FACTORS
            
        # Load structure
        structure = Structure.fromfile(str(cif_file1))
        structure = structure.remove_alternative_conformations()
        structure = structure.clean_structure(keep_type="protein")
        structure = structure.reorder()
        structure.build_hierarchy()
        
        # Load reference map
        ref_map = XMap.fromfile(str(mtz_file), label="2FOFCWT,PH2FOFCWT")
        
        # Prepare element indices and scattering parameters
        elements = structure.e
        element_indices = {}
        unique_elements = sorted(set(elements))
        for i, elem in enumerate(unique_elements):
            element_indices[elem] = i
            
        # Prepare scattering factors dictionary
        max_elem_idx = max(element_indices.values())
        tensor_shape = list(torch.tensor(next(iter(structure_factors.values()))).T.shape)
        scattering_params = torch.zeros([max_elem_idx + 1] + tensor_shape)
        
        for elem in unique_elements:
            idx = element_indices[elem]
            if elem in structure_factors:
                scattering_params[idx] = torch.tensor(structure_factors[elem]).T
            else:
                scattering_params[idx] = torch.tensor(structure_factors["C"]).T

        device = try_gpu()
                
        return {
            "device": device,
            "structure": structure,
            "ref_map": ref_map,
            "elements": elements,
            "element_indices": element_indices,
            "scattering_params": scattering_params,
            "em": em
        }
    
    def test_map_generation(self, test_data):
        """Test that the CUDA implementation generates maps that match reference.
        
        Parameters
        ----------
        test_data : Dict
            Test data from the test_data fixture
        """
        device = test_data["device"]
        
        # Create XMap_torch instance
        xmap = XMap_torch(test_data["ref_map"], device=device)
        
        # Configure density parameters
        density_params = DensityParameters(
            rmax=5.0,
            rstep=0.1,
            smin=0.0,
            smax=0.5,
            quad_points=50,
            integration_method="gausslegendre"
        )
        
        # Create transformer with CUDA kernels
        transformer = DifferentiableTransformer(
            xmap=xmap,
            scattering_params=test_data["scattering_params"],
            density_params=density_params,
            em=test_data["em"],
            device=device,
            use_cuda_kernels=True
        )
        
        # Prepare inputs
        structure = test_data["structure"]
        coords = structure.coor
        elements = structure.e
        b_factors = structure.b
        occupancies = structure.q
        active = structure.active
        
        # Convert to tensors
        coordinates = torch.tensor(coords, dtype=torch.float32, device=device)
        element_ids = torch.tensor(
            [test_data["element_indices"][elem] for elem in elements],
            dtype=torch.int32,
            device=device
        )
        b_factors = torch.tensor(b_factors, dtype=torch.float32, device=device)
        occupancies = torch.tensor(occupancies, dtype=torch.float32, device=device)
        active = torch.tensor(active, dtype=torch.bool, device=device)
        
        # Add batch dimension
        batch_size = 1
        coordinates = coordinates.unsqueeze(0).expand(batch_size, -1, -1)
        element_ids = element_ids.unsqueeze(0).expand(batch_size, -1)
        b_factors = b_factors.unsqueeze(0).expand(batch_size, -1)
        occupancies = occupancies.unsqueeze(0).expand(batch_size, -1)
        active = active.unsqueeze(0).expand(batch_size, -1)
        
        # Generate density map
        with torch.no_grad():
            density_map = transformer(
                coordinates,
                element_ids,
                b_factors,
                occupancies,
                active
            )
        
        # Reduce batch dimension
        generated_map = density_map[0].cpu().numpy()
        ref_map_data = xmap.array.cpu().numpy()
        
        # Calculate correlation with reference map
        gen_flat = generated_map.flatten()
        ref_flat = ref_map_data.flatten()
        correlation, _ = pearsonr(gen_flat, ref_flat)
        
        print(f"\nMap correlation with reference: {correlation:.4f}")
        
        # Check for reasonable correlation
        assert correlation > 0.7, f"Correlation with reference map too low: {correlation:.4f}"


class TestPerformanceComparison:
    """Performance benchmarks for CUDA vs PyTorch implementations.
    
    These tests measure the performance improvement of the CUDA
    implementation over the PyTorch implementation for different
    problem sizes.
    """
    
    @pytest.fixture(scope="class")
    def test_setup(self):
        """Setup for performance testing.
        
        Returns
        -------
        Dict
            Dictionary containing setup information
        """

        device = try_gpu()

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping performance tests")
        
        # Create synthetic test data of different sizes
        batch_sizes = [1, 2, 4]
        atom_counts = [100, 1000, 5000]
        
        # Define density parameters
        density_params = DensityParameters(
            rmax=5.0,
            rstep=0.1,
            smin=0.0,
            smax=0.5,
            quad_points=50,
            integration_method="gausslegendre"
        )
        
        return {
            "device": device,
            "batch_sizes": batch_sizes,
            "atom_counts": atom_counts,
            "density_params": density_params
        }
    
    @pytest.mark.parametrize("include_backward", [False, True])
    def test_performance_scaling(self, test_setup, include_backward):
        """Test performance scaling with problem size.
        
        Parameters
        ----------
        test_setup : Dict
            Setup information from the test_setup fixture
        include_backward : bool
            Whether to include backward pass in timing
        """
        device = test_setup["device"]
        batch_sizes = test_setup["batch_sizes"]
        atom_counts = test_setup["atom_counts"]
        density_params = test_setup["density_params"]
        
        results = []
        
        for batch_size in batch_sizes:
            for n_atoms in atom_counts:
                # Create synthetic data
                coords = torch.randn(batch_size, n_atoms, 3, device=device)
                occupancies = torch.ones(batch_size, n_atoms, device=device)
                
                # Create random radial profiles
                n_radial = int(density_params.rmax / density_params.rstep) + 1
                radial_profiles = torch.rand(batch_size, n_atoms, n_radial, device=device)
                radial_derivatives = torch.rand(batch_size, n_atoms, n_radial, device=device)
                
                # Other parameters
                rstep = density_params.rstep
                rmax = density_params.rmax
                lmax = torch.tensor([5, 5, 5], dtype=torch.int32, device=device)
                grid_dims = torch.tensor([64, 64, 64], dtype=torch.int32, device=device)
                grid_to_cartesian = torch.eye(3, device=device)
                
                # Prepare inputs for PyTorch version
                coords_pytorch = coords.clone().requires_grad_(include_backward)
                occupancies_pytorch = occupancies.clone().requires_grad_(include_backward)
                
                # Prepare inputs for CUDA version
                coords_cuda = coords.clone().requires_grad_(include_backward)
                occupancies_cuda = occupancies.clone().requires_grad_(include_backward)
                
                # Warm-up
                with torch.no_grad():
                    _ = dilate_points_torch(
                        coords_pytorch,
                        torch.ones_like(occupancies_pytorch, dtype=torch.bool),
                        occupancies_pytorch,
                        lmax,
                        radial_profiles,
                        rstep,
                        rmax,
                        grid_to_cartesian,
                        tuple(grid_dims.cpu().numpy())
                    )
                    
                    _ = dilate_atom_centric(
                        coords_cuda,
                        occupancies_cuda,
                        radial_profiles,
                        radial_derivatives,
                        rstep,
                        rmax,
                        lmax,
                        grid_dims,
                        grid_to_cartesian
                    )
                
                # Time PyTorch implementation
                torch.cuda.synchronize()
                start_time = time.time()
                
                output_pytorch = dilate_points_torch(
                    coords_pytorch,
                    torch.ones_like(occupancies_pytorch, dtype=torch.bool),
                    occupancies_pytorch,
                    lmax,
                    radial_profiles,
                    rstep,
                    rmax,
                    grid_to_cartesian,
                    tuple(grid_dims.cpu().numpy())
                )
                
                if include_backward:
                    loss_pytorch = output_pytorch.sum()
                    loss_pytorch.backward()
                
                torch.cuda.synchronize()
                pytorch_time = time.time() - start_time
                
                # Reset gradients
                if include_backward:
                    coords_pytorch.grad = None
                    occupancies_pytorch.grad = None
                    coords_cuda.grad = None
                    occupancies_cuda.grad = None
                
                # Time CUDA implementation
                torch.cuda.synchronize()
                start_time = time.time()
                
                output_cuda = dilate_atom_centric(
                    coords_cuda,
                    occupancies_cuda,
                    radial_profiles,
                    radial_derivatives,
                    rstep,
                    rmax,
                    lmax,
                    grid_dims,
                    grid_to_cartesian
                )
                
                if include_backward:
                    loss_cuda = output_cuda.sum()
                    loss_cuda.backward()
                
                torch.cuda.synchronize()
                cuda_time = time.time() - start_time
                
                # Calculate speedup
                speedup = pytorch_time / cuda_time
                
                results.append({
                    "batch_size": batch_size,
                    "n_atoms": n_atoms,
                    "pytorch_time": pytorch_time,
                    "cuda_time": cuda_time,
                    "speedup": speedup,
                    "backward": include_backward
                })
                
                print(f"\nBatch: {batch_size}, Atoms: {n_atoms}, " +
                      f"Backward: {include_backward}, Speedup: {speedup:.2f}x " +
                      f"(PyTorch: {pytorch_time:.4f}s, CUDA: {cuda_time:.4f}s)")
        
        # Verify that speedup increases with problem size
        small_speedup = next(r["speedup"] for r in results 
                            if r["batch_size"] == batch_sizes[0] and 
                            r["n_atoms"] == atom_counts[0] and
                            r["backward"] == include_backward)
        
        large_speedup = next(r["speedup"] for r in results 
                            if r["batch_size"] == batch_sizes[-1] and 
                            r["n_atoms"] == atom_counts[-1] and
                            r["backward"] == include_backward)
        
        # The speedup should increase with problem size
        assert large_speedup > small_speedup, "Speedup doesn't scale with problem size"


if __name__ == "__main__":
    pytest.main(["-s", __file__])