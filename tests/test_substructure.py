"""Tests for SubstructurePotential ensemble handling and RMSD reduction."""

import pytest
import torch
import numpy as np

from edg.edg.modules.potentials import SubstructurePotential


class TestSubstructurePotential:
    """Test SubstructurePotential ensemble handling and gradient behavior."""

    @pytest.fixture
    def ensemble_coords(self):
        """Create test ensemble coordinates."""
        torch.manual_seed(42)
        num_ensembles = 2
        ensemble_size = 3
        n_atoms = 10
        
        # Create base coordinates
        base_coords = torch.randn(n_atoms, 3) * 2.0
        
        # Create ensemble with some displacement from base
        coords = base_coords.unsqueeze(0).unsqueeze(0).expand(num_ensembles, ensemble_size, -1, -1).clone()
        coords += torch.randn_like(coords) * 5.0 # Add noise
        
        return coords, base_coords

    @pytest.fixture
    def mock_feats(self, ensemble_coords):
        """Create mock features for testing."""
        coords, _ = ensemble_coords
        num_ensembles, ensemble_size, n_atoms, _ = coords.shape
        batch_size = num_ensembles * ensemble_size
        
        return {
            "atom_pad_mask": torch.ones(batch_size, n_atoms, dtype=torch.bool),
            "atom_types": torch.randint(1, 10, (batch_size, n_atoms)),
        }

    @pytest.fixture
    def substructure_potential(self, ensemble_coords):
        """Create SubstructurePotential for testing."""
        coords, base_coords = ensemble_coords
        
        # Select subset of atoms for constraint
        selection = np.array([1, 3, 5])  # Exclude these from potential
        
        potential = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.1,  # Non-zero weight for testing
                "resampling_weight": 0.1,
                "buffer": 0.05,  # Small buffer to ensure potential is active
                "denoising_selection": selection,
                "reference_coords": base_coords,  # Reference coordinates
            }
        )
        
        return potential, selection

    def compute_rmsd(self, coords1, coords2):
        """Compute RMSD between two coordinate sets."""
        diff = coords1 - coords2
        msd = torch.mean(diff.pow(2), dim=-1).mean(dim=-1)  # Mean over atoms and coords
        return torch.sqrt(msd)

    def test_ensemble_shape_handling(self, ensemble_coords, mock_feats, substructure_potential):
        """Test that ensemble coordinates are handled with correct shapes."""
        coords, base_coords = ensemble_coords
        potential, selection = substructure_potential
        
        num_ensembles, ensemble_size, n_atoms, _ = coords.shape
        
        # Test compute_ensemble
        energy = potential.compute_ensemble(coords, mock_feats, potential.parameters)
        
        # Energy should have shape [num_ensembles]
        assert energy.shape == (num_ensembles,), f"Expected energy shape {(num_ensembles,)}, got {energy.shape}"
        
        # Energy should be finite
        assert torch.isfinite(energy).all(), "Energy should be finite"

    def test_gradient_ensemble_shape_handling(self, ensemble_coords, mock_feats, substructure_potential):
        """Test that ensemble gradients are computed with correct shapes."""
        coords, base_coords = ensemble_coords
        potential, selection = substructure_potential
        
        num_ensembles, ensemble_size, n_atoms, _ = coords.shape
        
        # Test compute_gradient_ensemble method directly
        grad = potential.compute_gradient_ensemble(coords, mock_feats, potential.parameters)
        
        # Gradient should have same shape as coordinates
        assert grad.shape == coords.shape, f"Expected gradient shape {coords.shape}, got {grad.shape}"
        
        # Gradients should be finite
        assert torch.isfinite(grad).all(), "Gradients should be finite"

    def test_rmsd_reduction_with_gradient_steps(self, ensemble_coords, mock_feats, substructure_potential):
        """Test that RMSD to reference coordinates decreases with gradient updates."""
        coords, base_coords = ensemble_coords
        potential, selection = substructure_potential
        
        num_ensembles, ensemble_size, n_atoms, _ = coords.shape
        
        # Create constraint mask (atoms NOT in denoising_selection)
        constraint_mask = torch.ones(n_atoms, dtype=torch.bool)
        constraint_mask[selection] = False
        constrained_atoms = torch.where(constraint_mask)[0]
        
        # Compute initial RMSD to base coordinates for constrained atoms only
        coords_detached = coords.detach().clone()
        coords_detached.requires_grad_(True)
        
        initial_rmsd = self.compute_rmsd(
            coords_detached[:, :, constrained_atoms, :], 
            base_coords[constrained_atoms, :].unsqueeze(0).unsqueeze(0).expand(num_ensembles, ensemble_size, -1, -1)
        )
        
        # Perform gradient descent steps
        learning_rate = potential.parameters["guidance_weight"]
        n_steps = 10
        
        for step in range(n_steps):
            # Compute gradients using the potential's method
            grad = potential.compute_gradient_ensemble(coords_detached, mock_feats, potential.parameters)
            
            # Update coordinates (gradient descent to minimize energy/distance)
            with torch.no_grad():
                coords_detached -= learning_rate * grad
        
        # Compute final RMSD
        final_rmsd = self.compute_rmsd(
            coords_detached[:, :, constrained_atoms, :],
            base_coords[constrained_atoms, :].unsqueeze(0).unsqueeze(0).expand(num_ensembles, ensemble_size, -1, -1)
        )
        
        print(f"Initial RMSD: {initial_rmsd}")
        print(f"Final RMSD: {final_rmsd}")
        
        # RMSD should decrease for all ensemble members
        assert (final_rmsd < initial_rmsd).all(), f"RMSD should decrease for all ensemble members. Initial: {initial_rmsd}, Final: {final_rmsd}"
        
        # Check that the reduction is significant (at least 10% reduction)
        reduction_ratio = final_rmsd / initial_rmsd
        assert (reduction_ratio < 0.9).all(), f"RMSD reduction should be at least 10%. Reduction ratio: {reduction_ratio}"

    def test_ensemble_members_treated_equally(self, ensemble_coords, mock_feats, substructure_potential):
        """Test that all ensemble members are treated equally by the potential."""
        coords, base_coords = ensemble_coords
        potential, selection = substructure_potential
        
        num_ensembles, ensemble_size, n_atoms, _ = coords.shape
        
        # Create identical coordinates for all ensemble members to test equal treatment
        identical_coords = coords[:, :1, :, :].expand(-1, ensemble_size, -1, -1).contiguous()
        
        # Compute gradients using the potential's method
        grad = potential.compute_gradient_ensemble(identical_coords, mock_feats, potential.parameters)
        
        # All ensemble members should have identical gradients since coordinates are identical
        for ensemble_idx in range(num_ensembles):
            ensemble_grads = grad[ensemble_idx]  # [ensemble_size, n_atoms, 3]
            
            # Check that all members within this ensemble have the same gradients
            for member_idx in range(1, ensemble_size):
                grad_diff = torch.abs(ensemble_grads[0] - ensemble_grads[member_idx])
                max_diff = grad_diff.max()
                assert max_diff < 1e-6, f"Ensemble members should have identical gradients. Max diff: {max_diff}"

    def test_reference_coordinate_broadcasting(self, ensemble_coords, mock_feats):
        """Test different reference coordinate broadcasting scenarios."""
        coords, base_coords = ensemble_coords
        num_ensembles, ensemble_size, n_atoms, _ = coords.shape
        
        # Test case 1: 2D reference coordinates (should broadcast to all ensemble members)
        potential_2d = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.2,
                "denoising_selection": np.array([]),
                "reference_coords": base_coords,  # [n_atoms, 3]
            }
        )
        
        energy_2d = potential_2d.compute_ensemble(coords, mock_feats, potential_2d.parameters)
        assert energy_2d.shape == (num_ensembles,)
        assert torch.isfinite(energy_2d).all()
        
        # Test case 2: 3D reference coordinates with num_ensembles dimension
        ref_coords_3d = base_coords.unsqueeze(0).expand(num_ensembles, -1, -1)
        potential_3d = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.2,
                "denoising_selection": np.array([]),
                "reference_coords": ref_coords_3d,  # [num_ensembles, n_atoms, 3]
            }
        )

        energy_3d = potential_3d.compute_ensemble(coords, mock_feats, potential_3d.parameters)
        assert energy_3d.shape == (num_ensembles,)
        assert torch.isfinite(energy_3d).all()

        # Test case 3: 4D reference coordiantes with num_ensembles and ensemble_size dimensions
        ref_coords_4d = base_coords.unsqueeze(0).unsqueeze(0).expand(num_ensembles, ensemble_size, -1, -1)
        potential_4d = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.2,
                "denoising_selection": np.array([]),
                "reference_coords": ref_coords_4d,  # [num_ensembles, ensemble_size, n_atoms, 3]
            }
        )

        coords = coords.reshape(num_ensembles, ensemble_size, n_atoms, 3)  # Ensure coords are 4D also
        
        energy_4d = potential_4d.compute_ensemble(coords, mock_feats, potential_4d.parameters)
        assert energy_4d.shape == (num_ensembles,)
        assert torch.isfinite(energy_4d).all()

    def test_denoising_selection_correctness(self, ensemble_coords, mock_feats):
        """Test that denoising_selection properly excludes atoms from potential."""
        coords, base_coords = ensemble_coords
        num_ensembles, ensemble_size, n_atoms, _ = coords.shape
        
        # Create two potentials: one with none being affected (all selected), one with all atoms being affected (none selected)
        selection = np.arange(n_atoms)
        
        potential_all_selection = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.05,
                "denoising_selection": np.array([]),
                "reference_coords": base_coords,
            }
        )
        
        potential_no_selection = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.05,
                "denoising_selection": selection,
                "reference_coords": base_coords,
            }
        )
        
        # Compute energies
        energy_all_sel = potential_all_selection.compute_ensemble(coords, mock_feats, potential_all_selection.parameters)
        energy_no_sel = potential_no_selection.compute_ensemble(coords, mock_feats, potential_no_selection.parameters)
        
        # Energy with selection should be nonzero, while energy without selection should be zero
        assert torch.all(energy_no_sel < energy_all_sel)
        assert torch.all(energy_all_sel > 0)
        assert torch.allclose(energy_no_sel, torch.zeros_like(energy_no_sel))
        
        # Both should be finite
        assert torch.isfinite(energy_all_sel).all() and torch.isfinite(energy_no_sel).all()

    def test_gradient_direction_consistency(self, ensemble_coords, mock_feats):
        """Test that gradient directions are consistent across ensemble members."""
        coords, base_coords = ensemble_coords
        num_ensembles, ensemble_size, n_atoms, _ = coords.shape
        
        # Create a simple test case with known displacement
        test_coords = torch.zeros(num_ensembles, ensemble_size, n_atoms, 3)
        ref_coords = torch.zeros(n_atoms, 3)
        
        # Place first ensemble member at [1, 0, 0] offset from reference
        test_coords[0, 0, :, 0] = 1.0  # x = 1, y = 0, z = 0
        
        # Place second ensemble member at same offset
        test_coords[0, 1, :, 0] = 1.0  # x = 1, y = 0, z = 0
        
        # Both should get same gradient direction (pointing toward reference at origin)
        potential = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.5,
                "denoising_selection": np.array([]),  # No exclusions - all atoms constrained
                "reference_coords": ref_coords,
            }
        )
        
        test_coords.requires_grad_(True)
        grad = potential.compute_gradient_ensemble(test_coords, mock_feats, potential.parameters)
        
        # Print detailed gradient information for debugging
        print(f"Gradient for first ensemble member (atom 0): {grad[0, 0, 0, :]}")
        print(f"Gradient for second ensemble member (atom 0): {grad[0, 1, 0, :]}")
        
        # Both gradients should point in positive x direction (away from reference at origin)
        # Since coords are at x=1 and reference is at x=0, and distance > buffer
        # Energy increases with distance, so gradient points away from reference
        # During optimization, we move in -gradient direction (toward reference)
        grad_member_0 = grad[0, 0, 0, :]  # First ensemble member, first atom
        grad_member_1 = grad[0, 1, 0, :]  # Second ensemble member, first atom
        
        # Check that x-component is positive for both (pointing away from origin)
        assert grad_member_0[0] > 0, f"First member should have positive x gradient, got {grad_member_0[0]}"
        assert grad_member_1[0] > 0, f"Second member should have positive x gradient, got {grad_member_1[0]}"
        
        # Check that gradients have same direction and magnitude
        grad_diff = torch.abs(grad_member_0 - grad_member_1)
        max_diff = grad_diff.max()
        print(f"Max gradient difference between ensemble members: {max_diff}")
        assert max_diff < 1e-6, f"Gradient directions should be identical for identical displacements, max diff: {max_diff}"

    def test_gradient_direction_single_vs_ensemble(self, ensemble_coords, mock_feats):
        """Test that single structure gradients match ensemble gradients."""
        coords, base_coords = ensemble_coords
        num_ensembles, ensemble_size, n_atoms, _ = coords.shape
        
        # Create simple test case
        test_coords_single = torch.zeros(1, n_atoms, 3)
        test_coords_single[0, :, 0] = 2.0  # All atoms at x=2
        
        ref_coords = torch.zeros(n_atoms, 3)  # Reference at origin
        
        # Single structure potential
        potential_single = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.5,
                "denoising_selection": np.array([]),
                "reference_coords": ref_coords,
            }
        )
        
        # Ensemble potential with same coordinates replicated
        test_coords_ensemble = test_coords_single.unsqueeze(1).expand(1, 2, -1, -1).contiguous()
        potential_ensemble = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.5,
                "denoising_selection": np.array([]),
                "reference_coords": ref_coords,
            }
        )
        
        # Compute gradients
        mock_feats_single = {
            "atom_pad_mask": torch.ones(1, n_atoms, dtype=torch.bool),
            "atom_types": torch.randint(1, 10, (1, n_atoms)),
        }
        
        mock_feats_ensemble = {
            "atom_pad_mask": torch.ones(2, n_atoms, dtype=torch.bool),  # 2 ensemble members
            "atom_types": torch.randint(1, 10, (2, n_atoms)),
        }
        
        test_coords_single.requires_grad_(True)
        test_coords_ensemble.requires_grad_(True)
        
        grad_single = potential_single.compute_gradient(test_coords_single, mock_feats_single, potential_single.parameters)
        grad_ensemble = potential_ensemble.compute_gradient_ensemble(test_coords_ensemble, mock_feats_ensemble, potential_ensemble.parameters)
        
        print(f"Single structure gradient (atom 0): {grad_single[0, 0, :]}")
        print(f"Ensemble member 0 gradient (atom 0): {grad_ensemble[0, 0, 0, :]}")
        print(f"Ensemble member 1 gradient (atom 0): {grad_ensemble[0, 1, 0, :]}")
        
        # Single gradient should match both ensemble member gradients
        grad_diff_0 = torch.abs(grad_single[0, 0, :] - grad_ensemble[0, 0, 0, :])
        grad_diff_1 = torch.abs(grad_single[0, 0, :] - grad_ensemble[0, 1, 0, :])
        
        assert grad_diff_0.max() < 1e-6, f"Single vs ensemble member 0 mismatch: {grad_diff_0.max()}"
        assert grad_diff_1.max() < 1e-6, f"Single vs ensemble member 1 mismatch: {grad_diff_1.max()}"

    def test_gradient_magnitude_scaling(self, ensemble_coords, mock_feats):
        """Test that gradient magnitudes scale correctly with distance from reference."""
        coords, base_coords = ensemble_coords
        num_ensembles, ensemble_size, n_atoms, _ = coords.shape
        
        # Create test coordinates at different distances from origin
        test_coords = torch.zeros(1, 3, n_atoms, 3)  # 1 ensemble, 3 members
        test_coords[0, 0, :, 0] = 1.0  # Member 0: distance = 1.0
        test_coords[0, 1, :, 0] = 2.0  # Member 1: distance = 2.0  
        test_coords[0, 2, :, 0] = 3.0  # Member 2: distance = 3.0
        
        ref_coords = torch.zeros(n_atoms, 3)  # Reference at origin
        
        potential = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.5,  # All members are outside buffer
                "denoising_selection": np.array([]),  # No exclusions
                "reference_coords": ref_coords,
            }
        )
        
        # Test with appropriate mock features for 3 ensemble members
        mock_feats_3 = {
            "atom_pad_mask": torch.ones(3, n_atoms, dtype=torch.bool),
            "atom_types": torch.randint(1, 10, (3, n_atoms)),
        }
        
        test_coords.requires_grad_(True)
        grad = potential.compute_gradient_ensemble(test_coords, mock_feats_3, potential.parameters)
        
        grad_mag_0 = torch.linalg.norm(grad[0, 0, 0, :])  # Gradient magnitude for member 0
        grad_mag_1 = torch.linalg.norm(grad[0, 1, 0, :])  # Gradient magnitude for member 1  
        grad_mag_2 = torch.linalg.norm(grad[0, 2, 0, :])  # Gradient magnitude for member 2
        
        print(f"Gradient magnitudes: {grad_mag_0:.3f}, {grad_mag_1:.3f}, {grad_mag_2:.3f}")
        print("Expected relationship: grad_mag_2 > grad_mag_1 > grad_mag_0")
        
        # For harmonic potential, gradient should increase with distance beyond buffer
        # All members are outside buffer (0.5), so gradients should increase with distance
        assert grad_mag_1 > grad_mag_0, f"Member 1 should have larger gradient than member 0: {grad_mag_1} vs {grad_mag_0}"
        assert grad_mag_2 > grad_mag_1, f"Member 2 should have larger gradient than member 1: {grad_mag_2} vs {grad_mag_1}"

    def test_ensemble_gradient_direction_issue(self):
        """Test the specific gradient direction reversal issue reported by user."""
        n_atoms = 5
        
        # Create test coordinates: 1 ensemble with 2 members at different positions
        test_coords = torch.zeros(1, 2, n_atoms, 3)
        test_coords[0, 0, :, 0] = 1.0  # Member 0: distance = 1.0
        test_coords[0, 1, :, 0] = -2.0  # Member 1: distance = 2.0
        
        ref_coords = torch.zeros(n_atoms, 3)  # Reference at origin
        
        potential = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.5,
                "denoising_selection": np.array([]),  # No exclusions
                "reference_coords": ref_coords,
            }
        )
        
        # Create correct mock features for 2 structures (flattened ensemble)
        mock_feats = {
            "atom_pad_mask": torch.ones(2, n_atoms, dtype=torch.bool),
            "atom_types": torch.randint(1, 10, (2, n_atoms)),
        }
        
        test_coords.requires_grad_(True)
        grad = potential.compute_gradient_ensemble(test_coords, mock_feats, potential.parameters)
        
        print(f"Test coordinates shapes: {test_coords.shape}")
        print(f"Member 0 position (atom 0): {test_coords[0, 0, 0, :]}")
        print(f"Member 1 position (atom 0): {test_coords[0, 1, 0, :]}")
        print(f"Gradient for member 0 (atom 0): {grad[0, 0, 0, :]}")
        print(f"Gradient for member 1 (atom 0): {grad[0, 1, 0, :]}")
        
        grad_0 = grad[0, 0, 0, :]  # Member 0, atom 0
        grad_1 = grad[0, 1, 0, :]  # Member 1, atom 0
        
        # Member 0 at x=+1.0 should have positive gradient (pointing away from origin)
        # Member 1 at x=-2.0 should have negative gradient (pointing away from origin)
        print("Member 0 (x=+1.0) should have positive gradient (pointing away from origin)")
        print("Member 1 (x=-2.0) should have negative gradient (pointing away from origin)")
        
        assert grad_0[0] > 0, f"Member 0 gradient should be positive, got {grad_0[0]}"
        assert grad_1[0] < 0, f"Member 1 gradient should be negative, got {grad_1[0]}"
        
        # Member 1 should have larger gradient magnitude (farther from reference)
        grad_mag_0 = torch.linalg.norm(grad_0)
        grad_mag_1 = torch.linalg.norm(grad_1)
        
        print(f"Gradient magnitudes: {grad_mag_0:.3f} vs {grad_mag_1:.3f}")
        assert grad_mag_1 > grad_mag_0, f"Member 1 should have larger gradient: {grad_mag_1} vs {grad_mag_0}"
        
        # Test the specific issue: do directions get reversed?
        grad_direction_0 = grad_0 / torch.linalg.norm(grad_0)
        grad_direction_1 = grad_1 / torch.linalg.norm(grad_1)
        
        # Should point in opposite directions
        direction_similarity = torch.dot(grad_direction_0, grad_direction_1)
        print(f"Direction similarity (should be close to -1.0): {direction_similarity}")
        assert direction_similarity < -0.5, f"Gradient directions should be opposing, got {direction_similarity}"
    
    def test_gradient_direction_reversal_bug(self):
        """Test for the specific bug: first member correct, others reversed."""
        n_atoms = 3
        
        # Test case that might trigger the bug: coordinates on opposite sides of reference
        test_coords = torch.zeros(1, 3, n_atoms, 3)
        
        # All members displaced in positive x direction from reference
        test_coords[0, 0, :, 0] = 1.0  # Member 0: +1 from origin  
        test_coords[0, 1, :, 0] = 1.5  # Member 1: +1.5 from origin
        test_coords[0, 2, :, 0] = 2.0  # Member 2: +2 from origin
        
        ref_coords = torch.zeros(n_atoms, 3)  # Reference at origin
        
        potential = SubstructurePotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 1.0,
                "resampling_weight": 0.1,
                "buffer": 0.2,  # Small buffer so all are in penalty region
                "denoising_selection": np.array([]),
                "reference_coords": ref_coords,
            }
        )
        
        # Create mock features for 3 structures
        mock_feats = {
            "atom_pad_mask": torch.ones(3, n_atoms, dtype=torch.bool),
            "atom_types": torch.randint(1, 10, (3, n_atoms)),
        }
        
        test_coords.requires_grad_(True)
        grad = potential.compute_gradient_ensemble(test_coords, mock_feats, potential.parameters)
        
        # Check gradients for all members
        grad_0_x = grad[0, 0, 0, 0]  # Member 0, atom 0, x-component
        grad_1_x = grad[0, 1, 0, 0]  # Member 1, atom 0, x-component  
        grad_2_x = grad[0, 2, 0, 0]  # Member 2, atom 0, x-component
        
        print("Positions: Member 0 at x=1.0, Member 1 at x=1.5, Member 2 at x=2.0")
        print("Reference at x=0.0, buffer=0.2")
        print("All members should have POSITIVE gradients (pointing away from origin)")
        print(f"Gradient x-components: {grad_0_x:.3f}, {grad_1_x:.3f}, {grad_2_x:.3f}")
        
        # All should be positive (pointing away from reference)
        assert grad_0_x > 0, f"Member 0 gradient should be positive, got {grad_0_x}"
        assert grad_1_x > 0, f"Member 1 gradient should be positive, got {grad_1_x}"  
        assert grad_2_x > 0, f"Member 2 gradient should be positive, got {grad_2_x}"
        
        # Check for direction reversal: if bug exists, members 1,2 might be negative
        if grad_1_x < 0 or grad_2_x < 0:
            print("🐛 BUG DETECTED: Gradient direction reversal!")
            print(f"Member 0: {grad_0_x} (should be positive)")
            print(f"Member 1: {grad_1_x} (should be positive, but got negative!)")
            print(f"Member 2: {grad_2_x} (should be positive, but got negative!)")
            
        # Magnitude should increase with distance
        assert grad_1_x > grad_0_x, f"Member 1 should have larger gradient than 0: {grad_1_x} vs {grad_0_x}"
        assert grad_2_x > grad_1_x, f"Member 2 should have larger gradient than 1: {grad_2_x} vs {grad_1_x}"


if __name__ == "__main__":
    pytest.main(["-vs", f"{__file__}"])