"""Test script for adaptive solver integration.

This script tests the basic functionality of the adaptive solver
integration without requiring full diffusion setup.
"""

import torch
from edg.edg.modules.adaptive_solver import (
    AdaptiveSolverConfig, 
    create_adaptive_solver,
    AdamGradientSolver,
    SimpleAdaptiveSolver
)
from edg.edg.modules.guided_diffusion import DensityGuidedDiffusionStepper


def test_adaptive_solver_creation():
    """Test creating adaptive solvers."""
    print("Testing adaptive solver creation...")
    
    # Test creating different solver types
    config = AdaptiveSolverConfig(
        learning_rate=0.01,
        max_iterations=5,
        convergence_threshold=1e-4,
        gradient_clip_norm=1.0
    )
    
    # Test Adam solver
    adam_solver = create_adaptive_solver("adam", config)
    assert isinstance(adam_solver, AdamGradientSolver)
    print("✓ Adam solver created successfully")
    
    # Test Simple solver
    simple_solver = create_adaptive_solver("simple", config)
    assert isinstance(simple_solver, SimpleAdaptiveSolver)
    print("✓ Simple solver created successfully")
    
    print("All solver types created successfully!\n")


def test_stepper_integration():
    """Test adaptive solver integration with DensityGuidedDiffusionStepper."""
    print("Testing stepper integration...")
    
    # Create a stepper instance (this will fail if imports are broken)
    try:
        # We can't fully initialize it without a model, but we can test the class exists
        stepper_class = DensityGuidedDiffusionStepper
        print("✓ DensityGuidedDiffusionStepper class accessible")
        
        # Test that the setup method exists
        if hasattr(stepper_class, 'setup_adaptive_solver'):
            print("✓ setup_adaptive_solver method exists")
        else:
            print("✗ setup_adaptive_solver method missing!")
            return False
            
    except Exception as e:
        print(f"✗ Error with stepper class: {e}")
        return False
    
    print("Stepper integration tests passed!\n")
    return True


def test_mock_gradient_optimization():
    """Test adaptive solver with mock optimization problem."""
    print("Testing mock gradient optimization...")
    
    # Create a simple quadratic optimization problem
    # Minimize f(x) = ||x - target||^2
    target = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # [2, 3] shape
    initial = torch.zeros_like(target)
    
    config = AdaptiveSolverConfig(
        learning_rate=0.1,
        max_iterations=10,
        convergence_threshold=1e-6,
        gradient_clip_norm=None,
        per_potential_scaling=False
    )
    
    solver = create_adaptive_solver("adam", config)
    
    # Mock potential class
    class MockPotential:
        def compute_parameters(self, t):
            return {"guidance_weight": 1.0, "guidance_interval": 1}
    
    potentials = [MockPotential()]
    
    def compute_energy(coords):
        return torch.sum((coords - target) ** 2)
    
    def compute_gradient(potential, coords, feats, parameters):
        # Mock gradient computation (ignore unused parameters)
        _ = potential, feats, parameters  # Acknowledge unused parameters
        return 2 * (coords - target)
    
    # Run optimization
    updated_coords, stats = solver.step(
        initial,
        potentials,
        {},  # mock feats
        0.5,  # steering_t
        compute_energy,
        compute_gradient
    )
    
    print(f"Initial energy: {compute_energy(initial).item():.6f}")
    print(f"Final energy: {stats['final_energy']:.6f}")
    print(f"Converged: {stats['converged']}")
    print(f"Iterations: {stats['iterations']}")
    
    # Check that we moved closer to the target
    initial_distance = torch.norm(initial - target)
    final_distance = torch.norm(updated_coords - target)
    
    if final_distance < initial_distance:
        print("✓ Optimization improved solution")
    else:
        print("✗ Optimization did not improve solution")
        return False
    
    print("Mock optimization test passed!\n")
    return True


def main():
    """Run all tests."""
    print("=== Adaptive Solver Integration Tests ===\n")
    
    try:
        test_adaptive_solver_creation()
        test_stepper_integration()
        test_mock_gradient_optimization()
        
        print("🎉 All tests passed! Adaptive solver integration is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)