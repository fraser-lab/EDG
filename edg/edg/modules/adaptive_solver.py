"""Adaptive gradient solvers for density guided diffusion.

This module provides adaptive optimization methods for the gradient update steps
in density guided diffusion, replacing fixed-step gradient descent with more
sophisticated approaches that can adapt step sizes and handle multiple potential
scales automatically.

Author: Karson Chrispens
Created: 2025-01-02
"""

import torch
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AdaptiveSolverConfig:
    """Configuration for adaptive gradient solvers.
    
    Parameters
    ----------
    learning_rate : float
        Base learning rate for the optimizer
    beta1 : float
        Exponential decay rate for first moment estimates (Adam-style)
    beta2 : float
        Exponential decay rate for second moment estimates (Adam-style)
    eps : float
        Small constant for numerical stability
    max_iterations : int
        Maximum number of gradient steps per solve
    convergence_threshold : float
        Gradient norm threshold for early stopping
    gradient_clip_norm : Optional[float]
        Maximum gradient norm for clipping (None to disable)
    per_potential_scaling : bool
        Whether to normalize gradients per potential type
    line_search : bool
        Whether to use backtracking line search
    line_search_c1 : float
        Armijo condition parameter for line search
    line_search_backtrack : float
        Backtracking factor for line search
    max_line_search_steps : int
        Maximum line search iterations
    """
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_iterations: int = 10
    convergence_threshold: float = 1e-4
    gradient_clip_norm: Optional[float] = 1.0
    per_potential_scaling: bool = True
    line_search: bool = False
    line_search_c1: float = 1e-4
    line_search_backtrack: float = 0.5
    max_line_search_steps: int = 5


class AdaptiveGradientSolver(ABC):
    """Base class for adaptive gradient solvers."""
    
    def __init__(self, config: AdaptiveSolverConfig):
        self.config = config
        self.reset()
    
    @abstractmethod
    def reset(self):
        """Reset the solver state."""
        pass
    
    @abstractmethod
    def step(
        self,
        coords: torch.Tensor,
        potentials: List[Any],
        feats: Dict[str, Any],
        steering_t: float,
        compute_energy_fn: Callable,
        compute_gradient_fn: Callable,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform an adaptive gradient step.
        
        Parameters
        ----------
        coords : torch.Tensor
            Current coordinates
        potentials : List[Any]
            List of potential objects
        feats : Dict[str, Any]
            Features dictionary
        steering_t : float
            Current diffusion time parameter
        compute_energy_fn : Callable
            Function to compute total energy
        compute_gradient_fn : Callable
            Function to compute gradients per potential
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            Updated coordinates and statistics dict
        """
        pass


class AdamGradientSolver(AdaptiveGradientSolver):
    """Adam-style adaptive gradient solver with additional features."""
    
    def reset(self):
        """Reset Adam state variables."""
        self.step_count = 0
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.potential_scales = {}  # Running estimates of gradient scales per potential
    
    def step(
        self,
        coords: torch.Tensor,
        potentials: List[Any],
        feats: Dict[str, Any], 
        steering_t: float,
        compute_energy_fn: Callable,
        compute_gradient_fn: Callable,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform adaptive gradient optimization using Adam algorithm."""
        
        self.step_count += 1
        stats = {
            'iterations': 0,
            'initial_energy': 0.0,
            'final_energy': 0.0,
            'gradient_norm': 0.0,
            'converged': False
        }
        
        current_coords = coords.clone()
        
        # Compute initial energy
        initial_energy = compute_energy_fn(current_coords)
        stats['initial_energy'] = initial_energy.item() if hasattr(initial_energy, 'item') else float(initial_energy)
        
        for iteration in range(self.config.max_iterations):
            # Compute gradients for each potential
            total_gradient = torch.zeros_like(current_coords)
            gradient_norms = {}
            
            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                if parameters["guidance_weight"] > 0 and iteration % parameters["guidance_interval"] == 0:
                    grad = compute_gradient_fn(potential, current_coords, feats, parameters)
                    
                    # Track gradient scales per potential type
                    potential_name = type(potential).__name__
                    grad_norm = torch.linalg.norm(grad).item()
                    gradient_norms[potential_name] = grad_norm
                    
                    if self.config.per_potential_scaling:
                        # Update running scale estimate
                        if potential_name not in self.potential_scales:
                            self.potential_scales[potential_name] = grad_norm
                        else:
                            # Exponential moving average
                            alpha = 0.1
                            self.potential_scales[potential_name] = (
                                alpha * grad_norm + (1 - alpha) * self.potential_scales[potential_name]
                            )
                        
                        # Scale gradient
                        if self.potential_scales[potential_name] > 1e-8:
                            grad = grad / (self.potential_scales[potential_name] + self.config.eps)
                    
                    total_gradient += parameters["guidance_weight"] * grad
            
            # Check convergence
            grad_norm = torch.linalg.norm(total_gradient).item()
            stats['gradient_norm'] = grad_norm
            
            if grad_norm < self.config.convergence_threshold:
                stats['converged'] = True
                break
            
            # Gradient clipping
            if self.config.gradient_clip_norm is not None and grad_norm > self.config.gradient_clip_norm:
                total_gradient = total_gradient * (self.config.gradient_clip_norm / grad_norm)
            
            # Adam update
            if self.m is None:
                self.m = torch.zeros_like(total_gradient)
                self.v = torch.zeros_like(total_gradient)
            
            # Update biased first moment estimate
            self.m = self.config.beta1 * self.m + (1 - self.config.beta1) * total_gradient
            
            # Update biased second moment estimate
            self.v = self.config.beta2 * self.v + (1 - self.config.beta2) * (total_gradient ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m / (1 - self.config.beta1 ** self.step_count)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v / (1 - self.config.beta2 ** self.step_count)
            
            # Compute update direction
            update_direction = m_hat / (torch.sqrt(v_hat) + self.config.eps)
            
            # Apply update
            if self.config.line_search:
                step_size = self._backtracking_line_search(
                    current_coords, 
                    update_direction, 
                    compute_energy_fn,
                    initial_energy
                )
            else:
                step_size = self.config.learning_rate
            
            current_coords = current_coords - step_size * update_direction
            stats['iterations'] = iteration + 1
        
        # Compute final energy
        final_energy = compute_energy_fn(current_coords)
        stats['final_energy'] = final_energy.item() if hasattr(final_energy, 'item') else float(final_energy)
        
        return current_coords, stats
    
    def _backtracking_line_search(
        self,
        coords: torch.Tensor,
        direction: torch.Tensor,
        compute_energy_fn: Callable,
        initial_energy: float,
    ) -> float:
        """Perform backtracking line search to find good step size."""
        
        alpha = self.config.learning_rate
        grad_dot_dir = torch.sum(direction * direction).item()  # For descent, this should be positive
        
        for _ in range(self.config.max_line_search_steps):
            new_coords = coords - alpha * direction
            new_energy = compute_energy_fn(new_coords)
            
            # Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*grad^T*d
            if new_energy <= initial_energy - self.config.line_search_c1 * alpha * grad_dot_dir:
                return alpha
            
            alpha *= self.config.line_search_backtrack
        
        # If line search fails, use small step
        return alpha


class SimpleAdaptiveSolver(AdaptiveGradientSolver):
    """Simplified adaptive solver with just gradient scaling and clipping."""
    
    def reset(self):
        """Reset state."""
        self.potential_scales = {}
    
    def step(
        self,
        coords: torch.Tensor,
        potentials: List[Any],
        feats: Dict[str, Any],
        steering_t: float,
        compute_energy_fn: Callable,
        compute_gradient_fn: Callable,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform simple adaptive gradient step."""
        
        stats = {
            'iterations': 0,
            'initial_energy': 0.0,
            'final_energy': 0.0,
            'gradient_norm': 0.0,
            'converged': False
        }
        
        current_coords = coords.clone()
        
        # Compute initial energy
        initial_energy = compute_energy_fn(current_coords)
        stats['initial_energy'] = initial_energy.item() if hasattr(initial_energy, 'item') else float(initial_energy)
        
        for iteration in range(self.config.max_iterations):
            # Compute gradients for each potential
            total_gradient = torch.zeros_like(current_coords)
            
            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                if parameters["guidance_weight"] > 0 and iteration % parameters["guidance_interval"] == 0:
                    grad = compute_gradient_fn(potential, current_coords, feats, parameters)
                    
                    # Simple per-potential scaling
                    if self.config.per_potential_scaling:
                        potential_name = type(potential).__name__
                        grad_norm = torch.linalg.norm(grad).item()
                        
                        if potential_name not in self.potential_scales:
                            self.potential_scales[potential_name] = grad_norm
                        else:
                            # Simple running average
                            self.potential_scales[potential_name] = (
                                0.9 * self.potential_scales[potential_name] + 0.1 * grad_norm
                            )
                        
                        # Scale gradient
                        if self.potential_scales[potential_name] > 1e-8:
                            grad = grad / (self.potential_scales[potential_name] + self.config.eps)
                    
                    total_gradient += parameters["guidance_weight"] * grad
            
            # Check convergence
            grad_norm = torch.linalg.norm(total_gradient).item()
            stats['gradient_norm'] = grad_norm
            
            if grad_norm < self.config.convergence_threshold:
                stats['converged'] = True
                break
            
            # Gradient clipping
            if self.config.gradient_clip_norm is not None and grad_norm > self.config.gradient_clip_norm:
                total_gradient = total_gradient * (self.config.gradient_clip_norm / grad_norm)
            
            # Simple gradient descent step
            current_coords = current_coords - self.config.learning_rate * total_gradient
            stats['iterations'] = iteration + 1
        
        # Compute final energy
        final_energy = compute_energy_fn(current_coords)
        stats['final_energy'] = final_energy.item() if hasattr(final_energy, 'item') else float(final_energy)
        
        return current_coords, stats


def create_adaptive_solver(solver_type: str = "adam", config: Optional[AdaptiveSolverConfig] = None) -> AdaptiveGradientSolver:
    """Factory function to create adaptive solvers.
    
    Parameters
    ----------
    solver_type : str
        Type of solver ("adam", "simple")
    config : Optional[AdaptiveSolverConfig]
        Solver configuration
        
    Returns
    -------
    AdaptiveGradientSolver
        Configured solver instance
    """
    if config is None:
        config = AdaptiveSolverConfig()
    
    if solver_type.lower() == "adam":
        return AdamGradientSolver(config)
    elif solver_type.lower() == "simple":
        return SimpleAdaptiveSolver(config)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")