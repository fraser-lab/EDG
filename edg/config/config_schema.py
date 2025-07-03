"""Configuration schema definitions for EDG experiments.

This module defines the complete configuration schema using dataclasses
that can be serialized to/from YAML and validated for correctness.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import torch

from .schedules import ParameterSchedule, ConstantScheduleConfig


@dataclass
class ModelConfig:
    """Configuration for the diffusion model."""
    version: str = "boltz2"  # "boltz1" or "boltz2"
    checkpoint_path: Optional[str] = None  # Auto-detect if None
    ccd_path: Optional[str] = None  # Auto-detect if None
    device: Optional[str] = None  # Auto-detect if None


@dataclass 
class DiffusionConfig:
    """Configuration for diffusion parameters."""
    step_scale: Optional[float] = None  # Auto-set based on model version if None
    num_steps: int = 200
    noise_scale: float = 1.0
    gamma_0: float = 1.0
    gamma_min: float = 0.01


@dataclass
class SteeringConfig:
    """Configuration for FK steering parameters."""
    enabled: bool = True
    guidance_update: bool = True
    num_particles: int = 3
    fk_resampling_interval: int = 1
    fk_lambda: float = 0.5
    num_gd_steps: int = 10


@dataclass
class AdaptiveSolverConfig:
    """Configuration for adaptive gradient solver."""
    type: str = "adam"  # "adam", "simple", or "none"
    learning_rate: float = 0.02
    max_iterations: int = 10
    convergence_threshold: float = 1e-4
    gradient_clip_norm: float = 1.0
    per_potential_scaling: bool = True
    line_search: bool = False
    
    # Adam-specific parameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Line search parameters
    line_search_c1: float = 1e-4
    line_search_backtrack: float = 0.5
    max_line_search_steps: int = 3
    adaptive_line_search: bool = False
    adaptive_backtrack_min: float = 0.1
    adaptive_backtrack_max: float = 0.8
    violation_scaling: float = 0.5


@dataclass
class DensityGuidanceConfig:
    """Configuration for density guidance parameters."""
    base_weight: Union[float, ParameterSchedule] = field(
        default_factory=lambda: ConstantScheduleConfig(0.4)
    )
    guidance_interval: int = 1
    resampling_weight: Union[float, ParameterSchedule] = field(
        default_factory=lambda: ConstantScheduleConfig(0.1)
    )
    resolution_adaptive: bool = True
    scale_guidance_to_denoising: bool = True
    max_guidance_denoising_ratio: Union[float, ParameterSchedule] = field(
        default_factory=lambda: ConstantScheduleConfig(0.2)
    )


@dataclass
class SubstructureConfig:
    """Configuration for substructure conditioning."""
    enabled: bool = False
    selection: Optional[str] = None  # Selection string like "chain A and resi 120-140"
    guidance_weight: float = 0.05
    resampling_weight: float = 0.0
    buffer: float = 0.5


@dataclass
class DensityConfig:
    """Configuration for density map and calculation."""
    map_path: str
    resolution: Optional[float] = None  # Required for CCP4/MRC files
    em_mode: bool = False  # Use electron scattering factors
    
    # Resolution scheduling
    resolution_schedule: Optional[ParameterSchedule] = None


@dataclass
class StructureConfig:
    """Configuration for input structure."""
    structure_path: str
    clean_structure: bool = True
    keep_type: str = "protein"  # "protein", "all", etc.
    remove_alternative_conformations: bool = True
    complete_residues: bool = True


@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""
    ensemble_size: int = 1
    partial_diffusion: bool = False
    noising_steps: Optional[int] = None  # Auto-set to num_steps//4 if None
    representation_noise_scale: Optional[float] = None
    
    # Output configuration
    save_interval: int = 10  # Save intermediate structures every N steps
    save_maps: bool = True
    save_scores: bool = True


@dataclass
class PotentialConfig:
    """Configuration for additional potentials."""
    # Individual potential configurations can be added here
    # For now, using default potentials from get_potentials()
    use_default_potentials: bool = True
    
    # Custom potential configurations can be added here in the future
    custom_potentials: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Required configurations
    name: str
    structure: StructureConfig
    density: DensityConfig
    
    # Output configuration  
    output_dir: str
    input_data_dir: str  # Path for temporary input data (YAML file)
    
    # Optional configurations with defaults
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    adaptive_solver: AdaptiveSolverConfig = field(default_factory=AdaptiveSolverConfig)
    density_guidance: DensityGuidanceConfig = field(default_factory=DensityGuidanceConfig)
    substructure: SubstructureConfig = field(default_factory=SubstructureConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    potentials: PotentialConfig = field(default_factory=PotentialConfig)
    
    def validate(self) -> List[str]:
        """Validate the configuration and return any errors.
        
        Returns
        -------
        List[str]
            List of validation error messages. Empty if valid.
        """
        errors = []
        
        # Check required files exist
        if not Path(self.structure.structure_path).exists():
            errors.append(f"Structure file not found: {self.structure.structure_path}")
        
        if not Path(self.density.map_path).exists():
            errors.append(f"Density map file not found: {self.density.map_path}")
        
        # Check density resolution for CCP4/MRC files
        map_ext = Path(self.density.map_path).suffix.lower()
        if map_ext in ['.ccp4', '.mrc', '.map'] and self.density.resolution is None:
            errors.append(f"Resolution must be specified for {map_ext} files")
        
        # Check model version
        if self.model.version not in ["boltz1", "boltz2"]:
            errors.append(f"Invalid model version: {self.model.version}")
        
        # Check adaptive solver type
        if self.adaptive_solver.type not in ["adam", "simple", "none"]:
            errors.append(f"Invalid adaptive solver type: {self.adaptive_solver.type}")
        
        # Check positive values
        if self.diffusion.num_steps <= 0:
            errors.append("Number of diffusion steps must be positive")
        
        if self.optimization.ensemble_size <= 0:
            errors.append("Ensemble size must be positive")
        
        if self.steering.num_particles <= 0:
            errors.append("Number of particles must be positive")
        
        return errors
    
    def to_optimizer_kwargs(self) -> Dict[str, Any]:
        """Convert config to kwargs for DensityGuidedDiffusion constructor.
        
        Returns
        -------
        Dict[str, Any]
            Keyword arguments for optimizer initialization
        """
        from boltz.main import BoltzSteeringParams
        
        # Create steering args
        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = self.steering.enabled
        steering_args.guidance_update = self.steering.guidance_update  
        steering_args.num_particles = self.steering.num_particles
        
        # Create adaptive solver config
        from edg.edg.modules.adaptive_solver import AdaptiveSolverConfig as AdaptiveSolverConfigClass
        
        adaptive_solver_config = AdaptiveSolverConfigClass(
            learning_rate=self.adaptive_solver.learning_rate,
            max_iterations=self.adaptive_solver.max_iterations,
            convergence_threshold=self.adaptive_solver.convergence_threshold,
            gradient_clip_norm=self.adaptive_solver.gradient_clip_norm,
            per_potential_scaling=self.adaptive_solver.per_potential_scaling,
            line_search=self.adaptive_solver.line_search,
            beta1=self.adaptive_solver.beta1,
            beta2=self.adaptive_solver.beta2,
            eps=self.adaptive_solver.eps,
            line_search_c1=self.adaptive_solver.line_search_c1,
            line_search_backtrack=self.adaptive_solver.line_search_backtrack,
            max_line_search_steps=self.adaptive_solver.max_line_search_steps,
            adaptive_line_search=self.adaptive_solver.adaptive_line_search,
            adaptive_backtrack_min=self.adaptive_solver.adaptive_backtrack_min,
            adaptive_backtrack_max=self.adaptive_solver.adaptive_backtrack_max,
            violation_scaling=self.adaptive_solver.violation_scaling,
        )
        
        return {
            "input_path": Path(self.input_data_dir),
            "y": self.density.map_path,
            "structure": self.structure.structure_path,
            "output_path": self.output_dir,
            "em": self.density.em_mode,
            "resolution": self.density.resolution,
            "step_scale": self.diffusion.step_scale,
            "ckpt_path": Path(self.model.checkpoint_path) if self.model.checkpoint_path else None,
            "model_version": self.model.version,
            "ccd_path": Path(self.model.ccd_path) if self.model.ccd_path else None,
            "device": self.model.device,
            "adaptive_solver": self.adaptive_solver.type,
            "adaptive_solver_config": adaptive_solver_config if self.adaptive_solver.type != "none" else None,
            "steering_args": steering_args,
        }