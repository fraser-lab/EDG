"""Configuration system for EDG (Ensembles from Density Generator).

This module provides a comprehensive configuration system that supports:
- YAML-based configuration files
- Command-line parameter overrides  
- Complex parameter scheduling
- Parameter validation and type checking
- Backward compatibility with existing API

Usage:
    from edg.config import ExperimentConfig, load_config
    
    # Load from YAML file
    config = load_config("path/to/config.yaml")
    
    # Load with CLI overrides
    config = load_config("path/to/config.yaml", overrides={"num_steps": 300})
"""

from .config_schema import (
    ExperimentConfig,
    DiffusionConfig,
    DensityConfig,
    DensityGuidanceConfig,
    AdaptiveSolverConfig,
    SteeringConfig,
    OptimizationConfig,
)
from .config_loader import load_config, merge_overrides, save_config
from .schedules import (
    ParameterSchedule,
    PiecewiseScheduleConfig,
    ExponentialInterpolationConfig,
    ResolutionScalingConfig,
)

__all__ = [
    "ExperimentConfig",
    "DiffusionConfig", 
    "DensityConfig",
    "DensityGuidanceConfig",
    "AdaptiveSolverConfig",
    "SteeringConfig",
    "OptimizationConfig",
    "load_config",
    "merge_overrides",
    "save_config",
    "ParameterSchedule",
    "PiecewiseScheduleConfig",
    "ExponentialInterpolationConfig", 
    "ResolutionScalingConfig",
]