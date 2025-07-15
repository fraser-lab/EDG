"""Configuration system for EDG (Ensembles from Density Generator).

Provides YAML-based configuration with CLI overrides, parameter scheduling,
and validation for EDG experiments.
"""

from .config_schema import (
    ExperimentConfig,
    BatchExperimentConfig,
    DiffusionConfig,
    DensityConfig,
    DensityGuidanceConfig,
    AdaptiveSolverConfig,
    SteeringConfig,
    OptimizationConfig,
)
from .config_loader import (
    load_config,
    load_batch_config,
    detect_config_type,
    merge_overrides,
    save_config,
    save_batch_config,
)
from .schedules import (
    ParameterSchedule,
    PiecewiseScheduleConfig,
    ExponentialInterpolationConfig,
    ResolutionScalingConfig,
)

__all__ = [
    "ExperimentConfig",
    "BatchExperimentConfig",
    "DiffusionConfig",
    "DensityConfig",
    "DensityGuidanceConfig",
    "AdaptiveSolverConfig",
    "SteeringConfig",
    "OptimizationConfig",
    "load_config",
    "load_batch_config",
    "detect_config_type",
    "merge_overrides",
    "save_config",
    "save_batch_config",
    "ParameterSchedule",
    "PiecewiseScheduleConfig",
    "ExponentialInterpolationConfig",
    "ResolutionScalingConfig",
]
