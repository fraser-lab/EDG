"""Configuration loading and processing utilities.

This module provides functionality to load configurations from YAML files,
apply command-line overrides, and convert schedule configurations to actual
schedule objects.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import yaml
from dataclasses import fields, is_dataclass, asdict

from .config_schema import ExperimentConfig, BatchExperimentConfig
from .schedules import ParameterSchedule, parse_schedule_config


def load_config(
    config_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None
) -> ExperimentConfig:
    """Load experiment configuration from YAML file with optional overrides.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to YAML configuration file
    overrides : Optional[Dict[str, Any]], optional
        Dictionary of parameter overrides from command line, by default None

    Returns
    -------
    ExperimentConfig
        Loaded and validated configuration

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If configuration is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML file
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        raise ValueError(f"Empty or invalid YAML file: {config_path}")

    # Apply command-line overrides
    if overrides:
        config_data = merge_overrides(config_data, overrides)

    # Parse schedule configurations
    config_data = parse_schedule_configs(config_data)

    # Create ExperimentConfig object with proper nested dataclass construction
    try:
        config = create_experiment_config(config_data)
    except TypeError as e:
        raise ValueError(f"Invalid configuration: {e}")

    # Validate configuration
    errors = config.validate()
    if errors:
        error_msg = "Configuration validation errors:\n" + "\n".join(
            f"  - {err}" for err in errors
        )
        raise ValueError(error_msg)

    return config


def merge_overrides(
    config_data: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge command-line overrides into configuration data.

    Supports nested parameter overrides using dot notation:
    - "num_steps" → {"optimization": {"num_steps": value}}
    - "optimization.num_steps" → {"optimization": {"num_steps": value}}
    - "density_guidance.base_weight" → {"density_guidance": {"base_weight": value}}

    Parameters
    ----------
    config_data : Dict[str, Any]
        Base configuration data
    overrides : Dict[str, Any]
        Override parameters with dot notation support

    Returns
    -------
    Dict[str, Any]
        Merged configuration data
    """
    config_data = config_data.copy()

    for key, value in overrides.items():
        # Map common CLI parameter names to config paths
        if "." not in key:
            key = map_override_key(key)
        
        # Handle dot notation for nested parameters
        if "." in key:
            parts = key.split(".")
            target = config_data

            # Navigate to parent dict
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]

            # Set final value
            target[parts[-1]] = value
        else:
            # Direct assignment for top-level parameters
            config_data[key] = value

    return config_data


def map_override_key(key: str) -> str:
    """Map common CLI parameter names to config paths.

    Parameters
    ----------
    key : str
        CLI parameter name

    Returns
    -------
    str
        Mapped configuration path (may include dots for nested access)
    """
    # Common parameter mappings
    mapping = {
        # Optimization parameters
        "num_steps": "diffusion.num_steps",
        "ensemble_size": "optimization.ensemble_size",
        "step_scale": "diffusion.step_scale",
        # Density parameters
        "resolution": "density.resolution",
        "map_path": "density.map_path",
        "em_mode": "density.em_mode",
        # Structure parameters
        "structure_path": "structure.structure_path",
        # Guidance parameters
        "guidance_weight": "density_guidance.base_weight",
        "resampling_weight": "density_guidance.resampling_weight",
        "max_guidance_denoising_ratio": "density_guidance.max_guidance_denoising_ratio",
        "guidance_interval": "density_guidance.guidance_interval",
        # Steering parameters
        "num_particles": "steering.num_particles",
        "guidance_update": "steering.guidance_update",
        "fk_lambda": "steering.fk_lambda",
        "fk_resampling_interval": "steering.fk_resampling_interval",
        # Solver parameters
        "learning_rate": "adaptive_solver.learning_rate",
        "solver_type": "adaptive_solver.type",
        "max_iterations": "adaptive_solver.max_iterations",
        "convergence_threshold": "adaptive_solver.convergence_threshold",
        "gradient_clip_norm": "adaptive_solver.gradient_clip_norm",
        "line_search": "adaptive_solver.line_search",
        # Model parameters
        "model_version": "model.version",
        "checkpoint_path": "model.checkpoint_path",
        "device": "model.device",
        # Output parameters
        "output_dir": "output_dir",
        "name": "name",
        # Substructure parameters
        "substructure_selection": "substructure.selection",
        "substructure_enabled": "substructure.enabled",
    }

    return mapping.get(key, key)


def infer_schedule_type(config_dict: Dict[str, Any]) -> Optional[str]:
    """Infer schedule type from configuration fields.
    
    Parameters
    ----------
    config_dict : Dict[str, Any]
        Dictionary that may represent a schedule configuration
        
    Returns
    -------
    Optional[str]
        Inferred schedule type or None if not a schedule
    """
    if not isinstance(config_dict, dict):
        return None
    
    # Check for exponential with bounds
    if all(field in config_dict for field in ["start", "end", "alpha", "start_t", "end_t"]):
        return "exponential_bounds"
    
    # Check for basic exponential
    if all(field in config_dict for field in ["start", "end", "alpha"]):
        return "exponential"
    
    # Check for piecewise step
    if "thresholds" in config_dict and "values" in config_dict:
        return "piecewise_step"
    
    # Check for piecewise 
    if "breakpoints" in config_dict and "values" in config_dict:
        return "piecewise"
    
    # Check for resolution scaling
    if "resolution_schedule" in config_dict and "reference_resolution" in config_dict:
        return "resolution_scaling"
    
    return None


def parse_schedule_configs(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse schedule configurations in nested config data.

    Recursively finds and converts schedule configurations to ParameterSchedule objects.

    Parameters
    ----------
    config_data : Dict[str, Any]
        Configuration data that may contain schedule configurations

    Returns
    -------
    Dict[str, Any]
        Configuration data with schedule configs converted to ParameterSchedule objects
    """
    result = {}

    for key, value in config_data.items():
        if isinstance(value, dict):
            # Check if this looks like a schedule config
            if "type" in value and isinstance(value["type"], str):
                try:
                    result[key] = parse_schedule_config(value)
                    continue
                except ValueError:
                    # Not a valid schedule config, treat as nested dict
                    pass
            # Check for implicit schedule based on field patterns
            else:
                inferred_type = infer_schedule_type(value)
                if inferred_type:
                    try:
                        # Add implicit type for schedule
                        schedule_config = dict(value)
                        schedule_config["type"] = inferred_type
                        result[key] = parse_schedule_config(schedule_config)
                        continue
                    except ValueError:
                        # Not a valid schedule config, treat as nested dict
                        pass

            # Recursively process nested dictionaries
            result[key] = parse_schedule_configs(value)
        else:
            result[key] = value

    return result


def save_config(config: ExperimentConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to save
    output_path : Union[str, Path]
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary and handle special types
    config_dict = asdict(config)
    config_dict = convert_schedules_to_dict(config_dict)

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def convert_schedules_to_dict(data: Any) -> Any:
    """Convert ParameterSchedule objects to dictionaries for YAML serialization.

    Parameters
    ----------
    data : Any
        Data that may contain ParameterSchedule objects

    Returns
    -------
    Any
        Data with ParameterSchedule objects converted to dictionaries
    """
    if isinstance(data, ParameterSchedule):
        # Convert schedule to dict representation
        schedule_dict = asdict(data)
        schedule_dict["type"] = data.__class__.__name__.replace("Config", "").lower()
        return schedule_dict
    elif isinstance(data, dict):
        return {k: convert_schedules_to_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_schedules_to_dict(item) for item in data]
    else:
        return data


def create_experiment_config(config_data: Dict[str, Any]) -> ExperimentConfig:
    """Create ExperimentConfig with proper nested dataclass construction.

    Parameters
    ----------
    config_data : Dict[str, Any]
        Configuration data dictionary

    Returns
    -------
    ExperimentConfig
        Constructed configuration object
    """
    from .config_schema import (
        StructureConfig,
        DensityConfig,
        ModelConfig,
        DiffusionConfig,
        SteeringConfig,
        AdaptiveSolverConfig,
        DensityGuidanceConfig,
        SubstructureConfig,
        OptimizationConfig,
        PotentialConfig,
    )

    # Create nested configuration objects
    nested_configs = {}

    if "structure" in config_data:
        nested_configs["structure"] = StructureConfig(**config_data["structure"])

    if "density" in config_data:
        nested_configs["density"] = DensityConfig(**config_data["density"])

    if "model" in config_data:
        nested_configs["model"] = ModelConfig(**config_data["model"])

    if "diffusion" in config_data:
        nested_configs["diffusion"] = DiffusionConfig(**config_data["diffusion"])

    if "steering" in config_data:
        nested_configs["steering"] = SteeringConfig(**config_data["steering"])

    if "adaptive_solver" in config_data:
        nested_configs["adaptive_solver"] = AdaptiveSolverConfig(
            **config_data["adaptive_solver"]
        )

    if "density_guidance" in config_data:
        nested_configs["density_guidance"] = DensityGuidanceConfig(
            **config_data["density_guidance"]
        )

    if "substructure" in config_data:
        nested_configs["substructure"] = SubstructureConfig(
            **config_data["substructure"]
        )

    if "optimization" in config_data:
        nested_configs["optimization"] = OptimizationConfig(
            **config_data["optimization"]
        )

    if "potentials" in config_data:
        nested_configs["potentials"] = PotentialConfig(**config_data["potentials"])

    # Create main config with nested objects
    main_config_data = {k: v for k, v in config_data.items() if k not in nested_configs}
    main_config_data.update(nested_configs)

    return ExperimentConfig(**main_config_data)


def load_batch_config(
    config_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None
) -> BatchExperimentConfig:
    """Load batch experiment configuration from YAML file with optional overrides.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to YAML configuration file
    overrides : Optional[Dict[str, Any]], optional
        Dictionary of parameter overrides from command line, by default None

    Returns
    -------
    BatchExperimentConfig
        Loaded and validated batch configuration

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If configuration is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML file
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        raise ValueError(f"Empty or invalid YAML file: {config_path}")

    # Apply command-line overrides (but don't merge into shared_config yet)
    if overrides:
        # For batch configs, we need to handle overrides differently
        # Store them to apply to individual experiments later
        pass

    # Parse schedule configurations
    config_data = parse_schedule_configs(config_data)

    # Create BatchExperimentConfig object with proper nested dataclass construction
    try:
        config = create_batch_experiment_config(config_data)
    except TypeError as e:
        raise ValueError(f"Invalid batch configuration: {e}")
    
    # Apply overrides to shared_config if they exist
    if overrides and config.shared_config:
        # Apply overrides to the shared config
        shared_config_dict = asdict(config.shared_config)
        shared_config_dict = merge_overrides(shared_config_dict, overrides)
        shared_config_dict = parse_schedule_configs(shared_config_dict)
        config.shared_config = create_experiment_config(shared_config_dict)

    # Validate configuration
    errors = config.validate()
    if errors:
        error_msg = "Batch configuration validation errors:\n" + "\n".join(
            f"  - {err}" for err in errors
        )
        raise ValueError(error_msg)

    return config


def create_batch_experiment_config(config_data: Dict[str, Any]) -> BatchExperimentConfig:
    """Create BatchExperimentConfig with proper nested dataclass construction.

    Parameters
    ----------
    config_data : Dict[str, Any]
        Configuration data dictionary

    Returns
    -------
    BatchExperimentConfig
        Constructed batch configuration object
    """
    # Handle shared_config if present
    shared_config = None
    if "shared_config" in config_data:
        shared_config = create_experiment_config(config_data["shared_config"])

    # Store raw experiment YAML data for proper merging
    experiment_yaml_data = config_data.get("experiments", [])

    # Get valid fields for BatchExperimentConfig
    from .config_schema import BatchExperimentConfig
    valid_fields = set(BatchExperimentConfig.__dataclass_fields__.keys())
    
    # Create main batch config with only valid fields
    batch_config_data = {k: v for k, v in config_data.items() 
                        if k in valid_fields and k not in ["shared_config", "experiments"]}
    batch_config_data["shared_config"] = shared_config
    batch_config_data["experiments"] = []  # Will be populated via get_experiment_configs_from_yaml

    # Create batch config
    batch_config = BatchExperimentConfig(**batch_config_data)
    
    # Store the raw YAML data for later use
    batch_config._experiment_yaml_data = experiment_yaml_data
    
    return batch_config


def detect_config_type(config_path: Union[str, Path]) -> str:
    """Detect whether a config file is for single or batch experiments.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to YAML configuration file

    Returns
    -------
    str
        "single" for single experiment configs, "batch" for batch configs

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML file
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        raise ValueError(f"Empty or invalid YAML file: {config_path}")

    # Check for batch-specific fields
    batch_indicators = ["protein_directory", "experiments", "shared_config", "output_base_dir"]
    
    if any(key in config_data for key in batch_indicators):
        return "batch"
    else:
        return "single"


def save_batch_config(config: BatchExperimentConfig, output_path: Union[str, Path]) -> None:
    """Save batch configuration to YAML file.

    Parameters
    ----------
    config : BatchExperimentConfig
        Batch configuration to save
    output_path : Union[str, Path]
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary and handle special types
    config_dict = asdict(config)
    config_dict = convert_schedules_to_dict(config_dict)

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
