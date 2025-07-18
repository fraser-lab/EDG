"""Configuration schema definitions for EDG experiments.

This module defines the complete configuration schema using dataclasses
that can be serialized to/from YAML and validated for correctness.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from .schedules import ParameterSchedule, ConstantScheduleConfig


@dataclass
class ModelConfig:
    """Configuration for the diffusion model."""

    version: str = "boltz2"  # "boltz1" or "boltz2"
    checkpoint_path: Optional[str] = None  # Auto-detect if None
    ccd_path: Optional[str] = None  # Auto-detect if None
    device: Optional[str] = None  # Auto-detect if None
    pre_loaded_model: Optional[object] = (
        None  # Pre-loaded model instance for parallel execution
    )


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
    physical_guidance_update: bool = True
    contact_guidance_update: bool = False
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
    resolution: Optional[Union[float, ParameterSchedule]] = None
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


@dataclass
class StructureConfig:
    """Configuration for input structure."""

    structure_path: str
    clean_structure: bool = True
    keep_type: str = "protein"  # "protein", "all", etc.
    remove_alternative_conformations: bool = True
    complete_residues: bool = True
    remove_all_ligands: bool = True  # Remove all non-protein ligands


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

    use_default_potentials: bool = True
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
    boltz_input_yaml: Optional[str] = None  # Optional path to existing Boltz input YAML
    shared_input_dir: Optional[str] = (
        None  # Optional path to shared input directory for reusing processed data
    )

    # Optional configurations with defaults
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    adaptive_solver: AdaptiveSolverConfig = field(default_factory=AdaptiveSolverConfig)
    density_guidance: DensityGuidanceConfig = field(
        default_factory=DensityGuidanceConfig
    )
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
        if (
            self.structure.structure_path is not None
            and not Path(self.structure.structure_path).exists()
        ):
            errors.append(f"Structure file not found: {self.structure.structure_path}")

        if (
            self.density.map_path is not None
            and not Path(self.density.map_path).exists()
        ):
            errors.append(f"Density map file not found: {self.density.map_path}")

        # Check density resolution for CCP4/MRC files
        if self.density.map_path is not None:
            map_ext = Path(self.density.map_path).suffix.lower()
            if map_ext in [".ccp4", ".mrc", ".map"] and self.density.resolution is None:
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


@dataclass
class BatchExperimentConfig:
    """Configuration for running multiple experiments in batch."""

    # Required configurations
    name: str

    # Batch-specific configuration
    experiments: List[ExperimentConfig] = field(default_factory=list)

    # Optional: Auto-discovery configuration
    protein_directory: Optional[str] = (
        None  # Directory containing protein subdirectories
    )
    structure_pattern: str = "{protein_id}_main.cif"  # Pattern for structure files
    density_pattern: str = "{protein_id}_main_2.0A.ccp4"  # Pattern for density files

    # Shared configuration that applies to all experiments
    shared_config: Optional[ExperimentConfig] = None

    # Batch execution configuration
    output_base_dir: str = "results/batch"
    input_base_dir: str = "input/batch"
    continue_on_error: bool = True
    max_parallel: int = 1  # Number of parallel experiments (uses multiple GPUs)

    def validate(self) -> List[str]:
        """Validate the batch configuration and return any errors.

        Returns
        -------
        List[str]
            List of validation error messages. Empty if valid.
        """
        errors = []

        # Check that we have experiments or auto-discovery config
        has_experiments = self.experiments or (
            hasattr(self, "_experiment_yaml_data") and self._experiment_yaml_data
        )
        if not has_experiments and not self.protein_directory:
            errors.append(
                "Must specify either 'experiments' list or 'protein_directory' for auto-discovery"
            )

        # Validate protein directory if specified
        if self.protein_directory:
            if not Path(self.protein_directory).exists():
                errors.append(f"Protein directory not found: {self.protein_directory}")
            elif not Path(self.protein_directory).is_dir():
                errors.append(
                    f"Protein directory is not a directory: {self.protein_directory}"
                )

        # Validate shared config if provided (skip file existence checks for placeholders and null values)
        if self.shared_config:
            shared_errors = self.shared_config.validate()
            # Filter out file existence errors for placeholder paths AND null values
            filtered_shared_errors = [
                error
                for error in shared_errors
                if not (
                    ("not found" in error)
                    and ("placeholder" in error or "None" in error)
                )
            ]
            if filtered_shared_errors:
                errors.extend(
                    [
                        f"Shared config error: {error}"
                        for error in filtered_shared_errors
                    ]
                )

        # Validate individual experiments
        for i, experiment in enumerate(self.experiments):
            exp_errors = experiment.validate()
            if exp_errors:
                errors.extend(
                    [
                        f"Experiment {i} ({experiment.name}): {error}"
                        for error in exp_errors
                    ]
                )

        # Check parallel execution setting
        if self.max_parallel < 1:
            errors.append("max_parallel must be at least 1")

        return errors

    def get_experiment_configs(self) -> List[ExperimentConfig]:
        """Get all experiment configurations, including auto-discovered ones.

        Returns
        -------
        List[ExperimentConfig]
            List of all experiment configurations ready for execution
        """
        configs = []

        # Use YAML-based merging if available (preferred method)
        if hasattr(self, "_experiment_yaml_data") and self._experiment_yaml_data:
            configs.extend(
                self.get_experiment_configs_from_yaml(self._experiment_yaml_data)
            )
        else:
            # Fallback to dataclass-based merging for legacy support
            for experiment in self.experiments:
                merged_config = self._merge_with_shared_config(experiment)
                configs.append(merged_config)

        # Add auto-discovered experiments with shared config merging
        if self.protein_directory:
            discovered_experiments = self._discover_protein_experiments()
            for experiment in discovered_experiments:
                merged_config = self._merge_with_shared_config(experiment)
                configs.append(merged_config)

        return configs

    def get_experiment_configs_from_yaml(
        self, experiment_yaml_data: List[Dict[str, Any]]
    ) -> List[ExperimentConfig]:
        """Get experiment configurations from raw YAML data with proper shared config merging.

        This method works with the original YAML data to avoid dataclass default contamination.

        Parameters
        ----------
        experiment_yaml_data : List[Dict[str, Any]]
            List of experiment configuration dictionaries from YAML

        Returns
        -------
        List[ExperimentConfig]
            List of all experiment configurations ready for execution
        """
        configs = []

        # Merge each experiment YAML with shared config
        for exp_yaml in experiment_yaml_data:
            merged_config = self._merge_yaml_with_shared_config(exp_yaml)
            configs.append(merged_config)

        return configs

    def _merge_yaml_with_shared_config(
        self, experiment_yaml: Dict[str, Any]
    ) -> ExperimentConfig:
        """Merge experiment YAML data with shared config, preserving explicit values only.

        This method ensures that only explicitly specified values in the experiment YAML
        override the shared config, avoiding dataclass default contamination.

        Parameters
        ----------
        experiment_yaml : Dict[str, Any]
            Raw experiment configuration from YAML

        Returns
        -------
        ExperimentConfig
            Merged configuration with proper priority handling
        """
        if self.shared_config is None:
            # No shared config, just create from YAML
            from .config_loader import parse_schedule_configs, create_experiment_config

            parsed_yaml = parse_schedule_configs(experiment_yaml)
            return create_experiment_config(parsed_yaml)

        # Convert shared config to dict for merging
        from dataclasses import asdict

        shared_dict = asdict(self.shared_config)

        # Use shared config as base, only override with explicitly set experiment values
        merged_dict = self._deep_merge_dicts(shared_dict, experiment_yaml)

        # Parse schedule configurations in the merged dict
        from .config_loader import parse_schedule_configs, create_experiment_config

        merged_dict = parse_schedule_configs(merged_dict)

        return create_experiment_config(merged_dict)

    def _merge_with_shared_config(
        self, experiment_config: ExperimentConfig
    ) -> ExperimentConfig:
        """Merge an experiment config with shared config (legacy method).

        This method is kept for backward compatibility and auto-discovered experiments.

        Parameters
        ----------
        experiment_config : ExperimentConfig
            Individual experiment configuration

        Returns
        -------
        ExperimentConfig
            Experiment configuration merged with shared config
        """
        if self.shared_config is None:
            return experiment_config

        # Convert both configs to dictionaries for merging
        from dataclasses import asdict

        shared_dict = asdict(self.shared_config)
        experiment_dict = asdict(experiment_config)

        # Simple deep merge - shared config provides defaults, experiment config overrides
        merged_dict = self._deep_merge_dicts(shared_dict, experiment_dict)

        # Parse schedule configurations in the merged dict
        from .config_loader import parse_schedule_configs, create_experiment_config

        merged_dict = parse_schedule_configs(merged_dict)

        return create_experiment_config(merged_dict)

    def _deep_merge_dicts(
        self, base_dict: Dict[str, Any], override_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override_dict taking precedence.

        Parameters
        ----------
        base_dict : Dict[str, Any]
            Base dictionary (shared config)
        override_dict : Dict[str, Any]
            Override dictionary (individual experiment config)

        Returns
        -------
        Dict[str, Any]
            Merged dictionary
        """
        import copy

        result = copy.deepcopy(base_dict)

        for key, value in override_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Special handling for schedule configurations
                if self._is_schedule_config(value) or self._is_schedule_config(
                    result[key]
                ):
                    # Don't merge schedule configs - let experiment config fully override
                    result[key] = value
                else:
                    # Recursively merge nested dictionaries for non-schedule configs
                    result[key] = self._deep_merge_dicts(result[key], value)
            elif value is not None:  # Only override if experiment value is not None
                result[key] = value
            # If value is None, keep the base_dict value (don't override with None)

        return result

    def _is_schedule_config(self, config: Dict[str, Any]) -> bool:
        """Check if a dictionary looks like a schedule configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary to check

        Returns
        -------
        bool
            True if this looks like a schedule config
        """
        if not isinstance(config, dict):
            return False

        # Check for schedule-specific keys
        schedule_keys = [
            "value",
            "breakpoints",
            "values",
            "start",
            "end",
            "alpha",
            "thresholds",
        ]
        return any(key in config for key in schedule_keys)

    def _discover_protein_experiments(self) -> List[ExperimentConfig]:
        """Discover protein experiments from directory structure.

        Returns
        -------
        List[ExperimentConfig]
            List of auto-discovered experiment configurations
        """
        from pathlib import Path
        import copy

        experiments = []
        protein_dir = Path(self.protein_directory)

        # Find all protein subdirectories
        for protein_subdir in protein_dir.iterdir():
            if not protein_subdir.is_dir():
                continue

            protein_id = protein_subdir.name

            # Look for structure and density files
            structure_file = protein_subdir / self.structure_pattern.format(
                protein_id=protein_id
            )
            density_file = protein_subdir / self.density_pattern.format(
                protein_id=protein_id
            )

            if structure_file.exists() and density_file.exists():
                # Create experiment config
                if self.shared_config:
                    # Use shared config as base
                    exp_config = copy.deepcopy(self.shared_config)
                    exp_config.name = protein_id
                    exp_config.structure.structure_path = str(structure_file)
                    exp_config.density.map_path = str(density_file)
                    exp_config.output_dir = str(Path(self.output_base_dir) / protein_id)
                    exp_config.input_data_dir = str(
                        Path(self.input_base_dir) / protein_id
                    )
                else:
                    # Create minimal config
                    exp_config = ExperimentConfig(
                        name=protein_id,
                        structure=StructureConfig(structure_path=str(structure_file)),
                        density=DensityConfig(
                            map_path=str(density_file), resolution=2.0
                        ),
                        output_dir=str(Path(self.output_base_dir) / protein_id),
                        input_data_dir=str(Path(self.input_base_dir) / protein_id),
                    )

                experiments.append(exp_config)

        return experiments
