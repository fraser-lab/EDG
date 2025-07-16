"""Command-line interface for EDG (Ensembles from Density Generator).

This module provides the main CLI entry point that supports:
- YAML configuration files
- Command-line parameter overrides
- Helpful error messages and validation
- Experiment execution and logging
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
import yaml

from edg.config import (
    load_config,
    load_batch_config,
    detect_config_type,
    save_config,
    save_batch_config,
    ExperimentConfig,
    BatchExperimentConfig,
)
from edg.experiment_runner import run_experiment
from edg.batch_runner import run_batch_experiment


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="edg",
        description="EDG (Ensembles from Density Generator) - Density-guided diffusion for atomic model optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  edg --config configs/ptp1b.yaml
  
  # With parameter overrides
  edg --config configs/ptp1b.yaml --num-steps 300 --resolution 1.5
  
  # Override complex parameters
  edg --config configs/mac1.yaml --guidance-weight 0.8 --output-dir results/test1
  
  # Validation only
  edg --config configs/ptp1b.yaml --validate-only
  
Parameter override syntax:
  - Simple parameters: --num-steps 300
  - Nested parameters: --density-guidance.base-weight 0.5
  - Boolean flags: --substructure-enabled (sets to True)
  
Common parameters:
  --num-steps N             Number of diffusion steps
  --resolution R            Map resolution in Angstroms  
  --ensemble-size N         Ensemble size
  --guidance-weight W       Density guidance weight
  --num-particles N         Number of steering particles
  --learning-rate LR        Adaptive solver learning rate
  --output-dir DIR          Output directory
        """,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    # Optional arguments
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration, don't run experiment",
    )

    parser.add_argument(
        "--save-config",
        type=str,
        help="Save the final configuration (after overrides) to this file",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-essential output"
    )

    parser.add_argument(
        "--batch", 
        action="store_true", 
        help="Enable batch processing mode (auto-detected if not specified)"
    )

    # Common parameter overrides
    optimization_group = parser.add_argument_group("Optimization Parameters")
    optimization_group.add_argument(
        "--num-steps", type=int, help="Number of diffusion steps"
    )
    optimization_group.add_argument(
        "--ensemble-size", type=int, help="Number of ensemble members"
    )
    optimization_group.add_argument(
        "--step-scale", type=float, help="Diffusion step scale factor"
    )

    density_group = parser.add_argument_group("Density Parameters")
    density_group.add_argument(
        "--resolution", type=float, help="Map resolution in Angstroms"
    )
    density_group.add_argument("--map-path", type=str, help="Path to density map file")
    density_group.add_argument(
        "--em-mode", action="store_true", help="Use electron microscopy mode"
    )

    structure_group = parser.add_argument_group("Structure Parameters")
    structure_group.add_argument(
        "--structure-path", type=str, help="Path to input structure file"
    )

    guidance_group = parser.add_argument_group("Guidance Parameters")
    guidance_group.add_argument(
        "--guidance-weight", type=float, help="Density guidance weight"
    )
    guidance_group.add_argument(
        "--resampling-weight", type=float, help="Particle resampling weight"
    )
    guidance_group.add_argument(
        "--max-guidance-denoising-ratio",
        type=float,
        help="Maximum guidance to denoising ratio",
    )
    guidance_group.add_argument(
        "--guidance-interval", type=int, help="Guidance application interval"
    )

    steering_group = parser.add_argument_group("Steering Parameters")
    steering_group.add_argument(
        "--num-particles", type=int, help="Number of steering particles"
    )
    steering_group.add_argument(
        "--guidance-update", action="store_true", help="Enable guidance updates"
    )
    steering_group.add_argument(
        "--no-guidance-update", action="store_true", help="Disable guidance updates"
    )
    steering_group.add_argument(
        "--fk-lambda", type=float, help="FK lambda parameter for steering"
    )
    steering_group.add_argument(
        "--fk-resampling-interval", type=int, help="FK resampling interval"
    )

    solver_group = parser.add_argument_group("Adaptive Solver Parameters")
    solver_group.add_argument(
        "--learning-rate", type=float, help="Solver learning rate"
    )
    solver_group.add_argument(
        "--solver-type", choices=["adam", "simple", "none"], help="Adaptive solver type"
    )
    solver_group.add_argument(
        "--max-iterations", type=int, help="Maximum solver iterations"
    )
    solver_group.add_argument(
        "--convergence-threshold",
        type=float,
        help="Convergence threshold for early stopping",
    )
    solver_group.add_argument(
        "--gradient-clip-norm", type=float, help="Gradient clipping norm"
    )
    solver_group.add_argument(
        "--line-search", action="store_true", help="Enable backtracking line search"
    )

    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument(
        "--model-version", choices=["boltz1", "boltz2"], help="Model version"
    )
    model_group.add_argument(
        "--checkpoint-path", type=str, help="Path to model checkpoint"
    )
    model_group.add_argument(
        "--device", type=str, help="Compute device (cpu, cuda, etc.)"
    )

    output_group = parser.add_argument_group("Output Parameters")
    output_group.add_argument("--output-dir", type=str, help="Output directory")
    output_group.add_argument("--input-data-dir", type=str, help="Input data directory")
    output_group.add_argument("--shared-input-dir", type=str, help="Shared input directory for parameter sweeps")
    output_group.add_argument("--name", type=str, help="Experiment name")

    substructure_group = parser.add_argument_group("Substructure Parameters")
    substructure_group.add_argument(
        "--substructure-selection", type=str, help="Substructure selection string"
    )
    substructure_group.add_argument(
        "--substructure-enabled",
        action="store_true",
        help="Enable substructure conditioning",
    )

    # Allow additional override parameters using special syntax
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        metavar="KEY=VALUE",
        help="Override any parameter using KEY=VALUE syntax (can be used multiple times)",
    )

    return parser


def parse_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Parse command-line arguments into override dictionary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    Dict[str, Any]
        Override parameters
    """
    overrides = {}

    # Standard overrides (excluding boolean flags)
    param_mapping = {
        "num_steps": args.num_steps,
        "ensemble_size": args.ensemble_size,
        "step_scale": args.step_scale,
        "resolution": args.resolution,
        "map_path": args.map_path,
        "structure_path": args.structure_path,
        "guidance_weight": args.guidance_weight,
        "resampling_weight": args.resampling_weight,
        "max_guidance_denoising_ratio": args.max_guidance_denoising_ratio,
        "guidance_interval": args.guidance_interval,
        "num_particles": args.num_particles,
        "fk_lambda": args.fk_lambda,
        "fk_resampling_interval": args.fk_resampling_interval,
        "learning_rate": args.learning_rate,
        "solver_type": args.solver_type,
        "max_iterations": args.max_iterations,
        "convergence_threshold": args.convergence_threshold,
        "gradient_clip_norm": args.gradient_clip_norm,
        "model_version": args.model_version,
        "checkpoint_path": args.checkpoint_path,
        "device": args.device,
        "output_dir": args.output_dir,
        "input_data_dir": args.input_data_dir,
        "shared_input_dir": args.shared_input_dir,
        "name": args.name,
        "substructure_selection": args.substructure_selection,
    }

    # Add non-None values
    for key, value in param_mapping.items():
        if value is not None:
            overrides[key] = value

    # Handle boolean flags - only add if explicitly set to True
    if args.em_mode:
        overrides["em_mode"] = True
    if args.line_search:
        overrides["line_search"] = True
    if args.substructure_enabled:
        overrides["substructure_enabled"] = True

    # Handle boolean flags that need special logic
    if args.guidance_update:
        overrides["guidance_update"] = True
    elif args.no_guidance_update:
        overrides["guidance_update"] = False

    # Parse additional override parameters
    if args.override:
        for override_str in args.override:
            if "=" not in override_str:
                raise ValueError(
                    f"Invalid override format: {override_str}. Use KEY=VALUE"
                )

            key, value = override_str.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Try to parse value as appropriate type
            parsed_value = parse_override_value(value)
            overrides[key] = parsed_value

    return {k: v for k, v in overrides.items() if v is not None}


def parse_override_value(value_str: str) -> Any:
    """Parse a string override value to appropriate Python type.

    Parameters
    ----------
    value_str : str
        String value to parse

    Returns
    -------
    Any
        Parsed value
    """
    value_str = value_str.strip()

    # Boolean values
    if value_str.lower() in ("true", "yes", "1", "on"):
        return True
    elif value_str.lower() in ("false", "no", "0", "off"):
        return False

    # Try numeric values
    try:
        if "." in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass

    # Try YAML parsing for complex values
    try:
        return yaml.safe_load(value_str)
    except (yaml.YAMLError, ValueError):
        pass

    # Return as string
    return value_str


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Set up logging configuration.

    Parameters
    ----------
    verbose : bool
        Enable verbose logging
    quiet : bool
        Suppress non-essential output
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main CLI entry point.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger(__name__)

    try:
        # Parse command-line overrides
        overrides = parse_overrides(args)

        # Auto-detect or use explicit batch mode
        if args.batch:
            config_type = "batch"
        else:
            config_type = detect_config_type(args.config)

        # Load configuration based on type
        if config_type == "batch":
            logger.info(f"Loading batch configuration from {args.config}")
            config = load_batch_config(args.config, overrides)
            
            logger.info(f"Loaded batch experiment: {config.name}")
            experiment_configs = config.get_experiment_configs()
            logger.info(f"Found {len(experiment_configs)} experiments in batch")
            
            if overrides:
                logger.info(f"Applied overrides: {list(overrides.keys())}")

            # Save final configuration if requested
            if args.save_config:
                logger.info(f"Saving final batch configuration to {args.save_config}")
                save_batch_config(config, args.save_config)

            # Validation only mode
            if args.validate_only:
                logger.info("Batch configuration validation passed!")
                print("✓ Batch configuration is valid")
                print(f"✓ Found {len(experiment_configs)} experiments")
                return 0

            # Run batch experiment
            logger.info("Starting batch experiment...")
            results = run_batch_experiment(config)

            logger.info("Batch experiment completed!")
            print(f"✓ Batch '{config.name}' completed")
            print(f"✓ {results['completed_experiments']}/{results['total_experiments']} experiments successful")
            print(f"✓ Results saved to: {config.output_base_dir}")
            
            if results['failed_experiments'] > 0:
                print(f"⚠ {results['failed_experiments']} experiments failed")
                return 1

        else:
            # Single experiment mode
            logger.info(f"Loading configuration from {args.config}")
            config = load_config(args.config, overrides)

            logger.info(f"Loaded experiment: {config.name}")
            if overrides:
                logger.info(f"Applied overrides: {list(overrides.keys())}")

            # Save final configuration if requested
            if args.save_config:
                logger.info(f"Saving final configuration to {args.save_config}")
                save_config(config, args.save_config)

            # Validation only mode
            if args.validate_only:
                logger.info("Configuration validation passed!")
                print("✓ Configuration is valid")
                return 0

            # Run experiment
            logger.info("Starting experiment...")
            results = run_experiment(config)

            logger.info("Experiment completed successfully!")
            print(f"✓ Experiment '{config.name}' completed")
            print(f"✓ Results saved to: {config.output_dir}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"✗ Configuration error: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        print("\n✗ Experiment interrupted", file=sys.stderr)
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
