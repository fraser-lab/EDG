#!/usr/bin/env python3
"""
Generate batch configuration YAML file for altloc substructure conditioning experiments.

This script reads an altloc_summary.csv file and generates a batch configuration
that runs experiments with different substructure conditioning settings based on
the CSV data.
"""

import argparse
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any
import sys

# Add the edg package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from edg.config import load_config


def load_base_template(template_path: str) -> Dict[str, Any]:
    """Load the base template configuration from a YAML file.

    Parameters
    ----------
    template_path : str
        Path to the template YAML configuration file

    Returns
    -------
    Dict[str, Any]
        Template configuration dictionary
    """
    template_config = load_config(template_path)

    # Convert to dictionary format suitable for batch configuration
    template_dict = {
        "name": None,  # Required field, will be overridden
        "output_dir": None,  # Required field, will be overridden
        "input_data_dir": None,  # Required field, will be overridden
        "structure": {
            "structure_path": None,  # Required field, will be overridden
            "clean_structure": template_config.structure.clean_structure,
            "keep_type": template_config.structure.keep_type,
            "remove_all_ligands": getattr(
                template_config.structure, "remove_all_ligands", False
            ),
            "remove_alternative_conformations": template_config.structure.remove_alternative_conformations,
            "complete_residues": template_config.structure.complete_residues,
        },
        "density": {
            "map_path": None,  # Required field, will be overridden
            "resolution": template_config.density.resolution,
            "em_mode": template_config.density.em_mode,
        },
        "model": {
            "version": template_config.model.version,
            "checkpoint_path": getattr(template_config.model, "checkpoint_path", None),
        },
        "diffusion": {
            "num_steps": template_config.diffusion.num_steps,
            "step_scale": template_config.diffusion.step_scale,
            "noise_scale": template_config.diffusion.noise_scale,
            "gamma_0": template_config.diffusion.gamma_0,
            "gamma_min": template_config.diffusion.gamma_min,
        },
        "steering": {
            "enabled": template_config.steering.enabled,
            "guidance_update": template_config.steering.guidance_update,
            "num_particles": template_config.steering.num_particles,
            "fk_resampling_interval": template_config.steering.fk_resampling_interval,
            "fk_lambda": template_config.steering.fk_lambda,
            "num_gd_steps": template_config.steering.num_gd_steps,
        },
        "adaptive_solver": {
            "type": template_config.adaptive_solver.type,
            "learning_rate": template_config.adaptive_solver.learning_rate,
            "max_iterations": template_config.adaptive_solver.max_iterations,
            "convergence_threshold": template_config.adaptive_solver.convergence_threshold,
            "gradient_clip_norm": template_config.adaptive_solver.gradient_clip_norm,
            "per_potential_scaling": template_config.adaptive_solver.per_potential_scaling,
            "line_search": template_config.adaptive_solver.line_search,
            "adaptive_line_search": getattr(
                template_config.adaptive_solver, "adaptive_line_search", False
            ),
            "beta1": getattr(template_config.adaptive_solver, "beta1", 0.9),
            "beta2": getattr(template_config.adaptive_solver, "beta2", 0.999),
            "eps": getattr(template_config.adaptive_solver, "eps", 1e-8),
        },
        "density_guidance": {
            "base_weight": template_config.density_guidance.base_weight,
            "guidance_interval": template_config.density_guidance.guidance_interval,
            "resampling_weight": template_config.density_guidance.resampling_weight,
            "scale_guidance_to_denoising": template_config.density_guidance.scale_guidance_to_denoising,
            "max_guidance_denoising_ratio": template_config.density_guidance.max_guidance_denoising_ratio,
            "resolution": getattr(template_config.density_guidance, "resolution", None),
        },
        "substructure": {
            "enabled": True,  # Will be enabled for all experiments
            "guidance_weight": template_config.substructure.guidance_weight,
            "resampling_weight": template_config.substructure.resampling_weight,
            "buffer": template_config.substructure.buffer,
        },
        "optimization": {
            "ensemble_size": template_config.optimization.ensemble_size,
            "partial_diffusion": template_config.optimization.partial_diffusion,
            "save_interval": template_config.optimization.save_interval,
            "save_maps": template_config.optimization.save_maps,
            "save_scores": template_config.optimization.save_scores,
        },
        "potentials": {
            "use_default_potentials": template_config.potentials.use_default_potentials,
        },
    }

    # Handle complex parameter schedules by converting them to dictionaries
    from dataclasses import asdict, is_dataclass

    def convert_schedule_to_dict(schedule):
        """Convert a parameter schedule to a dictionary representation."""
        if is_dataclass(schedule):
            # Use asdict to convert dataclass to dictionary
            return asdict(schedule)
        elif isinstance(schedule, list):
            # Handle lists that might contain schedule objects
            return [convert_schedule_to_dict(item) for item in schedule]
        else:
            return schedule

    # Convert complex schedules to dictionaries
    template_dict["density_guidance"]["base_weight"] = convert_schedule_to_dict(
        template_config.density_guidance.base_weight
    )
    template_dict["density_guidance"]["resampling_weight"] = convert_schedule_to_dict(
        template_config.density_guidance.resampling_weight
    )

    if (
        hasattr(template_config.density_guidance, "resolution")
        and template_config.density_guidance.resolution is not None
    ):
        template_dict["density_guidance"]["resolution"] = convert_schedule_to_dict(
            template_config.density_guidance.resolution
        )

    return template_dict


def generate_experiment_config(
    row: pd.Series, structure_base_path: str, output_base_dir: str, input_base_dir: str
) -> Dict[str, Any]:
    """Generate a minimal experiment configuration with only overrides.

    Parameters
    ----------
    row : pd.Series
        Row from the altloc_summary.csv file
    structure_base_path : str
        Base path for structure files
    output_base_dir : str
        Base output directory
    input_base_dir : str
        Base input directory

    Returns
    -------
    Dict[str, Any]
        Minimal experiment configuration with only overrides
    """
    pdb_code = row["pdb_code"].lower()
    target_chain = row["target_chain"]
    segment_start = int(row["segment_start"])
    segment_end = int(row["segment_end"])

    # Set density map path (use synthetic density output if available)
    synthetic_density_path = (
        f"synthetic_density_output/{pdb_code}/{pdb_code}_main_2.0A.ccp4"
    )
    if Path(synthetic_density_path).exists():
        density_map_path = synthetic_density_path
    else:
        # Fallback to structure-based path (would need to be generated)
        density_map_path = f"{structure_base_path}/{pdb_code}_2.0A.ccp4"

    # Create minimal experiment config with only overrides
    exp_config = {
        "name": pdb_code,
        "structure": {"structure_path": f"{structure_base_path}/{pdb_code}.cif"},
        "density": {
            "map_path": density_map_path,
            "resolution": 2.0,  # Add explicit resolution for .ccp4 files
        },
        "output_dir": f"{output_base_dir}/{pdb_code}",
        "input_data_dir": f"{input_base_dir}/{pdb_code}",
        "substructure": {
            "enabled": True,
            "selection": f"chain {target_chain} and resi {segment_start}-{segment_end}",
        },
    }

    return exp_config


def main():
    """Main function to generate batch configuration."""
    parser = argparse.ArgumentParser(
        description="Generate batch configuration for altloc substructure conditioning experiments"
    )
    parser.add_argument("--csv", required=True, help="Path to altloc_summary.csv file")
    parser.add_argument(
        "--template",
        required=True,
        help="Path to base template configuration YAML file",
    )
    parser.add_argument(
        "--output", required=True, help="Path to output batch configuration YAML file"
    )
    parser.add_argument(
        "--structure-base-path",
        default="tests/resources/altloc_data/mmcif_files",
        help="Base path for structure files (default: tests/resources/altloc_data/mmcif_files)",
    )
    parser.add_argument(
        "--output-base-dir",
        default="results/synthetic_density_test",
        help="Base output directory (default: results/synthetic_density_test)",
    )
    parser.add_argument(
        "--input-base-dir",
        default="input/synthetic_density_test",
        help="Base input directory (default: input/synthetic_density_test)",
    )
    parser.add_argument(
        "--batch-name",
        default="synthetic_density_test",
        help="Name for the batch experiment (default: synthetic_density_test)",
    )

    args = parser.parse_args()

    # Load CSV data
    print(f"Loading CSV data from {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Found {len(df)} proteins in CSV")

    # Load base template
    print(f"Loading template from {args.template}")
    template = load_base_template(args.template)

    # Generate experiments
    print("Generating experiment configurations...")
    experiments = []

    for _, row in df.iterrows():
        try:
            exp_config = generate_experiment_config(
                row, args.structure_base_path, args.output_base_dir, args.input_base_dir
            )
            experiments.append(exp_config)
            print(
                f"  Generated config for {row['pdb_code']}: chain {row['target_chain']} resi {row['segment_start']}-{row['segment_end']}"
            )
        except Exception as e:
            print(f"  Error generating config for {row['pdb_code']}: {e}")
            continue

    # Create batch configuration using shared_config approach
    batch_config = {
        "name": args.batch_name,
        "output_base_dir": args.output_base_dir,
        "input_base_dir": args.input_base_dir,
        "continue_on_error": True,
        "max_parallel": 1,
        "shared_config": template,
        "experiments": experiments,
    }

    # Write output file
    print(f"Writing batch configuration to {args.output}")
    with open(args.output, "w") as f:
        yaml.dump(batch_config, f, default_flow_style=False, indent=2, sort_keys=False)

    print(f"Generated batch configuration with {len(experiments)} experiments")
    print("\nTo run the batch:")
    print(f"  pixi run python -m edg --config {args.output}")
    print("\nTo validate the configuration:")
    print(f"  pixi run python -m edg --config {args.output} --validate-only")


if __name__ == "__main__":
    main()
