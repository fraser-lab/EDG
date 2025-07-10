"""Factory for creating optimizers from configuration.

This module provides utilities to create properly configured
DensityGuidedDiffusion optimizers from ExperimentConfig objects.
"""

from pathlib import Path
from typing import Optional, Any, Dict
import logging

from .config_schema import ExperimentConfig
from edg.edg.optimizer import DensityGuidedDiffusion
from edg.utils.utility import try_gpu
from boltz.main import BoltzSteeringParams
from edg.edg.modules.adaptive_solver import (
    AdaptiveSolverConfig as AdaptiveSolverConfigClass,
)


logger = logging.getLogger(__name__)


def create_optimizer_from_config(
    config: ExperimentConfig, input_yaml_path: Path
) -> DensityGuidedDiffusion:
    """Create DensityGuidedDiffusion optimizer from configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment configuration
    input_yaml_path : Path
        Path to input YAML file for the optimizer

    Returns
    -------
    DensityGuidedDiffusion
        Configured optimizer instance
    """
    logger.debug(f"Creating optimizer with model version: {config.model.version}")

    # Create steering args
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = config.steering.enabled
    steering_args.guidance_update = config.steering.guidance_update
    steering_args.num_particles = config.steering.num_particles

    # Create adaptive solver config if enabled
    adaptive_solver_config = None
    if config.adaptive_solver.type != "none":
        adaptive_solver_config = AdaptiveSolverConfigClass(
            learning_rate=config.adaptive_solver.learning_rate,
            max_iterations=config.adaptive_solver.max_iterations,
            convergence_threshold=config.adaptive_solver.convergence_threshold,
            gradient_clip_norm=config.adaptive_solver.gradient_clip_norm,
            per_potential_scaling=config.adaptive_solver.per_potential_scaling,
            line_search=config.adaptive_solver.line_search,
            beta1=config.adaptive_solver.beta1,
            beta2=config.adaptive_solver.beta2,
            eps=config.adaptive_solver.eps,
            line_search_c1=config.adaptive_solver.line_search_c1,
            line_search_backtrack=config.adaptive_solver.line_search_backtrack,
            max_line_search_steps=config.adaptive_solver.max_line_search_steps,
            adaptive_line_search=config.adaptive_solver.adaptive_line_search,
            adaptive_backtrack_min=config.adaptive_solver.adaptive_backtrack_min,
            adaptive_backtrack_max=config.adaptive_solver.adaptive_backtrack_max,
            violation_scaling=config.adaptive_solver.violation_scaling,
        )

    # Set checkpoint path
    checkpoint_path = config.model.checkpoint_path
    if checkpoint_path is None:
        if config.model.version == "boltz1":
            checkpoint_path = str(Path("~/.boltz/boltz1_conf.ckpt").expanduser())
        else:
            checkpoint_path = str(Path("~/.boltz/boltz2_conf.ckpt").expanduser())

    # Set CCD path
    ccd_path = config.model.ccd_path
    if ccd_path is None:
        ccd_path = str(Path("~/.boltz/ccd.pkl").expanduser())

    # Set device
    device = config.model.device
    if device is None:
        device = try_gpu()

    # Create optimizer
    optimizer = DensityGuidedDiffusion(
        input_path=input_yaml_path,
        y=config.density.map_path,
        structure=config.structure.structure_path,
        output_path=config.output_dir,
        em=config.density.em_mode,
        resolution=config.density.resolution,
        step_scale=config.diffusion.step_scale,
        ckpt_path=Path(checkpoint_path),
        model_version=config.model.version,
        ccd_path=Path(ccd_path),
        device=device,
        adaptive_solver=config.adaptive_solver.type,
        adaptive_solver_config=adaptive_solver_config,
        steering_args=steering_args,
        config=config,
    )

    return optimizer


def process_structure_from_config(
    optimizer: DensityGuidedDiffusion, config: ExperimentConfig
) -> DensityGuidedDiffusion:
    """Process structure according to configuration.

    Parameters
    ----------
    optimizer : DensityGuidedDiffusion
        Optimizer with loaded structure
    config : ExperimentConfig
        Experiment configuration

    Returns
    -------
    DensityGuidedDiffusion
        Optimizer with processed structure
    """
    structure_config = config.structure

    # Apply structure processing steps
    if structure_config.remove_alternative_conformations:
        logger.debug("Removing alternative conformations")
        optimizer.structure = optimizer.structure.remove_alternative_conformations()
        optimizer.structure = optimizer.structure.reorder()
        optimizer.structure.build_hierarchy()

    if structure_config.clean_structure:
        logger.debug(f"Cleaning structure (keep_type: {structure_config.keep_type})")
        optimizer.structure = optimizer.structure.clean_structure(
            keep_type=structure_config.keep_type,
            remove_all_ligands=structure_config.remove_all_ligands,
        )
        optimizer.structure = optimizer.structure.reorder()
        optimizer.structure.build_hierarchy()

    if structure_config.complete_residues:
        logger.debug("Completing residues")
        optimizer.structure = optimizer.structure.complete_residues()
        optimizer.structure = optimizer.structure.reorder()
        optimizer.structure.build_hierarchy()

    # Final processing
    optimizer.structure = optimizer.structure.reorder().extract(
        optimizer.structure.select("active", True)
    )

    num_atoms = optimizer.structure.active.sum()
    logger.info(f"Processed structure: {num_atoms} active atoms")

    return optimizer


def prepare_optimization_kwargs_from_config(
    config: ExperimentConfig, output_dir: Path, optimizer: DensityGuidedDiffusion
) -> Dict[str, Any]:
    """Prepare optimization keyword arguments from configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    output_dir : Path
        Output directory for this run
    optimizer : DensityGuidedDiffusion
        Configured optimizer

    Returns
    -------
    Dict[str, Any]
        Keyword arguments for optimizer.optimize()
    """
    kwargs = {
        "output_dir": str(output_dir),
        "num_steps": config.diffusion.num_steps,
        "ensemble_size": config.optimization.ensemble_size,
        "partial_diffusion": config.optimization.partial_diffusion,
        "steering": config.steering.enabled,
        "representation_noise_scale": config.optimization.representation_noise_scale,
    }

    # Add substructure conditioning if enabled
    if config.substructure.enabled and config.substructure.selection:
        # Parse selection string to get actual atom indices
        try:
            selector = optimizer.structure.select(config.substructure.selection)
            kwargs["substructure_conditioning_kwargs"] = {
                "selection": selector,
            }
            logger.info(
                f"Enabled substructure conditioning for {config.substructure.selection}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to parse substructure selection '{config.substructure.selection}': {e}"
            )

    # Add diffusion kwargs for partial diffusion
    if config.optimization.partial_diffusion:
        diffusion_kwargs = {}
        if config.optimization.noising_steps is not None:
            diffusion_kwargs["noising_steps"] = config.optimization.noising_steps
        else:
            # Auto-set to num_steps//4
            diffusion_kwargs["noising_steps"] = config.diffusion.num_steps // 4
        kwargs["diffusion_kwargs"] = diffusion_kwargs

    return kwargs
