"""Experiment runner for EDG density-guided diffusion experiments.

This module provides the main experiment execution logic that:
- Creates and configures the optimizer from config
- Processes the structure according to config
- Sets up potentials and schedules  
- Runs the optimization
- Saves results and logs
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import copy
import numpy as np
import matplotlib.pyplot as plt

from edg.config import ExperimentConfig
from edg.config.optimizer_factory import (
    create_optimizer_from_config,
    process_structure_from_config, 
    prepare_optimization_kwargs_from_config
)
from edg.edg.optimizer import DensityGuidedDiffusion
from edg.data.structure import Structure, Ensemble
from edg.utils.utility import try_gpu


logger = logging.getLogger(__name__)


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a complete EDG experiment from configuration.
    
    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment configuration
        
    Returns
    -------
    Dict[str, Any]
        Experiment results including final structures and scores
    """
    logger.info(f"Running experiment: {config.name}")
    
    # Create output directories
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_data_dir = Path(config.input_data_dir)
    input_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the final configuration used
    config_output_path = output_dir / "experiment_config.yaml"
    from edg.config import save_config
    save_config(config, config_output_path)
    logger.info(f"Saved experiment configuration to {config_output_path}")
    
    # Create input YAML file for the optimizer
    input_yaml_path = create_input_yaml(config, input_data_dir)
    
    # Initialize the optimizer
    logger.info("Initializing optimizer...")
    optimizer = create_optimizer_from_config(config, input_yaml_path)
    
    # Process the structure
    logger.info("Processing input structure...")
    optimizer = process_structure_from_config(optimizer, config)
    
    # Create output directory for this specific run
    run_output_dir = create_run_output_dir(config, output_dir)
    
    # Prepare optimization arguments
    optimize_kwargs = prepare_optimization_kwargs_from_config(config, run_output_dir, optimizer)
    
    # Run the optimization
    logger.info(f"Starting optimization with {config.diffusion.num_steps} steps...")
    final_structures, scores = optimizer.optimize(**optimize_kwargs)
    
    # Save results
    results = save_results(final_structures, scores, run_output_dir, config)
    
    logger.info(f"Experiment completed. Results saved to {run_output_dir}")
    
    return results


def create_input_yaml(config: ExperimentConfig, input_data_dir: Path) -> Path:
    """Create the input YAML file required by the optimizer.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    input_data_dir : Path
        Directory for input data files
        
    Returns
    -------
    Path
        Path to created input YAML file
    """
    yaml_path = input_data_dir / f"{config.name}.yaml"
    
    # Create minimal YAML content (the optimizer mainly needs this for the path)
    yaml_content = f"""# Input data configuration for {config.name}
experiment_name: {config.name}
structure_path: {config.structure.structure_path}
density_path: {config.density.map_path}
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.debug(f"Created input YAML at {yaml_path}")
    return yaml_path




def create_run_output_dir(config: ExperimentConfig, base_output_dir: Path) -> Path:
    """Create output directory for this specific run.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    base_output_dir : Path
        Base output directory
        
    Returns
    -------
    Path
        Specific run output directory
    """
    # Create descriptive directory name
    dir_name_parts = [
        config.model.version,
        f"{config.density.resolution}A" if config.density.resolution else "auto_res",
        f"{config.diffusion.num_steps}steps",
        f"{config.steering.num_particles}particles" if config.steering.enabled else "no_steering",
    ]
    
    if config.adaptive_solver.type != "none":
        dir_name_parts.append(f"{config.adaptive_solver.type}_solver")
        dir_name_parts.append(f"lr{config.adaptive_solver.learning_rate}")
    
    dir_name = "_".join(dir_name_parts)
    run_output_dir = base_output_dir / dir_name
    
    # Handle directory conflicts
    counter = 1
    original_dir = run_output_dir
    while run_output_dir.exists():
        run_output_dir = original_dir.parent / f"{original_dir.name}_{counter}"
        counter += 1
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run output directory: {run_output_dir}")
    
    return run_output_dir




def save_results(
    final_structures: List[Structure], 
    scores: List[np.ndarray], 
    output_dir: Path,
    config: ExperimentConfig
) -> Dict[str, Any]:
    """Save experiment results.
    
    Parameters
    ----------
    final_structures : List[Structure]
        Final optimized structures
    scores : List[np.ndarray]
        Optimization scores per step
    output_dir : Path
        Output directory
    config : ExperimentConfig
        Experiment configuration
        
    Returns
    -------
    Dict[str, Any]
        Summary of saved results
    """
    results = {
        "final_structures": final_structures,
        "scores": scores,
        "output_dir": str(output_dir),
        "num_structures": len(final_structures),
        "num_steps": len(scores),
    }
    
    # Save scores plot
    if config.optimization.save_scores and scores:
        logger.debug("Saving scores plot")
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(scores)), scores)
        plt.xlabel("Diffusion Step")
        plt.ylabel("Score")
        plt.title(f"Score per Diffusion Step - {config.name}")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "scores.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "scores.pdf", bbox_inches="tight")
        plt.close()
        
        # Save scores as CSV
        np.savetxt(output_dir / "scores.csv", scores, delimiter=",", header="step,score")
    
    # Log final statistics
    if scores:
        final_score = scores[-1]
        initial_score = scores[0] if len(scores) > 0 else None
        logger.info(f"Final score: {final_score:.6f}")
        if initial_score is not None:
            logger.info(f"Score improvement: {initial_score:.6f} → {final_score:.6f}")
        
        results["final_score"] = float(final_score)
        results["initial_score"] = float(initial_score) if initial_score is not None else None
    
    return results