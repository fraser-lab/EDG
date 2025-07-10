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
import yaml

from edg.config import ExperimentConfig
from edg.config.optimizer_factory import (
    create_optimizer_from_config,
    process_structure_from_config, 
    prepare_optimization_kwargs_from_config
)
from edg.edg.optimizer import DensityGuidedDiffusion
from edg.data.structure import Structure, Ensemble
from edg.utils.utility import try_gpu

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from edg.edg.optimizer import DensityGuidedDiffusion
    from edg.data.structure import Structure


logger = logging.getLogger(__name__)


def validate_boltz_yaml_compatibility(boltz_yaml_path: Path, structure_path: str = None, model_config = None) -> bool:
    """Validate that the Boltz YAML is properly formatted and can be processed.
    
    Parameters
    ----------
    boltz_yaml_path : Path
        Path to the Boltz input YAML file
    structure_path : str, optional
        Path to the structure file (not used in current implementation)
    model_config : ModelConfig, optional
        Model configuration for Boltz processing (not used in current implementation)
        
    Returns
    -------
    bool
        True if compatible, False otherwise
    """
    try:
        # First, do basic YAML format validation
        with open(boltz_yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        if not isinstance(yaml_data, dict) or 'sequences' not in yaml_data:
            logger.error("Boltz YAML must contain a 'sequences' key")
            return False
        
        if not isinstance(yaml_data['sequences'], list) or len(yaml_data['sequences']) == 0:
            logger.error("Boltz YAML 'sequences' must be a non-empty list")
            return False
        
        # Check that sequences are properly formatted
        for i, seq_entry in enumerate(yaml_data['sequences']):
            if not isinstance(seq_entry, dict) or ('protein' not in seq_entry and 'ligand' not in seq_entry):
                logger.error(f"Sequence entry {i} must have a 'protein' or 'ligand' key")
                return False
                
            # Handle protein sequences
            if 'protein' in seq_entry:
                protein = seq_entry['protein']
                if not isinstance(protein, dict) or 'id' not in protein or 'sequence' not in protein:
                    logger.error(f"Protein sequence entry {i} must have 'id' and 'sequence' keys")
                    return False
                    
                if not isinstance(protein['sequence'], str) or len(protein['sequence']) == 0:
                    logger.error(f"Protein sequence entry {i} must have a non-empty sequence string")
                    return False
                    
            # Handle ligand sequences
            elif 'ligand' in seq_entry:
                ligand = seq_entry['ligand']
                if not isinstance(ligand, dict) or 'id' not in ligand:
                    logger.error(f"Ligand sequence entry {i} must have 'id' key")
                    return False
                
                # Check for either 'ccd' or 'smiles' key
                if 'ccd' not in ligand and 'smiles' not in ligand:
                    logger.error(f"Ligand sequence entry {i} must have either 'ccd' or 'smiles' key")
                    return False
                
                # Validate ccd if present
                if 'ccd' in ligand:
                    if not isinstance(ligand['ccd'], str) or len(ligand['ccd']) == 0:
                        logger.error(f"Ligand sequence entry {i} must have a non-empty ccd string")
                        return False
                
                # Validate smiles if present
                if 'smiles' in ligand:
                    if not isinstance(ligand['smiles'], str) or len(ligand['smiles']) == 0:
                        logger.error(f"Ligand sequence entry {i} must have a non-empty smiles string")
                        return False
        
        # Count sequence types for logging
        protein_count = sum(1 for seq in yaml_data['sequences'] if 'protein' in seq)
        ligand_count = sum(1 for seq in yaml_data['sequences'] if 'ligand' in seq)
        
        logger.info(f"Boltz YAML format validation passed with {len(yaml_data['sequences'])} sequences "
                   f"({protein_count} protein, {ligand_count} ligand)")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate Boltz YAML: {e}")
        return False


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
    
    # Create output directory for this specific run
    run_output_dir = create_run_output_dir(config, output_dir)
    
    # Save the final configuration used to the run-specific directory
    config_output_path = run_output_dir / "experiment_config.yaml"
    from edg.config import save_config
    save_config(config, config_output_path)
    logger.info(f"Saved experiment configuration to {config_output_path}")
    
    # Initialize the optimizer first (we need the structure to create proper Boltz YAML)
    logger.info("Initializing optimizer...")
    
    # Use existing Boltz input YAML if provided, otherwise create one after optimizer init
    if config.boltz_input_yaml:
        logger.info(f"Using existing Boltz input YAML: {config.boltz_input_yaml}")
        input_yaml_path = Path(config.boltz_input_yaml)
        
        # Validate that the file exists
        if not input_yaml_path.exists():
            raise FileNotFoundError(f"Boltz input YAML file not found: {input_yaml_path}")
        
        # Validate that the Boltz YAML is compatible with the structure
        logger.info("Validating Boltz YAML compatibility with structure...")
        if not validate_boltz_yaml_compatibility(input_yaml_path, config.structure.structure_path, config.model):
            raise ValueError("Boltz YAML is not compatible with the structure file. "
                           "Please check that the sequences in the YAML match the structure.")
        
        # Ensure the input_data_dir exists but is clean when using existing Boltz YAML
        input_data_dir.mkdir(parents=True, exist_ok=True)
        
        # If the existing Boltz YAML is not in the input_data_dir, copy it there
        # This ensures Boltz only processes the correct file
        if input_yaml_path.parent != input_data_dir:
            target_yaml_path = input_data_dir / f"{config.name}.yaml"
            logger.info(f"Copying existing Boltz YAML to {target_yaml_path}")
            
            # Copy the existing YAML to the input directory
            import shutil
            shutil.copy2(input_yaml_path, target_yaml_path)
            input_yaml_path = target_yaml_path
        else:
            # Clean up any other YAML files in the directory to avoid conflicts
            for yaml_file in input_data_dir.glob("*.yaml"):
                if yaml_file != input_yaml_path:
                    logger.debug(f"Removing conflicting YAML file: {yaml_file}")
                    yaml_file.unlink()
                    
        # Create optimizer with the validated Boltz YAML
        optimizer = create_optimizer_from_config(config, input_yaml_path)
        
        # Load the structure directly (no sequence extraction needed)
        logger.info("Loading structure from file...")
        extension = os.path.splitext(config.structure.structure_path)[1]
        if extension not in (".cif", ".pdb", ".mmcif"):
            raise ValueError("Structure file must be in mmCIF or PDB format.")
        if extension in (".pdb",):
            optimizer.structure = Structure.fromfile(config.structure.structure_path)
        else:
            optimizer.structure = Ensemble.fromfile(config.structure.structure_path)[0]  # Take first model
        
        logger.info("Processing input structure...")
        optimizer = process_structure_from_config(optimizer, config)
            
    else:
        # No Boltz YAML provided - warn user and fallback to sequence extraction
        logger.warning("No Boltz input YAML provided. Falling back to extracting sequences from structure file.")
        logger.warning("This may cause tensor size mismatches if the structure has unexpected atom counts.")
        logger.warning("Consider providing a Boltz YAML file for better compatibility.")
        
        # Create temporary optimizer with minimal YAML to get structure
        temp_yaml_path = create_temp_yaml(config, input_data_dir)
        temp_optimizer = create_optimizer_from_config(config, temp_yaml_path)
        
        # Process structure to extract sequence information
        temp_optimizer = process_structure_from_config(temp_optimizer, config)
        
        # Create proper Boltz YAML with sequences
        input_yaml_path = create_boltz_yaml_with_sequences(config, input_data_dir, temp_optimizer)
        
        # Clean up temporary optimizer
        del temp_optimizer
        
        # Create final optimizer with extracted sequences
        optimizer = create_optimizer_from_config(config, input_yaml_path)
        
        # Process the structure
        logger.info("Processing input structure...")
        optimizer = process_structure_from_config(optimizer, config)
    
    # Prepare optimization arguments
    optimize_kwargs = prepare_optimization_kwargs_from_config(config, run_output_dir, optimizer)
    
    # Run the optimization
    logger.info(f"Starting optimization with {config.diffusion.num_steps} steps...")
    final_structures, scores = optimizer.optimize(**optimize_kwargs)
    
    # Save results
    results = save_results(final_structures, scores, run_output_dir, config)
    
    logger.info(f"Experiment completed. Results saved to {run_output_dir}")
    
    return results


def create_temp_yaml(config: ExperimentConfig, input_data_dir: Path) -> Path:
    """Create temporary minimal YAML file for initial structure loading.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    input_data_dir : Path
        Directory for input data files
        
    Returns
    -------
    Path
        Path to created temporary YAML file
    """
    yaml_path = input_data_dir / f"{config.name}_temp.yaml"
    
    # Create minimal YAML content for structure loading
    yaml_content = f"""# Temporary input data configuration for {config.name}
experiment_name: {config.name}
structure_path: {config.structure.structure_path}
density_path: {config.density.map_path}
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.debug(f"Created temporary YAML at {yaml_path}")
    return yaml_path


def create_boltz_yaml_with_sequences(config: ExperimentConfig, input_data_dir: Path, optimizer: "DensityGuidedDiffusion") -> Path:
    """Create proper Boltz YAML file with sequences extracted from structure.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    input_data_dir : Path
        Directory for input data files
    optimizer : DensityGuidedDiffusion
        Optimizer with processed structure
        
    Returns
    -------
    Path
        Path to created Boltz YAML file
    """
    yaml_path = input_data_dir / f"{config.name}.yaml"
    
    # Extract sequences from processed structure
    sequences = extract_sequences_from_structure(optimizer.structure)
    
    # Create proper Boltz YAML format
    yaml_content = "sequences:\n"
    for seq_info in sequences:
        yaml_content += "  - protein:\n"
        yaml_content += f"      id: {seq_info['id']}\n"
        yaml_content += f"      sequence: {seq_info['sequence']}\n"
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Created Boltz YAML with {len(sequences)} sequences at {yaml_path}")
    return yaml_path


def extract_sequences_from_structure(structure: "Structure") -> List[Dict[str, str]]:
    """Extract protein sequences from structure for Boltz YAML format.
    
    Parameters
    ----------
    structure : Structure
        Processed structure object
        
    Returns
    -------
    List[Dict[str, str]]
        List of sequence dictionaries with 'id' and 'sequence' keys
    """
    sequences = []
    
    # Get unique chains from structure data
    chains = np.unique(structure.data['chain'])
    
    for chain in chains:
        # Filter to this chain and CA atoms
        chain_mask = (structure.data['chain'] == chain) & (structure.data['name'] == 'CA')
        if not chain_mask.any():
            continue
            
        # Get residue data for this chain
        chain_residue_numbers = structure.data['resi'][chain_mask]
        chain_residue_names = structure.data['resn'][chain_mask]
        
        # Create residue mapping
        residue_data = {}
        for res_num, res_name in zip(chain_residue_numbers, chain_residue_names):
            residue_data[res_num] = res_name
        
        # Sort by residue number and build sequence
        sorted_residues = sorted(residue_data.items())
        
        # Convert 3-letter amino acid codes to 1-letter
        aa_mapping = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        sequence = ""
        for res_num, res_name in sorted_residues:
            if res_name in aa_mapping:
                sequence += aa_mapping[res_name]
            else:
                logger.warning(f"Unknown residue {res_name} in chain {chain}, skipping")
        
        if sequence:
            sequences.append({
                'id': chain,
                'sequence': sequence
            })
    
    return sequences




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
        plt.close()
        
        # Save scores as CSV
        np.savetxt(output_dir / "scores.csv", scores, delimiter=",", header="step,score")
    
    # Log final statistics
    if scores:
        final_score = scores[-1]
        initial_score = scores[0] if len(scores) > 0 else None
        logger.info(f"Final score: {final_score.mean():.6f}")
        if initial_score is not None:
            logger.info(f"Score improvement: {initial_score.mean():.6f} → {final_score.mean():.6f}")
        
        results["final_score"] = float(final_score.mean())
        results["initial_score"] = float(initial_score.mean()) if initial_score is not None else None
    
    return results