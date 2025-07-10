"""Utilities for managing shared input directories in parameter sweeps.

This module provides functions to create and validate shared input directories
that can be reused across parameter sweep runs to avoid redundant Boltz preprocessing.
"""

import os
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import yaml
import shutil

from edg.config import ExperimentConfig

logger = logging.getLogger(__name__)


def create_shared_input_key(config: ExperimentConfig) -> str:
    """Create a unique key for shared input data based on config contents.
    
    Parameters
    ----------
    config : ExperimentConfig
        Configuration to generate key for
        
    Returns
    -------
    str
        Unique key string for this configuration's shared data
    """
    # Key components that affect Boltz preprocessing
    key_components = {
        'structure_path': config.structure.structure_path,
        'model_version': config.model.version,
        'boltz_input_yaml': config.boltz_input_yaml,
        'structure_cleaning': {
            'clean_structure': config.structure.clean_structure,
            'keep_type': config.structure.keep_type,
            'remove_alternative_conformations': config.structure.remove_alternative_conformations,
            'complete_residues': config.structure.complete_residues,
            'remove_all_ligands': config.structure.remove_all_ligands,
        }
    }
    
    # Create hash from key components
    key_str = json.dumps(key_components, sort_keys=True)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    return f"{config.model.version}_{key_hash}"


def create_shared_input_directory(config: ExperimentConfig, base_results_dir: Path) -> Tuple[Path, Path]:
    """Create shared input directory structure for parameter sweeps.
    
    Parameters
    ----------
    config : ExperimentConfig
        Configuration to create shared directory for
    base_results_dir : Path
        Base results directory where shared input will be created
        
    Returns
    -------
    Tuple[Path, Path]
        Tuple of (shared_input_dir, shared_output_dir) paths
    """
    shared_key = create_shared_input_key(config)
    
    # Create shared directories
    shared_input_dir = base_results_dir / "shared_input"
    shared_output_dir = base_results_dir / "shared_output"
    
    shared_input_dir.mkdir(parents=True, exist_ok=True)
    shared_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata file
    metadata = {
        'shared_key': shared_key,
        'structure_path': config.structure.structure_path,
        'model_version': config.model.version,
        'boltz_input_yaml': config.boltz_input_yaml,
        'created_by': config.name,
        'structure_config': {
            'clean_structure': config.structure.clean_structure,
            'keep_type': config.structure.keep_type,
            'remove_alternative_conformations': config.structure.remove_alternative_conformations,
            'complete_residues': config.structure.complete_residues,
            'remove_all_ligands': config.structure.remove_all_ligands,
        }
    }
    
    metadata_file = shared_input_dir / "shared_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created shared input directory: {shared_input_dir}")
    logger.info(f"Created shared output directory: {shared_output_dir}")
    
    return shared_input_dir, shared_output_dir


def validate_shared_input_compatibility(config: ExperimentConfig, shared_input_dir: Path) -> bool:
    """Validate that existing shared input directory is compatible with current config.
    
    Parameters
    ----------
    config : ExperimentConfig
        Configuration to validate against
    shared_input_dir : Path
        Path to shared input directory to validate
        
    Returns
    -------
    bool
        True if compatible, False otherwise
    """
    try:
        metadata_file = shared_input_dir / "shared_metadata.json"
        if not metadata_file.exists():
            logger.warning(f"No metadata file found in shared input directory: {shared_input_dir}")
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check key compatibility
        current_key = create_shared_input_key(config)
        stored_key = metadata.get('shared_key')
        
        if current_key != stored_key:
            logger.warning(f"Shared input key mismatch: {current_key} vs {stored_key}")
            return False
            
        # Check structure file exists and matches
        stored_structure_path = metadata.get('structure_path')
        if stored_structure_path != config.structure.structure_path:
            logger.warning(f"Structure path mismatch: {config.structure.structure_path} vs {stored_structure_path}")
            return False
            
        if not Path(config.structure.structure_path).exists():
            logger.error(f"Structure file not found: {config.structure.structure_path}")
            return False
            
        # Check model version compatibility
        stored_model_version = metadata.get('model_version')
        if stored_model_version != config.model.version:
            logger.warning(f"Model version mismatch: {config.model.version} vs {stored_model_version}")
            return False
            
        # Check Boltz input YAML compatibility
        stored_boltz_yaml = metadata.get('boltz_input_yaml')
        if stored_boltz_yaml != config.boltz_input_yaml:
            logger.warning(f"Boltz input YAML mismatch: {config.boltz_input_yaml} vs {stored_boltz_yaml}")
            return False
            
        logger.info("Shared input directory validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating shared input directory: {e}")
        return False


def check_shared_processed_data(shared_output_dir: Path) -> bool:
    """Check if shared output directory contains processed Boltz data.
    
    Parameters
    ----------
    shared_output_dir : Path
        Path to shared output directory
        
    Returns
    -------
    bool
        True if processed data exists, False otherwise
    """
    try:
        processed_dir = shared_output_dir / "processed"
        if not processed_dir.exists():
            logger.debug(f"No processed directory found: {processed_dir}")
            return False
            
        # Check for key files that indicate successful processing
        required_files = [
            "manifest.json",
        ]
        
        for file_name in required_files:
            file_path = processed_dir / file_name
            if not file_path.exists():
                logger.debug(f"Missing required processed file: {file_path}")
                return False
                
        logger.info(f"Found existing processed data in: {processed_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error checking processed data: {e}")
        return False


def copy_boltz_input_to_shared(config: ExperimentConfig, shared_input_dir: Path) -> Path:
    """Copy Boltz input YAML to shared input directory.
    
    Parameters
    ----------
    config : ExperimentConfig
        Configuration containing Boltz input YAML path
    shared_input_dir : Path
        Shared input directory path
        
    Returns
    -------
    Path
        Path to copied YAML file in shared directory
    """
    if not config.boltz_input_yaml:
        raise ValueError("No Boltz input YAML specified in config")
        
    source_yaml = Path(config.boltz_input_yaml)
    if not source_yaml.exists():
        raise FileNotFoundError(f"Boltz input YAML not found: {source_yaml}")
        
    # Copy to shared directory with consistent naming
    target_yaml = shared_input_dir / f"{config.name}_shared.yaml"
    shutil.copy2(source_yaml, target_yaml)
    
    logger.info(f"Copied Boltz input YAML to shared directory: {target_yaml}")
    return target_yaml


def create_config_with_shared_input(config: ExperimentConfig, shared_input_dir: Path, 
                                   shared_output_dir: Path, run_output_dir: Path) -> ExperimentConfig:
    """Create a new config that uses shared input directory.
    
    Parameters
    ----------
    config : ExperimentConfig
        Original configuration
    shared_input_dir : Path
        Shared input directory path
    shared_output_dir : Path
        Shared output directory path (for Boltz processing)
    run_output_dir : Path
        Run-specific output directory
        
    Returns
    -------
    ExperimentConfig
        New configuration with shared input settings
    """
    # Create a copy of the config
    import copy
    new_config = copy.deepcopy(config)
    
    # Update paths to use shared directories
    new_config.input_data_dir = str(shared_input_dir)
    new_config.output_dir = str(shared_output_dir)  # Boltz processes here
    new_config.shared_input_dir = str(shared_input_dir)
    
    # The actual run outputs will be handled separately by experiment_runner
    # to move them to run_output_dir
    
    return new_config