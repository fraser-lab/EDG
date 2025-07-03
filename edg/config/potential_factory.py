"""Factory for creating potential objects from configuration.

This module provides utilities to create potential objects with
proper scheduling from configuration specifications.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

from .config_schema import ExperimentConfig, DensityGuidanceConfig
from .schedules import ParameterSchedule
from edg.edg.modules.potentials import DensityPotential, SubstructurePotential, get_potentials
from edg.edg.modules.density.density import XMap_torch
from edg.qfit.volume import XMap


def create_potentials_from_config(
    config: ExperimentConfig,
    xmap: XMap_torch,
    elements: Any,
    b_factors: Any,
    occupancies: Any,
    scattering_params: Any,
    atom_selection: Optional[np.ndarray] = None
) -> List[Any]:
    """Create potential objects from experiment configuration.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    xmap : XMap_torch
        Density map object
    elements : Any
        Atomic elements tensor
    b_factors : Any
        B-factors tensor
    occupancies : Any
        Occupancies tensor
    scattering_params : Any
        Scattering parameters
    atom_selection : Optional[np.ndarray]
        Optional atom selection for density potential
        
    Returns
    -------
    List[Any]
        List of potential objects
    """
    potentials = []
    
    # Create density potential
    density_potential = create_density_potential(
        config.density_guidance,
        xmap,
        elements,
        b_factors,
        occupancies,
        scattering_params,
        config.density.em_mode,
        atom_selection
    )
    potentials.append(density_potential)
    
    # Create substructure potential if enabled
    if config.substructure.enabled and config.substructure.selection:
        substructure_potential = create_substructure_potential(
            config,
            atom_selection  # This will be processed to get reference coords
        )
        potentials.append(substructure_potential)
    
    # Add default potentials if requested
    if config.potentials.use_default_potentials:
        default_potentials = get_potentials()
        potentials.extend(default_potentials)
    
    return potentials


def create_density_potential(
    guidance_config: DensityGuidanceConfig,
    xmap: XMap_torch,
    elements: Any,
    b_factors: Any,
    occupancies: Any,
    scattering_params: Any,
    em_mode: bool,
    atom_selection: Optional[np.ndarray] = None
) -> DensityPotential:
    """Create density potential from guidance configuration.
    
    Parameters
    ----------
    guidance_config : DensityGuidanceConfig
        Density guidance configuration
    xmap : XMap_torch
        Density map object
    elements : Any
        Atomic elements tensor
    b_factors : Any
        B-factors tensor
    occupancies : Any
        Occupancies tensor
    scattering_params : Any
        Scattering parameters
    em_mode : bool
        Whether to use electron microscopy mode
    atom_selection : Optional[np.ndarray]
        Optional atom selection
        
    Returns
    -------
    DensityPotential
        Configured density potential
    """
    # Convert schedule configs to actual schedules
    guidance_weight = (
        guidance_config.base_weight.to_schedule()
        if isinstance(guidance_config.base_weight, ParameterSchedule)
        else guidance_config.base_weight
    )
    
    resampling_weight = (
        guidance_config.resampling_weight.to_schedule()
        if isinstance(guidance_config.resampling_weight, ParameterSchedule)
        else guidance_config.resampling_weight
    )
    
    max_guidance_denoising_ratio = (
        guidance_config.max_guidance_denoising_ratio.to_schedule()
        if isinstance(guidance_config.max_guidance_denoising_ratio, ParameterSchedule)
        else guidance_config.max_guidance_denoising_ratio
    )
    
    parameters = {
        "guidance_interval": guidance_config.guidance_interval,
        "guidance_weight": guidance_weight,
        "resampling_weight": resampling_weight,
        "elements": elements,
        "b_factors": b_factors,
        "occupancies": occupancies,
        "scattering_params": scattering_params,
        "em": em_mode,
        "scale_guidance_to_denoising": guidance_config.scale_guidance_to_denoising,
        "max_guidance_denoising_ratio": max_guidance_denoising_ratio,
    }
    
    return DensityPotential(
        xmap=xmap,
        parameters=parameters,
        atom_selection=atom_selection,
    )


def create_substructure_potential(
    config: ExperimentConfig,
    reference_coords: Any
) -> SubstructurePotential:
    """Create substructure potential from configuration.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    reference_coords : Any
        Reference coordinates for substructure constraint
        
    Returns
    -------
    SubstructurePotential
        Configured substructure potential
    """
    # Parse selection string to get atom indices
    # For now, we'll pass an empty array and let the potential handle it
    # In a full implementation, you'd parse the selection string here
    selection_indices = np.array([], dtype=int)
    
    parameters = {
        "guidance_interval": 1,
        "guidance_weight": config.substructure.guidance_weight,
        "resampling_weight": config.substructure.resampling_weight,
        "buffer": config.substructure.buffer,
        "denoising_selection": selection_indices,
        "reference_coords": reference_coords,
    }
    
    return SubstructurePotential(parameters=parameters)