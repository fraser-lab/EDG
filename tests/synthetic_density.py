"""Synthetic electron density map generation from atomic structures.

This module provides functionality to generate synthetic electron density maps from
atomic structures (mmCIF, PDB files, or Ensemble objects) with user-defined
experimental parameters.
"""

import os
from pathlib import Path
from typing import Optional, Union, Tuple, List, NamedTuple

import torch
import numpy as np

from adp3d.data import Structure
from adp3d.data.structure import Ensemble
from adp3d.qfit.volume import XMap, Resolution, GetSpaceGroup, GridParameters
from adp3d.qfit.unitcell import UnitCell
from adp3d.adp.modules.density import (
    DifferentiableTransformer,
    XMap_torch,
    DensityParameters,
)
from adp3d.data.io import structure_to_density_input
from adp3d.data.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
)
from adp3d.utils.utility import try_gpu


class SyntheticDensityGenerator:
    """Generate synthetic electron density maps from atomic structures.
    
    This class provides functionality to create synthetic electron density maps
    from atomic structures (Structure or Ensemble objects) with optional 
    experimental parameters. It supports both X-ray crystallography and electron 
    microscopy scattering factors.
    """
    
    def __init__(
        self,
        structure: Union[str, Structure, Ensemble],
        reference_map_file: Optional[str] = None,
        resolution: Optional[float] = None,
        unit_cell: Optional[NamedTuple] = None,
        em_mode: Optional[bool] = False,  
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize the synthetic density generator.
        
        Parameters
        ----------
        structure : Union[str, Structure, Ensemble]
            Path to a structure file (mmCIF or PDB format), Structure object, or Ensemble object.
        reference_map_file : str, optional
            Path to a reference map file (CCP4, MRC, or MTZ format) to use for grid parameters.
        resolution : float, optional
            Map resolution in Angstroms, required if no reference map is provided.
        unit_cell : NamedTuple, optional
            Unit cell parameters (a, b, c, alpha, beta, gamma) for the structure.
        em_mode : bool, optional
            Whether to use electron microscopy mode instead of X-ray crystallography.
        device : Optional[Union[str, torch.device]], optional
            Device to use for calculations ('cpu', 'cuda', etc.).
        """
        if device is None:
            self.device = try_gpu()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Load structure
        if isinstance(structure, str):
            self.structure = Structure.fromfile(structure)
            self.is_ensemble = False
        elif isinstance(structure, Structure):
            self.structure = structure
            self.is_ensemble = False
        elif isinstance(structure, Ensemble):
            self.structure = structure
            self.is_ensemble = True
        else:
            raise TypeError("Structure must be a file path, Structure object, or Ensemble object")
        
        self.em_mode = em_mode
        self.resolution = resolution
        
        self.scattering_factors = (
            ELECTRON_SCATTERING_FACTORS if em_mode else ATOM_STRUCTURE_FACTORS
        )
        
        if reference_map_file is not None:
            self._setup_from_reference(reference_map_file)
        elif resolution is not None:
            self._setup_from_parameters(resolution, unit_cell)
        else:
            raise ValueError("Either reference_map_file or resolution must be provided")
    
    def _setup_from_reference(self, reference_map_file: str) -> None:
        """Set up the generator using a reference map file.
        
        Parameters
        ----------
        reference_map_file : str
            Path to a reference map file (CCP4, MRC, or MTZ format).
        """
        extension = os.path.splitext(reference_map_file)[1]
        if extension not in (".ccp4", ".mrc", ".map", ".mtz"):
            raise ValueError("Reference map must be a CCP4, MRC, MAP, or MTZ file.")
        
        if extension in (".mtz"):
            try:
                ref_map = XMap.fromfile(reference_map_file)
            except:
                ref_map = XMap.fromfile(
                    reference_map_file, label="2FOFCWT,PH2FOFCWT"
                )  # try 2FO-FC map if initial fails
        else:
            if not hasattr(self, 'resolution') or self.resolution is None:
                raise ValueError("Resolution must be provided for CCP4, MRC, or MAP files.")
            ref_map = XMap.fromfile(reference_map_file, resolution=self.resolution)
        
        self.xmap = XMap_torch(ref_map, device=self.device)
        self._setup_scattering_params()
        self.density_calculator = DifferentiableTransformer(
            self.xmap,
            scattering_params=self.scattering_params,
            em=self.em_mode,
            device=self.device,
            # use_cuda_kernels=True,
        )
    
    def _setup_from_parameters(self, resolution: float, unit_cell: NamedTuple = None) -> None:
        """Set up the generator using specified parameters.
        
        Parameters
        ----------
        resolution : float
            Map resolution in Angstroms.
        """
        self.resolution = resolution
        
        # Get the unit cell from the first structure if using an ensemble
        structure_ref = self.structure[0] if self.is_ensemble else self.structure
        unit_cell = structure_ref.unit_cell if unit_cell is None else unit_cell
        
        # Create an appropriately-sized grid based on resolution
        grid_a = int(unit_cell.a / resolution * 4.0)
        grid_b = int(unit_cell.b / resolution * 4.0)
        grid_c = int(unit_cell.c / resolution * 4.0)
        
        empty_array = np.zeros((grid_a, grid_b, grid_c), dtype=np.float32)
        # original atom coordinates are at 0, 0, 0 for AF3 predictions, need to offset by half the unit cell
        offset = (
            unit_cell.a / resolution * 2.0,
            unit_cell.b / resolution * 2.0,
            unit_cell.c / resolution * 2.0,
        )

        ref_map = XMap(
            empty_array,
            grid_parameters=GridParameters(voxelspacing=resolution / 4, offset=offset),
            resolution=Resolution(high=resolution, low=1000.0),
            unit_cell=unit_cell,
        )
        
        self.xmap = XMap_torch(ref_map, device=self.device)
        self._setup_scattering_params()
        self.density_calculator = DifferentiableTransformer(
            self.xmap,
            scattering_params=self.scattering_params,
            em=self.em_mode,
            device=self.device,
            use_cuda_kernels=True,
        )
    
    def _setup_scattering_params(self) -> None:
        """Set up scattering parameters for density calculation."""
        # Get elements from the first structure if using an ensemble
        structure_ref = self.structure[0] if self.is_ensemble else self.structure
        unique_elements = set(structure_ref.e)
        
        # Normalize element names according to DensityGuidedDiffusion approach
        unique_elements = sorted(
            set(
                [
                    (
                        elem.upper()
                        if len(elem) == 1
                        else elem[0].upper() + elem[1:].lower()
                    )
                    for elem in unique_elements
                ]
            )
        )
        
        # Map elements to atomic numbers using the ATOMIC_NUM_TO_ELEMENT list
        atomic_num_dict = {
            elem: ATOMIC_NUM_TO_ELEMENT.index(elem) for elem in unique_elements
        }
        
        max_atomic_num = max(atomic_num_dict.values())
        # Use the max atomic number found in the structure for tensor size
        n_coeffs = len(self.scattering_factors["C"][0])
        dense_size = torch.Size([max_atomic_num + 1, n_coeffs, 2])
        scattering_dense_tensor = torch.zeros(dense_size, dtype=torch.float32, device=self.device)
        
        for elem in unique_elements:
            atomic_num = atomic_num_dict[elem]
            
            if elem in self.scattering_factors:
                factor = self.scattering_factors[elem]
            else:
                print(f"Warning: Scattering factors for {elem} not found, using C instead")
                factor = self.scattering_factors["C"]
            
            factor = torch.tensor(factor, dtype=torch.float32, device=self.device).T  # (2, range) -> (range, 2)
            scattering_dense_tensor[atomic_num, :, :] = factor
        
        self.scattering_params = scattering_dense_tensor
        self.atomic_num_dict = atomic_num_dict
    
    def generate_map_from_structure(
        self, 
        structure: Structure,
        b_factor_scale: float = 1.0,
        occupancy_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate a synthetic density map from a single structure.
        
        Parameters
        ----------
        structure : Structure
            Structure object to generate density from.
        b_factor_scale : float, optional
            Scale factor for B-factors.
        occupancy_scale : float, optional
            Scale factor for occupancies.
            
        Returns
        -------
        torch.Tensor
            Generated density map.
        """
        coords, elements, b_factors, occupancies, active, _ = structure_to_density_input(structure)
    
        coords = coords.to(self.device).float().unsqueeze(0)  # Add batch dim
        elements = elements.to(self.device).long().unsqueeze(0)  # Add batch dim
        b_factors = b_factors.to(self.device).float().unsqueeze(0) * b_factor_scale
        occupancies = occupancies.to(self.device).float().unsqueeze(0) * occupancy_scale
        active = active.to(self.device).bool().unsqueeze(0)
        
        with torch.no_grad():
            density_map = self.density_calculator(
                coords,
                elements,
                b_factors,
                occupancies,
                active,
            )
        
        return density_map.squeeze(0)  # Remove batch dimension
    
    def generate_map(
        self, 
        b_factor_scale: float = 1.0,
        occupancy_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate a synthetic density map from the structure or ensemble.
        
        Parameters
        ----------
        b_factor_scale : float, optional
            Scale factor for B-factors.
        occupancy_scale : float, optional
            Scale factor for occupancies.
            
        Returns
        -------
        torch.Tensor
            Generated density map.
        """
        if self.is_ensemble:
            # Generate map for each structure in the ensemble and sum them
            ensemble_map = None
            for structure in self.structure:
                structure_map = self.generate_map_from_structure(
                    structure, 
                    b_factor_scale, 
                    occupancy_scale / len(self.structure)  # Scale occupancy by ensemble size
                    # TODO: improve to use occupancies from the ensemble
                )
                
                if ensemble_map is None:
                    ensemble_map = structure_map
                else:
                    ensemble_map += structure_map
            
            self.last_density_map = ensemble_map
            return ensemble_map
        else:
            # Generate map from a single structure
            density_map = self.generate_map_from_structure(
                self.structure, 
                b_factor_scale, 
                occupancy_scale
            )
            
            self.last_density_map = density_map
            return density_map
    
    def save_map(
        self, 
        output_file: str, 
        density_map: Optional[torch.Tensor] = None,
        downsample_to: Optional[float] = None,
    ) -> None:
        """Save the generated density map to a file.
        
        Parameters
        ----------
        output_file : str
            Path to the output file (will be saved in CCP4 format).
        density_map : Optional[torch.Tensor], optional
            Density map to save, by default None (uses the last generated map).
        downsample_to : Optional[float], optional
            Downsample the density map to this resolution, by default None.
        """
        if density_map is None:
            if not hasattr(self, 'last_density_map'):
                raise ValueError("No density map available to save.")
            density_map = self.last_density_map
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        xmap = self.density_calculator.xmap
        xmap.array = density_map

        # Downsample the density map if requested
        if downsample_to is not None:
            xmap = xmap.downsample_to_resolution(downsample_to, apply_filter=True)
        
        xmap.tofile(output_file)
        print(f"Saved density map to {output_file}")


if __name__ == "__main__":
    # generate synthetic mac1 data
    # pdb_5sop = Structure.fromfile("/home/kchrispens/adp-replicate/tests/resources/mac1_synthetic/5SOP_modified.pdb")
    # pdb_5soq = Structure.fromfile("/home/kchrispens/adp-replicate/tests/resources/mac1_synthetic/5SOQ_modified.pdb")
    # pdb_5sq8 = Structure.fromfile("/home/kchrispens/adp-replicate/tests/resources/mac1_synthetic/5SQ8_modified.pdb")

    # ensemble = Ensemble([pdb_5sop, pdb_5soq, pdb_5sq8])
    # ref_map_file = "/home/kchrispens/adp-replication/tests/resources/mac1_synthetic/5soq-sf.mtz"

    # density_generator = SyntheticDensityGenerator(ensemble, ref_map_file)

    # density = density_generator.generate_map()

    # density_generator.save_map("/home/kchrispens/adp-replicate/tests/resources/mac1_synthetic/5sop_5soq_5sq8.ccp4", density)

    # generate synthetic AAAWAAA data
    pdb = Structure.fromfile("/home/kchrispens/adp-replicate/tests/resources/AAAWAAA/AAAWAAA_Waltconf.mmcif")

    unit_cell = UnitCell(30.0, 30.0, 30.0)

    density_generator = SyntheticDensityGenerator(structure=pdb, resolution=8.0, unit_cell=unit_cell, em_mode=False)
    density = density_generator.generate_map(b_factor_scale=1.0, occupancy_scale=1.0)
    density_generator.save_map("/home/kchrispens/adp-replicate/tests/resources/AAAWAAA/AAAWAAA_Waltconf_8.ccp4", density)

    density_generator = SyntheticDensityGenerator(structure=pdb, resolution=4.0, unit_cell=unit_cell, em_mode=False)
    density = density_generator.generate_map(b_factor_scale=1.0, occupancy_scale=1.0)
    density_generator.save_map("/home/kchrispens/adp-replicate/tests/resources/AAAWAAA/AAAWAAA_Waltconf_4.ccp4", density)

    density_generator = SyntheticDensityGenerator(structure=pdb, resolution=2.0, unit_cell=unit_cell, em_mode=False)
    density = density_generator.generate_map(b_factor_scale=1.0, occupancy_scale=1.0)
    density_generator.save_map("/home/kchrispens/adp-replicate/tests/resources/AAAWAAA/AAAWAAA_Waltconf_2.ccp4", density)

    density_generator = SyntheticDensityGenerator(structure=pdb, resolution=1.0, unit_cell=unit_cell, em_mode=False)
    density = density_generator.generate_map(b_factor_scale=1.0, occupancy_scale=1.0)
    density_generator.save_map("/home/kchrispens/adp-replicate/tests/resources/AAAWAAA/AAAWAAA_Waltconf_1.ccp4", density)

    density_generator = SyntheticDensityGenerator(structure=pdb, resolution=1.0, unit_cell=unit_cell, em_mode=False)
    density = density_generator.generate_map(b_factor_scale=1.0, occupancy_scale=1.0)
    density_generator.save_map("/home/kchrispens/adp-replicate/tests/resources/AAAWAAA/AAAWAAA_Waltconf_4_downsampled_from_1_brickwall_filtered.ccp4", density, downsample_to=4.0)