"""Synthetic electron density map generation from atomic structures.

This module provides functionality to generate synthetic electron density maps from
atomic structures (mmCIF, PDB files, or Ensemble objects) with user-defined
experimental parameters, including advanced loop processing and XMap enhancements.

Key Features:
- Generate synthetic density maps from single structures or ensembles
- Occupancy loops for altloc conformations and ensemble weighting
- P1 canonical unit cell expansion using crystallographic symmetry
- Map carving around selected atoms with customizable padding
- Batch processing from CSV files
- Flexible command-line interface with comprehensive options

Usage Examples:

1. Basic Single Structure Processing:
   python synthetic_density.py --input-structure path/to/structure.cif
   python synthetic_density.py --pdb-id 1abc --resolution 2.0

   # With reference map file
   python synthetic_density.py --input-structure structure.cif --reference-map reference.ccp4

   # Process specific chain or selection
   python synthetic_density.py --pdb-id 6b8x --selection "chain A"
   python synthetic_density.py --pdb-id 6b8x --selection "chain A and resi 10-50"
   python synthetic_density.py --pdb-id 6b8x --selection "protein and not altloc C"

   # Generate individual conformation maps
   python synthetic_density.py --pdb-id 6b8x --individual-maps

2. Occupancy Loop Processing:
   # Process altloc conformations with automatic occupancy pairs (0.25 steps)
   python synthetic_density.py --pdb-id 6b8x --occupancy-loop

   # Custom occupancy pairs
   python synthetic_density.py --pdb-id 6b8x --occupancy-loop --occupancy-pairs "0.0,1.0;0.25,0.75;0.5,0.5;0.75,0.25;1.0,0.0"

   # Include individual conformation maps
   python synthetic_density.py --pdb-id 6b8x --occupancy-loop --individual-conformations

3. Ensemble Processing:
   # Process multiple structures as ensemble
   python synthetic_density.py --ensemble-mode --structure-list struct1.cif struct2.cif struct3.cif

   # Ensemble with occupancy loop and custom names
   python synthetic_density.py --ensemble-mode --structure-list 5sop.cif 5soq.cif --structure-names 5SOP 5SOQ --occupancy-loop

   # Ensemble with individual conformations
   python synthetic_density.py --ensemble-mode --structure-list 5sop.cif 5soq.cif --individual-conformations

4. XMap Enhancements:
   # Expand to P1 canonical unit cell
   python synthetic_density.py --input-structure structure.cif --expand-to-p1

   # Carve around specific atoms
   python synthetic_density.py --input-structure structure.cif --carve-around --carve-selection "chain A and resi 120-140"

   # Combined P1 expansion and carving
   python synthetic_density.py --input-structure structure.cif --expand-to-p1 --carve-around --carve-selection "chain A and resi 120-140" --carve-padding 5.0

5. Advanced Combined Processing:
   # MAC1-style processing: occupancy loop with P1 expansion
   python synthetic_density.py --pdb-id mac1 --occupancy-loop --individual-conformations --expand-to-p1

   # PTP1B-style processing: occupancy loop with structure factors and carving
   python synthetic_density.py --pdb-id 6b8x --occupancy-loop --download-sf --carve-around --carve-selection "chain A and resi 215-220"

   # Use custom reference map with individual conformations
   python synthetic_density.py --pdb-id 6b8x --reference-map experimental.ccp4 --individual-maps --reweight-occupancies

   # Normalize occupancies and process with enhancements
   python synthetic_density.py --pdb-id 6b8x --reweight-occupancies --occupancy-loop --expand-to-p1

   # Full ensemble workflow with all enhancements
   python synthetic_density.py --ensemble-mode --structure-list 5sop.cif 5soq.cif --occupancy-loop --individual-conformations --expand-to-p1 --carve-around --carve-selection "chain A" --reweight-occupancies

6. Batch Processing:
   # Process multiple structures from CSV
   python synthetic_density.py --batch-csv altloc_summary.csv --altloc-data-dir /path/to/structures

   # With structure factors download
   python synthetic_density.py --batch-csv altloc_summary.csv --download-sf

Command-Line Arguments:

Input Options:
  --pdb-id                    PDB ID to process
  --input-structure           Path to input structure file
  --structure-list           List of structure files for ensemble processing
  --structure-names          Names for ensemble structures (must match --structure-list)
  --batch-csv                Path to altloc_summary.csv for batch processing
  --altloc-data-dir          Directory containing mmCIF files from altloc analysis
  --reference-map            Path to reference map file for unit cell parameters

Output Options:
  --output-dir               Output directory (default: synthetic_density_output)
  --resolution               Map resolution in Angstroms (default: 2.0)

Basic Processing Options:
  --download-sf              Download structure factors for unit cell parameters
  --selection                Atom selection string for processing (e.g., 'chain A', 'protein')
  --reweight-occupancies      Normalize occupancies within alternative conformations to sum to 1.0
  --individual-maps          Generate individual maps for each alternative conformation

Loop Processing Options:
  --occupancy-loop           Enable occupancy loop processing for altloc conformations
  --occupancy-pairs          Occupancy pairs as string (e.g., '0.0,1.0;0.25,0.75;0.5,0.5')
  --occupancy-step           Step size for automatic occupancy pair generation (default: 0.25)
  --ensemble-mode            Process multiple structures as ensemble (requires --structure-list)
  --individual-conformations Also create individual conformation maps during loop processing

XMap Enhancement Options:
  --expand-to-p1             Expand maps to P1 canonical unit cell
  --carve-around             Carve maps around selected atoms
  --carve-selection          Atom selection for carving (e.g., 'chain A and resi 120-140')
  --carve-padding            Padding around atoms for carving in Angstroms (default: 3.0)

File Naming Conventions:
- Basic maps: {pdb_id}_{resolution}A.ccp4
- Occupancy loops: {pdb_id}_{occA}occA_{occB}occB_{resolution}A.ccp4
- Individual conformations: {pdb_id}_1occ{conf_name}_{resolution}A.ccp4
- Ensemble maps: {ensemble_id}_ensemble_{resolution}A.ccp4
- P1 expanded: {basename}_p1_{resolution}A.ccp4
- Carved maps: {basename}_carved_{resolution}A.ccp4
- Combined: {basename}_p1_carved_{resolution}A.ccp4

Dependencies:
- edg.data.Structure, edg.data.structure.Ensemble
- edg.qfit.volume.XMap, edg.edg.modules.density
- PyTorch for GPU-accelerated calculations
- NumPy, pandas for data handling
- Requests for structure factor downloads

Notes:
- Requires CUDA-capable GPU for practical performance
- Map resolution must be specified for CCP4/MRC files
- Supports both X-ray crystallography and cryo-EM modes
- All loop processing maintains backward compatibility with existing workflows
- XMap enhancements use existing crystallographic methods from qfit.volume
"""

import os
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from typing import Optional, Union, List, NamedTuple, Literal
from copy import deepcopy

import torch
import numpy as np
import pandas as pd

from edg.data import Structure
from edg.data.structure import Ensemble
from edg.qfit.volume import XMap, Resolution, GridParameters
from edg.edg.modules.density.density import (
    DifferentiableTransformer,
    XMap_torch,
)
from edg.data.io import structure_to_density_input
from edg.data.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
)
from edg.utils.utility import try_gpu


def _set_default_b_factors(
    struct_or_ensemble, value: float = 20.0, force: bool = False
):
    """Ensure B-factors are set (and non-NaN) for a Structure or Ensemble.

    If any NaN values are found they are replaced with the provided value. If
    force is True, all B-factors are overwritten with the provided value.
    """
    try:
        if hasattr(struct_or_ensemble, "__iter__") and not isinstance(
            struct_or_ensemble, Structure
        ):
            # Treat as ensemble (list-like of structures)
            for s in struct_or_ensemble:
                _set_default_b_factors(s, value=value, force=force)
            return
        s = struct_or_ensemble
        if not hasattr(s, "data"):
            return
        if "b" not in s.data:
            return
        b_arr = s.data["b"]
        if force:
            b_arr[:] = value
        else:
            # Replace NaNs only
            if np.isnan(b_arr).any():
                b_arr[np.isnan(b_arr)] = value
    except Exception:
        # Silent fail – do not break main workflow
        pass


class SyntheticDensityGenerator:
    """Generate synthetic electron density maps from atomic structures.

    This class provides functionality to create synthetic electron density maps
    from atomic structures (Structure or Ensemble objects) with optional
    experimental parameters. It supports both X-ray crystallography and electron
    microscopy scattering factors.

    The class supports different coordinate system modes:
    - 'crystallographic': Use real unit cell from structure/reference map
    - 'synthetic': Create synthetic coordinate system centered at origin
    """

    def __init__(
        self,
        structure: Union[str, Structure, Ensemble],
        reference_map_file: Optional[str] = None,
        resolution: Optional[float] = None,
        unit_cell: Optional[NamedTuple] = None,
        em_mode: Optional[bool] = False,
        device: Optional[Union[str, torch.device]] = None,
        coordinate_mode: Literal["crystallographic", "synthetic"] = "crystallographic",
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
        coordinate_mode : Literal['crystallographic', 'synthetic'], optional
            Coordinate system handling mode:
            - 'crystallographic': Use real unit cell coordinates without shifting
            - 'synthetic': Create synthetic coordinate system with structure centered at origin
        """
        if device is None:
            self.device = try_gpu()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Load structure
        if isinstance(structure, str):
            # Load from file and apply standard preparation (reorder -> complete_residues -> reorder)
            loaded_structure = Structure.fromfile(structure)
            try:
                loaded_structure = (
                    loaded_structure.reorder().complete_residues().reorder()
                )
            except Exception:
                # If preparation fails, proceed with the raw loaded structure
                pass
            self.structure = loaded_structure
            self.is_ensemble = False
        elif isinstance(structure, Structure):
            self.structure = structure
            self.is_ensemble = False
        elif isinstance(structure, Ensemble):
            self.structure = structure
            self.is_ensemble = True
        else:
            raise TypeError(
                "Structure must be a file path, Structure object, or Ensemble object"
            )

        self.em_mode = em_mode
        self.resolution = resolution
        self.coordinate_mode = coordinate_mode

        self.scattering_factors = (
            ELECTRON_SCATTERING_FACTORS if em_mode else ATOM_STRUCTURE_FACTORS
        )

        if reference_map_file is not None:
            self._setup_from_reference(reference_map_file)
        elif resolution is not None:
            self._setup_from_parameters(resolution, unit_cell)
        else:
            raise ValueError("Either reference_map_file or resolution must be provided")

        # Set up coordinate transformation based on mode
        self._setup_coordinate_system()

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
            except Exception:
                ref_map = XMap.fromfile(
                    reference_map_file, label="2FOFCWT,PH2FOFCWT"
                )  # try 2FO-FC map if initial fails
        else:
            if not hasattr(self, "resolution") or self.resolution is None:
                raise ValueError(
                    "Resolution must be provided for CCP4, MRC, or MAP files."
                )
            ref_map = XMap.fromfile(reference_map_file, resolution=self.resolution)

        self.xmap = XMap_torch(ref_map, device=self.device)
        self.ref_map = ref_map  # Store reference for coordinate system setup
        self._setup_scattering_params()
        self.density_calculator = DifferentiableTransformer(
            self.xmap,
            scattering_params=self.scattering_params,
            em=self.em_mode,
            device=self.device,
            use_cuda_kernels=True,
        )

    def _setup_from_parameters(
        self, resolution: float, unit_cell: NamedTuple = None
    ) -> None:
        """Set up the generator using specified parameters.

        Parameters
        ----------
        resolution : float
            Map resolution in Angstroms.
        """
        self.resolution = resolution
        voxelspacing = resolution / 4.0

        # Get the unit cell from the first structure if using an ensemble
        structure_ref = self.structure[0] if self.is_ensemble else self.structure
        unit_cell = structure_ref.unit_cell if unit_cell is None else unit_cell

        # Create an appropriately-sized grid based on resolution
        grid_a = int(np.ceil(unit_cell.a / resolution * 4.0))
        grid_b = int(np.ceil(unit_cell.b / resolution * 4.0))
        grid_c = int(np.ceil(unit_cell.c / resolution * 4.0))

        actual_extent_a = grid_a * voxelspacing
        actual_extent_b = grid_b * voxelspacing
        actual_extent_c = grid_c * voxelspacing

        # Cartesian coordinate of grid index (0,0,0)
        # This centers the defined unit_cell within the actual grid extent
        map_origin_cartesian = np.array(
            [
                (unit_cell.a - actual_extent_a) / 2.0,
                (unit_cell.b - actual_extent_b) / 2.0,
                (unit_cell.c - actual_extent_c) / 2.0,
            ]
        )

        empty_array = np.zeros((grid_c, grid_b, grid_a), dtype=np.float32)

        ref_map = XMap(
            empty_array,
            grid_parameters=GridParameters(voxelspacing=voxelspacing),
            resolution=Resolution(high=resolution, low=1000.0),
            unit_cell=unit_cell,
            origin=map_origin_cartesian,
        )

        self.xmap = XMap_torch(ref_map, device=self.device)
        self.ref_map = ref_map  # Store reference for coordinate system setup
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
        scattering_dense_tensor = torch.zeros(
            dense_size, dtype=torch.float32, device=self.device
        )

        for elem in unique_elements:
            atomic_num = atomic_num_dict[elem]

            if elem in self.scattering_factors:
                factor = self.scattering_factors[elem]
            else:
                print(
                    f"Warning: Scattering factors for {elem} not found, using C instead"
                )
                factor = self.scattering_factors["C"]

            factor = torch.tensor(
                factor, dtype=torch.float32, device=self.device
            ).T  # (2, range) -> (range, 2)
            scattering_dense_tensor[atomic_num, :, :] = factor

        self.scattering_params = scattering_dense_tensor
        self.atomic_num_dict = atomic_num_dict

    def _setup_coordinate_system(self) -> None:
        """Set up coordinate transformation based on the specified mode."""
        structure_ref = self.structure[0] if self.is_ensemble else self.structure

        if self.coordinate_mode == "crystallographic":
            # Use crystallographic coordinates without shifting
            self.coord_shift = np.array([0.0, 0.0, 0.0])
            self.needs_centering = False

        elif self.coordinate_mode == "synthetic":
            # Center structure at origin of synthetic coordinate system
            # For AF3-style predictions where structure is at origin
            unit_cell = structure_ref.unit_cell
            self.coord_shift = np.array(
                [
                    unit_cell.c / 2.0,
                    unit_cell.b / 2.0,
                    unit_cell.a / 2.0,
                ]
            )
            self.needs_centering = True

        else:
            raise ValueError(f"Unknown coordinate_mode: {self.coordinate_mode}")

    def generate_map_from_structure(
        self,
        structure: Structure,
        b_factor_scale: float = 1.0,
        occupancy_scale: float = 1.0,
        shift: Optional[bool] = None,
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
        shift : Optional[bool], optional
            Whether to shift coordinates. If None, uses coordinate_mode setting.
            For backward compatibility, explicit True/False overrides coordinate_mode.

        Returns
        -------
        torch.Tensor
            Generated density map.
        """
        # Ensure B-factors are sane (set NaNs to 20.0)
        _set_default_b_factors(structure, value=20.0, force=False)
        coords, elements, b_factors, occupancies, active, _ = (
            structure_to_density_input(structure)
        )

        # Apply coordinate transformation based on mode
        if shift is None:
            # Use coordinate mode setting
            should_shift = self.needs_centering
        else:
            # Explicit override for backward compatibility
            should_shift = shift

        if should_shift:
            coords = coords - torch.tensor(
                self.coord_shift, dtype=coords.dtype, device=coords.device
            )

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
        shift: Optional[bool] = None,
    ) -> torch.Tensor:
        """Generate a synthetic density map from the structure or ensemble.

        Parameters
        ----------
        b_factor_scale : float, optional
            Scale factor for B-factors.
        occupancy_scale : float, optional
            Scale factor for occupancies.
        shift : Optional[bool], optional
            Whether to shift coordinates. If None, uses coordinate_mode setting.
            For backward compatibility, explicit True/False overrides coordinate_mode.

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
                    occupancy_scale
                    / len(self.structure),  # Scale occupancy by ensemble size
                    # TODO: improve to use occupancies from the ensemble
                    shift=shift,
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
                self.structure, b_factor_scale, occupancy_scale, shift=shift
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
            if not hasattr(self, "last_density_map"):
                raise ValueError("No density map available to save.")
            density_map = self.last_density_map

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        xmap = self.density_calculator.xmap
        xmap.array = (
            density_map
            if isinstance(density_map, torch.Tensor)
            else torch.tensor(density_map, device=self.device)
        )

        # Downsample the density map if requested
        if downsample_to is not None:
            xmap = xmap.downsample_to_resolution(downsample_to, apply_filter=True)

        xmap.tofile(output_file)
        print(f"Saved density map to {output_file}")

    def expand_and_carve_density_map(
        self,
        density_map: np.ndarray,
        expand_to_p1: bool = False,
        carve_selection: str = None,
        carve_padding: float = 3.0,
    ) -> np.ndarray:
        """Apply P1 expansion and carving operations to a generated density map.

        Parameters
        ----------
        density_map : np.ndarray
            Generated density map to process
        expand_to_p1 : bool, optional
            Whether to expand to P1 canonical unit cell
        carve_selection : str, optional
            Atom selection string for carving (e.g., "chain A and resi 120-140")
        carve_padding : float, optional
            Padding around atoms for carving in Angstroms

        Returns
        -------
        XMap
            Processed density map after applying P1 expansion and/or carving
        """
        if not hasattr(self, "ref_map"):
            raise ValueError("No reference map available.")

        # Start with current XMap and update its density array
        from copy import deepcopy

        current_xmap = deepcopy(self.ref_map)
        current_xmap.array = (
            density_map
            if isinstance(density_map, np.ndarray)
            else density_map.cpu().numpy()
        )

        # Apply P1 expansion if requested
        if expand_to_p1:
            current_xmap = current_xmap.canonical_unit_cell()
            print("Applied P1 expansion to density map")

        # Apply carving if requested
        if carve_selection:
            # Get reference structure for atom selection
            structure_ref = self.structure[0] if self.is_ensemble else self.structure

            # Apply atom selection
            try:
                selection = structure_ref.select(carve_selection)
                if selection.sum() == 0:
                    raise ValueError(
                        f"No atoms selected with selection: {carve_selection}"
                    )
                selected_coords = structure_ref.coor[selection]

                # If any coordinates are NaN, do not include them in the extraction
                valid_mask = ~np.isnan(selected_coords).any(axis=1)
                selected_coords = selected_coords[valid_mask]
                # Extract map around selected coordinates
                current_xmap = current_xmap.extract(
                    selected_coords, padding=carve_padding
                )
                print(
                    f"Applied carving around {len(selection)} atoms with {carve_padding}Å padding"
                )

            except Exception as e:
                raise ValueError(f"Invalid atom selection '{carve_selection}': {e}")

        self.density_calculator.xmap = XMap_torch(current_xmap, device=self.device)
        self.ref_map = current_xmap  # Update reference map

        return current_xmap.array


def parse_occupancy_pairs(occupancy_pairs_str: str) -> List[tuple]:
    """Parse occupancy pairs from command line string.

    Args:
        occupancy_pairs_str: String like "0.0,1.0;0.25,0.75;0.5,0.5"

    Returns:
        List of (occ_a, occ_b) tuples
    """
    pairs = []
    for pair_str in occupancy_pairs_str.split(";"):
        occ_a, occ_b = map(float, pair_str.split(","))
        pairs.append((occ_a, occ_b))
    return pairs


def generate_occupancy_pairs(step: float = 0.25) -> List[tuple]:
    """Generate occupancy pairs that sum to 1.0.

    Args:
        step: Step size for occupancy increments

    Returns:
        List of (occ_a, occ_b) tuples
    """
    pairs = []
    occ_a = 0.0
    while occ_a <= 1.0:
        occ_b = 1.0 - occ_a
        pairs.append((occ_a, occ_b))
        occ_a += step
    return pairs


def process_occupancy_loop(
    structure: Structure,
    pdb_id: str,
    output_dir: str,
    resolution: float,
    occupancy_pairs: List[tuple],
    reference_map_file: str = None,
    individual_conformations: bool = False,
    expand_to_p1: bool = False,
    carve_selection: str = None,
    carve_padding: float = 3.0,
    reweight_occupancies: bool = False,
):
    """Process occupancy loop for altloc conformations within a single structure.

    Args:
        structure: Structure with altloc conformations
        pdb_id: PDB ID for naming
        output_dir: Output directory
        resolution: Map resolution
        occupancy_pairs: List of (occ_a, occ_b) tuples
        reference_map_file: Optional reference map file
        individual_conformations: Whether to also create individual conformation maps
        expand_to_p1: Whether to expand maps to P1
        carve_selection: Optional atom selection for carving
        carve_padding: Padding for carving
    """
    print(f"\n=== Processing occupancy loop for {pdb_id} ===")

    # Create output directory
    pdb_output_dir = os.path.join(output_dir, pdb_id.lower())
    os.makedirs(pdb_output_dir, exist_ok=True)

    # Clean and prepare structure
    structure = structure.clean_structure(keep_type="protein")
    structure = structure.reorder().complete_residues().reorder()

    # Normalize occupancies if requested
    if reweight_occupancies:
        structure = structure.reweight_occupancies()
        print("Normalized occupancies within alternative conformations")
    _set_default_b_factors(structure, value=20.0, force=True)

    # Extract altloc conformations
    altA = structure.select("altloc A")
    altB = structure.select("altloc B")

    if altA.sum() == 0 and altB.sum() == 0:
        print(f"No altloc conformations found in {pdb_id}, skipping occupancy loop")
        return

    print(f"Found {altA.sum()} atoms in altloc A, {altB.sum()} atoms in altloc B")

    # Set consistent B-factors
    structure.data["b"][:] = 20.0

    # Process each occupancy pair
    for Aocc, Bocc in occupancy_pairs:
        print(f"\n--- Processing occupancies: A={Aocc}, B={Bocc} ---")

        # Create weighted structure
        structure_copy = deepcopy(structure)
        structure_copy.data["q"][altA] = Aocc
        structure_copy.data["q"][altB] = Bocc

        # Setup density generator
        if reference_map_file and os.path.exists(reference_map_file):
            density_generator = SyntheticDensityGenerator(
                structure_copy, reference_map_file
            )
            resolution = density_generator.xmap.resolution.high
        else:
            density_generator = SyntheticDensityGenerator(
                structure=structure_copy,
                resolution=resolution,
                coordinate_mode="crystallographic",
            )

        # Generate density map
        density = density_generator.generate_map(
            b_factor_scale=1.0,
            occupancy_scale=1.0,
        )

        # Apply XMap enhancements if requested
        if expand_to_p1 or carve_selection:
            density = density_generator.expand_and_carve_density_map(
                density, expand_to_p1, carve_selection, carve_padding
            )

        # Create filename
        suffix = ""
        if Aocc > 0:
            suffix += f"_{Aocc}occA"
        if Bocc > 0:
            suffix += f"_{Bocc}occB"
        if expand_to_p1:
            suffix += "_p1"
        if carve_selection:
            suffix += "_carved"

        density_file = os.path.join(
            pdb_output_dir, f"{pdb_id.lower()}{suffix}_{resolution:.2f}A.ccp4"
        )

        density_generator.save_map(density_file, density)

        # Save structure
        structure_file = os.path.join(pdb_output_dir, f"{pdb_id.lower()}{suffix}.cif")
        structure_copy.tofile(structure_file)

        print(f"✓ Generated files for A={Aocc}, B={Bocc}")

    # Create individual conformation maps if requested
    if individual_conformations:
        print("\n--- Creating individual conformation maps ---")

        for conf_name, selection in [("A", altA), ("B", altB)]:
            if selection.sum() == 0:
                continue

            structure_copy = deepcopy(structure)
            structure_copy.data["q"][:] = 0.0  # Zero all occupancies
            structure_copy.data["q"][selection] = (
                1.0  # Full occupancy for this conformation
            )

            if reference_map_file and os.path.exists(reference_map_file):
                density_generator = SyntheticDensityGenerator(
                    structure_copy, reference_map_file
                )
                resolution = density_generator.xmap.resolution.high
            else:
                density_generator = SyntheticDensityGenerator(
                    structure=structure_copy,
                    resolution=resolution,
                    coordinate_mode="crystallographic",
                )

            density = density_generator.generate_map()

            if expand_to_p1 or carve_selection:
                density = density_generator.expand_and_carve_density_map(
                    density, expand_to_p1, carve_selection, carve_padding
                )

            suffix = f"_1occ{conf_name}"
            if expand_to_p1:
                suffix += "_p1"
            if carve_selection:
                suffix += "_carved"

            density_file = os.path.join(
                pdb_output_dir, f"{pdb_id.lower()}{suffix}_{resolution:.2f}A.ccp4"
            )

            density_generator.save_map(density_file, density)

            structure_file = os.path.join(
                pdb_output_dir, f"{pdb_id.lower()}{suffix}.cif"
            )
            structure_copy.tofile(structure_file)

            print(f"✓ Generated individual conformation {conf_name}")


def process_ensemble_occupancy_loop(
    structures: List[Structure],
    structure_names: List[str],
    pdb_id: str,
    output_dir: str,
    resolution: float,
    occupancy_pairs: List[tuple],
    reference_map_file: str = None,
    individual_conformations: bool = False,
    expand_to_p1: bool = False,
    carve_selection: str = None,
    carve_padding: float = 3.0,
    reweight_occupancies: bool = False,
):
    """Process occupancy loop for ensemble of structures.

    Args:
        structures: List of Structure objects
        structure_names: Names for each structure
        pdb_id: PDB ID for naming
        output_dir: Output directory
        resolution: Map resolution
        occupancy_pairs: List of (occ_a, occ_b) tuples (will use first two structures)
        reference_map_file: Optional reference map file
        individual_conformations: Whether to also create individual conformation maps
        expand_to_p1: Whether to expand maps to P1
        carve_selection: Optional atom selection for carving
        carve_padding: Padding for carving
    """
    print(f"\n=== Processing ensemble occupancy loop for {pdb_id} ===")

    if len(structures) < 2:
        print("Need at least 2 structures for ensemble occupancy loop")
        return

    # Create output directory
    pdb_output_dir = os.path.join(output_dir, pdb_id.lower())
    os.makedirs(pdb_output_dir, exist_ok=True)

    # Clean and prepare structures
    cleaned_structures = []
    for structure in structures:
        cleaned = structure.clean_structure(keep_type="protein")
        cleaned = cleaned.reorder().complete_residues().reorder()

        # Normalize occupancies if requested
        if reweight_occupancies:
            cleaned = cleaned.reweight_occupancies()
            print(
                f"Normalized occupancies for structure {structure_names[len(cleaned_structures)] if len(cleaned_structures) < len(structure_names) else 'unknown'}"
            )
            _set_default_b_factors(cleaned, value=20.0, force=True)

        cleaned.data["b"][:] = 20.0  # Set consistent B-factors
        cleaned_structures.append(cleaned)

    # Create ensemble
    ensemble = Ensemble(cleaned_structures)

    # Process each occupancy pair (using first two structures)
    for Aocc, Bocc in occupancy_pairs:
        print(
            f"\n--- Processing ensemble occupancies: {structure_names[0]}={Aocc}, {structure_names[1]}={Bocc} ---"
        )

        # Create weighted ensemble
        ensemble_copy = deepcopy(ensemble)
        ensemble_copy[0].data["q"][:] = Aocc  # First structure
        ensemble_copy[1].data["q"][:] = Bocc  # Second structure

        # Zero out remaining structures
        for i in range(2, len(ensemble_copy)):
            ensemble_copy[i].data["q"][:] = 0.0

        # Setup density generator
        if reference_map_file and os.path.exists(reference_map_file):
            density_generator = SyntheticDensityGenerator(
                ensemble_copy, reference_map_file
            )
            resolution = density_generator.xmap.resolution.high
        else:
            density_generator = SyntheticDensityGenerator(
                structure=ensemble_copy,
                resolution=resolution,
                coordinate_mode="crystallographic",
            )

        # Generate density map
        density = density_generator.generate_map(
            b_factor_scale=1.0,
            occupancy_scale=1.0,
        )

        # Apply XMap enhancements if requested
        if expand_to_p1 or carve_selection:
            density = density_generator.expand_and_carve_density_map(
                density, expand_to_p1, carve_selection, carve_padding
            )

        # Create filename
        suffix = ""
        if Aocc > 0:
            suffix += f"_{Aocc}occ{structure_names[0]}"
        if Bocc > 0:
            suffix += f"_{Bocc}occ{structure_names[1]}"
        if expand_to_p1:
            suffix += "_p1"
        if carve_selection:
            suffix += "_carved"

        density_file = os.path.join(
            pdb_output_dir, f"{pdb_id.lower()}{suffix}_{resolution:.2f}A.ccp4"
        )

        density_generator.save_map(density_file, density)

        # Save ensemble
        ensemble_file = os.path.join(pdb_output_dir, f"{pdb_id.lower()}{suffix}.cif")
        ensemble_copy.tofile(ensemble_file)

        print(
            f"✓ Generated ensemble files for {structure_names[0]}={Aocc}, {structure_names[1]}={Bocc}"
        )

    # Create individual conformation maps if requested
    if individual_conformations:
        print("\n--- Creating individual conformation maps ---")

        for i, (structure, name) in enumerate(zip(cleaned_structures, structure_names)):
            structure_copy = deepcopy(structure)
            structure_copy.data["q"][:] = 1.0  # Full occupancy

            if reference_map_file and os.path.exists(reference_map_file):
                density_generator = SyntheticDensityGenerator(
                    structure_copy, reference_map_file
                )
                resolution = density_generator.xmap.resolution.high
            else:
                density_generator = SyntheticDensityGenerator(
                    structure=structure_copy,
                    resolution=resolution,
                    coordinate_mode="crystallographic",
                )

            density = density_generator.generate_map()

            if expand_to_p1 or carve_selection:
                density = density_generator.expand_and_carve_density_map(
                    density, expand_to_p1, carve_selection, carve_padding
                )

            suffix = f"_1occ{name}"
            if expand_to_p1:
                suffix += "_p1"
            if carve_selection:
                suffix += "_carved"

            density_file = os.path.join(
                pdb_output_dir, f"{pdb_id.lower()}{suffix}_{resolution:.2f}A.ccp4"
            )

            density_generator.save_map(density_file, density)

            structure_file = os.path.join(
                pdb_output_dir, f"{pdb_id.lower()}{suffix}.cif"
            )
            structure_copy.tofile(structure_file)

            print(f"✓ Generated individual conformation {name}")


def download_structure_factors(pdb_id: str, output_dir: str) -> Optional[str]:
    """Download structure factors for a PDB ID from PDB-REDO.

    Args:
        pdb_id: PDB ID to download
        output_dir: Directory to save the file

    Returns:
        Path to downloaded file or None if failed
    """
    url = f"https://pdb-redo.eu/db/{pdb_id.lower()}/{pdb_id.lower()}_final.mtz"
    output_path = os.path.join(output_dir, f"{pdb_id.lower()}_final.mtz")

    # Check if file already exists
    if os.path.exists(output_path):
        print(f"✓ Structure factors already exist: {output_path}")
        return output_path

    # Setup retry strategy
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=2,
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )

    # Create session with retry strategy
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        print(f"Downloading PDB-REDO structure factors for {pdb_id}...")
        response = session.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"✓ Downloaded PDB-REDO structure factors: {output_path}")
        time.sleep(1)  # Rate limiting
        return output_path
    except Exception as e:
        print(f"✗ Failed to download PDB-REDO structure factors for {pdb_id}: {e}")
        return None
    finally:
        session.close()


def extract_altloc_conformations(
    structure: Structure, selection: str = None
) -> List[Structure]:
    """Extract alternative conformations from a structure.

    Args:
        structure: Input structure
        selection: Atom selection string for filtering (optional)

    Returns:
        List of structures, one for each conformation
    """
    structure = structure.clean_structure()
    structure = structure.complete_residues()
    # Filter by selection if specified
    if selection:
        try:
            selection_mask = structure.select(selection)
            if selection_mask.sum() == 0:
                print(f"Warning: Selection '{selection}' matched no atoms")
            else:
                structure = structure.extract(selection_mask)
        except Exception as e:
            print(f"Error applying selection '{selection}': {e}")
            print("Proceeding with full structure")

    # Get unique alternative location IDs
    altlocs = set(structure.altloc)
    altlocs.discard("")  # Remove empty altloc

    conformations = []

    # Add main conformation (no altloc or altloc A)
    main_selection = structure.select("altloc '' or altloc A")
    if main_selection.sum() > 0:
        main_struct = deepcopy(
            structure.extract(main_selection)
        )  # Use deepcopy to avoid modifying original structure
        main_struct.altloc = ""  # Clear altloc identifiers
        main_struct.q = 1.0
        conformations.append(main_struct)

    # Add alternative conformations
    for altloc in sorted(altlocs):
        if altloc and altloc != "A":
            alt_selection = structure.select(f"altloc '' or altloc {altloc}")
            if alt_selection.sum() > 0:
                alt_struct = deepcopy(structure.extract(alt_selection))
                alt_struct.altloc = ""  # Clear altloc identifiers
                alt_struct.q = 1.0
                conformations.append(alt_struct)

    return conformations


def process_single_structure(
    pdb_id: str,
    structure_path: str,
    output_dir: str,
    resolution: float = 2.0,
    download_sf: bool = False,
    selection: str = None,
):
    """Process a single structure to generate synthetic density maps.

    Args:
        pdb_id: PDB ID for naming
        structure_path: Path to structure file
        output_dir: Output directory
        resolution: Map resolution
        download_sf: Whether to download structure factors
        selection: Atom selection string for processing
    """
    print(f"\nProcessing {pdb_id}...")

    # Create output directory
    pdb_output_dir = os.path.join(output_dir, pdb_id.lower())
    os.makedirs(pdb_output_dir, exist_ok=True)

    # Download structure factors if requested
    sf_file = None
    if download_sf:
        sf_file = download_structure_factors(pdb_id, pdb_output_dir)

    # Load structure
    try:
        structure = Structure.fromfile(structure_path)
        try:
            structure = structure.reorder().complete_residues().reorder()
        except Exception:
            pass
        print(f"Loaded structure: {structure}")
    except Exception as e:
        print(f"✗ Failed to load structure {structure_path}: {e}")
        return

    # Extract alternative conformations
    conformations = extract_altloc_conformations(structure, selection)
    print(f"Found {len(conformations)} conformations")
    ensemble = Ensemble(conformations)

    # Generate density maps for each conformation
    for i, conformation in enumerate(conformations):
        conformation_name = f"conf_{i + 1}" if i > 0 else "main"

        try:
            # Setup density generator
            if sf_file and os.path.exists(sf_file):
                # Use structure factors for unit cell
                density_generator = SyntheticDensityGenerator(conformation, sf_file)
                shift = False  # Don't shift if using real unit cell
                resolution = density_generator.xmap.resolution.high
            else:
                # Use resolution-based approach
                density_generator = SyntheticDensityGenerator(
                    conformation, resolution=resolution
                )
                shift = True  # Shift to center in unit cell

            # Generate density map
            density = density_generator.generate_map(shift=shift)

            # Save density map
            density_file = os.path.join(
                pdb_output_dir,
                f"{pdb_id.lower()}_{conformation_name}_{resolution:.2f}A.ccp4",
            )
            density_generator.save_map(density_file, density)

            # Save structure
            structure_file = os.path.join(
                pdb_output_dir, f"{pdb_id.lower()}_{conformation_name}.cif"
            )
            conformation.tofile(structure_file)

            print(f"✓ Generated {conformation_name}: {density_file}")

        except Exception as e:
            print(f"✗ Failed to generate density for {conformation_name}: {e}")

    try:
        if sf_file and os.path.exists(sf_file):
            density_generator = SyntheticDensityGenerator(ensemble, sf_file)
            shift = False  # Don't shift if using real unit cell
        else:
            density_generator = SyntheticDensityGenerator(
                ensemble, resolution=resolution
            )
            shift = True  # Shift to center in unit cell

        density = density_generator.generate_map(
            shift=shift, occupancy_scale=1.0 / len(ensemble)
        )
        density_file = os.path.join(
            pdb_output_dir, f"{pdb_id.lower()}_ensemble_{resolution:.2f}A.ccp4"
        )

        density_generator.save_map(density_file, density)
        ensemble_file = os.path.join(pdb_output_dir, f"{pdb_id.lower()}_ensemble.cif")
        ensemble.tofile(ensemble_file)
        print(f"✓ Generated ensemble: {density_file}")
    except Exception as e:
        print(f"✗ Failed to generate density for ensemble: {e}")


def process_batch_from_csv(
    csv_path: str,
    output_dir: str,
    resolution: float = 2.0,
    download_sf: bool = False,
    altloc_data_dir: str = None,
):
    """Process multiple structures from altloc analysis CSV.

    Args:
        csv_path: Path to altloc_summary.csv
        output_dir: Output directory
        resolution: Map resolution
        download_sf: Whether to download structure factors
        altloc_data_dir: Directory containing mmCIF files from altloc analysis
    """
    print(f"Processing batch from {csv_path}...")

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} structures from CSV")
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        return

    # Process each structure
    for _, row in df.iterrows():
        pdb_code = row["pdb_code"]
        target_chain = row.get("target_chain", None)
        # Convert chain ID to selection string for backward compatibility
        selection = f"chain {target_chain}" if target_chain else None

        # Look for structure file
        structure_path = None
        if altloc_data_dir:
            # Look in altloc data directory
            potential_paths = [
                os.path.join(altloc_data_dir, f"{pdb_code.lower()}.cif"),
                os.path.join(altloc_data_dir, f"{pdb_code.upper()}.cif"),
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    structure_path = path
                    break

        if not structure_path:
            print(f"✗ Structure file not found for {pdb_code}")
            continue

        # Process structure
        process_single_structure(
            pdb_code, structure_path, output_dir, resolution, download_sf, selection
        )


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic density maps from protein structures with loop processing and XMap enhancements"
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument("--pdb-id", help="PDB ID to process")
    input_group.add_argument("--input-structure", help="Path to input structure file")
    input_group.add_argument(
        "--structure-list",
        nargs="+",
        help="List of structure files for ensemble processing",
    )
    input_group.add_argument(
        "--structure-names",
        nargs="+",
        help="Names for ensemble structures (must match --structure-list)",
    )
    input_group.add_argument(
        "--batch-csv", help="Path to altloc_summary.csv for batch processing"
    )
    input_group.add_argument(
        "--altloc-data-dir",
        default="tests/resources/altloc_data/mmcif_files",
        help="Directory containing mmCIF files from altloc analysis",
    )
    input_group.add_argument(
        "--reference-map",
        help="Path to reference map file for unit cell parameters (overrides structure factor download)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir", default="synthetic_density_output", help="Output directory"
    )
    output_group.add_argument(
        "--resolution", type=float, default=2.0, help="Map resolution in Angstroms"
    )

    # Processing options
    processing_group = parser.add_argument_group("Basic Processing Options")
    processing_group.add_argument(
        "--download-sf",
        action="store_true",
        help="Download structure factors for unit cell parameters",
    )
    processing_group.add_argument(
        "--selection",
        help="Atom selection string for processing (e.g., 'chain A', 'chain A and resi 10-50', 'protein')",
    )
    processing_group.add_argument(
        "--reweight-occupancies",
        action="store_true",
        help="Normalize occupancies within alternative conformations to sum to 1.0",
    )
    processing_group.add_argument(
        "--individual-maps",
        action="store_true",
        help="Generate individual maps for each alternative conformation (in addition to main processing)",
    )

    # Loop processing options
    loop_group = parser.add_argument_group("Loop Processing Options")
    loop_group.add_argument(
        "--occupancy-loop",
        action="store_true",
        help="Enable occupancy loop processing for altloc conformations",
    )
    loop_group.add_argument(
        "--occupancy-pairs",
        help="Occupancy pairs as string (e.g., '0.0,1.0;0.25,0.75;0.5,0.5')",
    )
    loop_group.add_argument(
        "--occupancy-step",
        type=float,
        default=0.25,
        help="Step size for automatic occupancy pair generation (default: 0.25)",
    )
    loop_group.add_argument(
        "--ensemble-mode",
        action="store_true",
        help="Process multiple structures as ensemble (requires --structure-list)",
    )
    loop_group.add_argument(
        "--individual-conformations",
        action="store_true",
        help="Also create individual conformation maps during loop processing",
    )

    # XMap enhancement options
    xmap_group = parser.add_argument_group("XMap Enhancement Options")
    xmap_group.add_argument(
        "--expand-to-p1",
        action="store_true",
        help="Expand maps to P1 canonical unit cell",
    )
    xmap_group.add_argument(
        "--carve-around", action="store_true", help="Carve maps around selected atoms"
    )
    xmap_group.add_argument(
        "--carve-selection",
        help="Atom selection for carving (e.g., 'chain A and resi 120-140')",
    )
    xmap_group.add_argument(
        "--carve-padding",
        type=float,
        default=3.0,
        help="Padding around atoms for carving in Angstroms (default: 3.0)",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.carve_around and not args.carve_selection:
        print("✗ --carve-around requires --carve-selection")
        return

    if args.ensemble_mode and not args.structure_list:
        print("✗ --ensemble-mode requires --structure-list")
        return

    if args.structure_list and args.structure_names:
        if len(args.structure_list) != len(args.structure_names):
            print("✗ --structure-names must have same length as --structure-list")
            return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine carve selection from arguments
    carve_selection = args.carve_selection if args.carve_around else None

    # Parse or generate occupancy pairs for loop processing
    occupancy_pairs = None
    if args.occupancy_loop:
        if args.occupancy_pairs:
            occupancy_pairs = parse_occupancy_pairs(args.occupancy_pairs)
        else:
            occupancy_pairs = generate_occupancy_pairs(args.occupancy_step)
        print(f"Using occupancy pairs: {occupancy_pairs}")

    # Determine reference map file if structure factors are requested or reference map provided
    def get_reference_map_file(pdb_id, output_dir):
        if args.reference_map:
            if os.path.exists(args.reference_map):
                print(f"Using provided reference map: {args.reference_map}")
                return args.reference_map
            else:
                print(f"Warning: Reference map file not found: {args.reference_map}")
                return None
        elif args.download_sf:
            return download_structure_factors(pdb_id, output_dir)
        return None

    if args.batch_csv:
        # Batch processing mode
        print("Note: Loop processing and XMap enhancements not supported in batch mode")
        process_batch_from_csv(
            args.batch_csv,
            args.output_dir,
            args.resolution,
            args.download_sf,
            args.altloc_data_dir,
        )

    elif args.ensemble_mode:
        # Ensemble processing mode
        if not args.structure_list:
            print("✗ --ensemble-mode requires --structure-list")
            return

        # Load structures
        structures = []
        for structure_path in args.structure_list:
            if not os.path.exists(structure_path):
                print(f"✗ Structure file not found: {structure_path}")
                return
            try:
                structure = Structure.fromfile(structure_path)
                try:
                    structure = structure.reorder().complete_residues().reorder()
                except Exception:
                    pass
                structures.append(structure)
                print(f"Loaded structure: {structure_path}")
            except Exception as e:
                print(f"✗ Failed to load structure {structure_path}: {e}")
                return

        # Use provided names or generate from filenames
        if args.structure_names:
            structure_names = args.structure_names
        else:
            structure_names = [
                os.path.splitext(os.path.basename(path))[0]
                for path in args.structure_list
            ]

        # Generate ensemble PDB ID
        ensemble_id = "_".join(structure_names[:2])  # Use first two names

        if args.occupancy_loop and occupancy_pairs:
            # Ensemble occupancy loop
            reference_map_file = get_reference_map_file(ensemble_id, args.output_dir)
            process_ensemble_occupancy_loop(
                structures,
                structure_names,
                ensemble_id,
                args.output_dir,
                args.resolution,
                occupancy_pairs,
                reference_map_file,
                args.individual_conformations,
                args.expand_to_p1,
                carve_selection,
                args.carve_padding,
                args.reweight_occupancies,
            )
        else:
            # Single ensemble processing
            print(f"Processing ensemble: {ensemble_id}")

            # Normalize occupancies if requested
            if args.reweight_occupancies:
                normalized_structures = []
                for i, structure in enumerate(structures):
                    normalized = structure.reweight_occupancies()
                    normalized_structures.append(normalized)
                    print(f"Normalized occupancies for ensemble structure {i + 1}")
                    _set_default_b_factors(normalized, value=20.0, force=True)
                structures = normalized_structures

            ensemble = Ensemble(structures)

            # Setup density generator
            reference_map_file = get_reference_map_file(ensemble_id, args.output_dir)
            if reference_map_file and os.path.exists(reference_map_file):
                density_generator = SyntheticDensityGenerator(
                    ensemble, reference_map_file
                )
                args.resolution = (
                    density_generator.xmap.resolution.high
                )  # Use resolution from reference map
            else:
                density_generator = SyntheticDensityGenerator(
                    structure=ensemble,
                    resolution=args.resolution,
                    coordinate_mode="crystallographic",
                )

            # Generate density map
            density = density_generator.generate_map()

            # Apply XMap enhancements
            if args.expand_to_p1 or carve_selection:
                density = density_generator.expand_and_carve_density_map(
                    density, args.expand_to_p1, carve_selection, args.carve_padding
                )

            # Save ensemble map
            suffix = ""
            if args.expand_to_p1:
                suffix += "_p1"
            if carve_selection:
                suffix += "_carved"

            density_file = os.path.join(
                args.output_dir,
                f"{ensemble_id}_ensemble{suffix}_{args.resolution:.2f}A.ccp4",
            )
            density_generator.save_map(density_file, density)

            ensemble_file = os.path.join(
                args.output_dir, f"{ensemble_id}_ensemble{suffix}.cif"
            )
            ensemble.tofile(ensemble_file)

            print(f"✓ Generated ensemble files: {density_file}")

    elif args.pdb_id or args.input_structure:
        # Single structure mode
        if args.pdb_id:
            pdb_id = args.pdb_id
            structure_path = args.input_structure
            if not structure_path:
                # Look for structure in altloc data directory
                potential_paths = [
                    os.path.join(args.altloc_data_dir, f"{args.pdb_id.lower()}.cif"),
                    os.path.join(args.altloc_data_dir, f"{args.pdb_id.upper()}.cif"),
                ]
                for path in potential_paths:
                    if os.path.exists(path):
                        structure_path = path
                        break
        else:
            structure_path = args.input_structure
            pdb_id = os.path.splitext(os.path.basename(args.input_structure))[0]

        if not structure_path or not os.path.exists(structure_path):
            print(f"✗ Structure file not found for {pdb_id}")
            return

        # Load structure
        try:
            structure = Structure.fromfile(structure_path)
            try:
                structure = structure.reorder().complete_residues().reorder()
            except Exception:
                pass
            print(f"Loaded structure: {structure}")
        except Exception as e:
            print(f"✗ Failed to load structure {structure_path}: {e}")
            return

        if args.occupancy_loop and occupancy_pairs:
            # Single structure occupancy loop
            reference_map_file = get_reference_map_file(pdb_id, args.output_dir)
            process_occupancy_loop(
                structure,
                pdb_id,
                args.output_dir,
                args.resolution,
                occupancy_pairs,
                reference_map_file,
                args.individual_conformations,
                args.expand_to_p1,
                carve_selection,
                args.carve_padding,
                args.reweight_occupancies,
            )
        else:
            # Standard single structure processing with enhancements
            reference_map_file = get_reference_map_file(pdb_id, args.output_dir)

            # Normalize occupancies if requested
            if args.reweight_occupancies:
                structure = structure.reweight_occupancies()
                print("Normalized occupancies within alternative conformations")
                _set_default_b_factors(structure, value=20.0, force=True)

            # Filter by selection if specified
            if args.selection:
                try:
                    selection_mask = structure.select(args.selection)
                    if selection_mask.sum() == 0:
                        print(f"Warning: Selection '{args.selection}' matched no atoms")
                    else:
                        structure = structure.extract(selection_mask)
                        structure = structure.reorder()
                        print(
                            f"Applied selection '{args.selection}', {len(selection_mask)} atoms selected"
                        )
                except Exception as e:
                    print(f"Error applying selection '{args.selection}': {e}")
                    print("Proceeding with full structure")

            # Generate main structure map (use first conformation or cleaned structure)
            if args.individual_maps:
                # Extract alternative conformations
                conformations = extract_altloc_conformations(structure, args.selection)
                main_structure = conformations[0] if conformations else structure
            else:
                # Use cleaned structure as-is
                main_structure = structure

            # Generate main density map
            try:
                # Setup density generator
                if reference_map_file and os.path.exists(reference_map_file):
                    density_generator = SyntheticDensityGenerator(
                        main_structure, reference_map_file
                    )
                    args.resolution = (
                        density_generator.xmap.resolution.high
                    )  # Use resolution from reference map
                else:
                    density_generator = SyntheticDensityGenerator(
                        main_structure, resolution=args.resolution
                    )

                # Generate density map
                density = density_generator.generate_map()

                # Apply XMap enhancements
                if args.expand_to_p1 or carve_selection:
                    density = density_generator.expand_and_carve_density_map(
                        density, args.expand_to_p1, carve_selection, args.carve_padding
                    )

                # Save main files
                suffix = ""
                if args.expand_to_p1:
                    suffix += "_p1"
                if carve_selection:
                    suffix += "_carved"

                density_file = os.path.join(
                    args.output_dir,
                    f"{pdb_id.lower()}{suffix}_{args.resolution:.2f}A.ccp4",
                )
                density_generator.save_map(density_file, density)

                structure_file = os.path.join(
                    args.output_dir, f"{pdb_id.lower()}{suffix}.cif"
                )
                main_structure.tofile(structure_file)

                print(f"✓ Generated main map: {density_file}")

            except Exception as e:
                print(f"✗ Failed to generate main density map: {e}")

            # Generate individual conformation maps if requested
            if args.individual_maps:
                conformations = extract_altloc_conformations(structure, args.selection)
                print(
                    f"Generating individual maps for {len(conformations)} conformations"
                )

                for i, conformation in enumerate(conformations):
                    conformation_name = f"conf_{i + 1}" if i > 0 else "main"

                    try:
                        # Setup density generator
                        if reference_map_file and os.path.exists(reference_map_file):
                            density_generator = SyntheticDensityGenerator(
                                conformation, reference_map_file
                            )
                            args.resolution = (
                                density_generator.xmap.resolution.high
                            )  # Use resolution from reference map
                        else:
                            density_generator = SyntheticDensityGenerator(
                                conformation, resolution=args.resolution
                            )

                        # Generate density map
                        density = density_generator.generate_map()

                        # Apply XMap enhancements
                        if args.expand_to_p1 or carve_selection:
                            density = density_generator.expand_and_carve_density_map(
                                density,
                                args.expand_to_p1,
                                carve_selection,
                                args.carve_padding,
                            )

                        # Save individual conformation files
                        suffix = f"_{conformation_name}"
                        if args.expand_to_p1:
                            suffix += "_p1"
                        if carve_selection:
                            suffix += "_carved"

                        density_file = os.path.join(
                            args.output_dir,
                            f"{pdb_id.lower()}{suffix}_{args.resolution:.2f}A.ccp4",
                        )
                        density_generator.save_map(density_file, density)

                        structure_file = os.path.join(
                            args.output_dir, f"{pdb_id.lower()}{suffix}.cif"
                        )
                        conformation.tofile(structure_file)

                        print(f"✓ Generated {conformation_name}: {density_file}")

                    except Exception as e:
                        print(
                            f"✗ Failed to generate density for {conformation_name}: {e}"
                        )

    else:
        print(
            "Please specify --pdb-id, --input-structure, --structure-list, or --batch-csv"
        )
        parser.print_help()


if __name__ == "__main__":
    import sys

    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        main()
    else:
        # No arguments provided - show help
        print(
            "Synthetic density map generator with loop processing and XMap enhancements"
        )
        print("Use --help to see all available options")
        print()
        print("Examples:")
        print("  # Basic single structure processing:")
        print("  python synthetic_density.py --input-structure path/to/structure.cif")
        print()
        print("  # Occupancy loop for altloc conformations:")
        print("  python synthetic_density.py --pdb-id 6b8x --occupancy-loop")
        print()
        print("  # Ensemble processing with custom occupancy pairs:")
        print(
            "  python synthetic_density.py --ensemble-mode --structure-list struct1.cif struct2.cif --occupancy-loop --occupancy-pairs '0.0,1.0;0.5,0.5;1.0,0.0'"
        )
        print()
        print("  # P1 expansion and carving:")
        print(
            "  python synthetic_density.py --input-structure structure.cif --expand-to-p1 --carve-around --carve-selection 'chain A and resi 120-140'"
        )
        print()
        print("Run with --help for complete documentation.")
