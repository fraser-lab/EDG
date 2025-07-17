"""Synthetic electron density map generation from atomic structures.

This module provides functionality to generate synthetic electron density maps from
atomic structures (mmCIF, PDB files, or Ensemble objects) with user-defined
experimental parameters.
"""

import os
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from typing import Optional, Union, List, NamedTuple

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
            raise TypeError(
                "Structure must be a file path, Structure object, or Ensemble object"
            )

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
            if not hasattr(self, "resolution") or self.resolution is None:
                raise ValueError(
                    "Resolution must be provided for CCP4, MRC, or MAP files."
                )
            ref_map = XMap.fromfile(reference_map_file, resolution=self.resolution)

        self.xmap = XMap_torch(ref_map, device=self.device)
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

        empty_array = np.zeros((grid_a, grid_b, grid_c), dtype=np.float32)
        # original atom coordinates are at 0, 0, 0 for AF3 predictions, need to offset by half the unit cell
        self.center = (
            unit_cell.c / 2.0,
            unit_cell.b / 2.0,
            unit_cell.a / 2.0,
        )

        ref_map = XMap(
            empty_array,
            grid_parameters=GridParameters(voxelspacing=voxelspacing),
            resolution=Resolution(high=resolution, low=1000.0),
            unit_cell=unit_cell,
            origin=map_origin_cartesian,
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

    def generate_map_from_structure(
        self,
        structure: Structure,
        b_factor_scale: float = 1.0,
        occupancy_scale: float = 1.0,
        shift: bool = True,
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
        shift : bool, optional
            Whether to shift the coordinates to center in the unit cell.

        Returns
        -------
        torch.Tensor
            Generated density map.
        """
        coords, elements, b_factors, occupancies, active, _ = (
            structure_to_density_input(structure)
        )

        # move coords to center of unit cell (0, 0, 0) is 0 A, 0 A, 0 A
        if shift:
            coords = coords - torch.tensor(
                self.center, dtype=coords.dtype, device=coords.device
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
        shift: bool = True,
    ) -> torch.Tensor:
        """Generate a synthetic density map from the structure or ensemble.

        Parameters
        ----------
        b_factor_scale : float, optional
            Scale factor for B-factors.
        occupancy_scale : float, optional
            Scale factor for occupancies.
        shift : bool, optional
            Whether to shift the coordinates to center in the unit cell.

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
        xmap.array = density_map

        # Downsample the density map if requested
        if downsample_to is not None:
            xmap = xmap.downsample_to_resolution(downsample_to, apply_filter=True)

        xmap.tofile(output_file)
        print(f"Saved density map to {output_file}")


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
    structure: Structure, chain_id: str = None
) -> List[Structure]:
    """Extract alternative conformations from a structure.

    Args:
        structure: Input structure
        chain_id: Specific chain to extract (optional)

    Returns:
        List of structures, one for each conformation
    """
    structure = structure.clean_structure()
    structure = structure.complete_residues()
    # Filter by chain if specified
    if chain_id:
        structure = structure.extract(structure.select(f"chain {chain_id}"))

    # Get unique alternative location IDs
    altlocs = set(structure.altloc)
    altlocs.discard("")  # Remove empty altloc

    conformations = []

    # Add main conformation (no altloc or altloc A)
    main_selection = structure.select("altloc '' or altloc A")
    if main_selection.sum() > 0:
        main_struct = structure.extract(main_selection)
        main_struct.altloc = ""  # Clear altloc identifiers
        main_struct.q = 1.0
        conformations.append(main_struct)

    # Add alternative conformations
    for altloc in sorted(altlocs):
        if altloc and altloc != "A":
            alt_selection = structure.select(f"altloc '' or altloc {altloc}")
            if alt_selection.sum() > 0:
                alt_struct = structure.extract(alt_selection)
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
    chain_id: str = None,
):
    """Process a single structure to generate synthetic density maps.

    Args:
        pdb_id: PDB ID for naming
        structure_path: Path to structure file
        output_dir: Output directory
        resolution: Map resolution
        download_sf: Whether to download structure factors
        chain_id: Specific chain to process
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
        print(f"Loaded structure: {structure}")
    except Exception as e:
        print(f"✗ Failed to load structure {structure_path}: {e}")
        return

    # Extract alternative conformations
    conformations = extract_altloc_conformations(structure, chain_id)
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
                f"{pdb_id.lower()}_{conformation_name}_{resolution}A.ccp4",
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
            pdb_output_dir, f"{pdb_id.lower()}_ensemble_{resolution}A.ccp4"
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
        chain_id = row.get("target_chain", None)

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
            pdb_code, structure_path, output_dir, resolution, download_sf, chain_id
        )


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic density maps from protein structures"
    )

    # Input options
    parser.add_argument("--pdb-id", help="PDB ID to process")
    parser.add_argument("--input-structure", help="Path to input structure file")
    parser.add_argument(
        "--batch-csv", help="Path to altloc_summary.csv for batch processing"
    )
    parser.add_argument(
        "--altloc-data-dir",
        default="tests/resources/altloc_data/mmcif_files",
        help="Directory containing mmCIF files from altloc analysis",
    )

    # Output options
    parser.add_argument(
        "--output-dir", default="synthetic_density_output", help="Output directory"
    )
    parser.add_argument(
        "--resolution", type=float, default=2.0, help="Map resolution in Angstroms"
    )

    # Processing options
    parser.add_argument(
        "--download-sf",
        action="store_true",
        help="Download structure factors for unit cell parameters",
    )
    parser.add_argument("--chain-id", help="Specific chain to process")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.batch_csv:
        # Batch processing mode
        process_batch_from_csv(
            args.batch_csv,
            args.output_dir,
            args.resolution,
            args.download_sf,
            args.altloc_data_dir,
        )

    elif args.pdb_id:
        # Single PDB ID mode
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

        if not structure_path:
            print(f"✗ Structure file not found for {args.pdb_id}")
            return

        process_single_structure(
            args.pdb_id,
            structure_path,
            args.output_dir,
            args.resolution,
            args.download_sf,
            args.chain_id,
        )

    elif args.input_structure:
        # Custom structure file mode
        pdb_id = os.path.splitext(os.path.basename(args.input_structure))[0]
        process_single_structure(
            pdb_id,
            args.input_structure,
            args.output_dir,
            args.resolution,
            args.download_sf,
            args.chain_id,
        )

    else:
        print("Please specify --pdb-id, --input-structure, or --batch-csv")
        parser.print_help()


if __name__ == "__main__":
    import sys

    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        main()
    else:
        # Run original test code if no arguments
        print("Running original test code...")
        print("Use --help to see command line options")
        print()
    ### Mac1 synthetic data test case
    # pdb_5sop = Structure.fromfile("/home/kchrispens/adp-replicate/tests/resources/mac1_synthetic/5SOP_modified.pdb")
    # pdb_5soq = Structure.fromfile("/home/kchrispens/adp-replicate/tests/resources/mac1_synthetic/5SOQ_modified.pdb")
    # pdb_5sq8 = Structure.fromfile("/home/kchrispens/adp-replicate/tests/resources/mac1_synthetic/5SQ8_modified.pdb")
    # pdb = Structure.fromfile("/home/kchrispens/adp-replicate/_notebooks/testing_more_AWA/boltz_results_test_more_AWA/predictions/test_more_AWA/boltz_out_more_AWA.cif")
    # pdb = Structure.fromfile("/home/kchrispens/qfit-3.0/tests/more_AWA/multiconformer_model2.pdb")

    # ensemble = Ensemble([pdb_5sop, pdb_5soq, pdb_5sq8])

    # ref_map_file = "/home/kchrispens/adp-replicate/tests/resources/mac1_synthetic/5soq-sf.mtz"
    # ref_map_file = "/home/kchrispens/adp-replicate/tests/resources/more_AWA/rfree_2A_Waltconf_1.ccp4"

    # density_generator = SyntheticDensityGenerator(ensemble, ref_map_file)
    # density_generator = SyntheticDensityGenerator(pdb, ref_map_file, resolution=2.)

    # density = density_generator.generate_map(shift=False) # , occupancy_scale=0.25)

    # density_generator.save_map("/home/kchrispens/adp-replicate/tests/resources/more_AWA/qfit_out.ccp4", density)

    ### PEPTIDE FLIP TEST CASE
    # pdb_7kqp = Structure.fromfile(
    #     "/home/kchrispens/adp-replicate/tests/resources/mac1_adpr_peptideflip/7kqp.cif"
    # )
    # pdb_7tx5 = Structure.fromfile(
    #     "/home/kchrispens/adp-replicate/tests/resources/mac1_adpr_peptideflip/7tx5.cif"
    # )
    # pdb_7kqp = pdb_7kqp.extract(pdb_7kqp.select("(chain A and resi 4-169) or chain C"))
    # pdb_7tx5 = pdb_7tx5.extract(pdb_7tx5.select("(chain A and resi 3-168) or chain B"))
    # pdb_7kqp = pdb_7kqp.reorder()
    # pdb_7tx5 = pdb_7tx5.reorder()
    # pdb_7kqp = pdb_7kqp.remove_alternative_conformations()
    # pdb_7tx5 = pdb_7tx5.remove_alternative_conformations()
    # pdb_7kqp = pdb_7kqp.clean_structure(keep_type="all", remove_all_ligands=False)
    # pdb_7tx5 = pdb_7tx5.clean_structure(keep_type="all", remove_all_ligands=False)
    # pdb_7kqp = pdb_7kqp.reorder()
    # pdb_7tx5 = pdb_7tx5.reorder()
    # pdb_7kqp = pdb_7kqp.complete_residues()
    # pdb_7tx5 = pdb_7tx5.complete_residues()
    # pdb_7kqp = pdb_7kqp.reorder()
    # pdb_7tx5 = pdb_7tx5.reorder()
    # print(pdb_7kqp)
    # print(pdb_7tx5)

    # ensemble = Ensemble([pdb_7kqp, pdb_7tx5])
    # pdb_7kqp.coor = pdb_7kqp.coor - pdb_7kqp.coor.mean() - np.array([-17.0, 7.0, 5.0])
    # ensemble.align_to_reference(pdb_7kqp)

    # unit_cell = UnitCell(50.0, 50.0, 50.0)

    # density_generator = SyntheticDensityGenerator(
    #     structure=ensemble[0], resolution=2.0, unit_cell=unit_cell, em_mode=False
    # )
    # density = density_generator.generate_map(
    #     b_factor_scale=1.0, occupancy_scale=1.0, shift=True
    # )
    # density_generator.save_map(
    #     "/home/kchrispens/adp-replicate/tests/resources/mac1_adpr_peptideflip/mac1_adpr_peptideflip_Aconf_2A.ccp4",
    #     density,
    # )

    # ensemble[0].tofile(
    #     "/home/kchrispens/adp-replicate/tests/resources/mac1_adpr_peptideflip/mac1_adpr_peptideflip_Aconf.cif"
    # )
    # ensemble.coor = ensemble.coor - density_generator.center
    # ensemble[0].tofile(
    #     "/home/kchrispens/adp-replicate/tests/resources/mac1_adpr_peptideflip/mac1_adpr_peptideflip_Aconf_shifted.cif"
    # )

    ### PTP1B TEST CASE
    # pdb = Structure.fromfile(
    #     "/home/kchrispens/adp-replicate/tests/resources/6b8x/processed/6b8x-sf_single_001.cif"
    # )
    # pdb = pdb.clean_structure(keep_type="protein")
    # pdb = pdb.reorder()
    # pdb = pdb.complete_residues()
    # pdb = pdb.reorder()
    # pdb = pdb.extract(pdb.select("not altloc B and not altloc C"))
    # pdb.q = 1.0
    # pdb = pdb.reorder()
    # pdb.coor = pdb.coor - np.array([48.0, 18.0, 0.0])
    # print(pdb)

    # unit_cell = UnitCell(50.0, 60.0, 70.0)

    # density_generator = SyntheticDensityGenerator(
    #     structure=pdb, resolution=2.0, unit_cell=unit_cell, em_mode=False
    # )
    # density = density_generator.generate_map(
    #     b_factor_scale=1.0, occupancy_scale=1.0, shift=True
    # )
    # density_generator.save_map(
    #     "/home/kchrispens/adp-replicate/tests/resources/6b8x/6b8x_1occAconf_2A.ccp4",
    #     density,
    # )

    # pdb.tofile("/home/kchrispens/adp-replicate/tests/resources/6b8x/6b8x_synthetic_1occAconf.cif")
    # pdb.coor = pdb.coor - density_generator.center
    # pdb.tofile(
    #     "/home/kchrispens/adp-replicate/tests/resources/6b8x/6b8x_synthetic_1occAconf_shifted.cif"
    # )

    ### Synthetic AAAWAAA data
    # pdb = Structure.fromfile(
    #     "/home/kchrispens/adp-replicate/tests/resources/unfold/unfold_altconf.cif"
    # )

    # unit_cell = UnitCell(40.0, 40.0, 40.0)

    # density_generator = SyntheticDensityGenerator(
    #     structure=pdb, resolution=8.0, unit_cell=unit_cell, em_mode=False
    # )
    # density = density_generator.generate_map(
    #     b_factor_scale=1.0, occupancy_scale=1.0, shift=False
    # )
    # density_generator.save_map(
    #     "/home/kchrispens/adp-replicate/tests/resources/unfold/unfold_altconf_8.ccp4",
    #     density,
    # )

    # density_generator = SyntheticDensityGenerator(
    #     structure=pdb, resolution=4.0, unit_cell=unit_cell, em_mode=False
    # )
    # density = density_generator.generate_map(
    #     b_factor_scale=1.0, occupancy_scale=1.0, shift=False
    # )
    # density_generator.save_map(
    #     "/home/kchrispens/adp-replicate/tests/resources/unfold/unfold_altconf_4.ccp4",
    #     density,
    # )

    # density_generator = SyntheticDensityGenerator(
    #     structure=pdb, resolution=2.0, unit_cell=unit_cell, em_mode=False
    # )
    # density = density_generator.generate_map(
    #     b_factor_scale=1.0, occupancy_scale=1.0, shift=False
    # )
    # density_generator.save_map(
    #     "/home/kchrispens/adp-replicate/tests/resources/unfold/unfold_altconf_2.ccp4",
    #     density,
    # )

    # density_generator = SyntheticDensityGenerator(
    #     structure=pdb, resolution=1.0, unit_cell=unit_cell, em_mode=False
    # )
    # density = density_generator.generate_map(
    #     b_factor_scale=1.0, occupancy_scale=1.0, shift=False
    # )
    # density_generator.save_map(
    #     "/home/kchrispens/adp-replicate/tests/resources/unfold/unfold_altconf_1.ccp4",
    #     density,
    # )

    # density_generator = SyntheticDensityGenerator(
    #     structure=pdb, resolution=1.0, unit_cell=unit_cell, em_mode=False
    # )
    # density = density_generator.generate_map(
    #     b_factor_scale=1.0, occupancy_scale=1.0, shift=False
    # )
    # density_generator.save_map(
    #     "/home/kchrispens/adp-replicate/tests/resources/unfold/unfold_altconf_4_downsampled_from_1_brickwall_filtered.ccp4",
    #     density,
    #     downsample_to=4.0,
    # )

    # pdb.coor = pdb.coor - density_generator.center
    # pdb.tofile("/home/kchrispens/adp-replicate/tests/resources/unfold/unfold_altconf_shifted.cif")
