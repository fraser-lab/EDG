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
import numpy as np
import matplotlib.pyplot as plt
import yaml

from edg.config import ExperimentConfig
from edg.config.optimizer_factory import (
    create_optimizer_from_config,
    process_structure_from_config,
    prepare_optimization_kwargs_from_config,
)
from edg.data.structure import Structure, Ensemble
from edg.utils.shared_input import (
    validate_shared_input_compatibility,
    copy_boltz_input_to_shared,
)

# Type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edg.data.structure import Structure


logger = logging.getLogger(__name__)


def validate_boltz_yaml_compatibility(
    boltz_yaml_path: Path, structure_path: str = None, model_config=None
) -> bool:
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
    # Mark unused parameters to avoid linting warnings
    _ = structure_path
    _ = model_config
    try:
        # First, do basic YAML format validation
        with open(boltz_yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        if not isinstance(yaml_data, dict) or "sequences" not in yaml_data:
            logger.error("Boltz YAML must contain a 'sequences' key")
            return False

        if (
            not isinstance(yaml_data["sequences"], list)
            or len(yaml_data["sequences"]) == 0
        ):
            logger.error("Boltz YAML 'sequences' must be a non-empty list")
            return False

        # Check that sequences are properly formatted
        for i, seq_entry in enumerate(yaml_data["sequences"]):
            if not isinstance(seq_entry, dict) or (
                "protein" not in seq_entry and "ligand" not in seq_entry
            ):
                logger.error(
                    f"Sequence entry {i} must have a 'protein' or 'ligand' key"
                )
                return False

            # Handle protein sequences
            if "protein" in seq_entry:
                protein = seq_entry["protein"]
                if (
                    not isinstance(protein, dict)
                    or "id" not in protein
                    or "sequence" not in protein
                ):
                    logger.error(
                        f"Protein sequence entry {i} must have 'id' and 'sequence' keys"
                    )
                    return False

                if (
                    not isinstance(protein["sequence"], str)
                    or len(protein["sequence"]) == 0
                ):
                    logger.error(
                        f"Protein sequence entry {i} must have a non-empty sequence string"
                    )
                    return False

            # Handle ligand sequences
            elif "ligand" in seq_entry:
                ligand = seq_entry["ligand"]
                if not isinstance(ligand, dict) or "id" not in ligand:
                    logger.error(f"Ligand sequence entry {i} must have 'id' key")
                    return False

                # Check for either 'ccd' or 'smiles' key
                if "ccd" not in ligand and "smiles" not in ligand:
                    logger.error(
                        f"Ligand sequence entry {i} must have either 'ccd' or 'smiles' key"
                    )
                    return False

                # Validate ccd if present
                if "ccd" in ligand:
                    if not isinstance(ligand["ccd"], str) or len(ligand["ccd"]) == 0:
                        logger.error(
                            f"Ligand sequence entry {i} must have a non-empty ccd string"
                        )
                        return False

                # Validate smiles if present
                if "smiles" in ligand:
                    if (
                        not isinstance(ligand["smiles"], str)
                        or len(ligand["smiles"]) == 0
                    ):
                        logger.error(
                            f"Ligand sequence entry {i} must have a non-empty smiles string"
                        )
                        return False

        # Count sequence types for logging
        protein_count = sum(1 for seq in yaml_data["sequences"] if "protein" in seq)
        ligand_count = sum(1 for seq in yaml_data["sequences"] if "ligand" in seq)

        logger.info(
            f"Boltz YAML format validation passed with {len(yaml_data['sequences'])} sequences "
            f"({protein_count} protein, {ligand_count} ligand)"
        )

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

    # Check if we're using a shared input directory
    using_shared_input = config.shared_input_dir is not None

    if using_shared_input:
        logger.info("Using shared input directory for parameter sweep optimization")
        return run_experiment_with_shared_input(config)
    else:
        logger.info("Running standalone experiment")
        return run_experiment_standalone(config)


def run_experiment_standalone(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a standalone experiment without shared input.

    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment configuration

    Returns
    -------
    Dict[str, Any]
        Experiment results including final structures and scores
    """
    logger.info(f"Running standalone experiment: {config.name}")

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

    # Run the main experiment logic
    return run_experiment_main_logic(config, input_data_dir, run_output_dir)


def run_experiment_with_shared_input(config: ExperimentConfig) -> Dict[str, Any]:
    """Run an experiment using shared input directory.

    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment configuration

    Returns
    -------
    Dict[str, Any]
        Experiment results including final structures and scores
    """
    logger.info(f"Running experiment with shared input: {config.name}")

    shared_input_dir = Path(config.shared_input_dir)
    shared_output_dir = Path(
        config.output_dir
    )  # This should be the shared output directory

    # Validate shared input compatibility
    if not validate_shared_input_compatibility(config, shared_input_dir):
        raise ValueError(
            f"Shared input directory is not compatible with current configuration: {shared_input_dir}"
        )

    # Check if we need to copy Boltz input YAML to shared directory
    if config.boltz_input_yaml:
        shared_yaml_path = shared_input_dir / f"{config.name}_shared.yaml"
        if not shared_yaml_path.exists():
            logger.info("Copying Boltz input YAML to shared directory")
            copy_boltz_input_to_shared(config, shared_input_dir)

    # Create run-specific output directory (this will be moved later)
    run_output_dir = create_run_output_dir(config, shared_output_dir)

    # Save the final configuration used to the run-specific directory
    config_output_path = run_output_dir / "experiment_config.yaml"
    from edg.config import save_config

    save_config(config, config_output_path)
    logger.info(f"Saved experiment configuration to {config_output_path}")

    # Run the main experiment logic
    return run_experiment_main_logic(config, shared_input_dir, run_output_dir)


def run_experiment_main_logic(
    config: ExperimentConfig, input_data_dir: Path, run_output_dir: Path
) -> Dict[str, Any]:
    """Main experiment logic that can be shared between standalone and shared input modes.

    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment configuration
    input_data_dir : Path
        Input data directory (either standalone or shared)
    run_output_dir : Path
        Run-specific output directory

    Returns
    -------
    Dict[str, Any]
        Experiment results including final structures and scores
    """
    # Initialize the optimizer first (we need the structure to create proper Boltz YAML)
    logger.info("Initializing optimizer...")

    # Use existing Boltz input YAML if provided, otherwise create one after optimizer init
    if config.boltz_input_yaml:
        # For shared input mode, check if we need to handle the YAML path differently
        if config.shared_input_dir:
            # Use the shared YAML file that was copied earlier
            shared_yaml_path = input_data_dir / f"{config.name}_shared.yaml"
            if shared_yaml_path.exists():
                input_yaml_path = shared_yaml_path
            else:
                input_yaml_path = Path(config.boltz_input_yaml)
        else:
            input_yaml_path = Path(config.boltz_input_yaml)

        # Validate that the file exists
        if not input_yaml_path.exists():
            raise FileNotFoundError(
                f"Boltz input YAML file not found: {input_yaml_path}"
            )

        # Validate that the Boltz YAML is compatible with the structure
        logger.info("Validating Boltz YAML compatibility with structure...")
        if not validate_boltz_yaml_compatibility(
            input_yaml_path, config.structure.structure_path, config.model
        ):
            raise ValueError(
                "Boltz YAML is not compatible with the structure file. "
                "Please check that the sequences in the YAML match the structure."
            )

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
            optimizer.structure = Ensemble.fromfile(config.structure.structure_path)[
                0
            ]  # Take first model

        logger.info("Processing input structure...")
        optimizer.structure = process_structure_from_config(optimizer.structure, config)

    else:
        # No Boltz YAML provided - warn user and fallback to sequence extraction
        logger.warning(
            "No Boltz input YAML provided. Falling back to extracting sequences from structure file."
        )
        logger.warning(
            "This may cause tensor size mismatches if the structure has unexpected atom counts."
        )
        logger.warning("Consider providing a Boltz YAML file for better compatibility.")

        # Load structure directly (without Boltz processing)
        logger.info("Loading structure directly...")
        extension = os.path.splitext(config.structure.structure_path)[1]
        if extension not in (".cif", ".pdb", ".mmcif"):
            raise ValueError("Structure file must be in mmCIF or PDB format.")

        if extension in (".pdb",):
            structure = Structure.fromfile(config.structure.structure_path)
        else:
            structure = Ensemble.fromfile(config.structure.structure_path)[
                0
            ]  # Take first model

        # Process structure according to configuration
        logger.info("Processing structure according to configuration...")
        structure = process_structure_from_config(structure, config)

        # Extract sequences and ligands from processed structure
        logger.info("Extracting sequences and ligands from structure...")
        sequences, ligands = extract_sequences_and_ligands_from_structure(
            structure, config
        )

        # Create proper Boltz YAML with sequences
        input_yaml_path = create_boltz_yaml_with_sequences_direct(
            config, input_data_dir, sequences, ligands
        )

        # Create optimizer with proper Boltz YAML
        optimizer = create_optimizer_from_config(config, input_yaml_path)

        # Set the processed structure on the optimizer
        optimizer.structure = structure

    # Prepare optimization arguments
    optimize_kwargs = prepare_optimization_kwargs_from_config(
        config, run_output_dir, optimizer
    )

    # Run the optimization
    logger.info(f"Starting optimization with {config.diffusion.num_steps} steps...")
    final_structures, scores = optimizer.optimize(**optimize_kwargs)

    # Save results
    results = save_results(final_structures, scores, run_output_dir, config)

    logger.info(f"Experiment completed. Results saved to {run_output_dir}")

    return results


def create_boltz_yaml_with_sequences_direct(
    config: ExperimentConfig,
    input_data_dir: Path,
    sequences: List[Dict[str, str]],
    ligands: List[Dict[str, str]],
) -> Path:
    """Create proper Boltz YAML file with sequences and ligands.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    input_data_dir : Path
        Directory for input data files
    sequences : List[Dict[str, str]]
        List of protein sequences with 'id' and 'sequence' keys
    ligands : List[Dict[str, str]]
        List of ligands with 'id' and 'ccd' keys

    Returns
    -------
    Path
        Path to created Boltz YAML file
    """
    yaml_path = input_data_dir / f"{config.name}.yaml"

    # Create proper Boltz YAML format
    yaml_content = "sequences:\n"

    # Add protein sequences
    for seq_info in sequences:
        yaml_content += "  - protein:\n"
        yaml_content += f"      id: {seq_info['id']}\n"
        yaml_content += f"      sequence: {seq_info['sequence']}\n"

    # Add ligand sequences (if configuration allows)
    for ligand_info in ligands:
        yaml_content += "  - ligand:\n"
        yaml_content += f"      id: {ligand_info['id']}\n"
        yaml_content += f"      ccd: {ligand_info['ccd']}\n"

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    logger.info(
        f"Created Boltz YAML with {len(sequences)} protein sequences and {len(ligands)} ligands at {yaml_path}"
    )
    return yaml_path


def extract_sequences_and_ligands_from_structure(
    structure: "Structure", config: ExperimentConfig
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Extract protein sequences and ligands from structure for Boltz YAML format.

    Only includes residues that can be completed by Structure.complete_residues()
    and conditionally includes ligands based on structure cleaning configuration.

    Parameters
    ----------
    structure : Structure
        Processed structure object
    config : ExperimentConfig
        Experiment configuration for structure cleaning settings

    Returns
    -------
    Tuple[List[Dict[str, str]], List[Dict[str, str]]]
        Tuple of (protein sequences, ligand sequences) where:
        - protein sequences have 'id' and 'sequence' keys
        - ligand sequences have 'id' and 'ccd' keys
    """
    sequences = []
    ligands = []

    # Determine if ligands should be included based on configuration
    include_ligands = (
        config.structure.keep_type == "all" and not config.structure.remove_all_ligands
    )

    # Extract protein sequences from residues (property handles building internally)
    for residue in structure.single_conformer_residues:
        # Only include residues that have backbone atoms (completable by complete_residues)
        if hasattr(residue, "active") and len(residue.active) >= 4:
            # Check if first 4 atoms (backbone) are active - matching complete_residues() logic
            if not np.all(residue.active[:4]):
                logger.debug(
                    f"Skipping residue {residue.resn[0]} {residue.resi[0]} with missing backbone atoms"
                )
                continue

        # Get chain ID and residue name
        chain_id = residue.chain[0]
        resn = residue.resn[0]

        # Convert 3-letter amino acid codes to 1-letter
        # Includes standard amino acids and common non-canonical residues
        aa_mapping = {
            # Standard amino acids
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLU": "E",
            "GLN": "Q",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
            # Non-canonical and modified amino acids (NOTE: Some of these will still break due to atom count mismatches, but at least we can see the conversion is better)
            "MSE": "M",  # Selenomethionine -> Methionine
            "SEP": "S",  # Phosphoserine -> Serine
            "TPO": "T",  # Phosphothreonine -> Threonine
            "PTR": "Y",  # Phosphotyrosine -> Tyrosine
            "MLY": "K",  # N-dimethyl-lysine -> Lysine
            "MLZ": "K",  # N-methyl-lysine -> Lysine
            "MEN": "N",  # N-methyl-asparagine -> Asparagine
            "HYP": "P",  # 4-hydroxyproline -> Proline
            "CSO": "C",  # S-hydroxycysteine -> Cysteine
            "OCS": "C",  # Cysteinesulfenic acid -> Cysteine
            "PCA": "E",  # Pyroglutamic acid -> Glutamic acid
            "KCX": "K",  # Lysine NZ-carboxylic acid -> Lysine
            "CAS": "C",  # S-(dimethylarsenic)cysteine -> Cysteine
            "CSD": "C",  # 3-sulfinoalanine -> Cysteine
            "CME": "C",  # S,S-(2-hydroxyethyl)cysteine -> Cysteine
            "CXM": "M",  # N-carboxymethionine -> Methionine
            "DAL": "A",  # D-alanine -> Alanine
            "DAS": "D",  # D-aspartic acid -> Aspartic acid
            "DCY": "C",  # D-cysteine -> Cysteine
            "DGL": "E",  # D-glutamic acid -> Glutamic acid
            "DGN": "Q",  # D-glutamine -> Glutamine
            "DHI": "H",  # D-histidine -> Histidine
            "DIL": "I",  # D-isoleucine -> Isoleucine
            "DLE": "L",  # D-leucine -> Leucine
            "DLY": "K",  # D-lysine -> Lysine
            "DNE": "N",  # D-asparagine -> Asparagine
            "DPN": "F",  # D-phenylalanine -> Phenylalanine
            "DPR": "P",  # D-proline -> Proline
            "DSN": "S",  # D-serine -> Serine
            "DTH": "T",  # D-threonine -> Threonine
            "DTR": "W",  # D-tryptophan -> Tryptophan
            "DTY": "Y",  # D-tyrosine -> Tyrosine
            "DVA": "V",  # D-valine -> Valine
            "M3L": "K",  # N-trimethyllysine -> Lysine
            "FME": "M",  # N-formylmethionine -> Methionine
            "NEP": "H",  # N1-phosphonohistidine -> Histidine
            "ALY": "K",  # N(6)-acetyllysine -> Lysine
            "ARM": "R",  # Deoxy-D-arginine -> Arginine
            "ASX": "N",  # Asparagine or aspartic acid -> Asparagine
            "GLX": "Q",  # Glutamine or glutamic acid -> Glutamine
            "LLP": "K",  # 2-lysine -> Lysine
            "MGN": "Q",  # 2-methylglutamine -> Glutamine
            "MHS": "H",  # 1-N-methylhistidine -> Histidine
            "TRN": "W",  # AZA-tryptophan -> Tryptophan
            "SAC": "S",  # N-acetylserine -> Serine
            "SAR": "G",  # Sarcosine -> Glycine
            "SET": "S",  # S-ethylcysteine -> Serine
            "SUI": "G",  # 1,2-dihydro-2-oxopyrimidine-4-carboxylic acid -> Glycine
        }

        if resn in aa_mapping:
            # Log non-canonical residue conversions
            converted_aa = aa_mapping[resn]
            if resn not in [
                "ALA",
                "ARG",
                "ASN",
                "ASP",
                "CYS",
                "GLU",
                "GLN",
                "GLY",
                "HIS",
                "ILE",
                "LEU",
                "LYS",
                "MET",
                "PHE",
                "PRO",
                "SER",
                "THR",
                "TRP",
                "TYR",
                "VAL",
            ]:
                logger.debug(
                    f"Converting non-canonical residue {resn} {residue.resi[0]} to {converted_aa}"
                )

            # Find or create sequence for this chain
            chain_seq = None
            for seq in sequences:
                if seq["id"] == chain_id:
                    chain_seq = seq
                    break

            if chain_seq is None:
                chain_seq = {"id": chain_id, "sequence": "", "residues": []}
                sequences.append(chain_seq)

            # Add residue info for sorting
            chain_seq["residues"].append({"resi": residue.resi[0], "aa": converted_aa})
        else:
            logger.warning(
                f"Skipping unknown residue type: {resn} {residue.resi[0]} in chain {chain_id}"
            )

    # Sort residues by number and build final sequences
    for seq in sequences:
        # Sort residues by residue number
        seq["residues"].sort(key=lambda x: x["resi"])
        # Build sequence string
        seq["sequence"] = "".join([r["aa"] for r in seq["residues"]])
        # Remove temporary residues list
        del seq["residues"]

    # Extract ligands if configuration allows
    if include_ligands:
        for ligand in structure.ligands:
            # Get ligand identifier
            chain_id = ligand.chain[0]
            resi = ligand.resi[0]
            icode = ligand.icode[0] if hasattr(ligand, "icode") else ""

            # Create ligand ID
            ligand_id = f"{chain_id}_{resi}"
            if icode:
                ligand_id += f"_{icode}"

            # Get CCD code from residue name
            ccd_code = ligand.resn[0]

            ligands.append({"id": ligand_id, "ccd": ccd_code})

    logger.debug(
        f"Extracted {len(sequences)} protein chains and {len(ligands)} ligands (include_ligands={include_ligands})"
    )
    return sequences, ligands


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
        config.name,
        config.model.version,
        f"{config.density.resolution}A" if config.density.resolution else "auto_res",
        f"{config.diffusion.num_steps}steps",
        f"{config.steering.num_particles}particles"
        if config.steering.enabled
        else "no_steering",
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
    config: ExperimentConfig,
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
        np.savetxt(
            output_dir / "scores.csv", scores, delimiter=",", header="step,score"
        )

    # Log final statistics
    if scores:
        final_score = scores[-1]
        initial_score = scores[0] if len(scores) > 0 else None
        logger.info(f"Final score: {final_score.mean():.6f}")
        if initial_score is not None:
            logger.info(
                f"Score improvement: {initial_score.mean():.6f} → {final_score.mean():.6f}"
            )

        results["final_score"] = float(final_score.mean())
        results["initial_score"] = (
            float(initial_score.mean()) if initial_score is not None else None
        )

    return results
