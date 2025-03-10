"""Preprocess SF-CIF files into maps for utilization with this method."""

# James Holton's advice on how to make these CryoEM-similar maps

# expand to P1
# calculate the mFo-DFc difference map
# calculate an Fcalc map for just the protomer of interest -> done in PHENIX
# make your "carving" mask around the protomer of interest
# I prefer to make a "feathered" mask, dropping from "1" to "0" at non-bonding distances, say 3.5 to 4.5 A - I have a script for this
# apply the mask to the mFo-DFc map
# add the "carved" Fo-Fc map to the Fcalc map. this effectively subtracts the symmetry mates

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import subprocess
import urllib.request
import gemmi
import logging
import os


@dataclass
class ProcessingConfig:
    """Configuration for structure factor processing pipeline.

    Parameters
    ----------
    output_dir : str
        Directory for output files and processing results.
    selection : str
        Selection string for map calculation.
    refinement_script : str
        Path to refinement script.
    ignore_symmetry_conflicts : bool, optional
        Whether to ignore symmetry conflicts, by default True.
    mask_atoms : bool, optional
        Whether to mask atoms, by default True.
    wrapping : bool, optional
        Whether to enable wrapping, by default True.
    mtz_labels : List[str], optional
        Labels for MTZ processing.
    phenix_path : Optional[str], optional
        Path to PHENIX installation.
    max_workers : int, optional
        Maximum number of parallel workers.
    em : bool, optional
        Whether the input should be processed as a CryoEM map.
    """

    output_dir: str
    selection: str
    refinement_script: str
    map_box_script: str
    ignore_symmetry_conflicts: bool = True
    mask_atoms: bool = True
    wrapping: bool = True
    mtz_labels: List[str] = field(default_factory=lambda: ["2FOFCWT", "PH2FOFCWT"])
    phenix_path: Optional[str] = None
    max_workers: int = 4
    em: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if not Path(self.refinement_script).exists():
            raise ValueError(f"Refinement script not found: {self.refinement_script}")
        if self.phenix_path and not Path(self.phenix_path).exists():
            raise ValueError(f"PHENIX path not found: {self.phenix_path}")


class ProcessingError(Exception):
    """Custom exception for processing errors."""

    pass


class SFProcessor:
    """Structure factor processing pipeline.

    Handles downloading, conversion, refinement, and map calculation for
    structure factor data from the RCSB PDB.
    """

    def __init__(self, config: ProcessingConfig):
        """Initialize the processor with configuration.

        Parameters
        ----------
        config : ProcessingConfig
            Processing configuration parameters.
        """
        self.config = config
        self.download_dir = Path(config.output_dir)
        self.processed_dir = Path(config.output_dir) / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure logging for the processing pipeline."""
        self.log_file = self.processed_dir / "processing.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def process_structure(self, pdb_id: str, pdb_instead: bool = False) -> None:
        """Process a single structure through the pipeline.

        Parameters
        ----------
        pdb_id : str
            PDB ID to process.
        pdb_instead : bool
            Whether to download a PDB file instead of mmCIF.
        """
        sf_file = self.download_sf(pdb_id)
        cif_file, fasta_file = self.download_mmcif_and_sequence(pdb_id, pdb_instead)
        mtz_file = self.convert_to_mtz(sf_file)
        self.logger.info(f"Converted {pdb_id} to MTZ: {mtz_file}")

        refined_files = self.refine_structure(mtz_file, cif_file, fasta_file)
        p1_mtz = self.expand_to_p1(refined_files["mtz"])
        self.calculate_map(refined_files["cif"], p1_mtz)

    def run_subprocess(
        self, cmd: List[str], description: str, log_output: bool = False
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command with logging.

        Parameters
        ----------
        cmd : List[str]
            Command to execute.
        description : str
            Description of the command for logging.

        Returns
        -------
        subprocess.CompletedProcess
            Completed process result.
        """
        self.logger.info(f"Running {description}: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if log_output:
                self.logger.info(f"Command output:\n{result.stdout}")
            else:
                self.logger.debug(f"Command output:\n{result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed with exit code {e.returncode}")
            self.logger.error(f"Error output:\n{e.stderr}")
            self.logger.error(f"Command output:\n{e.stdout}")
            raise

    def download_sf(self, pdb_id: str) -> Path:
        """Download structure factors from RCSB.

        Parameters
        ----------
        pdb_id : str
            PDB ID to download.

        Returns
        -------
        Path
            Path to downloaded file.
        """
        sf_url = f"https://files.rcsb.org/download/{pdb_id}-sf.cif"
        output_file = self.download_dir / f"{pdb_id}-sf.cif"

        if not output_file.exists():
            try:
                urllib.request.urlretrieve(sf_url, output_file)
                self.logger.info(f"Downloaded {pdb_id} structure factors")
            except urllib.error.URLError as e:
                raise ProcessingError(f"Failed to download {pdb_id}") from e

        return output_file

    def download_mmcif_and_sequence(
        self, pdb_id: str, pdb_instead: bool = False
    ) -> Tuple[Path, Path]:
        """Download mmCIF/PDB file and FASTA from RCSB.

        Parameters
        ----------
        pdb_id : str
            PDB ID to download.
        pdb_instead : bool
            Whether to download a PDB file instead of mmCIF.

        Returns
        -------
        Tuple[Path, Path]
            Path to downloaded file.
        """
        if pdb_instead:
            mmcif_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            output_file = self.download_dir / f"{pdb_id}.pdb"
        else:
            mmcif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            output_file = self.download_dir / f"{pdb_id}.cif"

        fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
        fasta_file = self.download_dir / f"{pdb_id}.fa"

        if not output_file.exists():
            try:
                urllib.request.urlretrieve(mmcif_url, output_file)
                self.logger.info(
                    f"Downloaded {pdb_id} {'PDB' if pdb_instead else 'mmCIF'}"
                )
            except urllib.error.URLError as e:
                raise ProcessingError(f"Failed to download {pdb_id}") from e

        if not fasta_file.exists():
            try:
                urllib.request.urlretrieve(fasta_url, fasta_file)
                self.logger.info(f"Downloaded {pdb_id} FASTA")
            except urllib.error.URLError as e:
                raise ProcessingError(f"Failed to download {pdb_id} FASTA") from e

        return output_file, fasta_file

    def convert_to_mtz(self, sf_file: Path) -> Path:
        """Convert structure factor CIF to MTZ.

        Parameters
        ----------
        sf_file : Path
            Input CIF file.

        Returns
        -------
        Path
            Path to output MTZ file.
        """
        try:
            doc = gemmi.cif.read(str(sf_file))
            rblock = gemmi.as_refln_blocks(doc)[0]
            cif2mtz = gemmi.CifToMtz()
            mtz = cif2mtz.convert_block_to_mtz(rblock)

            output_file = sf_file.with_suffix(".mtz")
            mtz.write_to_file(str(output_file))
            return output_file

        except Exception as e:
            raise ProcessingError(f"Failed to convert {sf_file} to MTZ") from e

    def refine_structure(
        self, mtz_file: Path, cif_file: Path, fasta_file: Path
    ) -> Dict[str, Path]:
        """Refine structure with provided script.

        Parameters
        ----------
        mtz_file : Path
            Input MTZ file.
        cif_file : Path
            Downloaded mmCIF file being processed.

        Returns
        -------
        Dict[str, Path]
            Paths to refined output files.
        """
        cmd = [
            self.config.refinement_script,
            str(mtz_file),
            str(cif_file),
            str(fasta_file),
        ]
        self.run_subprocess(cmd, "structure refinement", True)

        output_prefix = f"{Path(mtz_file).stem}_single_001"
        return {
            "cif": self.processed_dir / f"{output_prefix}.cif",
            "mtz": self.processed_dir / f"{output_prefix}.mtz",
        }

    def expand_to_p1(self, mtz_file: Path) -> Path:
        """Expand MTZ file to P1 space group.

        Parameters
        ----------
        mtz_file : Path
            Input MTZ file.

        Returns
        -------
        Path
            Path to expanded MTZ file.
        """
        try:
            mtz = gemmi.read_mtz_file(str(mtz_file))
            mtz.expand_to_p1()

            output_file = mtz_file.with_name(f"{mtz_file.stem}_P1.mtz")
            mtz.write_to_file(str(output_file))
            return output_file

        except Exception as e:
            raise ProcessingError(f"Failed to expand {mtz_file} to P1") from e

    def calculate_map(self, model_file: Path, mtz_file: Path) -> Path:
        """Calculate electron density map.

        Parameters
        ----------
        model_file : Path
            Input model file.
        mtz_file : Path
            Input MTZ file.

        Returns
        -------
        Path
            Path to output map file.
        """
        cmd = [
            self.config.map_box_script,  # this script only outputs in the same directory as it is run, so need stem in the input models
            str(model_file.name),
            str(mtz_file.name),
            ",".join(self.config.mtz_labels),
            str(self.config.ignore_symmetry_conflicts),
            str(self.config.mask_atoms),
            str(self.config.wrapping),
            str(self.config.selection),
            str(model_file.parent),
        ]

        self.run_subprocess(cmd, "map calculation", log_output=True)
