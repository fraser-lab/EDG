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
from typing import Optional, Dict, List
import subprocess
import urllib.request
import gemmi
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial


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
        self.processed_dir = Path(config.output_dir) / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)

    def setup_logging(self) -> None:
        """Configure logging for the processing pipeline."""
        self.log_file = self.processed_dir / "processing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def process_structures(self, pdb_ids: List[str]) -> Dict[str, bool]:
        """Process multiple structures in parallel.
        
        Parameters
        ----------
        pdb_ids : List[str]
            List of PDB IDs to process.
            
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping PDB IDs to processing success status.
        """
        process_fn = partial(self.process_structure_safe)
        results = {}
        
        with self.executor as executor:
            futures = {executor.submit(process_fn, pdb_id): pdb_id 
                      for pdb_id in pdb_ids}
            
            for future in futures:
                pdb_id = futures[future]
                try:
                    future.result()
                    results[pdb_id] = True
                except Exception as e:
                    self.logger.error(f"Failed to process {pdb_id}: {str(e)}")
                    results[pdb_id] = False
                    
        return results

    def process_structure_safe(self, pdb_id: str) -> None:
        """Safely process a single structure with error handling.
        
        Parameters
        ----------
        pdb_id : str
            PDB ID to process.
        """
        try:
            self.logger.info(f"Starting processing for {pdb_id}")
            self.process_structure(pdb_id)
            self.logger.info(f"Completed processing for {pdb_id}")
        except Exception as e:
            self.logger.error(f"Error processing {pdb_id}: {str(e)}")
            raise ProcessingError(f"Failed to process {pdb_id}") from e

    def process_structure(self, pdb_id: str) -> None:
        """Process a single structure through the pipeline.
        
        Parameters
        ----------
        pdb_id : str
            PDB ID to process.
        """
        sf_file = self.download_sf(pdb_id)
        cif_file = self.download_mmcif(pdb_id)
        mtz_file = self.convert_to_mtz(sf_file)
        self.logger.info(f"Converted {pdb_id} to MTZ: {mtz_file}")
        
        refined_files = self.refine_structure(mtz_file, cif_file)
        p1_mtz = self.expand_to_p1(refined_files["mtz"])
        self.calculate_map(refined_files["cif"], p1_mtz)

    def run_subprocess(self, cmd: List[str], description: str) -> subprocess.CompletedProcess:
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
                text=True
            )
            self.logger.debug(f"Command output:\n{result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed with exit code {e.returncode}")
            self.logger.error(f"Error output:\n{e.stderr}")
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
        output_file = self.processed_dir / f"{pdb_id}-sf.cif"

        if not output_file.exists():
            try:
                urllib.request.urlretrieve(sf_url, output_file)
                self.logger.info(f"Downloaded {pdb_id} structure factors")
            except urllib.error.URLError as e:
                raise ProcessingError(f"Failed to download {pdb_id}") from e

        return output_file
    
    def download_mmcif(self, pdb_id: str) -> Path:
        """Download mmCIF file from RCSB.
        
        Parameters
        ----------
        pdb_id : str
            PDB ID to download.
            
        Returns
        -------
        Path
            Path to downloaded file.
        """
        mmcif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        output_file = self.processed_dir / f"{pdb_id}.cif"

        if not output_file.exists():
            try:
                urllib.request.urlretrieve(mmcif_url, output_file)
                self.logger.info(f"Downloaded {pdb_id} mmCIF")
            except urllib.error.URLError as e:
                raise ProcessingError(f"Failed to download {pdb_id}") from e

        return output_file

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

    def refine_structure(self, mtz_file: Path, cif_file: Path) -> Dict[str, Path]:
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
        cmd = [self.config.refinement_script, str(mtz_file), str(cif_file)]
        self.run_subprocess(cmd, "structure refinement")

        output_prefix = f"{cif_file}-refined"
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
            "phenix.map_box",
            str(model_file),
            str(mtz_file),
            f'label={",".join(self.config.mtz_labels)}',
            f"ignore_symmetry_conflicts={str(self.config.ignore_symmetry_conflicts)}",
            f"mask_atoms={str(self.config.mask_atoms)}",
            f"wrapping={str(self.config.wrapping)}",
            f'selection="{self.config.selection}"',
        ]
        
        self.run_subprocess(cmd, "map calculation")
        return model_file