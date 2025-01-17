"""
Process maps with pipeline from map.py

Author: Karson Chrispens (karson.chrispens@ucsf.edu)
Date: 16 Jan 2024
"""

import adp3d.utils.map as map

config = map.ProcessingConfig(
    output_dir="tests/resources/1AZ5",
    selection="A",
    refinement_script="adp3d/utils/single_refinement.sh",
    map_box_script="adp3d/utils/map_box.sh",
)

# TODO make this scale

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
        futures = {
            executor.submit(process_fn, pdb_id): pdb_id for pdb_id in pdb_ids
        }

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