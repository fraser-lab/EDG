"""
I/O for ADP3D data (density maps, PDBs, etc.)

Author: Karson Chrispens
Date: 22 Nov 2024
Updated: 22 Nov 2024
"""

import gemmi
import torch
import numpy as np
from typing import Union, Tuple
from adp3d.utils.utility import DotDict
from boltz.data.types import Structure as BoltzStructure
from .structure import Structure
from pathlib import Path
from dataclasses import replace
from boltz.data.write.mmcif import to_mmcif
from .sf import ATOMIC_NUM_TO_ELEMENT


def export_density_map(
    density: torch.Tensor, grid: Union[dict, DotDict, gemmi.FloatGrid], output_path: str
) -> None:
    """Export a density map to a file.

    Args:
        density (torch.Tensor): The density map to export.
        grid (Union[dict, DotDict, gemmi.FloatGrid]): The grid information.
        output_path (str): The output file path.

    Note:
        The grid information can be the grid information from a density map, or a dictionary with the following keys
        - unit_cell: The unit cell of the grid. (gemmi.UnitCell or tuple of (a, b, c, alpha, beta, gamma))
        - spacegroup: The spacegroup of the grid. (gemmi.SpaceGroup or str)
    """
    if isinstance(grid, (gemmi.FloatGrid, DotDict)):
        unit_cell = (
            grid.unit_cell
            if isinstance(grid.unit_cell, gemmi.UnitCell)
            else gemmi.UnitCell(*grid.unit_cell)
        )
        spacegroup = (
            grid.spacegroup
            if isinstance(grid.spacegroup, gemmi.SpaceGroup)
            else gemmi.SpaceGroup(grid.spacegroup)
        )
    else:
        unit_cell = (
            grid["unit_cell"]
            if isinstance(grid["unit_cell"], gemmi.UnitCell)
            else gemmi.UnitCell(*grid["unit_cell"])
        )
        spacegroup = (
            gemmi.SpaceGroup(grid["spacegroup"])
            if isinstance(grid["spacegroup"], str)
            else grid["spacegroup"]
        )

    density_np = density.cpu().numpy()
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = gemmi.FloatGrid(density_np, cell=unit_cell, spacegroup=spacegroup)
    ccp4.setup(np.NAN, gemmi.MapSetup.ReorderOnly)
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(output_path)


def write_mmcif(
    coords: torch.Tensor,
    structure: Union[Structure, BoltzStructure], # TODO: FIXME Structure API not being used properly??
    output_path: Path,
    elements: torch.Tensor = None,
    return_structure: bool = False,
) -> None | Structure:
    """
    Write one or more mmCIF files from a batch of coordinates and a structure.

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of shape [batch, n_atoms, 3] or [n_atoms, 3] containing atomic coordinates.
    structure : Structure
        A Structure object containing atom, residue, and chain information.
    output_path : Path
        The base file path for the output mmCIF files. For a batch, a suffix
        '_{batch_index}' will be appended to the base name.
    elements : torch.Tensor, optional
        An optional tensor containing atomix numbers for each atom.
        This can either be of shape [n_atoms] (to be used for all structures) or
        [batch, n_atoms] (to provide distinct elements per structure).
    return_structure : bool, optional
        Whether to return the modified Structure object. Defaults to False.
    """
    base_path = Path(output_path)
    batch_size = coords.shape[0] if coords.ndim == 3 else 1
    elements_batch_size = elements.shape[0] if elements is not None and elements.ndim == 2 else 1

    if elements is not None and batch_size != elements_batch_size:
        raise ValueError(
            f"Batch size or number of atoms in coords and elements must match. coords: {coords.shape} elements: {elements.shape}"
        )

    for i in range(batch_size):
        model_coords = coords[i] if batch_size > 1 else coords
        model_coords_np = (
            model_coords.cpu().numpy() if model_coords.is_cuda else model_coords.numpy()
        )

        atoms = structure.atoms.copy()
        atoms["coords"] = model_coords_np
        atoms["is_present"] = True

        # If element information is provided, update the atom table accordingly
        if elements is not None:
            # If elements are provided for each model separately, use those elements
            if elements.ndim == 2:
                elem_tensor = elements[i]
            else:
                elem_tensor = elements
            model_elements_np = (
                elem_tensor.cpu().numpy()
                if elem_tensor.is_cuda
                else elem_tensor.numpy()
            )
            atoms["element"] = model_elements_np

        new_structure = replace(structure, atoms=atoms)

        mmcif_str = to_mmcif(new_structure)

        # Construct the output file path by appending _{i} before the extension
        new_output_path = base_path.with_name(f"{base_path.stem}_{i}{base_path.suffix}") if batch_size > 1 else base_path

        # Make sure the directory exists
        new_output_path.parent.mkdir(parents=True, exist_ok=True)

        with new_output_path.open("w") as f:
            f.write(mmcif_str)

    if return_structure:
        return new_structure

def structure_to_density_input(
    structure: Union[Structure, BoltzStructure],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Turn a qFit or Boltz-1 Structure object into a data needed to compute density and coordinate updates based on density.

    Parameters
    ----------
    structure : Union[Structure, BoltzStructure]
        qFit or Boltz-1 Structure object describing the structure.
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]
        Tensor of Cartesian coordinates, tensor of elements (atomic numbers), b-factors, occupancies, and resolution of the structure (or 2.0 if not present).
    """
    if isinstance(structure, BoltzStructure):
        atoms = structure.atoms
        mask_not_present = atoms["is_present"]
        coords = torch.from_numpy(atoms["coords"][mask_not_present]).float()
        elements = torch.from_numpy(atoms["element"][mask_not_present]).long()
        b_factors = torch.tensor([25] * len(atoms))
        occupancies = torch.tensor([1.0] * len(atoms))
        return (
            coords,
            elements,
            b_factors,
            occupancies,
            structure.info.resolution if structure.info.resolution > 0 else 2.0,
        )
    elif isinstance(structure, Structure):
        mask_not_present = structure.active
        elements = [ATOMIC_NUM_TO_ELEMENT.index(e) for e in structure.e[mask_not_present]]
        coords = torch.from_numpy(structure.coor[mask_not_present]).float()
        elements = torch.tensor(elements).long()
        b_factors = torch.from_numpy(structure.b[mask_not_present]).float()
        occupancies = torch.from_numpy(structure.q[mask_not_present]).float()
        return (
            coords,
            elements,
            b_factors,
            occupancies,
            structure.info.resolution if structure.info.resolution > 0 else 2.0,
        )
    else:
        raise ValueError("structure must be a Boltz-1 or qFit Structure object.")
