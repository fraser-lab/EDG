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
from boltz.data.types import Structure
from pathlib import Path
from dataclasses import replace
from boltz.data.write.mmcif import to_mmcif


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
    structure: Structure,
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


# def ma_cif_to_XCS(
#     path: str, all_atom: bool = False
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Missing atoms CIF to X tensor. Changed from Axel's code.

#     NOTE: Cannot handle HETATM records.

#     Parameters
#     ----------
#     path : str
#         path to the missing atoms CIF file
#     all_atom : bool, optional
#         whether to use all atoms, by default False

#     Returns
#     -------
#     Tuple[Tensor, Tensor, Tensor]
#         X tensor, C tensor, and S tensor
#     """
#     dict = MMCIF2Dict(path)

#     try:
#         label_seq_id = np.array(dict["_atom_site.label_seq_id"], np.int32)
#     except ValueError:
#         raise ValueError(
#             'label_seq_id contains "." characters, suggesting HETATMs are present. This function cannot handle HETATMs.'
#         )

#     label_atom_id = np.array(dict["_atom_site.label_atom_id"])
#     label_comp_id = np.array(dict["_atom_site.label_comp_id"])
#     auth_asym_id = np.array(dict["_atom_site.auth_asym_id"])
#     xs = np.array(dict["_atom_site.Cartn_x"])
#     ys = np.array(dict["_atom_site.Cartn_y"])
#     zs = np.array(dict["_atom_site.Cartn_z"])

#     label_seq_id = label_seq_id - np.min(label_seq_id) + 1  # always start from 1
#     n_residues = int(np.max(label_seq_id))

#     X = torch.zeros(1, n_residues, 14 if all_atom else 4, 3).float()
#     C = -torch.ones(n_residues).float()
#     S = torch.zeros(n_residues).long()

#     chain_ids = {c: i + 1 for i, c in enumerate(np.unique(auth_asym_id))}
#     atom_idx = 0
#     old_idx = 0

#     for idx, element, aa, chain, x, y, z in zip(
#         label_seq_id, label_atom_id, label_comp_id, auth_asym_id, xs, ys, zs
#     ):
#         if (
#             idx != old_idx
#         ):  # idx starts at 1, so this will be true for the first atom of each residue
#             atom_idx = 0
#             atoms = AA_GEOMETRY[aa]["atoms"] if all_atom else ["N", "CA", "C", "O"]

#         if atom_idx >= len(atoms):
#             print(
#                 f"Too many atoms for residue {idx}, {aa}, {chain}"
#             )  # TODO: Fix later with hydrogens??
#             continue

#         if idx > n_residues:
#             break

#         coords = torch.tensor([float(x), float(y), float(z)]).float()
#         res_idx = int(idx) - 1

#         if element == atoms[atom_idx]:
#             X[0, res_idx, atom_idx] = coords
#             atom_idx += 1

#         C[res_idx] = chain_ids[chain]
#         if aa in AA20_3:
#             S[res_idx] = AA20_3.index(aa)

#         old_idx = idx

#     return X, C.reshape(1, -1), S.reshape(1, -1)


def structure_to_density_input(
    structure: Structure,
    coords: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Turn a Boltz-1 Structure object into a data needed to compute density and coordinate updates based on density.

    Parameters
    ----------
    structure : Structure
        Boltz-1 Structure object describing the structure.
    coords : torch.Tensor, optional
        Cartesian coordinates of atoms in the structure, by default None. Used to initialize density input from diffusion output.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, float]
        Tensor of Cartesian coordinates, tensor of elements in Boltz-1 elements encoding, and resolution of the structure (or 2.0 if not present).
    """
    atoms = structure.data.atoms
    mask_not_present = atoms["is_present"]
    coords = torch.from_numpy(atoms["coords"][mask_not_present]).float()
    elements = torch.from_numpy(atoms["element"][mask_not_present]).long()
    return (
        coords,
        elements,
        structure.info.resolution if structure.info.resolution > 0 else 2.0,
    )
