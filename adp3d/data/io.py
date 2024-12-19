"""
I/O for ADP3D data (density maps, PDBs, etc.)

Author: Karson Chrispens
Date: 22 Nov 2024
Updated: 22 Nov 2024
"""

import os
import subprocess
import gemmi
import torch
import numpy as np
from typing import Union, Tuple
from adp3d.utils.utility import DotDict
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from chroma.constants import AA20_3, AA_GEOMETRY
from boltz.data.types import Structure


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


def ma_cif_to_XCS(
    path: str, all_atom: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Missing atoms CIF to X tensor. Implemented from Axel's code.

    NOTE: Cannot handle HETATM records.

    Parameters
    ----------
    path : str
        path to the missing atoms CIF file
    all_atom : bool, optional
        whether to use all atoms, by default False

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        X tensor, C tensor, and S tensor
    """
    dict = MMCIF2Dict(path)

    try:
        label_seq_id = np.array(dict["_atom_site.label_seq_id"], np.int32)
    except ValueError:
        raise ValueError("label_seq_id contains \".\" characters, suggesting HETATMs are present. This function cannot handle HETATMs.")

    label_atom_id = np.array(dict["_atom_site.label_atom_id"])
    label_comp_id = np.array(dict["_atom_site.label_comp_id"])
    auth_asym_id = np.array(dict["_atom_site.auth_asym_id"])
    xs = np.array(dict["_atom_site.Cartn_x"])
    ys = np.array(dict["_atom_site.Cartn_y"])
    zs = np.array(dict["_atom_site.Cartn_z"])

    label_seq_id = label_seq_id - np.min(label_seq_id) + 1  # always start from 1
    n_residues = int(np.max(label_seq_id))

    X = torch.zeros(1, n_residues, 14 if all_atom else 4, 3).float()
    C = -torch.ones(n_residues).float()
    S = torch.zeros(n_residues).long()

    chain_ids = {c: i + 1 for i, c in enumerate(np.unique(auth_asym_id))}
    atom_idx = 0
    old_idx = 0

    for idx, element, aa, chain, x, y, z in zip(
        label_seq_id, label_atom_id, label_comp_id, auth_asym_id, xs, ys, zs
    ):
        if idx != old_idx: # idx starts at 1, so this will be true for the first atom of each residue
            atom_idx = 0
            atoms = AA_GEOMETRY[aa]["atoms"] if all_atom else ["N", "CA", "C", "O"]

        if atom_idx >= len(atoms):
            print(f"Too many atoms for residue {idx}, {aa}, {chain}") # TODO: Fix later with hydrogens??
            continue

        if idx > n_residues:
            break

        coords = torch.tensor([float(x), float(y), float(z)]).float()
        res_idx = int(idx) - 1

        if element == atoms[atom_idx]:
            X[0, res_idx, atom_idx] = coords
            atom_idx += 1

        C[res_idx] = chain_ids[chain]
        if aa in AA20_3:
            S[res_idx] = AA20_3.index(aa)

        old_idx = idx

    return X, C.reshape(1, -1), S.reshape(1, -1)


def structure_to_density_input(
    structure: Structure,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Turn a Boltz-1 Structure object into a data needed to compute density and coordinate updates based on density.

    Parameters
    ----------
    structure : Structure
        Boltz-1 Structure object describing the structure.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, float]
        Tensor of Cartesian coordinates, tensor of elements in Boltz-1 elements encoding, and resolution of the structure.
    """
    atoms = structure.data.atoms
    mask_not_present = atoms["is_present"]
    coords = torch.from_numpy(atoms["coords"][mask_not_present]).float()
    elements = torch.from_numpy(atoms["element"][mask_not_present]).long()
    return coords, elements, structure.info.resolution