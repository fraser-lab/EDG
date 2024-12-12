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
from typing import Union
from adp3d.utils.utility import DotDict
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def export_density_map(density: torch.Tensor, grid: Union[dict, DotDict, gemmi.FloatGrid], output_path: str):
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
        unit_cell = grid.unit_cell if isinstance(grid.unit_cell, gemmi.UnitCell) else gemmi.UnitCell(*grid.unit_cell)
        spacegroup = grid.spacegroup if isinstance(grid.spacegroup, gemmi.SpaceGroup) else gemmi.SpaceGroup(grid.spacegroup)
    else:
        unit_cell = grid["unit_cell"] if isinstance(grid["unit_cell"], gemmi.UnitCell) else gemmi.UnitCell(*grid["unit_cell"])
        spacegroup = gemmi.SpaceGroup(grid["spacegroup"]) if isinstance(grid["spacegroup"], str) else grid["spacegroup"]

    density_np = density.cpu().numpy()
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = gemmi.FloatGrid(
        density_np, cell=unit_cell, spacegroup=spacegroup
    )
    ccp4.setup(np.NAN, gemmi.MapSetup.ReorderOnly)
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(output_path)

def ma_cif_to_XCS(path, n_residues, all_atom=False):
    """Missing atoms CIF to X tensor. Implemented from Axel's code.

    Parameters
    ----------
    path : str
        path to the missing atoms CIF file
    n_residues : int
        number of total residues
    all_atom : bool, optional
        whether to use all atoms, by default False

    Returns
    -------
    Tensor
        X tensor.
    """
    dict = MMCIF2Dict(path)
    label_seq_id = np.array(dict['_atom_site.label_seq_id'], np.int32)
    label_atom_id = np.array(dict['_atom_site.label_atom_id'])
    xs = np.array(dict['_atom_site.Cartn_x'])
    ys = np.array(dict['_atom_site.Cartn_y'])
    zs = np.array(dict['_atom_site.Cartn_z'])

    label_seq_id = label_seq_id - np.min(label_seq_id) + 1 # always start from 1
    X = torch.zeros(1, n_residues, 14 if all_atom else 4, 3).float()
    C = -torch.ones(n_residues).float()
    atom_idx = 0
    old_idx = 0

    for idx, element, x, y, z in zip(label_seq_id, label_atom_id, xs, ys, zs):
        if idx == old_idx:
            atom_idx += 1
        else:
            atom_idx = 0
        if idx > n_residues:
            break
        if element == 'N':
            X[0, int(idx) - 1, atom_idx] = torch.tensor([float(x), float(y), float(z)]).float()
        elif element == 'CA':
            X[0, int(idx) - 1, atom_idx] = torch.tensor([float(x), float(y), float(z)]).float()
        elif element == 'C':
            X[0, int(idx) - 1, atom_idx] = torch.tensor([float(x), float(y), float(z)]).float()
        elif element == 'O':
            X[0, int(idx) - 1, atom_idx] = torch.tensor([float(x), float(y), float(z)]).float()
        else:
            if all_atom:
                X[0, int(idx) - 1, atom_idx] = torch.tensor([float(x), float(y), float(z)]).float()  
        
        C[int(idx) - 1] = 1.
        old_idx = idx

    return X, C.reshape(1, -1)