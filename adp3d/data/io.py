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
import numpy
from typing import Union
from adp3d.utils.utility import DotDict

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
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(output_path)