"""Optimize atomic models using density-guided diffusion.

Implements density-guided diffusion for conformational optimization of atomic models
using the Boltz-1 diffusion model as a prior and real-space map likelihood.

Author: Karson Chrispens
Created: 6 Aug 2024 
Updated: 19 Dec 2024
"""

from pathlib import Path
import os
import pickle
from typing import Optional, Dict, Tuple
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import gemmi

from boltz.data.types import Structure
from adp3d.adp.density import DensityCalculator, normalize, to_f_density
from adp3d.data.io import export_density_map, structure_to_density_input
from adp3d.data.mmcif import parse_mmcif
from adp3d.adp.diffusion import DiffusionStepper
from adp3d.utils.utility import try_gpu

@torch.jit.script
def cos_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes the cosine similarity between two tensors.

    Parameters
    ----------
    a : torch.Tensor
    b : torch.Tensor

    Returns
    -------
    torch.Tensor
        Scaled scalar product of a and b.
    """
    return torch.real(
        torch.sum(a * torch.conj(b)) / torch.linalg.norm(a) / torch.linalg.norm(b)
    )

class DensityGuidedDiffusion:
    """Density-guided diffusion optimizer.
    
    Uses Boltz-1 diffusion model as a prior combined with
    real-space map likelihood for atomic model optimization.
    """

    def __init__(
        self,
        y: str,
        structure: str,
        em: bool = False,
        device: Optional[str] = None,
    ):
        """Initialize the density-guided diffusion optimizer.

        Parameters
        ----------
        y : str
            Path to the density map (MRC/CCP4 format)
        structure : str
            Path to the input structure (mmCIF format)
        em : bool, optional
            Whether map is from electron microscopy, by default False
        device : Optional[str], optional
            Device to run optimization on, by default None
        """
        if device is None:
            self.device = try_gpu()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.em = em

        ccd_path = Path("~/.boltz/ccd.pkl").expanduser()
        with ccd_path.open("rb") as f:
            ccd = pickle.load(f)
        self.structure = parse_mmcif(structure, ccd)

        coords, _, _ = structure_to_density_input(self.structure)
        # self.center_shift = torch.mean(coords, dim=0) # FIXME: doesn't quite work right now for unknown reason
        # ^ not sure if we want to do this because we have possible spacegroup issues

        self.density_extension = os.path.splitext(y)[1]
        if self.density_extension not in (".ccp4", ".map", ".mtz", ".cif"):
            warnings.warn("Density map must be a CCP4, MRC, SF-CIF, or MTZ file.")

        if self.density_extension in (
            ".mtz",
            ".cif",
        ):  # NOTE: MTZ or SF-CIF aren't used currently
            raise NotImplementedError(
                "MTZ and SF-CIF files are not yet supported for density input."
            )
        elif self.density_extension in (".ccp4", ".map", ".mrc"):
            map = gemmi.read_ccp4_map(y)
            map.setup(np.nan, gemmi.MapSetup.ReorderOnly) # necessary to get the proper spacing
            self.grid = map.grid
            if map.grid.spacing == (0.0, 0.0, 0.0):
                raise ValueError("Spacing of the density map is zero. Make sure your input map is properly processed.")
            self.y = normalize(
                torch.from_numpy(np.ascontiguousarray(self.grid.array)).to(
                    self.device, dtype=torch.float32
                )
            )
            self.f_y = normalize(to_f_density(self.y))
        else:
            raise ValueError("Density map must be a CCP4, MRC, SF-CIF, or MTZ file.")

        self.density_calculator = torch.jit.script(
            DensityCalculator(self.grid, self.center_shift, self.device, em=em)
        )

        stepper = DiffusionStepper(Path("~/.boltz/boltz1_conf.pkl"), "data.json") # TODO: FIX

    def density_score( # FIXME: scaffold, doesn't work
        self,
        coords: torch.Tensor,
        elements: torch.Tensor,
        resolution: float = 2.0,
    ) -> torch.Tensor:
        """Calculate density score for current coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Current atomic coordinates
        elements : torch.Tensor
            Element types for each atom
        resolution : float, optional
            Map resolution to compare at, by default 2.0

        Returns
        -------
        torch.Tensor
            Density correlation score
        """
        model_map = self.density_calculator(
            coords,
            elements, 
            resolution=resolution,
            real=True,
            to_normalize=True
        )
        # SiLU (swish) to penalize the model going out into solvent, but not penalize being not in exactly the density as much
        return torch.linalg.norm(torch.nn.SiLU(torch.flatten(self.y) - torch.flatten(model_map)))

    def optimize( # FIXME: scaffold, doesn't work
        self,
        num_steps: int = 100,
        start_sigma: float = 2.0,
        final_sigma: float = 0.1, 
        density_weight: float = 1.0,
        output_dir: Optional[str] = None,
    ) -> Structure:
        """Run density-guided optimization.

        Parameters
        ----------
        num_steps : int, optional
            Number of optimization steps, by default 100
        start_sigma : float, optional
            Initial noise level, by default 2.0
        final_sigma : float, optional
            Final noise level, by default 0.1
        density_weight : float, optional
            Weight for density score, by default 1.0
        output_dir : Optional[str], optional
            Directory to save intermediate structures, by default None

        Returns
        -------
        Structure
            Optimized structure
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        coords, elements, _ = structure_to_density_input(self.structure) 
        coords = coords.to(self.device)
        elements = elements.to(self.device)
        
        sigmas = torch.exp(torch.linspace(
            np.log(start_sigma), np.log(final_sigma), num_steps
        ))

        for i, sigma_t in enumerate(sigmas):
            density_score = self.density_score(coords, elements) # FIXME: update coords with this


            
            step_coords, _ = single_diffusion_step(
                self.model,
                coords,
                torch.ones_like(coords[..., 0]),
                sigma_t.item(),
                network_kwargs={"coords": coords},
            )

            coords = (1 - density_weight) * step_coords + density_weight * coords

            if output_dir and i % 10 == 0:
                self.save_structure(coords, elements, f"{output_dir}/step_{i}.cif")

        final_structure = self.update_structure(coords, elements)
        return final_structure

    def save_structure( # FIXME: scaffold, doesn't work
        self,
        coords: torch.Tensor,
        elements: torch.Tensor, 
        path: str
    ) -> None:
        """Save current structure to file.

        Parameters
        ----------
        coords : torch.Tensor
            Current atomic coordinates
        elements : torch.Tensor
            Element types for each atom
        path : str
            Output file path
        """
        structure = self.update_structure(coords, elements)
        structure.to_file(path)

    def update_structure( # FIXME: scaffold, doesn't work
        self,
        coords: torch.Tensor,
        elements: torch.Tensor
    ) -> Structure:
        """Update structure with new coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            New atomic coordinates
        elements : torch.Tensor
            Element types for each atom

        Returns
        -------
        Structure
            Updated structure object
        """
        structure = self.structure.copy()
        structure.coords = coords.cpu().numpy()
        return structure