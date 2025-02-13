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
from typing import Optional, Dict, Tuple, Union, List
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import gemmi
from tqdm import tqdm

from boltz.data.types import Structure
from boltz.main import BoltzDiffusionParams
from adp3d.adp.density import DensityCalculator, normalize, to_f_density
from adp3d.data.io import export_density_map, structure_to_density_input, write_mmcif
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
        input_path: Path,
        y: str,
        structure: str,
        output_path: str,
        em: bool = False,
        step_scale: float = 1.638,  # default step scale, ok results down to 0.8 with higher diversity
        ckpt_path: Path = Path("~/.boltz/boltz1_conf.pkl").expanduser(),
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
            map.setup(
                np.nan, gemmi.MapSetup.ReorderOnly
            )  # necessary to get the proper spacing
            self.grid = map.grid
            if map.grid.spacing == (0.0, 0.0, 0.0):
                raise ValueError(
                    "Spacing of the density map is zero. Make sure your input map is properly processed."
                )
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

        diffusion_args = BoltzDiffusionParams(step_scale=step_scale)
        self.stepper = DiffusionStepper(
            checkpoint_path=ckpt_path,
            input_path=input_path,
            out_dir=output_path,
            diffusion_args=diffusion_args,
        )

    def density_score(
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
            coords, elements, resolution=resolution, real=True, to_normalize=True
        )  # TODO: dont use normalization, use e-/A^3
        return torch.linalg.norm(torch.flatten(self.y) - torch.flatten(model_map))
        # # SiLU (swish) to penalize the model going out into solvent, but not penalize being not in exactly the density as much
        # return torch.linalg.norm(torch.nn.SiLU(torch.flatten(self.y) - torch.flatten(model_map)))

    def optimize(  # TODO: compare score based guidance to DPS and DMAP
        self,
        output_dir: str,
        num_steps: int = 200,
        num_samples: int = 1,
        learning_rate: Union[List, float] = 1e-3,
        partial_diffusion: bool = False,
        **diffusion_kwargs,
    ) -> Structure:
        """Run density-guided optimization.

        Parameters
        ----------
        output_dir : str
            Output directory for optimized structure
        num_steps : int, optional
            Number of optimization steps, by default 200
        num_samples : int, optional
            Size of ensemble to generate, by default 1
        learning_rate : Union[List, float], optional
            Learning rate for density optimization, by default 1e-3
        partial_diffusion : bool, optional
            Whether to use partial diffusion, by default False
        **diffusion_kwargs : Dict[str, Any]
            Additional arguments for partial diffusion. May include:
                - noising_steps : int
                    Number of steps for noise addition, 25-30% of num_steps works well
                - structure_input : Structure
                    Input structure for partial diffusion
                - segment_selection : List[int]
                    Indices of segments for selective diffusion

        Returns
        -------
        Structure
            Optimized structure
        """
        os.makedirs(output_dir, exist_ok=True)

        coords, elements, resolution = structure_to_density_input(self.structure)

        if resolution == 0.0:
            warnings.warn(f"Resolution of input structure is {resolution}. Using 2.0 A instead.")

        coords = coords.to(self.device)
        elements = elements.to(self.device)

        if partial_diffusion:
            self.stepper.initialize_partial_diffusion(
                num_samples=num_samples, sampling_steps=num_steps, **diffusion_kwargs
            )
        else:
            self.stepper.initialize_diffusion(
                num_samples=num_samples, sampling_steps=num_steps
            )

        step_coords = self.stepper.cached_diffusion_init["atom_coords"]

        pbar = tqdm(range(num_steps), desc="Optimizing structure")
        for i in pbar:
            step_lr = (
                learning_rate if isinstance(learning_rate, float) else learning_rate[i]
            )

            step_coords = self.stepper.step(
                step_coords, augmentation=True, align_to_input=True
            )

            with torch.enable_grad():
                coords_to_grad = step_coords.clone().detach().requires_grad_(True)
                density_score = self.density_score(
                    coords_to_grad, elements, resolution
                )
                density_score.backward()

            coords_grad = coords_to_grad.grad

            step_coords = 0.9 * step_coords - step_lr * coords_grad # FIXME: 0.9 is momentum, somewhat arbitrary

            if i % 10 == 0:
                write_mmcif(step_coords, elements, f"{output_dir}/step_{i}.cif")

        structure = write_mmcif(step_coords, elements, f"{output_dir}/final.cif", return_structure=True)
        return structure