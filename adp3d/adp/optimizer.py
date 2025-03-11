"""Optimize atomic models using density-guided diffusion.

Implements density-guided diffusion for conformational optimization of atomic models
using the Boltz-1 diffusion model as a prior and real-space map likelihood.

Author: Karson Chrispens
Created: 6 Aug 2024
Updated: 19 Dec 2024
"""

from pathlib import Path
import os
import copy
from typing import Optional, Union, List
import warnings

import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat

from boltz.main import BoltzDiffusionParams
from boltz.model.model import Boltz1
from boltz.data.feature.pad import pad_dim

from adp3d.data import Structure
from adp3d.adp.density import (
    DifferentiableTransformer,
    XMap_torch,
    normalize,
    to_f_density,
)
from adp3d.qfit.volume import XMap
from adp3d.data.io import export_density_map, structure_to_density_input
from adp3d.data.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
)
from adp3d.adp.diffusion import DiffusionStepper, DensityGuidedDiffusionStepper
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
        resolution: float = None,
        step_scale: float = 1.638,  # default step scale, ok results down to 0.8 with higher diversity
        ckpt_path: Path = Path("~/.boltz/boltz1_conf.pkl").expanduser(),
        model: Optional[Boltz1] = None,
        ccd_path: Path = Path("~/.boltz/ccd.pkl").expanduser(),
        device: Optional[str] = None,
    ):
        """Initialize the density-guided diffusion optimizer.

        Parameters
        ----------
        input_path : Path
            Path to input data directory
        y : str
            Path to the density map file (CCP4, MRC, SF-CIF, or MTZ format)
        structure : str
            Path to the structure file in mmCIF or PDB format
        output_path : str
            Directory path for output files
        em : bool, optional
            Flag for electron microscopy mode, by default False
        resolution : float, optional
            Map resolution in Angstroms, by default None. MTZ files have resolution information, but CCP4, MRC, and MAP files do not.
        step_scale : float, optional
            Scale factor for diffusion steps, by default 1.638
        ckpt_path : Path, optional
            Path to the Boltz1 model checkpoint, by default "~/.boltz/boltz1_conf.pkl"
        model : Boltz1, optional
            Pre-loaded Boltz1 model instance, by default None
        ccd_path : Path, optional
            Path to the CCD dictionary file, by default "~/.boltz/ccd.pkl"
        device : str, optional
            Device to run computations on ('cpu', 'cuda', etc.), by default None

        Raises
        ------
        ValueError
            If density map spacing is zero or if unsupported file format is provided
        NotImplementedError
            If MTZ or SF-CIF files are provided (currently unsupported)

        Notes
        -----
        Density map must be in CCP4, MRC, SF-CIF, or MTZ format.
        Currently, MTZ and SF-CIF formats are not supported for density input.
        """
        if device is None:
            self.device = try_gpu()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.em = em

        st = Structure.fromfile(structure)
        # remove hydrogens
        st = st.extract(st.select("e", "H", comparison="!="))
        st = st.reorder()
        self.structure = st

        # coords, _, _ = structure_to_density_input(self.structure)

        extension = os.path.splitext(y)[1]
        if extension not in (".ccp4", ".mrc", ".map", ".mtz"):
            warnings.warn("Density map/reflections must be a CCP4, MRC, MTZ file.")
        if extension in (".ccp4", ".map", ".mrc"):
            if resolution is None:
                raise ValueError(
                    "Map resolution must be provided for CCP4, MRC, or MAP files."
                )
            xmap = XMap.fromfile(y, resolution=resolution)
        else:
            xmap = XMap.fromfile(y)

        xmap = XMap_torch(xmap, device=self.device)

        scattering_factors = (
            ELECTRON_SCATTERING_FACTORS if em else ATOM_STRUCTURE_FACTORS
        )

        self._setup_scattering_params(scattering_factors)

        self.density_calculator = torch.jit.script(
            DifferentiableTransformer(
                xmap,
                scattering_params=self.scattering_params,
                em=self.em,
                device=self.device,
            )
        )

        diffusion_args = BoltzDiffusionParams(step_scale=step_scale)
        self.stepper = DensityGuidedDiffusionStepper(
            checkpoint_path=ckpt_path,
            data_path=input_path,
            out_dir=output_path,
            diffusion_args=diffusion_args,
            device=self.device,
            model=model,
        )

    def _setup_scattering_params(self, structure_factors: dict):
        """Set up scattering parameters for density calculation."""
        unique_elements = sorted(set(self.structure.e))
        atomic_num_dict = {elem: ATOMIC_NUM_TO_ELEMENT.index(elem) for elem in unique_elements}
        
        max_atomic_num = max(atomic_num_dict.values())
        indices = []
        values = []

        for elem in unique_elements:
            atomic_num = atomic_num_dict[elem]

            if elem in structure_factors:
                factor = structure_factors[elem]
            else:
                print(
                    f"Warning: Scattering factors for {elem} not found, using C instead"
                )
                factor = structure_factors["C"]

            factor = torch.tensor(factor, dtype=torch.float32).T # (2, range) -> (range, 2)

            for i in range(factor.shape[0]):  # range
                for j in range(factor.shape[1]):  # 2
                    indices.append([atomic_num, i, j])
                    values.append(factor[i, j].item())

        sparse_indices = torch.tensor(indices, dtype=torch.long).t()
        sparse_values = torch.tensor(values, dtype=torch.float32)

        sparse_size = torch.Size([max_atomic_num + 1, len(structure_factors["C"][0]), 2])
        scattering_sparse_tensor = torch.sparse_coo_tensor(
            sparse_indices, sparse_values, sparse_size
        )

        self.scattering_params = scattering_sparse_tensor

    def density_score(
        self,
        coords: torch.Tensor,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        occupancies: torch.Tensor,
        norm: int = 1,
    ) -> torch.Tensor:
        """Calculate density score for current coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Current atomic coordinates, shape [batch, atoms, 3]
        elements : torch.Tensor
            Element atomic numbers for each atom, shape [batch, atoms]
        b_factors : torch.Tensor
            B-factors for each atom, shape [batch, atoms]
        occupancies : torch.Tensor
            Occupancies for each atom, shape [batch, atoms]
        norm : int, optional
            Which norm to use for the score, by default 1

        Returns
        -------
        torch.Tensor
            Density correlation score
        """
        element_ids = (
            torch.tensor([self.elements_to_ids[e] for e in elements.flatten()])
            .reshape(elements.shape)
            .to(self.device)
        )
        model_map = self.density_calculator(
            coords, element_ids, b_factors, occupancies, chunk_size=50000
        ).sum(
            0
        )  # TODO: dont use normalization, use e-/A^3
        return torch.linalg.norm(
            torch.flatten(model_map) - torch.flatten(self.y), ord=norm
        )
        # # SiLU (swish) to penalize the model going out into solvent, but not penalize being not in exactly the density as much
        # return torch.linalg.norm(torch.nn.SiLU(torch.flatten(self.y) - torch.flatten(model_map)))

    def optimize(  # TODO: compare score based guidance to DPS and DMAP
        self,
        output_dir: str,
        num_steps: int = 200,
        num_samples: int = 1,
        learning_rate: Union[List, float] = 1e-1,
        partial_diffusion: bool = False,
        diffusion_kwargs: dict = None,
    ) -> Structure:
        """Run density-guided optimization.

        Parameters
        ----------
        output_dir : str
            Output directory for optimized structure
        num_steps : int, optional
            Number of optimization steps, by default 200
        num_sample  : int, optional
            Size of ensemble to generate, by default 1
        learning_rate : Union[List, float], optional
            Learning rate for density optimization, by default 1e-1
        partial_diffusion : bool, optional
            Whether to use partial diffusion, by default False
        diffusion_kwargs : Dict[str, Any]
            Additional arguments for partial diffusion. May include:
                - noising_steps : int
                    Number of steps for noise addition, 25-30% of num_steps works well
                - structure : Structure
                    Input structure for partial diffusion
                - selector : List[int]
                    Indices of segments for selective diffusion

        Returns
        -------
        Structure
            Optimized structure
        """
        os.makedirs(output_dir, exist_ok=True)

        coords, elements, b_factors, occupancies, resolution = structure_to_density_input(self.structure)
        coords = repeat(coords, "a c -> n a c", n=num_samples)
        elements = repeat(elements, "e -> n e", n=num_samples)
        # FIXME: using uniform b-factors and occupancies for now
        b_factors = repeat(b_factors, "b -> n b", n=num_samples) / num_samples
        occupancies = repeat(occupancies, "q -> n q", n=num_samples) / num_samples

        if resolution == 0.0 or resolution is None:
            if self.density_calculator.xmap.resolution is not None:
                resolution = self.density_calculator.xmap.resolution.high
            warnings.warn(
                f"Resolution of input structure is {resolution}. Using 2.0 A instead."
            )

        coords = coords.to(self.device)
        elements = elements.to(self.device)
        b_factors = b_factors.to(self.device)
        occupancies = occupancies.to(self.device)

        if partial_diffusion:
            self.stepper.initialize_partial_diffusion(
                num_samples=num_samples, sampling_steps=num_steps, **diffusion_kwargs
            )
        else:
            self.stepper.initialize_diffusion(
                num_samples=num_samples, sampling_steps=num_steps
            )

        step_coords = self.stepper.cached_diffusion_init["atom_coords"]
        pad_mask = (
            self.stepper.cached_representations["feats"]["atom_pad_mask"]
            .squeeze()
            .bool()
        )

        # v_density = torch.zeros_like(step_coords)
        scores = []

        if partial_diffusion:
            pbar = tqdm(
                range(diffusion_kwargs["noising_steps"]), desc="Optimizing structure"
            )
        else:
            pbar = tqdm(range(num_steps), desc="Optimizing structure")
        for i in pbar:
            step_lr = (
                learning_rate if isinstance(learning_rate, float) else learning_rate[i]
            )

            with torch.set_grad_enabled(True):  # Explicit gradient context
                masked_coords = step_coords.clone().squeeze()[pad_mask, :]
                coords_to_grad = masked_coords.detach().clone()
                coords_to_grad = coords_to_grad.requires_grad_(True)

                density_score = self.density_score(coords_to_grad, elements, b_factors, occupancies, resolution)
                density_score.backward()

                if coords_to_grad.grad is None:
                    raise ValueError(
                        "Gradient computation failed - tensor is not a leaf"
                    )

                # Create gradient update tensor of original shape
                full_grad = torch.zeros_like(step_coords.squeeze())
                full_grad[pad_mask, :] = coords_to_grad.grad

                # only do gradient on partially diffused atoms
                if diffusion_kwargs["selector"] is not None:
                    selector = torch.from_numpy(diffusion_kwargs["selector"]).to(
                        self.device
                    )
                    selector = pad_dim(
                        selector, 0, step_coords.shape[1] - selector.shape[0]
                    )
                    full_grad[~selector, :] = 0

            pbar.set_postfix(
                {
                    "score": f"{density_score.item():.4f}",
                }
            )
            scores.append(density_score.item())

            step_coords = self.stepper.step(
                step_coords,
                full_grad,
                guidance_scale=step_lr,
                augmentation=True,
                align_to_input=True,
            )

            # Gradient descent with momentum # TODO: try others?
            # v_density = 0.9 * v_density + step_lr * full_grad.unsqueeze(0)
            # step_coords = step_coords - v_density

            # Raw gradient descent
            # step_coords = step_coords - step_lr * full_grad.unsqueeze(
            #     0
            # )
            structure = copy.deepcopy(self.structure) # q and b will need to be updated in the future
            structure.coor = self.stepper.diffusion_trajectory[
                    f"step_{self.stepper.current_step - 1}"
                ]["coords"].cpu().numpy()
            structure.tofile(f"{output_dir}/step_{i}.cif")
        
        structure = copy.deepcopy(self.structure) # q and b will need to be updated in the future
        structure.coor = self.stepper.diffusion_trajectory[
                f"step_{self.stepper.current_step - 1}"
            ]["coords"].cpu().numpy()
        structure.tofile(f"{output_dir}/final.cif")
        
        return structure, scores
