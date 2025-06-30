"""Optimize atomic models using density-guided diffusion.

Implements density-guided diffusion for conformational optimization of atomic models
using Boltz diffusion models (Boltz-1 or Boltz-2) as a prior and real-space map likelihood.

Author: Karson Chrispens
Created: 6 Aug 2024
Updated: 19 Dec 2024
"""

from pathlib import Path
import os
import copy
from functools import partial
from typing import Optional, Tuple, Union, List
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat

from boltz.main import BoltzDiffusionParams, Boltz2DiffusionParams
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2
from boltz.data.pad import pad_dim
from boltz.model.potentials.schedules import (
    PiecewiseStepFunction,
    PiecewiseSchedule,
    ExponentialInterpolation,
    ExponentialInterpolationWithBounds,
    ResolutionScaling,
)

from edg.data import Structure
from edg.edg.modules.density.density import (
    DifferentiableTransformer,
    XMap_torch,
    normalize,
    to_f_density,
    scale_map,
)
from edg.data.structure import Ensemble
from edg.qfit.volume import XMap
from edg.data.io import structure_to_density_input
from edg.data.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
)
from edg.edg.modules.diffusion import DiffusionStepper
from edg.edg.modules.guided_diffusion import DensityGuidedDiffusionStepper
from edg.utils.utility import try_gpu
from edg.edg.modules.potentials import SubstructurePotential, DensityPotential


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

    Uses Boltz diffusion models (Boltz-1 or Boltz-2) as a prior combined with
    real-space map likelihood for atomic model optimization.
    """

    def __init__(
        self,
        input_path: Path,
        y: str,
        structure: str,
        output_path: str,
        em: bool = False,
        resolution: Optional[float] = None,
        step_scale: Optional[float] = None,  # Auto-set based on model version if None
        ckpt_path: Path = Path("~/.boltz/boltz1_conf.pkl").expanduser(),
        model: Optional[Union[Boltz1, Boltz2]] = None,
        model_version: str = "boltz1",  # "boltz1" or "boltz2"
        ccd_path: Path = Path("~/.boltz/ccd.pkl").expanduser(),
        device: Optional[str] = None,
        **kwargs,
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
            Scale factor for diffusion steps. Auto-set to 1.638 for Boltz-1 or 1.5 for Boltz-2 if None, by default None
        ckpt_path : Path, optional
            Path to the model checkpoint, by default "~/.boltz/boltz1_conf.pkl"
        model : Union[Boltz1, Boltz2], optional
            Pre-loaded model instance, by default None
        model_version : str, optional
            Model version to use ("boltz1" or "boltz2"), by default "boltz1"
        ccd_path : Path, optional
            Path to the CCD dictionary file, by default "~/.boltz/ccd.pkl"
        device : str, optional
            Device to run computations on ('cpu', 'cuda', etc.), by default None
        **kwargs
            Additional keyword arguments.

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
        self.model_version = model_version.lower()
        if self.model_version not in ["boltz1", "boltz2"]:
            raise ValueError(f"model_version must be 'boltz1' or 'boltz2', got {model_version}")
        
        # Set default step_scale based on model version if not provided
        if step_scale is None:
            step_scale = 1.638 if self.model_version == "boltz1" else 1.5
        
        self.output_path = Path(output_path)
        os.makedirs(self.output_path, exist_ok=True)

        st = Structure.fromfile(structure)
        self.structure = st

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
            try:
                xmap = XMap.fromfile(y)
            except:  # noqa: E722
                xmap = XMap.fromfile(
                    y, label=kwargs.get("label", "2FOFCWT,PH2FOFCWT")
                )  # try 2FO-FC map if initial fails

        xmap = XMap_torch(xmap, device=self.device)
        self.y = xmap.array.float()

        scattering_factors = (
            ELECTRON_SCATTERING_FACTORS if em else ATOM_STRUCTURE_FACTORS
        )

        self._setup_scattering_params(scattering_factors)

        self.density_calculator = DifferentiableTransformer(
            xmap,
            scattering_params=self.scattering_params,
            em=self.em,
            device=self.device,
        )

        # Scale experimental map
        coords, elements, b_factors, occupancies, active, _ = (
            structure_to_density_input(self.structure)
        )
        coords = coords.to(self.device).float().unsqueeze(0)  # Add batch dim
        elements = elements.to(self.device).long().unsqueeze(0)
        b_factors = b_factors.to(self.device).float().unsqueeze(0)
        occupancies = occupancies.to(self.device).float().unsqueeze(0)
        active = active.to(self.device).bool().unsqueeze(0)

        # Calculate initial model map
        with torch.no_grad():
            initial_model_map = self.density_calculator(
                coords, elements, b_factors, occupancies, active
            ).squeeze(0)
            mask = self.density_calculator.create_mask(coords.squeeze(0), 5.0)

        self.y = scale_map(self.y, initial_model_map, mask)
        self.density_calculator.xmap.array = self.y  # set array to appropriate level

        scaled_map_path = self.output_path / "scaled_experimental_map.ccp4"
        self.density_calculator.xmap.tofile(str(scaled_map_path))
        print(f"Saved scaled experimental map to {scaled_map_path}")

        # Create diffusion args based on model version
        if self.model_version == "boltz1":
            diffusion_args = BoltzDiffusionParams(step_scale=step_scale)
        else:  # boltz2
            diffusion_args = Boltz2DiffusionParams(step_scale=step_scale)
        
        self.stepper = DensityGuidedDiffusionStepper(
            checkpoint_path=ckpt_path,
            data_path=input_path,
            out_dir=output_path,
            diffusion_args=diffusion_args,
            model_version=self.model_version,
            device=self.device,
            model=model,
            steering_args=kwargs["steering_args"] if "steering_args" in kwargs else None,
        )

    def _setup_scattering_params(self, structure_factors: dict):
        """Set up scattering parameters for density calculation."""
        unique_elements = set(self.structure.e)
        unique_elements = sorted(
            set(
                [
                    (
                        elem.upper()
                        if len(elem) == 1
                        else elem[0].upper() + elem[1:].lower()
                    )
                    for elem in unique_elements
                ]
            )
        )
        atomic_num_dict = {
            elem: ATOMIC_NUM_TO_ELEMENT.index(elem) for elem in unique_elements
        }

        max_atomic_num = max(atomic_num_dict.values())
        # Use the max atomic number found in the structure for tensor size
        n_coeffs = len(structure_factors["C"][0])
        dense_size = torch.Size([max_atomic_num + 1, n_coeffs, 2])
        scattering_dense_tensor = torch.zeros(dense_size, dtype=torch.float32)

        for elem in unique_elements:
            atomic_num = atomic_num_dict[elem]

            if elem in structure_factors:
                factor = structure_factors[elem]
            else:
                print(
                    f"Warning: Scattering factors for {elem} not found, using C instead"
                )
                factor = structure_factors["C"]

            factor = torch.tensor(
                factor, dtype=torch.float32
            ).T  # (2, range) -> (range, 2)

            scattering_dense_tensor[atomic_num, :, :] = factor

        self.scattering_params = scattering_dense_tensor

    def optimize(
        self,
        output_dir: str,
        num_steps: int = 200,
        ensemble_size: int = 1,
        partial_diffusion: bool = False,
        steering: bool = False,
        diffusion_kwargs: Optional[dict] = None,
        substructure_conditioning_kwargs: Optional[dict] = None,
        representation_noise_scale: Optional[float] = None,
    ) -> Structure:
        os.makedirs(output_dir, exist_ok=True)

        coords, elements, b_factors, occupancies, active, resolution = (
            structure_to_density_input(self.structure)
        )

        if steering:  # make sure particles are all updated by density
            num_particles = self.stepper.model.steering_args["num_particles"]
            num_samples = num_particles * ensemble_size

        coords = repeat(coords, "a c -> n e a c", n=num_particles, e=ensemble_size)
        elements = repeat(elements, "e -> n e", n=num_samples)
        b_factors = repeat(b_factors, "b -> n b", n=num_samples)
        # b_factors = torch.full(elements.size(), 4 / num_samples)
        occupancies = (
            repeat(occupancies, "q -> n q", n=num_samples)
            / ensemble_size  # FIXME is this right?
        )
        active = repeat(active, "a -> n a", n=num_samples)

        if resolution is None:
            if self.density_calculator.xmap.resolution is not None:
                resolution = self.density_calculator.xmap.resolution.high
            else:
                warnings.warn(
                    f"Resolution of input structure is {resolution}. Using 2.0 A instead."
                )
                resolution = 2.0

        coords = coords.to(self.device).float()
        elements = elements.to(self.device).long()
        b_factors = b_factors.to(self.device).float()
        occupancies = occupancies.to(self.device).float()
        active = active.to(self.device).bool()

        resolution_scale = ExponentialInterpolationWithBounds(
            start=resolution,
            end=8.0,
            alpha=5.0,
            start_t=(1 - 175 / 200),
            end_t=(1 - 150 / 200),
        )
        density_schedule = ExponentialInterpolationWithBounds(
            start=0.0,
            end=150,
            alpha=90,
            start_t=(1 - 175 / 200),
            end_t=(1 - 150 / 200),
        )

        density_potential = DensityPotential(
            xmap=self.density_calculator.xmap,
            parameters={
                "guidance_interval": 1,
                "guidance_weight": PiecewiseSchedule([1-175/200], [0, 1.0]),  # NOTE: for L2
                # "guidance_weight": ResolutionScaling(
                #     resolution_scale, resolution, base=density_schedule # NOTE: for L2
                #     # resolution_scale, resolution, base=0.5 # NOTE: for hybrid
                # ),
                # "guidance_weight": PiecewiseSchedule(
                #     [(1 - 175 / 200), (1 - 150 / 200), (1 - 125 / 200)],
                #     [
                #         0,
                #         density_schedule,
                #         150,
                #         0,
                #     ],
                # ),
                "resolution": resolution_scale,
                "resampling_weight": PiecewiseSchedule(
                    [(1 - 175 / 200), (1 - 150 / 200), (1 - 125 / 200)],
                    [
                        0.005,
                        ExponentialInterpolationWithBounds(
                            start=0.005,
                            end=20,
                            alpha=10,
                            start_t=(1 - 175 / 200),
                            end_t=(1 - 150 / 200),
                        ),
                        20,
                        0.,
                    ],
                ),
                "elements": elements,
                "b_factors": b_factors,
                "occupancies": occupancies,
                "scattering_params": self.scattering_params,
                "em": self.em,
            },
        )

        potentials = [density_potential]

        if partial_diffusion:
            if diffusion_kwargs is None:
                diffusion_kwargs = {}
            if "structure" not in diffusion_kwargs:
                diffusion_kwargs["structure"] = coords

            self.stepper.initialize_partial_diffusion(
                ensemble_size=ensemble_size,
                sampling_steps=num_steps,
                extra_potentials=potentials,
                representation_noise_scale=representation_noise_scale,
                **diffusion_kwargs,
            )
        elif substructure_conditioning_kwargs is not None:
            substructure_potential = SubstructurePotential(
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": 0.05,
                    "resampling_weight": 0.0,
                    "buffer": 0.5,
                    "denoising_selection": substructure_conditioning_kwargs.get(
                        "selection", np.array([], dtype=int)
                    ),
                    "reference_coords": coords,
                }
            )
            potentials.append(substructure_potential)
            self.stepper.initialize_substructure_conditioned_diffusion(
                ensemble_size=ensemble_size,
                sampling_steps=num_steps,
                structure=coords,
                selection=substructure_conditioning_kwargs.get(
                    "selection", np.array([], dtype=int)
                ),
                invert=True,
                extra_potentials=potentials,
                representation_noise_scale=representation_noise_scale,
            )
        else:
            self.stepper.initialize_diffusion(
                ensemble_size=ensemble_size,
                sampling_steps=num_steps,
                init_coords=coords,
                extra_potentials=potentials,
                representation_noise_scale=representation_noise_scale,
            )

        step_coords = self.stepper.cached_diffusion_init["atom_coords"]

        # TODO: implement Pseudo-B alignment weights from ROCKET

        scores = []

        if partial_diffusion:
            noising_steps = (
                diffusion_kwargs.get("noising_steps", num_steps // 4)
                if diffusion_kwargs
                else num_steps // 4
            )
            pbar = tqdm(range(noising_steps), desc="Optimizing structure")
        else:
            pbar = tqdm(range(num_steps), desc="Optimizing structure")

        for i in pbar:
            step_coords, loss = self.stepper.step_steering_ensemble(
                step_coords,
                augmentation=True,
                align_to_input=True,
                alignment_reverse_diffusion=True,
                selection=(
                    substructure_conditioning_kwargs["selection"]
                    if substructure_conditioning_kwargs is not None
                    else np.array([], dtype=int)
                ),
                ensemble_size=ensemble_size,
            )

            pbar.set_postfix(
                {
                    "score": f"{loss.mean().item():.4f}",
                }
            )
            scores.append(loss.cpu().numpy())

            coords_tensor = self.stepper.diffusion_trajectory[
                f"step_{self.stepper.current_step - 1}"
            ]["coords"]

            # FIXME: debugging, save calculated model and map every 10 steps
            if i % 10 == 0:
                with torch.no_grad():
                    model_map_ensemble = self.density_calculator(
                        coords_tensor.flatten(0, 1),
                        elements,
                        b_factors,
                        occupancies,
                        active,
                    )
                    summed_map_array = model_map_ensemble.reshape(
                        num_particles, ensemble_size, *model_map_ensemble.shape[1:]
                    ).sum(1)
                    # Use the existing XMap object to save the calculated density
                    for i in range(summed_map_array.shape[0]):
                        self.density_calculator.xmap.tofile(
                            f"{output_dir}/step_{self.stepper.current_step - 1}_map_{i}.ccp4",
                            density=summed_map_array[i],
                        )
                    torch.cuda.empty_cache()

                for i in range(coords_tensor.shape[0]):
                    step_structures = []
                    for j in range(coords_tensor.shape[1]):
                        structure = copy.deepcopy(self.structure)
                        structure.coor = coords_tensor[i, j].cpu().numpy()
                        step_structures.append(structure)

                    step_ensemble = Ensemble(step_structures)
                    step_ensemble.tofile(
                        f"{output_dir}/step_{self.stepper.current_step - 1}_ensemble_{i}.cif"
                    )

        final_coords_tensor = self.stepper.diffusion_trajectory[
            f"step_{self.stepper.current_step - 1}"
        ]["coords"]

        with torch.no_grad():
            model_map_ensemble = self.density_calculator(
                final_coords_tensor.flatten(0, 1),
                elements,
                b_factors,
                occupancies,
                active,
            )
            summed_map_array = model_map_ensemble.reshape(
                num_particles, ensemble_size, *model_map_ensemble.shape[1:]
            ).sum(1)
            # Use the existing XMap object to save the calculated density
            for i in range(summed_map_array.shape[0]):
                self.density_calculator.xmap.tofile(
                    f"{output_dir}/final_map_{i}.ccp4", density=summed_map_array[i]
                )
            torch.cuda.empty_cache()

        for i in range(final_coords_tensor.shape[0]):
            final_structures = []
            for j in range(final_coords_tensor.shape[1]):
                structure = copy.deepcopy(self.structure)
                structure.coor = final_coords_tensor[i, j].cpu().numpy()
                final_structures.append(structure)
            # TODO: Update q and b factors if necessary for the final state

            final_ensemble = Ensemble(final_structures)
            final_ensemble.tofile(f"{output_dir}/final_ensemble_{i}.cif")

        return final_structures, scores
