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
from functools import partial
from typing import Optional, Tuple, Union, List
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat

from boltz.main import BoltzDiffusionParams
from boltz.model.model import Boltz1
from boltz.data.feature.pad import pad_dim
from boltz.model.potentials.schedules import (
    PiecewiseStepFunction,
    PiecewiseStepComposer,
    ExponentialInterpolation,
)

from adp3d.data import Structure
from adp3d.adp.modules.density import (
    DifferentiableTransformer,
    XMap_torch,
    normalize,
    to_f_density,
    scale_map,
)
from adp3d.data.structure import Ensemble
from adp3d.qfit.volume import XMap
from adp3d.data.io import structure_to_density_input
from adp3d.data.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
)
from adp3d.adp.modules.diffusion import DiffusionStepper
from adp3d.adp.modules.guided_diffusion import DensityGuidedDiffusionStepper
from adp3d.utils.utility import try_gpu
from adp3d.adp.modules.potentials import SubstructurePotential, DensityPotential


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
        resolution: Optional[float] = None,
        step_scale: float = 1.638,  # default step scale, ok results down to 0.8 with higher diversity
        ckpt_path: Path = Path("~/.boltz/boltz1_conf.pkl").expanduser(),
        model: Optional[Boltz1] = None,
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
            Scale factor for diffusion steps, by default 1.638
        ckpt_path : Path, optional
            Path to the Boltz1 model checkpoint, by default "~/.boltz/boltz1_conf.pkl"
        model : Boltz1, optional
            Pre-loaded Boltz1 model instance, by default None
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

        scaled_map_path = self.output_path / "scaled_experimental_map.ccp4"
        self.density_calculator.xmap.tofile(str(scaled_map_path), density=self.y)
        print(f"Saved scaled experimental map to {scaled_map_path}")

        diffusion_args = BoltzDiffusionParams(step_scale=step_scale)
        self.stepper = DensityGuidedDiffusionStepper(
            checkpoint_path=ckpt_path,
            data_path=input_path,
            out_dir=output_path,
            diffusion_args=diffusion_args,
            device=self.device,
            model=model,
        )

        self.initial_centroid: Optional[torch.Tensor] = None

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
        num_samples: int = 1,
        partial_diffusion: bool = False,
        steering: bool = False,
        diffusion_kwargs: Optional[dict] = None,
        substructure_conditioning_kwargs: Optional[dict] = None,
    ) -> Structure:
        """Run density-guided optimization.

        Parameters
        ----------
        output_dir : str
            Output directory for optimized structure
        num_steps : int, optional
            Number of optimization steps, by default 200
        num_sample  : int, optional
            Size of ensemble to generate, by default 1.
            NOTE: If steering, this will be ignored and the number of particles in FK steering will be used.
        partial_diffusion : bool, optional
            Whether to use partial diffusion, by default False
        diffusion_kwargs : Dict[str, Any]
            Additional arguments for partial diffusion. May include:
                - noising_steps : int
                    Number of steps for noise addition, 25-30% of num_steps works well
                - structure : Structure
                    Input structure for partial diffusion
                - selection : NDArray[np.int]
                    Indices of segments for selective diffusion
        substructure_conditioning_kwargs : dict, optional
            Keyword arguments for substructure conditioning, by default None
            Values should be:
                - selection : NDArray[np.int]
                    Indices of atoms to leave out of conditioning
                - scale : float
                    Scale factor for the conditioning
        Returns
        -------
        Tuple[Ensemble, List[float]]
            Ensemble of optimized structures and their scores
        """
        os.makedirs(output_dir, exist_ok=True)

        # FIXME: resolution not needed here, will come from experimental map
        coords, elements, b_factors, occupancies, active, resolution = (
            structure_to_density_input(self.structure)
        )

        if steering:  # make sure particles are all updated by density
            num_samples = self.stepper.model.steering_args["num_particles"]

        coords = repeat(coords, "a c -> n a c", n=num_samples)
        elements = repeat(elements, "e -> n e", n=num_samples)
        # FIXME: using uniform b-factors and occupancies for now
        b_factors = repeat(b_factors, "b -> n b", n=num_samples)
        # b_factors = torch.full(elements.size(), 4 / num_samples)
        occupancies = (
            repeat(occupancies, "q -> n q", n=num_samples) # / num_samples # FIXME
        )
        active = repeat(active, "a -> n a", n=num_samples)

        if resolution == 0.0 or resolution is None:
            if self.density_calculator.xmap.resolution is not None:
                resolution = self.density_calculator.xmap.resolution.high
            warnings.warn(
                f"Resolution of input structure is {resolution}. Using 2.0 A instead."
            )

        coords = coords.to(self.device).float()
        elements = elements.to(self.device).long()
        b_factors = b_factors.to(self.device).float()
        occupancies = occupancies.to(self.device).float()
        active = active.to(self.device).bool()

        # Calculate and store initial centroid, then center coordinates
        self.initial_centroid = (coords * active.unsqueeze(-1)).sum(
            dim=1, keepdim=True
        ) / active.sum(dim=1, keepdim=True).unsqueeze(-1)
        coords_centered = coords - self.initial_centroid

        initialized_density_calculator = partial(
            self.density_calculator,
            elements=elements,
            b_factors=b_factors,
            occupancies=occupancies,
            active=active,
        )

        density_potential = DensityPotential(
            xmap=self.density_calculator.xmap,
            parameters={
                "guidance_interval": 1,
                "guidance_weight": PiecewiseStepFunction(
                    [0.25, 0.5, 0.75], [0.0001, 0.001, 0.01, 0.1]
                ),
                "resolution": PiecewiseStepComposer(
                    [0.25, 0.5, 0.75], [ExponentialInterpolation(resolution, 4., 0), 4., 8., 10.]
                ),
                "resampling_weight": PiecewiseStepFunction(
                    [0.25, 0.5, 0.75], [0.0001, 0.001, 0.01, 0.1]
                ),
                "occupancies": occupancies,
                "b_factors": b_factors,
                "initial_centroid": self.initial_centroid,
                "scattering_params": self.scattering_params,
                "em": self.em,
            },
        )

        potentials = [density_potential]
        # potentials = []

        if partial_diffusion:
            if diffusion_kwargs is None:
                diffusion_kwargs = {}
            if "structure" not in diffusion_kwargs:
                diffusion_kwargs["structure"] = coords_centered

            self.stepper.initialize_partial_diffusion(
                num_samples=num_samples,
                sampling_steps=num_steps,
                extra_potentials=potentials,
                **diffusion_kwargs,
            )
        elif substructure_conditioning_kwargs is not None:
            substructure_potential = SubstructurePotential(
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": 0.1,
                    "resampling_weight": 0.0,
                    "buffer": 0.5,
                    "denoising_selection": substructure_conditioning_kwargs.get(
                        "selection", np.array([], dtype=int)
                    ),
                    "reference_coords": coords_centered,
                }
            )
            potentials.append(substructure_potential)
            self.stepper.initialize_substructure_conditioned_diffusion(
                num_samples=num_samples,
                sampling_steps=num_steps,
                structure=coords_centered,
                selection=substructure_conditioning_kwargs.get(
                    "selection", np.array([], dtype=int)
                ),
                invert=True,
                extra_potentials=potentials,
            )
        else:
            self.stepper.initialize_diffusion(
                num_samples=num_samples,
                sampling_steps=num_steps,
                init_coords=coords_centered,
                extra_potentials=potentials,
            )

        step_coords = self.stepper.cached_diffusion_init["atom_coords"]

        # TODO: implement Pseudo-B alignment weights from ROCKET

        # v_density = torch.zeros_like(step_coords)
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
            # density guided step using self.density_score
            step_coords, loss = self.stepper.step_steering(
                step_coords,
                augmentation=True,
                align_to_input=True,
                alignment_reverse_diffusion=True,
                selection=(
                    substructure_conditioning_kwargs["selection"]
                    if substructure_conditioning_kwargs is not None
                    else np.array([], dtype=int)
                ),
                ensemble_size = num_samples,
            )

            # update the progress bar with negative log likelihood
            pbar.set_postfix(
                {
                    "score": f"{loss:.4f}",
                }
            )
            scores.append(loss)

            coords_tensor = self.stepper.diffusion_trajectory[
                f"step_{self.stepper.current_step - 1}"
            ]["coords"]
            # Translate coords back before saving intermediate results
            coords_tensor_translated = coords_tensor + self.initial_centroid.repeat(
                coords_tensor.shape[0] // self.initial_centroid.shape[0], 1, 1
            )

            step_structures = []

            # FIXME: debugging, save calculated model and map every 10 steps
            if i % 10 == 0:
                with torch.no_grad():
                    model_map_ensemble = self.density_calculator(
                        coords_tensor_translated,
                        elements,
                        b_factors,
                        occupancies,
                        active,
                    )
                    summed_map_array = model_map_ensemble.sum(0)
                    # Use the existing XMap object to save the calculated density
                    self.density_calculator.xmap.tofile(
                        f"{output_dir}/step_{self.stepper.current_step - 1}_map.ccp4",
                        density=summed_map_array,
                    )
                    torch.cuda.empty_cache()

                for j in range(coords_tensor_translated.shape[0]):
                    structure = copy.deepcopy(self.structure)

                    structure.coor = coords_tensor_translated[j].cpu().numpy()
                    # TODO: Update q and b factors if they are also being optimized or change
                    step_structures.append(structure)

                step_ensemble = Ensemble(step_structures)
                step_ensemble.tofile(
                    f"{output_dir}/step_{self.stepper.current_step - 1}_ensemble.cif"
                )

        final_coords_tensor = self.stepper.diffusion_trajectory[
            f"step_{self.stepper.current_step - 1}"
        ]["coords"]
        # Translate final coords back before saving
        final_coords_tensor_translated = final_coords_tensor + self.initial_centroid
        final_structures = []

        with torch.no_grad():
            model_map_ensemble = self.density_calculator(
                final_coords_tensor_translated,
                elements,
                b_factors,
                occupancies,
                active,
            )
            summed_map_array = model_map_ensemble.sum(0)
            # Use the existing XMap object to save the calculated density
            self.density_calculator.xmap.tofile(
                f"{output_dir}/final_map.ccp4", density=summed_map_array
            )
            torch.cuda.empty_cache()

        for j in range(final_coords_tensor_translated.shape[0]):
            structure = copy.deepcopy(self.structure)
            structure.coor = final_coords_tensor_translated[j].cpu().numpy()
            # TODO: Update q and b factors if necessary for the final state
            final_structures.append(structure)

        final_ensemble = Ensemble(final_structures)
        final_ensemble.tofile(f"{output_dir}/final_ensemble.cif")

        return final_structures, scores
