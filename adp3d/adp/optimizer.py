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

import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat

from boltz.main import BoltzDiffusionParams
from boltz.model.model import Boltz1
from boltz.data.feature.pad import pad_dim

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
            except:
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
        coords = coords.to(self.device).float().unsqueeze(0) # Add batch dim
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

    def density_score(
        self,
        coords: torch.Tensor,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        occupancies: torch.Tensor,
        active: torch.Tensor,
        initial_centroid: torch.Tensor,
        norm: int = 1,
        substructure_conditioning_kwargs: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate density score and optionally substructure score for current coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Current atomic coordinates, shape [batch, atoms, 3]
            NOTE: This should be just the atoms, not padded.
        elements : torch.Tensor
            Element atomic numbers for each atom, shape [batch, atoms]
        b_factors : torch.Tensor
            B-factors for each atom, shape [batch, atoms]
        occupancies : torch.Tensor
            Occupancies for each atom, shape [batch, atoms]
        active : torch.Tensor
            Mask for active atoms, shape [batch, atoms]
        initial_centroid : torch.Tensor
            Centroid of the original input coordinates, shape [batch, 1, 3]
        norm : int, optional
            Which norm to use for the score, by default 1
        substructure_conditioning_kwargs : dict, optional
            Keyword arguments for substructure conditioning, by default None
            Values should be:
                - selection : NDArray[np.bool_]
                    Indices of atoms to leave out of conditioning
                - coords : torch.Tensor
                    Coordinates of atoms to condition on, shape [batch, atoms, 3]
                - scale : float
                    Scale factor for the conditioning

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Density score and substructure score
        """
        # Translate coordinates back to the original frame for density calculation
        coords_translated = coords + initial_centroid

        substructure_score = None
        if substructure_conditioning_kwargs is not None:
            # Translate conditioning coords as well
            conditioning_coords = substructure_conditioning_kwargs["coords"] + initial_centroid
            selection = substructure_conditioning_kwargs["selection"]
            scale = substructure_conditioning_kwargs["scale"]
            inverse_selector = torch.ones(conditioning_coords.shape[1], device=self.device).bool()
            inverse_selector[selection] = False
            # Use translated coordinates for comparison
            conditioning_coords_subset = conditioning_coords[:, inverse_selector, :]
            current_coords_subset = coords_translated[:, inverse_selector, :]

            # compute the difference between the conditioning coordinates and the current coords
            substructure_score = - (
                scale
                / (coords.shape[0])
                * torch.linalg.norm(current_coords_subset - conditioning_coords_subset)
            )

        model_map = self.density_calculator(
            coords_translated, elements, b_factors, occupancies, active
        ).sum(
            0
        )  # TODO: dont use normalization, use e-/A^3

        density_correlation_score = -torch.linalg.norm(
                torch.flatten(self.y) - torch.flatten(model_map), ord=norm
            )

        return density_correlation_score, substructure_score
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
        substructure_conditioning_kwargs: dict = None,
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
        coords = repeat(coords, "a c -> n a c", n=num_samples)
        elements = repeat(elements, "e -> n e", n=num_samples)
        # FIXME: using uniform b-factors and occupancies for now
        b_factors = repeat(b_factors, "b -> n b", n=num_samples) / num_samples
        occupancies = repeat(occupancies, "q -> n q", n=num_samples) / num_samples
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
        self.initial_centroid = (coords * active.unsqueeze(-1)).sum(dim=1, keepdim=True) / active.sum(dim=1, keepdim=True).unsqueeze(-1)
        coords_centered = coords - self.initial_centroid

        if partial_diffusion:
            if "structure" in diffusion_kwargs:
                warnings.warn("Partial diffusion with structure input not fully handled regarding centering. Ensure provided structure coords are centered if needed.")
            
            if 'init_coords' not in diffusion_kwargs:
                 diffusion_kwargs['init_coords'] = coords_centered

            self.stepper.initialize_partial_diffusion(
                num_samples=num_samples, sampling_steps=num_steps, **diffusion_kwargs
            )
        else:
            self.stepper.initialize_diffusion(
                num_samples=num_samples, sampling_steps=num_steps, init_coords=coords_centered
            )

        step_coords = self.stepper.cached_diffusion_init["atom_coords"]

        # TODO: implement Pseudo-B alignment weights from ROCKET

        if substructure_conditioning_kwargs is not None:
            substructure_conditioning_kwargs["coords"] = coords
            selection = torch.from_numpy(substructure_conditioning_kwargs["selection"]).to(
                self.device
            )
            inverse_selector = torch.ones(
                step_coords.shape[1], device=self.device
            ).bool()
            inverse_selector[selection] = False
            coords_centered_padded = pad_dim(coords_centered, 1, step_coords.shape[1] - coords_centered.shape[1])
            # replace the unselected (not in segment) atoms in denoised with the initial structure coords for constraint
            # NOTE: inverse_selector[:coords.shape[1]] is used to ensure the shape matches, as coords is not padded
            # and padded 0s are added at the end of step_coords
            step_coords[:, inverse_selector, :] = coords_centered_padded[:, inverse_selector, :]
            density_loss = partial(
                self.density_score,
                elements=elements,
                b_factors=b_factors,
                occupancies=occupancies,
                active=active,
                initial_centroid=self.initial_centroid,
                norm=1,
                substructure_conditioning_kwargs=substructure_conditioning_kwargs,
            )
        else:
            density_loss = partial( # TODO: this will break downstream as substructure_score will be None
                self.density_score,
                elements=elements,
                b_factors=b_factors,
                occupancies=occupancies,
                active=active,
                initial_centroid=self.initial_centroid,
                norm=1,
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

            # density guided step using self.density_score
            step_coords, loss = self.stepper.step(
                step_coords,
                density_loss=density_loss,
                guidance_scale=step_lr,
                augmentation=True,
                align_to_input=True,
                alignment_reverse_diffusion=False, # FIXME: Breaks the computational graph
                selection=(
                    substructure_conditioning_kwargs["selection"]
                    if substructure_conditioning_kwargs is not None
                    else None
                ),
            )

            # update the progress bar with negative log likelihood
            pbar.set_postfix(
                {
                    "score": f"{loss:.4f}",
                }
            )
            scores.append(loss)

            # Gradient descent with momentum # TODO: try others?
            # v_density = 0.9 * v_density + step_lr * full_grad.unsqueeze(0)
            # step_coords = step_coords - v_density

            # Raw gradient descent
            # step_coords = step_coords - step_lr * full_grad.unsqueeze(
            #     0
            # )
            coords_tensor = self.stepper.diffusion_trajectory[
                f"step_{i}"
            ]["coords"]
            # Translate coords back before saving intermediate results
            coords_tensor_translated = coords_tensor + self.initial_centroid

            ensemble_size = coords_tensor_translated.shape[0]
            step_structures = []

            # FIXME: debugging, save calculated model and map every 10 steps
            if i % 10 == 0:
                with torch.no_grad():
                    model_map_ensemble = self.density_calculator(
                        coords_tensor_translated,
                        elements,
                        b_factors,
                        occupancies,
                        active
                    )
                summed_map_array = model_map_ensemble.sum(0) 
                # Use the existing XMap object to save the calculated density
                self.density_calculator.xmap.tofile(f"{output_dir}/step_{i}_map.ccp4", density=summed_map_array)

                for j in range(ensemble_size):
                    structure = copy.deepcopy(self.structure)
                    
                    structure.coor = coords_tensor_translated[j].cpu().numpy()
                    # TODO: Update q and b factors if they are also being optimized or change
                    step_structures.append(structure)

                step_ensemble = Ensemble(step_structures)
                step_ensemble.tofile(f"{output_dir}/step_{i}_ensemble.cif")

        final_coords_tensor = self.stepper.diffusion_trajectory[
            f"step_{i}"
        ]["coords"]
        # Translate final coords back before saving
        final_coords_tensor_translated = final_coords_tensor + self.initial_centroid
        ensemble_size = final_coords_tensor_translated.shape[0]
        final_structures = []

        with torch.no_grad():
            model_map_ensemble = self.density_calculator(
                final_coords_tensor_translated,
                elements,
                b_factors,
                occupancies,
                active
            )
        summed_map_array = model_map_ensemble.sum(0) 
        # Use the existing XMap object to save the calculated density
        self.density_calculator.xmap.tofile(f"{output_dir}/final_map.ccp4", density=summed_map_array)

        for j in range(ensemble_size):
            structure = copy.deepcopy(self.structure)
            structure.coor = final_coords_tensor_translated[j].cpu().numpy()
            # TODO: Update q and b factors if necessary for the final state
            final_structures.append(structure)

        final_ensemble = Ensemble(final_structures)
        final_ensemble.tofile(f"{output_dir}/final_ensemble.cif")

        return final_structures, scores
