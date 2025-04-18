from typing import Callable, Dict, Optional, Tuple, Union
import torch
from math import sqrt

from pathlib import Path
from dataclasses import asdict
from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
from boltz.model.modules.utils import default, center_random_augmentation
from boltz.model.loss.diffusion import weighted_rigid_align
from dataclasses import asdict, dataclass

from adp3d.utils.utility import try_gpu
from adp3d.data.structure import Structure
from boltz.main import check_inputs, process_inputs, BoltzProcessedInput
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest
from boltz.data.feature.pad import pad_dim
import numpy as np
from numpy.typing import NDArray


@dataclass
class PredictArgs:
    """Arguments for model prediction."""

    recycling_steps: int = 3  # default in Boltz1
    sampling_steps: int = 200
    diffusion_samples: int = (
        1  # number of samples you want to generate, will be used as multiplicity
    )
    write_confidence_summary: bool = True
    write_full_pae: bool = False
    write_full_pde: bool = False


class DiffusionStepper:
    """Controls fine-grained diffusion steps using the pretrained Boltz1 model.

    This class provides granular control over the diffusion process by:
    1. Loading and caching model representations after the pairformer stage
    2. Enabling step-by-step diffusion with custom parameters
    3. Maintaining the original model weights and architecture
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        data_path: Union[str, Path],
        out_dir: Union[str, Path],
        model: Optional[Boltz1] = None,
        use_msa_server: bool = True,
        predict_args: PredictArgs = PredictArgs(),
        diffusion_args: BoltzDiffusionParams = BoltzDiffusionParams(),
        device: Optional[torch.device] = None,
    ) -> None:
        """Load Boltz-1 pretrained model weights and components from checkpoint.

        Parameters
        ----------
        checkpoint_path : Union[str, Path]
            Path to the model checkpoint file.
        data_path : Union[str, Path]
            Path to the input data (folder of YAML files, FASTA files, or a FASTA or YAML file).
        out_dir : Union[str, Path]
            Path to the output directory.
        model : Optional[Boltz1], optional
            Preloaded model, by default None.
        use_msa_server : bool, optional
            Whether to use the MSA server, by default True.
        predict_args : PredictArgs, optional
            Arguments for model prediction, by default PredictArgs().
        diffusion_args : BoltzDiffusionParams, optional
            Diffusion parameters, by default BoltzDiffusionParams(). step_scale is most useful,
            set to a lower value (default 1.638) to get more diversity.
        device : Optional[torch.device], optional
            Device to load the model to, by default None.

        Returns
        -------
        None
        """
        self.device = device or try_gpu()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.cache_path = Path(
            checkpoint_path
        ).parent  # NOTE: assumes checkpoint and ccd dictionary get downloaded to the same place

        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            self.model = (
                Boltz1.load_from_checkpoint(
                    checkpoint_path,
                    strict=True,
                    predict_args=asdict(predict_args),
                    map_location="cpu",
                    diffusion_process_args=asdict(diffusion_args),
                    ema=False,
                )
                .to(self.device)
                .eval()
            )

        self.setup(data_path=data_path, out_dir=out_dir, use_msa_server=use_msa_server)

        self.cached_representations: Dict[str, torch.Tensor] = {}
        self.cached_diffusion_init = {}
        self.diffusion_trajectory: Dict[str, torch.Tensor] = {}
        self.current_step: int = 0

    def setup(
        self,
        data_path: Union[str, Path],
        out_dir: Union[str, Path],
        use_msa_server: bool = True,
    ) -> BoltzInferenceDataModule:
        """Get BoltzInferenceDataModule set up so the stepper can run on a batch.

        Parameters
        ----------
        data_path : Union[str, Path]
            Path to the input data (folder of YAML files, FASTA files, or a FASTA or YAML file).

        Returns
        -------
        BoltzInferenceDataModule
            Data module containing processed inputs.
        """
        input_path = Path(data_path) if isinstance(data_path, str) else data_path
        out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        input_path = input_path.expanduser().resolve()
        ccd_path = self.cache_path / "ccd.pkl"
        data = check_inputs(input_path, out_dir, False)

        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            use_msa_server=use_msa_server,
            msa_server_url="https://api.colabfold.com",  # NOTE: this requires internet access on cluster
            msa_pairing_strategy="greedy",
        )

        # Load processed data
        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=Manifest.load(processed_dir / "manifest.json"),
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
        )

        # Create data module # TODO: set this up so batched will work with later functions? This will require getting density maps into the schema I think
        data_module = BoltzInferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            num_workers=2,  # NOTE: default in Boltz1
        )

        self.data_module = data_module

    def prepare_feats_from_datamodule_batch(
        self,
    ) -> Dict[str, torch.Tensor]:
        """Prepare features from a DataModule batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch from BoltzInferenceDataModule.

        Returns
        -------
        Dict[str, torch.Tensor]
            Processed features ready for the model.
        """
        return self.data_module.transfer_batch_to_device(
            next(iter(self.data_module.predict_dataloader())), self.device, 0
        )  # FIXME: I generally assume batch size of 1, which will break in the future.

    def compute_representations(
        self,
        feats: Dict[str, torch.Tensor],
        recycling_steps: Optional[int] = None,
    ) -> None:
        """Compute and cache main trunk representations.

        Parameters
        ----------
        feats : Dict[str, torch.Tensor]
            Input feats containing model features
        recycling_steps : Optional[int], optional
            Override default number of recycling steps, by default None
        """
        recycling_steps = recycling_steps or self.model.predict_args["recycling_steps"]

        with torch.no_grad():
            # Compute input embeddings
            s_inputs = self.model.input_embedder(feats)

            # Initialize sequence and pairwise embeddings
            s_init = self.model.s_init(s_inputs)
            z_init = (
                self.model.z_init_1(s_inputs)[:, :, None]
                + self.model.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.model.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.model.token_bonds(feats["token_bonds"].float())

            # Initialize tensors for recycling
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Compute pairwise mask
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            # Recycling iterations
            for i in range(recycling_steps + 1):
                s = s_init + self.model.s_recycle(self.model.s_norm(s))
                z = z_init + self.model.z_recycle(self.model.z_norm(z))

                if not self.model.no_msa:
                    z = z + self.model.msa_module(z, s_inputs, feats)

                s, z = self.model.pairformer_module(
                    s, z, mask=mask, pair_mask=pair_mask
                )

            # Cache outputs
            self.cached_representations = {
                "s": s,
                "z": z,
                "s_inputs": s_inputs,
                "relative_position_encoding": relative_position_encoding,
                "feats": feats,
            }

    def initialize_partial_diffusion(
        self,
        structure: Union[Structure, torch.Tensor],
        noising_steps: int = 0,
        num_samples: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        selector: NDArray[np.bool_] = None,
    ) -> None:
        """
        Initialize with a partial diffusion setup, starting from some initial set of coordinates. This allows denoising from
        a partially noised input, which is useful for perturbing from some base set of coordinates for an ensemble.

        Parameters
        ----------
        structure : Union[Structure, torch.Tensor]
            Initial structure or set of atomic coordinates. If not a tensor, it is assumed to
            have an attribute (e.g. `coords`) that contains the coordinates.
        noising_steps : int, optional
            Number of noising steps.
        num_samples : Optional[int], optional
            Number of samples to generate (used to determine diffusion multiplicity),
            by default the value from predict_args.
        sampling_steps : Optional[int], optional
            Total number of sampling steps in the diffusion process,
            by default the value from the model's structure_module.
        selector : NDArray[np.bool_], optional
            Selector mask for atoms to be noised, by default None (all atoms are noised).
        """
        self.diffusion_trajectory = {}

        batch = self.prepare_feats_from_datamodule_batch()
        self.compute_representations(batch)

        num_sampling_steps = default(
            sampling_steps, self.model.structure_module.num_sampling_steps
        )
        diffusion_samples = default(
            num_samples, self.model.predict_args["diffusion_samples"]
        )

        if noising_steps < 0 or num_sampling_steps - noising_steps <= 0:
            raise ValueError(
                f"Invalid number of noising steps: ({noising_steps}) or sampling steps: ({num_sampling_steps})."
            )
        self.current_step = num_sampling_steps - noising_steps

        atom_mask = self.cached_representations["feats"]["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(diffusion_samples, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.model.structure_module.sample_schedule(num_sampling_steps)
        gammas = torch.where(
            sigmas > self.model.structure_module.gamma_min,
            self.model.structure_module.gamma_0,
            0.0,
        )
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is noise at the beginning
        atom_coords = (
            torch.tensor(structure.coor, device=self.device)
            .float()
            .unsqueeze(0)
            .repeat(diffusion_samples, 1, 1)
        )

        atom_coords = pad_dim(atom_coords, 1, shape[1] - atom_coords.shape[1])
        init_coords = atom_coords.clone()
        eps = (
            self.model.structure_module.noise_scale
            * sigmas[-noising_steps - 1]
            * torch.randn(shape, device=self.device)
        )

        atom_coords = atom_coords + eps

        token_repr = None
        token_a = None

        self.cached_diffusion_init = {
            "init_coords": init_coords,
            "atom_coords": atom_coords,
            "atom_mask": atom_mask,
            "token_repr": token_repr,
            "token_a": token_a,
            "sigmas_and_gammas": sigmas_and_gammas,
            "diffusion_samples": diffusion_samples,
            "num_sampling_steps": num_sampling_steps,
        }

    def initialize_diffusion(
        self,
        num_samples: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        init_coords: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the diffusion process.

        Parameters
        ----------
        num_samples : Optional[int], optional
            Number of samples to generate, by default the number from predict_args in initialization
        sampling_steps : Optional[int], optional
            Number of sampling steps, by default the number from predict_args in initialization
        init_coords : Optional[torch.Tensor], optional
            Initial coordinates for downstream guidance, by default None
        """

        self.current_step = 0
        self.diffusion_trajectory = {}

        batch = self.prepare_feats_from_datamodule_batch()
        self.compute_representations(batch)

        num_sampling_steps = default(
            sampling_steps, self.model.structure_module.num_sampling_steps
        )
        diffusion_samples = default(
            num_samples, self.model.predict_args["diffusion_samples"]
        )
        atom_mask = self.cached_representations["feats"]["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(diffusion_samples, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.model.structure_module.sample_schedule(num_sampling_steps)
        gammas = torch.where(
            sigmas > self.model.structure_module.gamma_min,
            self.model.structure_module.gamma_0,
            0.0,
        )
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)

        token_repr = None
        token_a = None

        self.cached_diffusion_init = {
            "init_coords": pad_dim(init_coords, 1, shape[1] - init_coords.shape[1]),
            "atom_coords": atom_coords,
            "atom_mask": atom_mask,
            "token_repr": token_repr,
            "token_a": token_a,
            "sigmas_and_gammas": sigmas_and_gammas,
            "diffusion_samples": diffusion_samples,
            "num_sampling_steps": num_sampling_steps,
        }

    def step(
        self,
        atom_coords: torch.Tensor,
        return_denoised: bool = False,
        augmentation: bool = True,
        align_to_input: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Execute a single diffusion denoising step.

        Parameters
        ----------
        atom_coords : torch.Tensor
            Current atomic coordinates of shape (batch, num_atoms, 3)
        return_denoised : bool, optional
            Whether to return the fully denoised coordinate prediction, by default False
        augmentation : bool, optional
            Whether to apply augmentation, by default True
        align_to_input : bool, optional
            Whether to align the output coordinates to the initial input coordinates (if provided during initialization), by default True.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Denoised atomic coordinates after a single step in the trajectory, and optionally the fully denoised coordinate prediction.
        """
        # Get cached representations
        s = self.cached_representations["s"]
        z = self.cached_representations["z"]
        s_inputs = self.cached_representations["s_inputs"]
        relative_position_encoding = self.cached_representations[
            "relative_position_encoding"
        ]
        feats = self.cached_representations["feats"]
        multiplicity = self.cached_diffusion_init[
            "diffusion_samples"
        ]  # batch is regulated by dataloader, this lets you do ensemble prediction

        # Get cached diffusion info
        atom_mask: torch.Tensor = self.cached_diffusion_init["atom_mask"]
        sigma_tm, sigma_t, gamma = self.cached_diffusion_init["sigmas_and_gammas"][
            self.current_step
        ]
        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

        t_hat = sigma_tm * (1 + gamma)
        eps = (
            self.model.structure_module.noise_scale
            * sqrt(t_hat**2 - sigma_tm**2)
            * torch.randn(atom_coords.shape, device=self.device)
        )

        # NOTE: This might create some interesting pathologies, but in principle this augmentation should not be needed post-training
        if augmentation:
            atom_coords = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
            )

        atom_coords_noisy = atom_coords + eps

        with torch.no_grad():
            atom_coords_denoised, _ = (
                self.model.structure_module.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        s_trunk=s,
                        z_trunk=z,
                        s_inputs=s_inputs,
                        feats=feats,
                        relative_position_encoding=relative_position_encoding,
                        multiplicity=multiplicity,
                    ),
                )
            )

        # Alignment reverse diffusion
        atom_coords_noisy = weighted_rigid_align(
            atom_coords_noisy.float(),
            atom_coords_denoised.float(),
            atom_mask.float(),
            atom_mask.float(),
        )

        atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
        atom_coords_next: torch.Tensor = (
            atom_coords_noisy
            + self.model.structure_module.step_scale
            * (sigma_t - t_hat)
            * denoised_over_sigma
        )

        # Align to input
        if align_to_input:
            if self.cached_diffusion_init["init_coords"] is None:
                raise ValueError(
                    "No initial input coordinates found in cached diffusion init. Please change from align_to_input if you are not using partial diffusion."
                )
            atom_coords_next = weighted_rigid_align(
                atom_coords_next.float(),
                self.cached_diffusion_init["init_coords"].float(),
                atom_mask.float(),
                atom_mask.float(),
            ).to(atom_coords_next)

        pad_mask = feats["atom_pad_mask"].squeeze().bool()
        unpad_coords_next = atom_coords_next[
            :, pad_mask, :
        ]  # unpad the coords to B, N_unpad, 3
        unpad_coords_denoised = atom_coords_denoised[
            :, pad_mask, :
        ]  # unpad the coords to B, N_unpad, 3

        # Store unpadded in trajectory (0 indexed)
        self.diffusion_trajectory[f"step_{self.current_step}"] = {
            "coords": unpad_coords_next.clone(),
            "denoised": unpad_coords_denoised.clone(),  # the overall prediction from this current level (no noise mixture)
        }

        self.current_step += 1  # NOTE: current step to execute

        if return_denoised:
            return atom_coords_next, atom_coords_denoised
        else:
            return atom_coords_next


class DensityGuidedDiffusionStepper(DiffusionStepper):
    """Controls fine-grained diffusion steps using the pretrained Boltz1 model and guidance via the diffusion update"""

    def step(
        self,
        atom_coords: torch.Tensor,
        density_loss: Callable,
        guidance_scale: float = 0.1,
        return_denoised: bool = False,
        augmentation: bool = True,
        selection: Optional[NDArray[np.bool_]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Execute a single diffusion denoising step with density guidance.

        Parameters
        ----------
        atom_coords : torch.Tensor
            Current atomic coordinates of shape (batch, num_atoms, 3).
        density_score : Callable
            Function that takes in the current atomic coordinates and returns the loss as a Tensor
        guidance_scale : float, optional
            Scale factor for applying the density gradient guidance, by default 0.1.
        return_denoised : bool, optional
            Whether to return the fully denoised coordinate prediction alongside the next step coordinates, by default False.
        augmentation : bool, optional
            Whether to apply random centering augmentation, by default True.
        selection : Optional[NDArray[int]], optional
            Indices of atoms to apply diffusion to. If None, applies to all atoms. By default None.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor, float]
            Coordinates after a single guided diffusion step.
            If `return_denoised` is True, returns a tuple containing the next step
            coordinates and the fully denoised coordinate prediction for the current step.
            The third element is the guidance loss.
        """
        # Get cached representations
        s = self.cached_representations["s"]
        z = self.cached_representations["z"]
        s_inputs = self.cached_representations["s_inputs"]
        relative_position_encoding = self.cached_representations[
            "relative_position_encoding"
        ]
        feats = self.cached_representations["feats"]
        multiplicity = self.cached_diffusion_init[
            "diffusion_samples"
        ]  # batch is regulated by dataloader, this lets you do ensemble prediction
        pad_mask = feats["atom_pad_mask"].squeeze().bool()

        # Get cached diffusion info
        atom_mask: torch.Tensor = self.cached_diffusion_init["atom_mask"]
        sigma_tm, sigma_t, gamma = self.cached_diffusion_init["sigmas_and_gammas"][
            self.current_step
        ]
        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

        t_hat = sigma_tm * (1 + gamma)
        eps = (
            self.model.structure_module.noise_scale
            * sqrt(t_hat**2 - sigma_tm**2)
            * torch.randn(atom_coords.shape, device=self.device)
        )

        # NOTE: This might create some interesting pathologies, but in principle this augmentation should not be needed post-training
        if augmentation:
            atom_coords = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
            )

        # NOTE: only apply noise to the selected atoms, this is probably not good for staying on the diffusion manifold
        # atom_coords_noisy = atom_coords.clone()
        # if selection is not None:
        #     selection = torch.from_numpy(selection).to(self.device)
        #     atom_coords_noisy[:, selection, :] = atom_coords[:, selection, :] + eps[:, selection, :]
        # else:
        #     atom_coords_noisy += eps
        atom_coords_noisy = atom_coords + eps

        with torch.no_grad():
            atom_coords_denoised, _ = (
                self.model.structure_module.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        s_trunk=s,
                        z_trunk=z,
                        s_inputs=s_inputs,
                        feats=feats,
                        relative_position_encoding=relative_position_encoding,
                        multiplicity=multiplicity,
                    ),
                )
            )

        # replace the unselected (not in segment) atoms in denoised with the initial structure coords
        # NOTE: This is from the Maddipatla paper, but I would probably do something different?
        if selection is not None:
            selection = torch.from_numpy(selection).to(self.device) # TODO: set this up in a more efficient way
            inverse_selector = torch.ones(atom_coords_denoised.shape[1], device=self.device).bool()
            inverse_selector[selection] = False
            atom_coords_denoised[:, inverse_selector, :] = self.cached_diffusion_init[
                "init_coords"
            ][:, inverse_selector, :]

        if augmentation:
            atom_coords_noisy = weighted_rigid_align(
                atom_coords_noisy.float(),
                atom_coords_denoised.float(),
                atom_mask.float(),
                atom_mask.float(),
            )

        with torch.set_grad_enabled(True):  # Explicit gradient context
            masked_coords = atom_coords_noisy[:, pad_mask, :]
            coords_to_grad = masked_coords.detach().clone()
            coords_to_grad = coords_to_grad.requires_grad_(True)

            # TODO: only compute density and gradient for partially diffused atoms in segment (requires map subtraction)
            loss = density_loss(coords_to_grad)
            loss.backward()

            if coords_to_grad.grad is None:
                raise ValueError("Gradient computation failed - tensor is not a leaf")

            full_grad = torch.zeros_like(atom_coords_noisy)

            # only use gradient on partially diffused atoms in segment
            # if selection is not None:
            #     selector = torch.from_numpy(selection).to(self.device)
            #     full_grad[:, selector, :] = coords_to_grad.grad[:, selector, :]
            # else:
            full_grad[:, pad_mask, :] = coords_to_grad.grad # use whole grad each time

        atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat # NOTE: try atom coords?

        scaled_guidance_grad = (
            torch.linalg.norm(denoised_over_sigma)
            / torch.linalg.norm(full_grad)
            * full_grad
        )

        denoised_over_sigma = (
            denoised_over_sigma + scaled_guidance_grad * guidance_scale
        )

        atom_coords_next: torch.Tensor = (
            atom_coords_noisy
            + self.model.structure_module.step_scale
            * (sigma_t - t_hat)
            * denoised_over_sigma
        )

        # Align to input instead of alignment reverse diffusion
        # if align_to_input: # TODO: I don't think this is needed when replacing all but segment
        #     if self.cached_diffusion_init["init_coords"] is None:
        #         raise ValueError(
        #             "No initial input coordinates found in cached diffusion init. Please change from align_to_input if you are not using partial diffusion."
        #         )
        #     atom_coords_next = weighted_rigid_align(
        #         atom_coords_next.float(),
        #         self.cached_diffusion_init["init_coords"].float(),
        #         atom_mask.float(),
        #         atom_mask.float(),
        #     ).to(atom_coords_next)

        unpad_coords_next = atom_coords_next[
            :, pad_mask, :
        ]  # unpad the coords to B, N_unpad, 3
        unpad_coords_denoised = atom_coords_denoised[
            :, pad_mask, :
        ]  # unpad the coords to B, N_unpad, 3

        # Store unpadded in trajectory (0 indexed)
        self.diffusion_trajectory[f"step_{self.current_step}"] = {
            "coords": unpad_coords_next.clone(),
            "denoised": unpad_coords_denoised.clone(),  # the overall prediction from this current level (no noise mixture)
        }

        self.current_step += 1  # NOTE: current step to execute

        if return_denoised:
            return atom_coords_next, atom_coords_denoised, -loss.item()
        else:
            return atom_coords_next, -loss.item()
