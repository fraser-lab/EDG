from typing import Dict, Optional, Tuple, Union
import torch
from math import sqrt

from pathlib import Path
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2
from boltz.main import (
    BoltzDiffusionParams,
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    PairformerArgs,
    PairformerArgsV2,
    MSAModuleArgs,
)

# from boltz.model.potentials.potentials import get_potentials
from boltz.model.modules.utils import default, center_random_augmentation
from boltz.model.loss.diffusion import weighted_rigid_align
from dataclasses import asdict, dataclass

from edg.utils.utility import try_gpu
from edg.data.structure import Structure
from edg.edg.modules.potentials import get_potentials
from boltz.main import check_inputs, process_inputs, BoltzProcessedInput
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.data.pad import pad_dim
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
    """Controls fine-grained diffusion steps using pretrained Boltz models.

    This class provides granular control over the diffusion process by:
    1. Loading and caching model representations after the pairformer stage
    2. Enabling step-by-step diffusion with custom parameters
    3. Maintaining the original model weights and architecture
    4. Supporting both Boltz-1 and Boltz-2 models
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        data_path: Union[str, Path],
        out_dir: Union[str, Path],
        model: Optional[Union[Boltz1, Boltz2]] = None,
        model_version: str = "boltz1",  # "boltz1" or "boltz2"
        use_msa_server: bool = True,
        predict_args: PredictArgs = PredictArgs(),
        diffusion_args: Union[BoltzDiffusionParams, Boltz2DiffusionParams] = None,
        steering_args: BoltzSteeringParams = BoltzSteeringParams(),
        method: str = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load pretrained Boltz model weights and components from checkpoint.

        Parameters
        ----------
        checkpoint_path : Union[str, Path]
            Path to the model checkpoint file.
        data_path : Union[str, Path]
            Path to the input data (folder of YAML files, FASTA files, or a FASTA or YAML file).
        out_dir : Union[str, Path]
            Path to the output directory.
        model : Optional[Union[Boltz1, Boltz2]], optional
            Preloaded model, by default None.
        model_version : str, optional
            Model version ("boltz1" or "boltz2"), by default "boltz1".
        use_msa_server : bool, optional
            Whether to use the MSA server, by default True.
        predict_args : PredictArgs, optional
            Arguments for model prediction, by default PredictArgs().
        diffusion_args : Union[BoltzDiffusionParams, Boltz2DiffusionParams], optional
            Diffusion parameters, by default None (auto-selected based on model version).
        steering_args : BoltzSteeringParams, optional
            Steering parameters, by default BoltzSteeringParams().
        method : str, optional
            Method name for method conditioning, by default None (which is X-ray Diffraction).
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

        self.model_version = model_version.lower()
        if self.model_version not in ["boltz1", "boltz2"]:
            raise ValueError(
                f"model_version must be 'boltz1' or 'boltz2', got {model_version}"
            )

        # Set default diffusion args based on model version if not provided
        if diffusion_args is None:
            if self.model_version == "boltz1":
                diffusion_args = BoltzDiffusionParams()
            else:  # boltz2
                diffusion_args = Boltz2DiffusionParams()

        # Set args based on model version
        if self.model_version == "boltz1":
            pairformer_args = PairformerArgs()
        else:  # boltz2
            pairformer_args = PairformerArgsV2()

        # MSAModuleArgs with correct parameters based on boltz repo
        msa_args = MSAModuleArgs(
            subsample_msa=True,  # Default from boltz repo
            num_subsampled_msa=1024,  # Default from boltz repo
            use_paired_feature=(self.model_version == "boltz2"),
        )

        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            if self.model_version == "boltz1":
                self.model = (
                    Boltz1.load_from_checkpoint(
                        checkpoint_path,
                        strict=True,
                        predict_args=asdict(predict_args),
                        map_location="cpu",
                        diffusion_process_args=asdict(diffusion_args),
                        ema=False,
                        use_kernels=True,  # Required parameter for Boltz1
                        pairformer_args=asdict(pairformer_args),
                        msa_args=asdict(msa_args),  # Correct parameter name
                        steering_args=asdict(steering_args),
                    )
                    .to(self.device)
                    .eval()
                )
            else:  # boltz2
                self.model = (
                    Boltz2.load_from_checkpoint(
                        checkpoint_path,
                        strict=True,
                        predict_args=asdict(predict_args),
                        map_location="cpu",
                        diffusion_process_args=asdict(diffusion_args),
                        ema=False,
                        pairformer_args=asdict(pairformer_args),
                        msa_args=asdict(msa_args),  # Correct parameter name
                        steering_args=asdict(steering_args),
                        # affinity_mw_correction=True,  # Required parameter for Boltz2
                    )
                    .to(self.device)
                    .eval()
                )

        self.data_module = self.setup(
            data_path=data_path,
            out_dir=out_dir,
            use_msa_server=use_msa_server,
            method=method,
        )

        self.cached_representations: Dict[str, torch.Tensor] = {}
        self.cached_diffusion_init = {}
        self.diffusion_trajectory: Dict[str, torch.Tensor] = {}
        self.current_step: int = 0

    def setup(
        self,
        data_path: Union[str, Path],
        out_dir: Union[str, Path],
        use_msa_server: bool = True,
        method: str = None,
    ) -> Union[BoltzInferenceDataModule, Boltz2InferenceDataModule]:
        """Get BoltzInferenceDataModule set up so the stepper can run on a batch.

        Parameters
        ----------
        data_path : Union[str, Path]
            Path to the input data (folder of YAML files, FASTA files, or a FASTA or YAML file).

        Returns
        -------
        Union[BoltzInferenceDataModule, Boltz2InferenceDataModule]
            Data module containing processed inputs.
        """
        input_path = Path(data_path) if isinstance(data_path, str) else data_path
        out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        input_path = input_path.expanduser().resolve()
        ccd_path = self.cache_path / "ccd.pkl"
        mol_dir = self.cache_path / "mols"
        data = check_inputs(input_path)

        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,  # Required for Boltz2
            use_msa_server=use_msa_server,
            msa_server_url="https://api.colabfold.com",  # NOTE: this requires internet access on cluster
            msa_pairing_strategy="greedy",
            boltz2=(self.model_version == "boltz2"),  # Required parameter
            preprocessing_threads=1,  # Default value
        )

        # Load processed data
        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=Manifest.load(processed_dir / "manifest.json"),
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            constraints_dir=(processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None,
            template_dir=processed_dir / "templates"
            if (processed_dir / "templates").exists()
            else None,
            extra_mols_dir=processed_dir / "mols"
            if (processed_dir / "mols").exists()
            else None,
        )

        # Create data module based on model version
        if self.model_version == "boltz1":
            data_module = BoltzInferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                num_workers=2,  # NOTE: default in Boltz1 is 2
                constraints_dir=processed.constraints_dir,
            )
        else:  # boltz2
            data_module = Boltz2InferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                mol_dir=mol_dir,  # Required for Boltz2
                num_workers=8,  # NOTE: default in Boltz2 is 8
                constraints_dir=processed.constraints_dir,
                template_dir=processed_dir / "templates"
                if (processed_dir / "templates").exists()
                else None,
                extra_mols_dir=processed_dir / "mols"
                if (processed_dir / "mols").exists()
                else None,
                override_method=method,  # Can be set if specific method conditioning is needed
            )

        return data_module

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
        )  # NOTE: I generally assume batch size of 1, which may break in the future.

    def compute_representations(
        self,
        feats: Dict[str, torch.Tensor],
        recycling_steps: Optional[int] = None,
        representation_noise_scale: Optional[float] = None,
    ) -> None:
        """Compute and cache main trunk representations.

        Parameters
        ----------
        feats : Dict[str, torch.Tensor]
            Input feats containing model features
        recycling_steps : Optional[int], optional
            Override default number of recycling steps, by default None
        representation_noise_scale : Optional[float], optional
            Scale for Gaussian noise added to single and pair representations for diversity, by default None
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

                if self.model.use_templates:
                    if self.model.is_template_compiled:
                        template_module = self.model.template_module._orig_mod  # noqa: SLF001
                    else:
                        template_module = self.model.template_module

                    z = z + template_module(
                        z, feats, pair_mask, use_kernels=self.model.use_kernels
                    )

                if self.model.is_msa_compiled:
                    msa_module = self.model.msa_module._orig_mod  # noqa: SLF001
                else:
                    msa_module = self.model.msa_module

                z = z + msa_module(
                    z, s_inputs, feats, use_kernels=self.model.use_kernels
                )

                if self.model.is_pairformer_compiled:
                    pairformer_module = self.model.pairformer_module._orig_mod  # noqa: SLF001
                else:
                    pairformer_module = self.model.pairformer_module

                s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask)

            # Add noise to representations for diversity if specified
            if (
                representation_noise_scale is not None
                and representation_noise_scale > 0
            ):
                s_noise = representation_noise_scale * torch.randn_like(s)
                z_noise = representation_noise_scale * torch.randn_like(z)
                s = s + s_noise
                z = z + z_noise

            q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                self.model.diffusion_conditioning(
                    s_trunk=s,
                    z_trunk=z,
                    relative_position_encoding=relative_position_encoding,
                    feats=feats,
                )
            )

            diffusion_conditioning = {
                "q": q,
                "c": c,
                "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            }

            # Cache outputs
            self.cached_representations = {
                "s": s,
                "z": z,
                "s_inputs": s_inputs,
                "relative_position_encoding": relative_position_encoding,
                "feats": feats,
                "diffusion_conditioning": diffusion_conditioning,
            }

    def initialize_diffusion(
        self,
        ensemble_size: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        init_coords: Optional[torch.Tensor] = None,
        extra_potentials: Optional[list] = None,
        representation_noise_scale: Optional[float] = None,
    ) -> None:
        """Initialize the diffusion process.

        Parameters
        ----------
        ensemble_size : Optional[int], optional
            Number of samples to generate, by default the number from predict_args in initialization
        sampling_steps : Optional[int], optional
            Number of sampling steps, by default the number from predict_args in initialization
        init_coords : Optional[torch.Tensor], optional
            Initial coordinates for downstream guidance, by default None
        representation_noise_scale : Optional[float], optional
            Scale for Gaussian noise added to representations for diversity, by default None
        """
        self.current_step = 0
        self.diffusion_trajectory = {}

        batch = self.prepare_feats_from_datamodule_batch()
        self.compute_representations(
            batch, representation_noise_scale=representation_noise_scale
        )

        num_sampling_steps = default(
            sampling_steps, self.model.structure_module.num_sampling_steps
        )
        diffusion_samples = default(
            ensemble_size, self.model.predict_args["diffusion_samples"]
        )
        atom_mask = self.cached_representations["feats"]["atom_pad_mask"]

        steering_vars = {}

        if self.model.steering_args["fk_steering"]:
            potentials = get_potentials()
            if extra_potentials is not None:
                potentials.extend(extra_potentials)
                # reverse the ordering so substructure and density come before physicality potentials
                potentials.reverse()
            num_particles = self.model.steering_args["num_particles"]
            energy_traj = torch.empty((num_particles, 0), device=self.device)
            resample_weights = torch.ones(num_particles, device=self.device).reshape(
                -1, self.model.steering_args["num_particles"]
            )
            steering_vars["energy_traj"] = energy_traj
            steering_vars["resample_weights"] = resample_weights
            steering_vars["potentials"] = potentials

        if self.model.steering_args["physical_guidance_update"]:
            scaled_guidance_update = torch.zeros(
                (diffusion_samples * num_particles, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )
            steering_vars["scaled_guidance_update"] = scaled_guidance_update

        atom_mask = atom_mask.repeat_interleave(diffusion_samples * num_particles, 0)
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

        init_coords = init_coords.reshape(-1, init_coords.shape[-2], 3)

        self.cached_diffusion_init = {
            "init_coords": pad_dim(init_coords, 1, shape[1] - init_coords.shape[1]),
            "atom_coords": atom_coords,
            "atom_mask": atom_mask,
            "token_repr": token_repr,
            "token_a": token_a,
            "sigmas_and_gammas": sigmas_and_gammas,
            "diffusion_samples": diffusion_samples,
            "num_sampling_steps": num_sampling_steps,
            "steering_vars": steering_vars,
        }

    def initialize_partial_diffusion(
        self,
        structure: Union[Structure, torch.Tensor],
        noising_steps: int = 0,
        ensemble_size: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        extra_potentials: Optional[list] = None,
        representation_noise_scale: Optional[float] = None,
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
        ensemble_size : Optional[int], optional
            Number of samples to generate (used to determine diffusion multiplicity),
            by default the value from predict_args.
        sampling_steps : Optional[int], optional
            Total number of sampling steps in the diffusion process,
            by default the value from the model's structure_module.
        selector : NDArray[np.bool_], optional
            Selector mask for atoms to be noised, by default None (all atoms are noised).
        potentials : Optional[list], optional
            List of potentials for steering, by default None.
        representation_noise_scale : Optional[float], optional
            Scale for Gaussian noise added to representations for diversity, by default None.
        """
        self.diffusion_trajectory = {}

        batch = self.prepare_feats_from_datamodule_batch()
        self.compute_representations(
            batch, representation_noise_scale=representation_noise_scale
        )

        num_sampling_steps = default(
            sampling_steps, self.model.structure_module.num_sampling_steps
        )
        diffusion_samples = default(
            ensemble_size, self.model.predict_args["diffusion_samples"]
        )

        if noising_steps < 0 or num_sampling_steps - noising_steps <= 0:
            raise ValueError(
                f"Invalid number of noising steps: ({noising_steps}) or sampling steps: ({num_sampling_steps})."
            )
        self.current_step = num_sampling_steps - noising_steps

        atom_mask = self.cached_representations["feats"]["atom_pad_mask"]

        # Setup steering variables dictionary
        steering_vars = {}

        if self.model.steering_args["fk_steering"]:
            potentials = get_potentials()
            if extra_potentials is not None:
                potentials.extend(extra_potentials)
                # reverse the ordering so substructure and density come before physicality potentials
                potentials.reverse()
            num_particles = self.model.steering_args["num_particles"]
            energy_traj = torch.empty((num_particles, 0), device=self.device)
            resample_weights = torch.ones(num_particles, device=self.device).reshape(
                -1, self.model.steering_args["num_particles"]
            )
            steering_vars["energy_traj"] = energy_traj
            steering_vars["resample_weights"] = resample_weights
            steering_vars["potentials"] = potentials

        if self.model.steering_args["physical_guidance_update"]:
            scaled_guidance_update = torch.zeros(
                (diffusion_samples * num_particles, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )
            steering_vars["scaled_guidance_update"] = scaled_guidance_update

        atom_mask = atom_mask.repeat_interleave(diffusion_samples * num_particles, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.model.structure_module.sample_schedule(num_sampling_steps)
        gammas = torch.where(
            sigmas > self.model.structure_module.gamma_min,
            self.model.structure_module.gamma_0,
            0.0,
        )
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is based on the init coords
        if isinstance(structure, Structure):
            atom_coords = (
                torch.tensor(structure.coor, device=self.device)
                .float()
                .unsqueeze(0)
                .repeat(diffusion_samples * num_particles, 1, 1)
            )
        elif isinstance(structure, torch.Tensor):
            atom_coords = structure.reshape(-1, structure.shape[-2], 3)

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
            "steering_vars": steering_vars,
        }

    def initialize_substructure_conditioned_diffusion(
        self,
        structure: Union[Structure, torch.Tensor],
        selection: NDArray[np.int_],
        ensemble_size: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        invert: bool = False,
        extra_potentials: Optional[list] = None,
        representation_noise_scale: Optional[float] = None,
    ) -> None:
        """Initialize diffusion with substructure conditioning.

        This method allows for initializing the diffusion process with a specific substructure
        selected by an indexing selection. Applies the principles from Chroma supplement section N.2
        to generate conditional samples given a motif (except here the prior is isotropic Gaussian noise,
        so it works out easier).

        Parameters
        ----------
        structure : Union[Structure, torch.Tensor]
            Initial structure or set of atomic coordinates.
        selection : NDArray[np.int_]
            Selector indices for atoms in structure.
        ensemble_size : Optional[int], optional
            Number of samples to generate, by default None.
        sampling_steps : Optional[int], optional
            Total number of sampling steps in the diffusion process, by default None.
        invert : bool, optional
            Whether to invert the selection (e.g. if the selection provided is for the motif to be denoised).
        potentials : Optional[list], optional
            List of potentials for steering, by default None.
        representation_noise_scale : Optional[float], optional
            Scale for Gaussian noise added to representations for diversity, by default None.
        """
        self.diffusion_trajectory = {}
        self.current_step = 0

        batch = self.prepare_feats_from_datamodule_batch()
        self.compute_representations(
            batch, representation_noise_scale=representation_noise_scale
        )

        num_sampling_steps = default(
            sampling_steps, self.model.structure_module.num_sampling_steps
        )
        diffusion_samples = default(
            ensemble_size, self.model.predict_args["diffusion_samples"]
        )

        atom_mask = self.cached_representations["feats"]["atom_pad_mask"]

        # Setup steering variables dictionary
        steering_vars = {}

        if self.model.steering_args["fk_steering"]:
            potentials = get_potentials()
            if extra_potentials is not None:
                potentials.extend(extra_potentials)
                # reverse the ordering so substructure and density come before physicality potentials
                potentials.reverse()
            num_particles = self.model.steering_args["num_particles"]
            energy_traj = torch.empty((num_particles, 0), device=self.device)
            resample_weights = torch.ones(num_particles, device=self.device).reshape(
                -1, self.model.steering_args["num_particles"]
            )
            steering_vars["energy_traj"] = energy_traj
            steering_vars["resample_weights"] = resample_weights
            steering_vars["potentials"] = potentials

        if self.model.steering_args["physical_guidance_update"]:
            scaled_guidance_update = torch.zeros(
                (diffusion_samples * num_particles, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )
            steering_vars["scaled_guidance_update"] = scaled_guidance_update

        atom_mask = atom_mask.repeat_interleave(diffusion_samples * num_particles, 0)

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

        if isinstance(structure, Structure):
            init_coords = (
                torch.tensor(structure.coor, device=self.device)
                .float()
                .unsqueeze(0)
                .repeat(diffusion_samples * num_particles, 1, 1)
            )
        elif isinstance(structure, torch.Tensor):
            init_coords = structure.reshape(-1, structure.shape[-2], 3)

        init_coords = pad_dim(init_coords, 1, shape[1] - init_coords.shape[1])

        if invert:
            inverse_selector = torch.ones(
                init_coords.shape[1], device=self.device
            ).bool()
            inverse_selector[selection] = False
            selection = inverse_selector

        # set the initial coordinates for the selected substructure
        atom_coords[:, selection, :] = init_coords[:, selection, :]

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
            "steering_vars": steering_vars,
        }

    def step(  # FIXME: does not include steering here
        self,
        atom_coords: torch.Tensor,
        return_denoised: bool = False,
        augmentation: bool = True,
        align_to_input: bool = True,
        alignment_reverse_diffusion: bool = True,
        alignment_weights: Optional[torch.Tensor] = None,
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
        alignment_reverse_diffusion : bool, optional
            Whether to align the noised coordinates to the denoised coordinates, by default True.
        alignment_weights : Optional[torch.Tensor], optional
            Weights for alignment of shape (batch, num_atoms). If None, uses the identity matrix. By default None.

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
            if self.model_version == "boltz1":
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
            else:  # boltz2
                atom_coords_denoised = (
                    self.model.structure_module.preconditioned_network_forward(
                        atom_coords_noisy,
                        t_hat,
                        network_condition_kwargs=dict(
                            multiplicity=multiplicity,
                            s_inputs=s_inputs,
                            s_trunk=s,
                            feats=feats,
                            diffusion_conditioning=self.cached_representations[
                                "diffusion_conditioning"
                            ],
                        ),
                    )
                )

        # Alignment reverse diffusion
        if alignment_reverse_diffusion:
            alignment_weights_reverse = (
                alignment_weights.float()
                if alignment_weights is not None
                else atom_mask.float()
            )
            atom_coords_noisy = weighted_rigid_align(
                atom_coords_noisy.float(),
                atom_coords_denoised.float(),
                alignment_weights_reverse,
                atom_mask.float(),
            ).to(atom_coords_denoised)

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
            alignment_weights_input = (
                alignment_weights.float()
                if alignment_weights is not None
                else atom_mask.float()
            )
            atom_coords_next = weighted_rigid_align(
                atom_coords_next.float(),
                self.cached_diffusion_init["init_coords"].float(),
                alignment_weights_input,
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
