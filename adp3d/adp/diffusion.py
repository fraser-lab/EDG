from typing import Dict, Optional, Tuple, Union
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
from boltz.main import check_inputs, process_inputs, BoltzProcessedInput
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest


@dataclass
class PredictArgs:
    """Arguments for model prediction."""

    recycling_steps: int = 3  # default in Boltz1
    sampling_steps: int = 1
    diffusion_samples: int = 1 # number of samples you want to generate, will be used as multiplicity
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
        device : Optional[torch.device], optional
            Device to load the model to, by default None.

        Returns
        -------
        Dict[str, torch.nn.Module]
            Dictionary containing the loaded model components.
        """
        self.device = device or try_gpu()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.cache_path = Path(
            checkpoint_path
        ).parent  # NOTE: assumes checkpoint and ccd dictionary get downloaded to the same place

        diffusion_params = BoltzDiffusionParams()
        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            self.model = (
                Boltz1.load_from_checkpoint(
                    checkpoint_path,
                    strict=True,
                    predict_args=asdict(predict_args),
                    map_location="cpu",
                    diffusion_process_args=asdict(diffusion_params),
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
        )  # NOTE: assumes batch size of 1

    def compute_representations(
        self,
        feats: Dict[str, torch.Tensor],
        recycling_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute and cache main trunk representations.

        Parameters
        ----------
        feats : Dict[str, torch.Tensor]
            Input feats containing model features
        recycling_steps : Optional[int], optional
            Override default number of recycling steps, by default None

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing cached model representations
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

    def initialize_diffusion(
        self,
        num_samples: Optional[int] = None,
        sampling_steps: Optional[int] = None,
    ) -> torch.Tensor:

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
        atom_coords_denoised = None

        token_repr = None
        token_a = None

        self.cached_diffusion_init = {
            "atom_coords": atom_coords,
            "atom_mask": atom_mask,
            "token_repr": token_repr,
            "token_a": token_a,
            "atom_coords_denoised": atom_coords_denoised,
            "sigmas_and_gammas": sigmas_and_gammas,
            "diffusion_samples": diffusion_samples,
            "num_sampling_steps": num_sampling_steps,
        }

    def step(
        self,
        atom_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Execute a single diffusion denoising step.

        Parameters
        ----------
        atom_coords : torch.Tensor
            Current atomic coordinates of shape (batch, num_atoms, 3)

        Returns
        -------
        torch.Tensor
            Denoised atomic coordinates after a single step in the trajectory.
        """
        # Get cached representations
        s = self.cached_representations["s"]
        z = self.cached_representations["z"]
        s_inputs = self.cached_representations["s_inputs"]
        relative_position_encoding = self.cached_representations[
            "relative_position_encoding"
        ]
        feats = self.cached_representations["feats"]
        multiplicity = self.cached_diffusion_init["diffusion_samples"] # batch is regulated by dataloader, this lets you do 

        # Get cached diffusion info
        atom_mask: torch.Tensor = self.cached_diffusion_init["atom_mask"]
        token_repr: torch.Tensor = self.cached_diffusion_init["token_repr"]
        token_a: torch.Tensor = self.cached_diffusion_init["token_a"]
        atom_coords_denoised: torch.Tensor = self.cached_diffusion_init[
            "atom_coords_denoised"
        ]
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
        atom_coords_noisy = atom_coords + eps

        atom_coords, atom_coords_denoised = center_random_augmentation(
            atom_coords,
            atom_mask,
            augmentation=True,
            return_second_coords=True,
            second_coords=atom_coords_denoised,
        )

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

        # Store in trajectory
        self.diffusion_trajectory[f"step_{self.current_step}"] = {
            "coords": atom_coords_next.clone(),
            "denoised": atom_coords_denoised.clone(),
        }

        self.current_step += 1
        return atom_coords_denoised
