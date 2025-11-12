# type: ignore

from typing import Optional, Tuple, Union
import torch
from einops import einsum
import torch.nn.functional as F

from math import sqrt

from boltz.model.modules.utils import (
    compute_random_augmentation,
)
from edg.edg.modules.schedules import ParameterSchedule
import numpy as np
from numpy.typing import NDArray

from edg.edg.modules.diffusion import DiffusionStepper
from edg.edg.modules.adaptive_solver import (
    create_adaptive_solver,
    AdaptiveSolverConfig,
)


def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights,
    mask,
):
    """Compute weighted alignment in a differentiable manner.
    This is the same as the weighted_rigid_align function in
    boltz/src/boltz/model/loss/diffusion.py, but with the final
    detach_() removed.

    Parameters
    ----------
    true_coords: torch.Tensor
        The ground truth atom coordinates
    pred_coords: torch.Tensor
        The predicted atom coordinates
    weights: torch.Tensor
        The weights for alignment
    mask: torch.Tensor
        The atoms mask

    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """

    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )

    return aligned_coords


class DensityGuidedDiffusionStepper(DiffusionStepper):
    """Controls fine-grained diffusion steps using pretrained Boltz models with density guidance via the diffusion update"""

    def __init__(self, *args, **kwargs):
        """Initialize the density guided diffusion stepper with adaptive solver support."""
        super().__init__(*args, **kwargs)
        self.adaptive_solver = None
        self.adaptive_solver_config = None

    def setup_adaptive_solver(
        self,
        solver_type: str = "adam",
        config: Optional[AdaptiveSolverConfig] = None,
        enable: bool = True,
    ):
        """Setup adaptive gradient solver.

        Parameters
        ----------
        solver_type : str
            Type of adaptive solver ("adam", "simple", or "none" for disabled)
        config : Optional[AdaptiveSolverConfig]
            Solver configuration
        enable : bool
            Whether to enable adaptive solver
        """
        if not enable or solver_type.lower() == "none":
            self.adaptive_solver = None
            return

        if config is None:
            # Create default config
            config = AdaptiveSolverConfig(
                learning_rate=0.01,
                max_iterations=10,
                convergence_threshold=1e-4,
                gradient_clip_norm=1.0,
                per_potential_scaling=True,
                line_search=False,
            )

        self.adaptive_solver_config = config
        self.adaptive_solver = create_adaptive_solver(solver_type, config)

    def _adaptive_guidance_update(
        self,
        atom_coords_denoised: torch.Tensor,
        potentials: list,
        feats: dict,
        steering_t: float,
        denoising_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        """Compute guidance update using adaptive solver."""

        def compute_energy(coords):
            """Compute total energy for current coordinates."""
            total_energy = 0.0
            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                if parameters["guidance_weight"] > 0:
                    energy = potential.compute(coords, feats, parameters)
                    total_energy += parameters["guidance_weight"] * energy.sum()
            return total_energy

        def compute_gradient(potential, coords, feats, parameters):
            """Compute gradient for a single potential with scaling."""
            grad = potential.compute_gradient(coords, feats, parameters)

            if parameters.get("scale_guidance_to_denoising", False):
                guidance_magnitude = torch.linalg.norm(grad, dim=-1, keepdim=True)

                max_ratio_schedule = parameters.get("max_guidance_denoising_ratio", 1.0)
                if isinstance(max_ratio_schedule, ParameterSchedule):
                    max_ratio_val = max_ratio_schedule.compute(steering_t)
                    if isinstance(max_ratio_val, torch.Tensor):
                        max_ratio_val = max_ratio_val.item()
                else:
                    max_ratio_val = float(max_ratio_schedule)

                guidance_ratio = guidance_magnitude / (denoising_magnitude + 1e-8)
                scale_factor = torch.minimum(
                    torch.ones_like(guidance_ratio),
                    max_ratio_val / (guidance_ratio + 1e-8),
                )
                grad = grad * scale_factor

            return grad

        updated_coords, stats = self.adaptive_solver.step(
            atom_coords_denoised,
            potentials,
            feats,
            steering_t,
            compute_energy,
            compute_gradient,
        )

        # Optional: store or log solver statistics for debugging
        # print(f"Adaptive solver stats: {stats}")

        guidance_update = updated_coords - atom_coords_denoised
        return guidance_update

    def _fixed_guidance_update(
        self,
        atom_coords_denoised: torch.Tensor,
        potentials: list,
        feats: dict,
        steering_t: float,
        denoising_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        """Compute guidance update using original fixed-step approach."""

        guidance_update = torch.zeros_like(atom_coords_denoised)
        for guidance_step in range(self.model.steering_args["num_gd_steps"]):
            energy_gradient = torch.zeros_like(atom_coords_denoised)
            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                if (
                    parameters["guidance_weight"] > 0
                    and (guidance_step) % parameters["guidance_interval"] == 0
                ):
                    grad = potential.compute_gradient(
                        atom_coords_denoised + guidance_update,
                        feats,
                        parameters,
                    )

                    if parameters.get("scale_guidance_to_denoising", False):
                        guidance_magnitude = torch.linalg.norm(
                            grad, dim=-1, keepdim=True
                        )

                        max_ratio_schedule = parameters.get(
                            "max_guidance_denoising_ratio", 1.0
                        )
                        if isinstance(max_ratio_schedule, ParameterSchedule):
                            max_ratio_val = max_ratio_schedule.compute(steering_t)
                            if isinstance(max_ratio_val, torch.Tensor):
                                max_ratio_val = max_ratio_val.item()
                        else:
                            max_ratio_val = float(max_ratio_schedule)

                        # Apply clipping
                        guidance_ratio = guidance_magnitude / (
                            denoising_magnitude + 1e-8
                        )
                        scale_factor = torch.minimum(
                            torch.ones_like(guidance_ratio),
                            max_ratio_val / (guidance_ratio + 1e-8),
                        )
                        grad = grad * scale_factor

                    energy_gradient += parameters["guidance_weight"] * grad

            guidance_update -= energy_gradient

        return guidance_update

    def _adaptive_guidance_update_ensemble(
        self,
        atom_coords_denoised: torch.Tensor,
        potentials: list,
        feats: dict,
        steering_t: float,
        denoising_magnitude: torch.Tensor,
        num_ensembles: int,
        ensemble_size: int,
        n_atoms: int,
    ) -> torch.Tensor:
        """Compute guidance update for ensemble using adaptive solver."""

        def compute_energy(coords):
            """Compute total energy for current coordinates."""
            total_energy = 0.0
            coords_ensemble = coords.reshape(num_ensembles, ensemble_size, n_atoms, 3)

            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                if parameters["guidance_weight"] > 0:
                    if hasattr(potential, "compute_ensemble"):
                        energy = potential.compute_ensemble(
                            coords_ensemble, feats, parameters
                        )
                    else:
                        energy = potential.compute(coords, feats, parameters)
                        energy = energy.reshape(num_ensembles, ensemble_size).sum(dim=1)
                    total_energy += parameters["guidance_weight"] * energy.sum()
            return total_energy

        def compute_gradient(potential, coords, feats, parameters):
            """Compute gradient for a single potential with scaling."""
            if hasattr(potential, "compute_gradient_ensemble"):
                grad = potential.compute_gradient_ensemble(
                    coords.reshape(num_ensembles, ensemble_size, n_atoms, 3),
                    feats,
                    parameters,
                ).reshape(-1, n_atoms, 3)
            else:
                grad = potential.compute_gradient(coords, feats, parameters)

            if parameters.get("scale_guidance_to_denoising", False):
                guidance_magnitude = torch.linalg.norm(grad, dim=-1, keepdim=True)

                max_ratio_schedule = parameters.get("max_guidance_denoising_ratio", 1.0)
                if isinstance(max_ratio_schedule, ParameterSchedule):
                    max_ratio_val = max_ratio_schedule.compute(steering_t)
                    if isinstance(max_ratio_val, torch.Tensor):
                        max_ratio_val = max_ratio_val.item()
                else:
                    max_ratio_val = float(max_ratio_schedule)

                guidance_ratio = guidance_magnitude / (denoising_magnitude + 1e-8)
                scale_factor = torch.minimum(
                    torch.ones_like(guidance_ratio),
                    max_ratio_val / (guidance_ratio + 1e-8),
                )
                grad = grad * scale_factor

            return grad

        updated_coords, stats = self.adaptive_solver.step(
            atom_coords_denoised,
            potentials,
            feats,
            steering_t,
            compute_energy,
            compute_gradient,
        )

        # Optional: store or log solver statistics for debugging
        # print(f"Adaptive solver stats: {stats}")

        guidance_update = updated_coords - atom_coords_denoised
        return guidance_update

    def _fixed_guidance_update_ensemble(
        self,
        atom_coords_denoised: torch.Tensor,
        potentials: list,
        feats: dict,
        steering_t: float,
        denoising_magnitude: torch.Tensor,
        num_ensembles: int,
        ensemble_size: int,
        n_atoms: int,
    ) -> torch.Tensor:
        """Compute guidance update for ensemble using original fixed-step approach."""

        guidance_update = torch.zeros_like(atom_coords_denoised)

        for guidance_step in range(self.model.steering_args["num_gd_steps"]):
            energy_gradient = torch.zeros_like(atom_coords_denoised)

            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                if (
                    parameters["guidance_weight"] > 0
                    and guidance_step % parameters["guidance_interval"] == 0
                ):
                    if hasattr(potential, "compute_gradient_ensemble"):
                        grad = potential.compute_gradient_ensemble(
                            (atom_coords_denoised + guidance_update).reshape(
                                num_ensembles, ensemble_size, n_atoms, 3
                            ),
                            feats,
                            parameters,
                        ).reshape(-1, n_atoms, 3)
                    else:
                        grad = potential.compute_gradient(
                            atom_coords_denoised + guidance_update,
                            feats,
                            parameters,
                        )

                    # Scale guidance update relative to denoising magnitude
                    if parameters.get("scale_guidance_to_denoising", False):
                        guidance_magnitude = torch.linalg.norm(
                            grad, dim=-1, keepdim=True
                        )

                        # Compute max allowed ratio using schedule
                        max_ratio_schedule = parameters.get(
                            "max_guidance_denoising_ratio", 1.0
                        )
                        if isinstance(max_ratio_schedule, ParameterSchedule):
                            max_ratio_val = max_ratio_schedule.compute(steering_t)
                            if isinstance(max_ratio_val, torch.Tensor):
                                max_ratio_val = max_ratio_val.item()
                        else:
                            max_ratio_val = float(max_ratio_schedule)

                        # Apply clipping
                        guidance_ratio = guidance_magnitude / (
                            denoising_magnitude + 1e-8
                        )
                        scale_factor = torch.minimum(
                            torch.ones_like(guidance_ratio),
                            max_ratio_val / (guidance_ratio + 1e-8),
                        )
                        grad = grad * scale_factor

                    energy_gradient += parameters["guidance_weight"] * grad

            guidance_update -= energy_gradient

        return guidance_update

    def step_steering(
        self,
        atom_coords: torch.Tensor,
        return_denoised: bool = False,
        augmentation: bool = False,
        align_to_input: bool = True,
        alignment_reverse_diffusion: bool = True,
        selection: Optional[NDArray[np.bool_]] = None,
        alignment_weights: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Execute a single diffusion denoising step with density guidance.

        Parameters
        ----------
        atom_coords : torch.Tensor
            Current atomic coordinates of shape (batch, num_atoms, 3).
        return_denoised : bool, optional
            Whether to return the fully denoised coordinate prediction alongside the next step coordinates, by default False.
        augmentation : bool, optional
            Whether to apply random centering augmentation, by default True.
        align_to_input : bool, optional
            Whether to align the denoised coordinates to the initial input coordinates (if provided during initialization), by default True.
        alignment_reverse_diffusion : bool, optional
            Whether to align the noised coordinates to the denoised coordinates, by default False.
            (This is the Kabsch alignment used in the Boltz-1 paper, they say not critical with the full model)
        selection : Optional[NDArray[int]], optional
            Indices of atoms to apply diffusion to. If None, applies to all atoms. By default None.
        alignment_weights : Optional[torch.Tensor], optional
            Weights for alignment of shape (batch, num_atoms). If None, uses the identity matrix. By default None.

        Returns
        -------
        Tuple[torch.Tensor, float] or Tuple[torch.Tensor, torch.Tensor, float]
            Coordinates after a single guided diffusion step and the guidance loss.
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
        num_sampling_steps = self.cached_diffusion_init["num_sampling_steps"]
        atom_mask: torch.Tensor = self.cached_diffusion_init["atom_mask"]
        sigma_tm, sigma_t, gamma = self.cached_diffusion_init["sigmas_and_gammas"][
            self.current_step
        ]
        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

        # get steering info
        if self.model.steering_args["fk_steering"]:
            energy_traj: torch.Tensor = self.cached_diffusion_init["steering_vars"][
                "energy_traj"
            ]
            potentials: list = self.cached_diffusion_init["steering_vars"]["potentials"]
            scaled_guidance_update: torch.Tensor = self.cached_diffusion_init[
                "steering_vars"
            ]["scaled_guidance_update"]
            resample_weights: torch.Tensor = self.cached_diffusion_init[
                "steering_vars"
            ]["resample_weights"]
            score = None

        # Conditionally construct network_condition_kwargs based on model version
        if self.model_version == "boltz1":
            network_condition_kwargs = dict(
                s_trunk=s,
                z_trunk=z,
                s_inputs=s_inputs,
                feats=feats,
                relative_position_encoding=relative_position_encoding,
                multiplicity=multiplicity,
            )
        else:  # boltz2
            network_condition_kwargs = dict(
                multiplicity=multiplicity,
                s_inputs=s_inputs,
                s_trunk=s,
                feats=feats,
                diffusion_conditioning=self.cached_representations[
                    "diffusion_conditioning"
                ],
            )

        steering_t = 1.0 - (self.current_step / num_sampling_steps)
        t_hat = sigma_tm * (1 + gamma)
        noise_var = self.model.structure_module.noise_scale**2 * (
            t_hat**2 - sigma_tm**2
        )
        eps = sqrt(noise_var) * torch.randn(atom_coords.shape, device=self.device)

        selection = torch.from_numpy(selection).to(self.device)
        inverse_selector = torch.ones(atom_coords.shape[1], device=self.device).bool()
        inverse_selector[selection] = False

        # NOTE: This might create some interesting pathologies if off
        if augmentation:
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )

            if (
                self.model.steering_args["physical_guidance_update"]
                and scaled_guidance_update is not None
            ):
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

        atom_coords_noisy = atom_coords + eps

        if self.model_version == "boltz1":
            atom_coords_denoised, _ = (
                self.model.structure_module.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=network_condition_kwargs,
                )
            )
        else:  # boltz2
            atom_coords_denoised = (
                self.model.structure_module.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    network_condition_kwargs=network_condition_kwargs,
                )
            )

        if align_to_input:
            alignment_weights = (
                alignment_weights.float()
                if alignment_weights is not None
                else inverse_selector.float()
            )

            atom_coords_denoised = weighted_rigid_align(
                atom_coords_denoised.float(),
                self.cached_diffusion_init["init_coords"].float(),
                alignment_weights,
                atom_mask.float(),
            )

        # Clamp to the motif
        atom_coords_denoised[:, inverse_selector, :] = self.cached_diffusion_init[
            "init_coords"
        ][:, inverse_selector, :]

        if self.model.steering_args["fk_steering"] and (
            (
                self.current_step % self.model.steering_args["fk_resampling_interval"]
                == 0
                and noise_var > 0
            )
            or self.current_step == num_sampling_steps - 1
        ):
            # Compute energy of x_0 prediction
            energy = torch.zeros(multiplicity, device=self.device)
            score = torch.zeros(multiplicity, device=self.device)
            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                if parameters["resampling_weight"] > 0:
                    component_energy = potential.compute(
                        atom_coords_denoised,
                        network_condition_kwargs["feats"],
                        parameters,
                    )
                    score += component_energy
                    energy += parameters["resampling_weight"] * component_energy
            energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

            # Compute log G values
            if self.current_step == 0:
                log_G = -1 * energy
            else:
                log_G = energy_traj[:, -2] - energy_traj[:, -1]

            # Compute ll difference between guided and unguided transition distribution
            if self.model.steering_args["physical_guidance_update"] and noise_var > 0:
                ll_difference = (eps**2 - (eps + scaled_guidance_update) ** 2).sum(
                    dim=(-1, -2)
                ) / (2 * noise_var)
            else:
                ll_difference = torch.zeros_like(energy)

            # Compute resampling weights
            resample_weights = F.softmax(
                (ll_difference + self.model.steering_args["fk_lambda"] * log_G).reshape(
                    -1, self.model.steering_args["num_particles"]
                ),
                dim=1,
            )

        # Compute guidance update to x_0 prediction
        if (
            self.model.steering_args["physical_guidance_update"]
            # and self.current_step < num_sampling_steps - 1
        ):
            # Compute original denoising magnitude before guidance
            original_denoised_over_sigma = (
                atom_coords_noisy - atom_coords_denoised
            ) / t_hat
            denoising_magnitude = torch.linalg.norm(
                original_denoised_over_sigma, dim=(-1, -2), keepdim=True
            )

            if self.adaptive_solver is not None:
                guidance_update = self._adaptive_guidance_update(
                    atom_coords_denoised,
                    potentials,
                    network_condition_kwargs["feats"],
                    steering_t,
                    denoising_magnitude,
                )
            else:
                guidance_update = self._fixed_guidance_update(
                    atom_coords_denoised,
                    potentials,
                    network_condition_kwargs["feats"],
                    steering_t,
                    denoising_magnitude,
                )

            atom_coords_denoised += guidance_update
            scaled_guidance_update = (
                guidance_update
                * -1
                * self.model.structure_module.step_scale
                * (sigma_t - t_hat)
                / t_hat
            )

        if (
            self.model.steering_args["fk_steering"]
            and (
                self.current_step % self.model.steering_args["fk_resampling_interval"]
                == 0
                and noise_var > 0
            )
            and self.current_step != num_sampling_steps - 1
            # or self.current_step == num_sampling_steps - 1 # Changed from Boltz, since I want ensemble at the end.
        ):
            resample_indices = (
                torch.multinomial(
                    resample_weights,
                    resample_weights.shape[
                        1
                    ],  # Changed from Boltz, since I want ensemble at the end.
                    replacement=True,
                )
                + resample_weights.shape[1]
                * torch.arange(
                    resample_weights.shape[0], device=resample_weights.device
                ).unsqueeze(-1)
            ).flatten()

            atom_coords = atom_coords[resample_indices]
            atom_coords_noisy = atom_coords_noisy[resample_indices]
            atom_mask = atom_mask[resample_indices]
            if atom_coords_denoised is not None:
                atom_coords_denoised = atom_coords_denoised[resample_indices]
            energy_traj = energy_traj[resample_indices]
            if self.model.steering_args["physical_guidance_update"]:
                scaled_guidance_update = scaled_guidance_update[resample_indices]

        # cache FK steering variables
        steering_vars = {
            "energy_traj": energy_traj,
            "resample_weights": resample_weights,
            "scaled_guidance_update": scaled_guidance_update,
        }
        self.cached_diffusion_init["steering_vars"].update(steering_vars)

        if alignment_reverse_diffusion:
            # align noisy coords to match the denoised coords,
            # this is what Boltz-1 paper talks about for proper interpolation
            # NOTE: I think this is pretty necessary here, since we align the denoised coords to the input
            atom_coords_noisy = weighted_rigid_align(
                atom_coords_noisy.float(),
                atom_coords_denoised.float(),
                alignment_weights,
                atom_mask.float(),
            )

        atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat

        # E[Xt-1 | Xt] = ut-1 (posterior mean of Xt-1 given Xt)
        atom_coords_next: torch.Tensor = (
            atom_coords_noisy
            + self.model.structure_module.step_scale
            * (sigma_t - t_hat)
            * denoised_over_sigma
        )

        # if self.current_step == num_sampling_steps - 1:
        #     # Take top K ensemble members
        #     resample_indices = torch.topk(
        #         energy_traj[:, -1], ensemble_size, largest=False
        #     ).indices.flatten()
        #     atom_coords = atom_coords[resample_indices]
        #     atom_coords_noisy = atom_coords_noisy[resample_indices]
        #     atom_mask = atom_mask[resample_indices]
        #     if atom_coords_denoised is not None:
        #         atom_coords_denoised = atom_coords_denoised[resample_indices]
        #     energy_traj = energy_traj[resample_indices]
        #     if self.model.steering_args["physical_guidance_update"]:
        #         scaled_guidance_update = scaled_guidance_update[resample_indices]

        unpad_coords_next = atom_coords_next[
            :, pad_mask, :
        ]  # unpad the coords to B, N_unpad, 3
        unpad_coords_denoised = atom_coords_denoised[
            :, pad_mask, :
        ]  # unpad the coords to B, N_unpad, 3

        # Store unpadded in trajectory (0 indexed)
        self.diffusion_trajectory[f"step_{self.current_step}"] = {
            "coords": unpad_coords_next.detach().clone(),
            "denoised": unpad_coords_denoised.detach().clone(),  # the overall prediction from this current level (no noise mixture)
        }

        self.current_step += 1  # NOTE: current step to execute

        if score is None:
            score = torch.zeros_like(energy_traj[:, -1])
            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                component_energy = potential.compute(
                    atom_coords_denoised,
                    network_condition_kwargs["feats"],
                    parameters,
                )
                score += component_energy

        if return_denoised:
            return (
                atom_coords_next.detach(),
                atom_coords_denoised.detach(),
                score.mean().item(),
            )
        else:
            return (
                atom_coords_next.detach(),
                score.mean().item(),
            )  # return minimum energy of ensemble

    def step_steering_ensemble(
        self,
        atom_coords: torch.Tensor,
        ensemble_size: int = 1,
        return_denoised: bool = False,
        augmentation: bool = False,
        align_to_input: bool = True,
        alignment_reverse_diffusion: bool = True,
        selection: Optional[NDArray[np.bool_]] = None,
        alignment_weights: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Execute a single diffusion denoising step with ensemble particle steering. Assumes FK steering is enabled.

        Parameters
        ----------
        atom_coords : torch.Tensor
            Current atomic coordinates (batch, num_atoms, 3) or (num_ensembles, ensemble_size, num_atoms, 3)
        ensemble_size : int, optional
            Number of ensemble members to sample from the diffusion process
        return_denoised : bool, optional
            Whether to return the fully denoised coordinate prediction
        augmentation : bool, optional
            Whether to apply random centering augmentation
        align_to_input : bool, optional
            Whether to align the denoised coordinates to the initial input
        alignment_reverse_diffusion : bool, optional
            Whether to align the noised coordinates to the denoised coordinates
        selection : Optional[NDArray[int]], optional
            Indices of atoms to apply diffusion to
        alignment_weights : Optional[torch.Tensor], optional
            Weights for alignment of shape [num_ensembles * ensemble_size, n_atoms]

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]
            Coordinates after steering step and the mean score
        """
        if atom_coords.ndim == 3:
            multiplicity, n_atoms, _ = atom_coords.shape
            num_ensembles = multiplicity // ensemble_size
        elif atom_coords.ndim == 4:
            num_ensembles, ensemble_size, n_atoms, _ = atom_coords.shape
            multiplicity = num_ensembles * ensemble_size
            atom_coords = atom_coords.reshape(multiplicity, n_atoms, 3)
        else:
            raise ValueError(
                f"atom_coords must be 3D or 4D, but got {atom_coords.ndim}D"
            )

        if multiplicity % ensemble_size != 0:
            raise ValueError(
                f"Total size {multiplicity} must be divisible by ensemble size {ensemble_size}"
            )

        s = self.cached_representations["s"]
        z = self.cached_representations["z"]
        s_inputs = self.cached_representations["s_inputs"]
        relative_position_encoding = self.cached_representations[
            "relative_position_encoding"
        ]
        feats = self.cached_representations["feats"]
        pad_mask = feats["atom_pad_mask"].squeeze().bool()

        num_sampling_steps = self.cached_diffusion_init["num_sampling_steps"]
        atom_mask: torch.Tensor = self.cached_diffusion_init["atom_mask"]
        sigma_tm, sigma_t, gamma = self.cached_diffusion_init["sigmas_and_gammas"][
            self.current_step
        ]
        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

        energy_traj: torch.Tensor = self.cached_diffusion_init["steering_vars"][
            "energy_traj"
        ]
        potentials: list = self.cached_diffusion_init["steering_vars"]["potentials"]
        scaled_guidance_update: torch.Tensor = self.cached_diffusion_init[
            "steering_vars"
        ]["scaled_guidance_update"]
        resample_weights: torch.Tensor = self.cached_diffusion_init["steering_vars"][
            "resample_weights"
        ]
        score = None

        # Conditionally construct network_condition_kwargs based on model version
        if self.model_version == "boltz1":
            network_condition_kwargs = dict(
                s_trunk=s,
                z_trunk=z,
                s_inputs=s_inputs,
                feats=feats,
                relative_position_encoding=relative_position_encoding,
                multiplicity=multiplicity,
            )
        else:  # boltz2
            network_condition_kwargs = dict(
                multiplicity=multiplicity,
                s_inputs=s_inputs,
                s_trunk=s,
                feats=feats,
                diffusion_conditioning=self.cached_representations[
                    "diffusion_conditioning"
                ],
            )

        steering_t = 1.0 - (self.current_step / num_sampling_steps)
        t_hat = sigma_tm * (1 + gamma)
        noise_var = self.model.structure_module.noise_scale**2 * (
            t_hat**2 - sigma_tm**2
        )

        eps = sqrt(noise_var) * torch.randn_like(atom_coords)

        selection = torch.from_numpy(selection).to(self.device)
        inverse_selector = torch.ones(n_atoms, device=self.device).bool()
        inverse_selector[selection] = False

        if augmentation:
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )

            if scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

        atom_coords_noisy = atom_coords + eps

        if self.model_version == "boltz1":
            atom_coords_denoised, _ = (
                self.model.structure_module.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=network_condition_kwargs,
                )
            )
        else:  # boltz2
            atom_coords_denoised = (
                self.model.structure_module.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    network_condition_kwargs=network_condition_kwargs,
                )
            )

        if align_to_input:
            if alignment_weights is not None:
                alignment_weights = alignment_weights.float()
            else:
                alignment_weights = (
                    inverse_selector.float().unsqueeze(0).expand(multiplicity, -1)
                )

            atom_coords_denoised = weighted_rigid_align(
                atom_coords_denoised.float(),
                self.cached_diffusion_init["init_coords"].float(),
                alignment_weights,
                atom_mask.float(),
            )

        # Clamp to motif
        # atom_coords_denoised[:, inverse_selector, :] = self.cached_diffusion_init[
        #     "init_coords"
        # ][:, inverse_selector, :]

        atom_coords_denoised_ensemble = atom_coords_denoised.reshape(
            num_ensembles, ensemble_size, n_atoms, 3
        )

        if (
            self.current_step % self.model.steering_args["fk_resampling_interval"] == 0
            and noise_var > 0
        ) or self.current_step == num_sampling_steps - 1:
            energy = torch.zeros(num_ensembles, device=self.device)
            score = torch.zeros(num_ensembles, device=self.device)
            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                if parameters["resampling_weight"] > 0:
                    if hasattr(potential, "compute_ensemble"):
                        component_energy = potential.compute_ensemble(
                            atom_coords_denoised_ensemble,  # [num_ensembles, ensemble_size, n_atoms, 3]
                            network_condition_kwargs["feats"],
                            parameters,
                        )
                    else:
                        component_energy = (
                            potential.compute(
                                atom_coords_denoised,  # [multiplicity, n_atoms, 3]
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            .reshape(num_ensembles, ensemble_size)
                            .sum(dim=1)
                        )  # Sum over ensemble members
                    energy += parameters["resampling_weight"] * component_energy
                    score += component_energy  # Total ensemble energy for reporting

            energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

            if self.current_step == 0:
                log_G = -1 * energy
            else:
                log_G = energy_traj[:, -2] - energy

            if scaled_guidance_update is not None and noise_var > 0:
                eps_ensemble = eps.reshape(num_ensembles, ensemble_size, n_atoms, 3)
                scaled_guidance_ensemble = scaled_guidance_update.reshape(
                    num_ensembles, ensemble_size, n_atoms, 3
                )
                ll_difference = (
                    eps_ensemble**2 - (eps_ensemble + scaled_guidance_ensemble) ** 2
                ).sum(dim=(1, 2, 3)) / (2 * noise_var)
            else:
                ll_difference = torch.zeros_like(energy)

            resample_logits = (
                ll_difference + self.model.steering_args["fk_lambda"] * log_G
            )
            resample_weights = F.softmax(resample_logits, dim=0)

        if score is None:
            score = torch.zeros(num_ensembles, device=self.device)
            for potential in potentials:
                parameters = potential.compute_parameters(steering_t)
                if parameters["resampling_weight"] > 0:
                    if hasattr(potential, "compute_ensemble"):
                        component_energy = potential.compute_ensemble(
                            atom_coords_denoised_ensemble,  # [num_ensembles, ensemble_size, n_atoms, 3]
                            network_condition_kwargs["feats"],
                            parameters,
                        )
                    else:
                        component_energy = (
                            potential.compute(
                                atom_coords_denoised.reshape(multiplicity, n_atoms, 3),
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            .reshape(num_ensembles, ensemble_size)
                            .sum(dim=1)
                        )  # Sum over ensemble members
                    score += component_energy  # Total ensemble energy for reporting

        if (
            self.model.steering_args["physical_guidance_update"]
            and self.current_step < num_sampling_steps - 1
        ):
            # Compute original denoising magnitude before guidance
            original_denoised_over_sigma = (
                atom_coords_noisy - atom_coords_denoised
            ) / t_hat
            denoising_magnitude = torch.linalg.norm(
                original_denoised_over_sigma, dim=-1, keepdim=True
            )

            if self.adaptive_solver is not None:
                guidance_update = self._adaptive_guidance_update_ensemble(
                    atom_coords_denoised,
                    potentials,
                    feats,
                    steering_t,
                    denoising_magnitude,
                    num_ensembles,
                    ensemble_size,
                    n_atoms,
                )
            else:
                guidance_update = self._fixed_guidance_update_ensemble(
                    atom_coords_denoised,
                    potentials,
                    feats,
                    steering_t,
                    denoising_magnitude,
                    num_ensembles,
                    ensemble_size,
                    n_atoms,
                )

            atom_coords_denoised += guidance_update
            scaled_guidance_update = (
                guidance_update
                * -1
                * self.model.structure_module.step_scale
                * (sigma_t - t_hat)
                / t_hat
            )

        if (
            self.current_step % self.model.steering_args["fk_resampling_interval"] == 0
            and noise_var > 0
        ) and self.current_step != num_sampling_steps - 1:
            resample_indices = torch.multinomial(
                resample_weights, num_ensembles, replacement=True
            )

            atom_coords = atom_coords.reshape(num_ensembles, ensemble_size, n_atoms, 3)[
                resample_indices
            ].reshape(-1, n_atoms, 3)
            atom_coords_noisy = atom_coords_noisy.reshape(
                num_ensembles, ensemble_size, n_atoms, 3
            )[resample_indices].reshape(-1, n_atoms, 3)
            atom_mask = atom_mask.reshape(num_ensembles, ensemble_size, n_atoms)[
                resample_indices
            ].reshape(-1, n_atoms)
            if atom_coords_denoised is not None:
                atom_coords_denoised = atom_coords_denoised.reshape(
                    num_ensembles, ensemble_size, n_atoms, 3
                )[resample_indices].reshape(-1, n_atoms, 3)
            energy_traj = energy_traj[resample_indices]
            if scaled_guidance_update is not None:
                scaled_guidance_update = scaled_guidance_update.reshape(
                    num_ensembles, ensemble_size, n_atoms, 3
                )[resample_indices].reshape(-1, n_atoms, 3)

        steering_vars = {
            "energy_traj": energy_traj,
            "resample_weights": resample_weights,
            "scaled_guidance_update": scaled_guidance_update,
        }
        self.cached_diffusion_init["steering_vars"].update(steering_vars)

        if alignment_reverse_diffusion:
            atom_coords_noisy = weighted_rigid_align(
                atom_coords_noisy.float(),
                atom_coords_denoised.float(),
                alignment_weights,
                atom_mask.float(),
            )

        atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)
        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat

        atom_coords_next = (
            atom_coords_noisy
            + self.model.structure_module.step_scale
            * (sigma_t - t_hat)
            * denoised_over_sigma
        )

        atom_coords_next = atom_coords_next.reshape(
            num_ensembles, ensemble_size, n_atoms, 3
        )
        atom_coords_denoised = atom_coords_denoised.reshape(
            num_ensembles, ensemble_size, n_atoms, 3
        )

        unpad_coords_next = atom_coords_next[:, :, pad_mask, :]
        unpad_coords_denoised = atom_coords_denoised[:, :, pad_mask, :]

        self.diffusion_trajectory[f"step_{self.current_step}"] = {
            "coords": unpad_coords_next.detach().clone(),
            "denoised": unpad_coords_denoised.detach().clone(),
        }

        self.current_step += 1

        if return_denoised:
            return (
                atom_coords_next.detach(),
                atom_coords_denoised.detach(),
                score,
            )
        else:
            return (
                atom_coords_next.detach(),
                score,
            )
