# type: ignore

from typing import Callable, Dict, Optional, Tuple, Union
import torch
from einops import einsum
import torch.nn.functional as F

from math import sqrt

from boltz.model.modules.utils import (
    center_random_augmentation,
    compute_random_augmentation,
)
from boltz.model.potentials.potentials import get_potentials
import numpy as np
from numpy.typing import NDArray

from adp3d.adp.modules.diffusion import DiffusionStepper


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
    """Controls fine-grained diffusion steps using the pretrained Boltz1 model and guidance via the diffusion update"""

    def step(
        self,
        atom_coords: torch.Tensor,
        density_loss: Callable,
        guidance_scale: float = 0.1,
        return_denoised: bool = False,
        augmentation: bool = True,
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
        density_score : Callable
            Function that takes in the current atomic coordinates and returns the loss as a Tensor
        guidance_scale : float, optional
            Scale factor for applying the density gradient guidance, by default 0.1.
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

        selection = torch.from_numpy(selection).to(self.device)
        inverse_selector = torch.ones(atom_coords.shape[1], device=self.device).bool()
        inverse_selector[selection] = False

        # NOTE: This might create some interesting pathologies if off
        if augmentation:
            atom_coords = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
            )

        atom_coords_noisy = atom_coords + eps

        # if selection is not None:
        #     selection = torch.from_numpy(selection).to(
        #         self.device
        #     )
        #     inverse_selector = torch.ones(
        #         atom_coords_noisy.shape[1], device=self.device
        #     ).bool()
        #     inverse_selector[selection] = False
        #     atom_coords_noisy[:, inverse_selector, :] = self.cached_diffusion_init[
        #         "init_coords"
        #     ][:, inverse_selector, :]

        # need to update noisy coords even though loss is on denoised coords,
        # therefore we need to track the gradient of the noisy coords
        atom_coords_noisy = atom_coords_noisy.detach().requires_grad_(True)

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

        # if align_to_input: # FIXME: testing with atom_coords_next
        #     alignment_weights = (
        #         alignment_weights.float()
        #         if alignment_weights is not None
        #         else atom_mask.float()
        #     )

        #     # selection is going to be noisier, want to align to scaffold
        #     alignment_weights[:, selection] = 0.0

        #     # align denoised coordinates to match the initial structure
        #     atom_coords_denoised_aligned = weighted_rigid_align(
        #         atom_coords_denoised.float(),
        #         self.cached_diffusion_init["init_coords"].float(),
        #         alignment_weights,
        #         atom_mask.float(),
        #     )
        # else:
        #     atom_coords_denoised_aligned = atom_coords_denoised

        # TODO: this breaks the computational graph, so there needs to be some other method to do this
        if alignment_reverse_diffusion:
            # align noisy coords to match the denoised coords,
            # this is what Boltz-1 paper talks about for proper interpolation
            atom_coords_noisy_aligned = weighted_rigid_align(
                atom_coords_noisy.float(),
                # atom_coords_denoised_aligned.float(), # FIXME
                atom_coords_denoised.float(),
                alignment_weights,
                atom_mask.float(),
            )
        else:
            atom_coords_noisy_aligned = atom_coords_noisy

        # masked_coords = atom_coords_denoised_aligned[:, pad_mask, :]
        masked_coords = atom_coords_denoised[:, pad_mask, :]

        density_score, substructure_score = density_loss(masked_coords)
        total_loss = density_score + substructure_score
        total_loss.backward()
        grad_noisy = atom_coords_noisy.grad.clone()

        atom_coords_noisy_aligned = atom_coords_noisy_aligned.to(
            # atom_coords_denoised_aligned
            atom_coords_denoised
        )

        denoised_over_sigma = (
            # atom_coords_noisy_aligned - atom_coords_denoised_aligned
            atom_coords_noisy_aligned - atom_coords_denoised
        ) / t_hat

        scaled_guidance_grad = (
            torch.linalg.norm(denoised_over_sigma)
            / torch.linalg.norm(grad_noisy)
            * grad_noisy
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

        # Align to input for next step rather than denoised coords
        if align_to_input:
            if self.cached_diffusion_init["init_coords"] is None:
                raise ValueError(
                    "No initial input coordinates found in cached diffusion init. Please change from align_to_input if you are not using partial diffusion."
                )
            # align all except the motif

            alignment_weights = (
                alignment_weights.float()
                if alignment_weights is not None
                else atom_mask.float()
            )
            alignment_weights[:, selection] = 0.0

            atom_coords_next = weighted_rigid_align(
                atom_coords_next.float(),
                self.cached_diffusion_init["init_coords"].float(),
                alignment_weights,
                atom_mask.float(),
            ).to(atom_coords_next)

        # # Clamp atom_coords_next to the motif
        # atom_coords_next[:, inverse_selector, :] = self.cached_diffusion_init[
        #     "init_coords"
        # ][:, inverse_selector, :]

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

        if return_denoised:
            return (
                atom_coords_next.detach(),
                atom_coords_denoised.detach(),
                -(density_score + substructure_score).item(),
            )
        else:
            return atom_coords_next.detach(), -(
                density_score + substructure_score
            ).item()

    def dmap_step(
        self,
        atom_coords: torch.Tensor,
        density_loss: Callable,
        zeta: float = 0.1,
        dmap_steps: int = 3,
        return_denoised: bool = False,
        augmentation: bool = True,
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
        density_loss : Callable
            Function that takes in the current atomic coordinates and returns the loss as a Tensor
        zeta: float, optional
            Scale factor for applying the density gradient guidance in DMAP, by default 0.1.
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

        network_condition_kwargs = dict(
            s_trunk=s,
            z_trunk=z,
            s_inputs=s_inputs,
            feats=feats,
            relative_position_encoding=relative_position_encoding,
            multiplicity=multiplicity,
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
                self.model.steering_args["guidance_update"]
                and scaled_guidance_update is not None
            ):
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

        with torch.no_grad():
            atom_coords_noisy = atom_coords + eps

            atom_coords_denoised, _ = (
                self.model.structure_module.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=network_condition_kwargs,
                )
            )

            if self.model.steering_args["fk_steering"] and (
                (
                    self.current_step
                    % self.model.steering_args["fk_resampling_interval"]
                    == 0
                    and noise_var > 0
                )
                or self.current_step == num_sampling_steps - 1
            ):
                # Compute energy of x_0 prediction
                energy = torch.zeros(multiplicity, device=self.device)
                for potential in potentials:
                    parameters = potential.compute_parameters(steering_t)
                    if parameters["resampling_weight"] > 0:
                        component_energy = potential.compute(
                            atom_coords_denoised,
                            network_condition_kwargs["feats"],
                            parameters,
                        )
                        energy += parameters["resampling_weight"] * component_energy
                energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                # Compute log G values
                if self.current_step == 0:
                    log_G = -1 * energy
                else:
                    log_G = energy_traj[:, -2] - energy_traj[:, -1]

                # Compute ll difference between guided and unguided transition distribution
                if self.model.steering_args["guidance_update"] and noise_var > 0:
                    ll_difference = (eps**2 - (eps + scaled_guidance_update) ** 2).sum(
                        dim=(-1, -2)
                    ) / (2 * noise_var)
                else:
                    ll_difference = torch.zeros_like(energy)

                # Compute resampling weights
                resample_weights = F.softmax(
                    (
                        ll_difference + self.model.steering_args["fk_lambda"] * log_G
                    ).reshape(-1, self.model.steering_args["num_particles"]),
                    dim=1,
                )

            # Compute guidance update to x_0 prediction
            if (
                self.model.steering_args["guidance_update"]
                and self.current_step < num_sampling_steps - 1
            ):
                guidance_update = torch.zeros_like(atom_coords_denoised)
                for guidance_step in range(self.model.steering_args["num_gd_steps"]):
                    energy_gradient = torch.zeros_like(atom_coords_denoised)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if (
                            parameters["guidance_weight"] > 0
                            and (guidance_step) % parameters["guidance_interval"] == 0
                        ):
                            energy_gradient += parameters[
                                "guidance_weight"
                            ] * potential.compute_gradient(
                                atom_coords_denoised + guidance_update,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                    guidance_update -= energy_gradient
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
                    self.current_step
                    % self.model.steering_args["fk_resampling_interval"]
                    == 0
                    and noise_var > 0
                )
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
                if self.model.steering_args["guidance_update"]:
                    scaled_guidance_update = scaled_guidance_update[resample_indices]

            # cache FK steering variables
            steering_vars = {
                "energy_traj": energy_traj,
                "resample_weights": resample_weights,
                "scaled_guidance_update": scaled_guidance_update,
            }
            self.cached_diffusion_init["steering_vars"].update(steering_vars)

            atom_coords_noisy = atom_coords_noisy.to(
                # atom_coords_denoised_aligned
                atom_coords_denoised
            )

            denoised_over_sigma = (
                # atom_coords_noisy_aligned - atom_coords_denoised_aligned
                atom_coords_noisy - atom_coords_denoised
            ) / t_hat

            # E[Xt-1 | Xt] = ut-1 (posterior mean of Xt-1 given Xt)
            atom_coords_next: torch.Tensor = (
                atom_coords_noisy
                + self.model.structure_module.step_scale
                * (sigma_t - t_hat)
                * denoised_over_sigma
            )

            t_hat_next = sigma_t * (1 + gamma)
            eps = (
                self.model.structure_module.noise_scale
                * sqrt(t_hat_next**2 - sigma_t**2)
                * torch.randn(atom_coords_next.shape, device=self.device)
            )
            # Xt-1 = ut-1 + eps
            atom_coords_next_noisy = atom_coords_next + eps

        for _ in range(dmap_steps):
            atom_coords_next_noisy = atom_coords_next_noisy.detach().requires_grad_(
                True
            )

            # E[X0 | Xt-1]
            atom_coords_next_denoised, _ = (
                self.model.structure_module.preconditioned_network_forward(
                    atom_coords_next_noisy,
                    t_hat_next,
                    training=False,
                    network_condition_kwargs=network_condition_kwargs,
                )
            )
            masked_coords = atom_coords_next_denoised[:, pad_mask, :]

            density_score, substructure_score = density_loss(masked_coords)
            total_loss = density_score + substructure_score
            total_loss.backward()
            grad_noisy = atom_coords_next_noisy.grad.clone()

            # only apply gradient for atoms in selection
            # grad_noisy = torch.zeros_like(atom_coords_next_noisy)
            # if selection is not None:
            #     grad_noisy[:, selection, :] = density_grad[:, selection, :]
            #     grad_noisy[:, inverse_selector, :] = substructure_grad[:, inverse_selector, :]
            # else:
            #     grad_noisy = density_grad + substructure_grad
            grad_noisy_scaled = (
                torch.linalg.norm(denoised_over_sigma)
                / torch.linalg.norm(grad_noisy)
                * grad_noisy
            )

            atom_coords_next_star = (
                atom_coords_next_noisy + zeta * grad_noisy_scaled
            )  # + because score is already negated
            diff_star = atom_coords_next_star - atom_coords_next
            # diff_star = diff_star / torch.linalg.norm(diff_star) * sqrt(atom_coords_next.shape[1] * 3) * sqrt(t_hat_next**2 - sigma_t**2) * self.model.structure_module.noise_scale # FIXME: testing unscaling
            atom_coords_next_noisy = atom_coords_next + diff_star

        atom_coords_next = (
            atom_coords_next_noisy.detach()
        )  # need to set DMAP updated coords

        # Align to input for next step rather than denoised coords
        if align_to_input:
            if self.cached_diffusion_init["init_coords"] is None:
                raise ValueError(
                    "No initial input coordinates found in cached diffusion init. Please change from align_to_input if you are not using partial diffusion."
                )
            # align all except the motif

            alignment_weights = (
                alignment_weights.float()
                if alignment_weights is not None
                else atom_mask.float()
            )
            alignment_weights[:, selection] = 0.0

            atom_coords_next = weighted_rigid_align(
                atom_coords_next.float(),
                self.cached_diffusion_init["init_coords"].float(),
                alignment_weights,
                atom_mask.float(),
            ).to(atom_coords_next)

        # Clamp atom_coords_next to the motif # FIXME
        atom_coords_next[:, inverse_selector, :] = self.cached_diffusion_init[
            "init_coords"
        ][:, inverse_selector, :]

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

        if return_denoised:
            return (
                atom_coords_next.detach(),
                atom_coords_denoised.detach(),
                -(total_loss).item(),
            )
        else:
            return atom_coords_next.detach(), -(total_loss).item()
