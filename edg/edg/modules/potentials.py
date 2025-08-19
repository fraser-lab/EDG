"""Copy of Boltz-1x potentials used for FK steering. Added my own density potential here.

Date: 6 May 2025
Author: Karson Chrispens (karson.chrispens@ucsf.edu)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import torch

from boltz.data import const
from boltz.model.potentials.schedules import *
from edg.utils.interpolation import (
    trilinear_interpolation_torch,
    tricubic_interpolation_torch,
)

from .density.density import XMap_torch, DifferentiableTransformer
from .density.loss import batched_squared_error


class Potential(ABC):
    def __init__(
        self,
        parameters: Optional[
            Dict[str, Union[ParameterSchedule, float, int, bool]]
        ] = None,
    ):
        self.parameters = parameters

    def compute(self, coords, feats, parameters):
        index, args, com_args = self.compute_args(feats, parameters)

        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[..., atom_pad_mask, :]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
        value = self.compute_variable(coords, index, compute_gradient=False)
        energy = self.compute_function(value, *args)
        return energy.sum(dim=-1)

    def compute_ensemble(self, coords, feats, parameters):
        num_ensembles, ensemble_size = coords.shape[:2]
        coords_flat = coords.reshape(-1, *coords.shape[2:])

        energy_flat = self.compute(coords_flat, feats, parameters)
        return energy_flat.reshape(num_ensembles, ensemble_size).sum(dim=1)

    def compute_gradient(self, coords, feats, parameters):
        index, args, com_args = self.compute_args(feats, parameters)
        if com_args is not None:
            com_index, atom_pad_mask = com_args
        else:
            com_index, atom_pad_mask = None, None

        if index.shape[1] == 0:
            return torch.zeros_like(coords)

        if com_index is not None:
            unpad_coords = coords[..., atom_pad_mask, :]
            unpad_com_index = com_index[atom_pad_mask]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
            com_counts = torch.bincount(com_index[atom_pad_mask])

        value, grad_value = self.compute_variable(coords, index, compute_gradient=True)
        energy, dEnergy = self.compute_function(value, *args, compute_derivative=True)

        grad_atom = torch.zeros_like(coords).scatter_reduce(
            -2,
            index.flatten(start_dim=0, end_dim=1)
            .unsqueeze(-1)
            .expand((*coords.shape[:-2], -1, 3)),
            dEnergy.tile(grad_value.shape[-3]).unsqueeze(-1)
            * grad_value.flatten(start_dim=-3, end_dim=-2),
            "sum",
        )

        if com_index is not None:
            grad_atom = grad_atom[..., com_index, :]

        return grad_atom

    def compute_gradient_ensemble(self, coords, feats, parameters):
        num_ensembles, ensemble_size = coords.shape[:2]
        coords_flat = coords.reshape(-1, *coords.shape[2:])

        grad_flat = self.compute_gradient(coords_flat, feats, parameters)
        return grad_flat.reshape(num_ensembles, ensemble_size, *grad_flat.shape[1:])

    def compute_parameters(self, t):
        if self.parameters is None:
            return None
        parameters = {
            name: parameter
            if not isinstance(parameter, ParameterSchedule)
            else parameter.compute(t)
            for name, parameter in self.parameters.items()
        }
        return parameters

    @abstractmethod
    def compute_function(self, value, *args, compute_derivative=False):
        raise NotImplementedError

    @abstractmethod
    def compute_variable(self, coords, index, compute_gradient=False):
        raise NotImplementedError

    @abstractmethod
    def compute_args(self, t, feats, **parameters):
        raise NotImplementedError


class FlatBottomPotential(Potential):
    def compute_function(
        self, value, k, lower_bounds, upper_bounds, compute_derivative=False
    ):
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))

        neg_overflow_mask = value < lower_bounds
        pos_overflow_mask = value > upper_bounds

        energy = torch.zeros_like(value)
        energy[neg_overflow_mask] = (k * (lower_bounds - value))[neg_overflow_mask]
        energy[pos_overflow_mask] = (k * (value - upper_bounds))[pos_overflow_mask]
        if not compute_derivative:
            return energy

        dEnergy = torch.zeros_like(value)
        dEnergy[neg_overflow_mask] = (
            -1 * k.expand_as(neg_overflow_mask)[neg_overflow_mask]
        )
        dEnergy[pos_overflow_mask] = (
            1 * k.expand_as(pos_overflow_mask)[pos_overflow_mask]
        )

        return energy, dEnergy


class HarmonicPotential(Potential):
    def compute_function(
        self, value, k, lower_bounds, upper_bounds, compute_derivative=False
    ):
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))

        neg_overflow_mask = value < lower_bounds
        pos_overflow_mask = value > upper_bounds

        energy = torch.zeros_like(value)
        energy[neg_overflow_mask] = (k * (lower_bounds - value) ** 2)[neg_overflow_mask]
        energy[pos_overflow_mask] = (k * (value - upper_bounds) ** 2)[pos_overflow_mask]
        if not compute_derivative:
            return energy

        dEnergy = torch.zeros_like(value)
        dEnergy[neg_overflow_mask] = (
            -2
            * k.expand_as(neg_overflow_mask)[neg_overflow_mask]
            * (lower_bounds - value)[neg_overflow_mask]
        )
        dEnergy[pos_overflow_mask] = (
            2
            * k.expand_as(pos_overflow_mask)[pos_overflow_mask]
            * (value - upper_bounds)[pos_overflow_mask]
        )

        return energy, dEnergy


class DistancePotential(Potential):
    def compute_variable(self, coords, index, compute_gradient=False):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
        r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)

        if not compute_gradient:
            return r_ij_norm

        grad_i = r_hat_ij
        grad_j = -1 * r_hat_ij
        grad = torch.stack((grad_i, grad_j), dim=1)

        return r_ij_norm, grad


class DihedralPotential(Potential):
    def compute_variable(self, coords, index, compute_gradient=False):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])

        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)

        r_kj_norm = torch.linalg.norm(r_kj, dim=-1)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)

        sign_phi = torch.sign(
            r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)
        ).squeeze(-1, -2)
        phi = sign_phi * torch.arccos(
            torch.clamp(
                (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1, -2)
                / (n_ijk_norm * n_jkl_norm),
                -1 + 1e-8,
                1 - 1e-8,
            )
        )

        if not compute_gradient:
            return phi

        a = (
            (r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)
        b = (
            (r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)

        grad_i = n_ijk * (r_kj_norm / n_ijk_norm**2).unsqueeze(-1)
        grad_l = -1 * n_jkl * (r_kj_norm / n_jkl_norm**2).unsqueeze(-1)
        grad_j = (a - 1) * grad_i - b * grad_l
        grad_k = (b - 1) * grad_l - a * grad_i
        grad = torch.stack((grad_i, grad_j, grad_k, grad_l), dim=1)
        return phi, grad


class AbsDihedralPotential(DihedralPotential):
    def compute_variable(self, coords, index, compute_gradient=False):
        if not compute_gradient:
            phi = super().compute_variable(
                coords, index, compute_gradient=compute_gradient
            )
            phi = torch.abs(phi)
            return phi

        phi, grad = super().compute_variable(
            coords, index, compute_gradient=compute_gradient
        )
        grad[(phi < 0)[..., None, :, None].expand_as(grad)] *= -1
        phi = torch.abs(phi)

        return phi, grad


# class PoseBustersPotential(FlatBottomPotential, DistancePotential):
class PoseBustersPotential(HarmonicPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["rdkit_bounds_index"][0]
        lower_bounds = feats["rdkit_lower_bounds"][0].clone()
        upper_bounds = feats["rdkit_upper_bounds"][0].clone()
        bond_mask = feats["rdkit_bounds_bond_mask"][0]
        angle_mask = feats["rdkit_bounds_angle_mask"][0]

        lower_bounds[bond_mask * ~angle_mask] *= 1.0 - parameters["bond_buffer"]
        upper_bounds[bond_mask * ~angle_mask] *= 1.0 + parameters["bond_buffer"]
        lower_bounds[~bond_mask * angle_mask] *= 1.0 - parameters["angle_buffer"]
        upper_bounds[~bond_mask * angle_mask] *= 1.0 + parameters["angle_buffer"]
        lower_bounds[bond_mask * angle_mask] *= 1.0 - min(
            parameters["angle_buffer"], parameters["angle_buffer"]
        )
        upper_bounds[bond_mask * angle_mask] *= 1.0 + min(
            parameters["angle_buffer"], parameters["angle_buffer"]
        )
        lower_bounds[~bond_mask * ~angle_mask] *= 1.0 - parameters["clash_buffer"]
        upper_bounds[~bond_mask * ~angle_mask] = float("inf")

        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None


class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["connected_atom_index"][0]
        lower_bounds = None
        upper_bounds = torch.full(
            (pair_index.shape[1],), parameters["buffer"], device=pair_index.device
        )
        k = torch.ones_like(upper_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None


class BondPotential(HarmonicPotential, DistancePotential):
    # class BondPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        device = feats["atom_pad_mask"].device
        pair_index = torch.empty(2, 0, dtype=torch.long, device=device)
        lower_bounds = torch.empty(0, dtype=torch.float, device=device)
        upper_bounds = torch.empty(0, dtype=torch.float, device=device)
        k = torch.ones_like(lower_bounds)

        lower_value = 1 - parameters["buffer"]
        upper_value = 1 + parameters["buffer"]

        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        atom_chain_id = atom_chain_id[atom_pad_mask]
        res_token_indices = torch.where(feats["res_type"][0])[1]

        is_aa_mask = (res_token_indices > 1) & (res_token_indices < 22)
        is_rna_mask = (res_token_indices > 22) & (res_token_indices < 27)
        is_dna_mask = (res_token_indices > 27) & (res_token_indices < 32)

        any_biomol = is_aa_mask | is_rna_mask | is_dna_mask
        if not any_biomol.any():
            return pair_index, (k, lower_bounds, upper_bounds), None

        rep_atoms = torch.where(feats["r_set_to_rep_atom"][0])[1]

        atom_offsets = torch.zeros_like(res_token_indices, dtype=torch.long)
        bond_lengths = torch.zeros_like(res_token_indices, dtype=torch.float)

        atom_offsets[is_aa_mask] = 1
        atom_offsets[is_rna_mask] = -3
        atom_offsets[is_dna_mask] = -2

        partner_offsets = torch.zeros_like(res_token_indices, dtype=torch.long)
        partner_offsets[is_aa_mask] = -1
        partner_offsets[is_rna_mask] = -11
        partner_offsets[is_dna_mask] = -10

        bond_lengths[is_aa_mask] = parameters["aa_bond_length"]
        bond_lengths[is_rna_mask | is_dna_mask] = parameters["nucleotide_bond_length"]

        atom_indices = rep_atoms + atom_offsets[any_biomol]
        partner_indices = rep_atoms + partner_offsets[any_biomol]

        chain_ids = atom_chain_id[atom_indices]
        unique_chains, inverse_indices = chain_ids.unique(return_inverse=True)

        is_chain_start = torch.zeros(any_biomol.sum(), dtype=torch.bool, device=device)
        is_chain_start[0] = True
        is_chain_start[1:] = chain_ids[1:] != chain_ids[:-1]

        valid_pairs = ~is_chain_start

        if not valid_pairs.any():
            return pair_index, (k, lower_bounds, upper_bounds), None

        biomol_indices = torch.where(any_biomol)[0]
        valid_biomol_indices = biomol_indices[valid_pairs]

        pair_index = torch.stack(
            [
                atom_indices[valid_biomol_indices - 1],
                partner_indices[valid_biomol_indices],
            ]
        )

        bond_lengths_selected = bond_lengths[valid_biomol_indices]
        lower_value = 1 - parameters["buffer"]
        upper_value = 1 + parameters["buffer"]

        lower_bounds = bond_lengths_selected * lower_value
        upper_bounds = bond_lengths_selected * upper_value
        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None


class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = (chain_sizes > 1)[atom_chain_id]

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=atom_chain_id.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=atom_chain_id.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]

        pair_index = torch.triu_indices(
            atom_chain_id.shape[0],
            atom_chain_id.shape[0],
            1,
            device=atom_chain_id.device,
        )

        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]

        num_chains = atom_chain_id.max() + 1
        connected_chain_index = feats["connected_chain_index"][0]
        connected_chain_matrix = torch.eye(
            num_chains, device=atom_chain_id.device, dtype=torch.bool
        )
        connected_chain_matrix[connected_chain_index[0], connected_chain_index[1]] = (
            True
        )
        connected_chain_matrix[connected_chain_index[1], connected_chain_index[0]] = (
            True
        )
        connected_chain_mask = connected_chain_matrix[
            atom_chain_id[pair_index[0]], atom_chain_id[pair_index[1]]
        ]

        pair_index = pair_index[
            :, pair_pad_mask * pair_ion_mask * ~connected_chain_mask
        ]

        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (
            1.0 - parameters["buffer"]
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = feats["symmetric_chain_index"][0]
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion_mask]
        lower_bounds = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            dtype=torch.float32,
            device=pair_index.device,
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return (
            pair_index,
            (k, lower_bounds, upper_bounds),
            (atom_chain_id, atom_pad_mask),
        )


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        stereo_bond_index = feats["stereo_bond_index"][0]
        stereo_bond_orientations = feats["stereo_bond_orientations"][0].bool()

        lower_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        upper_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        lower_bounds[stereo_bond_orientations] = torch.pi - parameters["buffer"]
        upper_bounds[stereo_bond_orientations] = float("inf")
        lower_bounds[~stereo_bond_orientations] = float("-inf")
        upper_bounds[~stereo_bond_orientations] = parameters["buffer"]

        k = torch.ones_like(lower_bounds)

        return stereo_bond_index, (k, lower_bounds, upper_bounds), None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats, parameters):
        chiral_atom_index = feats["chiral_atom_index"][0]
        chiral_atom_orientations = feats["chiral_atom_orientations"][0].bool()

        lower_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        upper_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        lower_bounds[chiral_atom_orientations] = parameters["buffer"]
        upper_bounds[chiral_atom_orientations] = float("inf")
        upper_bounds[~chiral_atom_orientations] = -1 * parameters["buffer"]
        lower_bounds[~chiral_atom_orientations] = float("-inf")

        k = torch.ones_like(lower_bounds)
        return chiral_atom_index, (k, lower_bounds, upper_bounds), None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        double_bond_index = feats["planar_bond_index"][0].T
        double_bond_improper_index = torch.tensor(
            [
                [1, 2, 3, 0],
                [4, 5, 0, 3],
            ],
            device=double_bond_index.device,
        ).T
        improper_index = (
            double_bond_index[:, double_bond_improper_index]
            .swapaxes(0, 1)
            .flatten(start_dim=1)
        )
        lower_bounds = None
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            parameters["buffer"],
            device=improper_index.device,
        )
        k = torch.ones_like(upper_bounds)

        return improper_index, (k, lower_bounds, upper_bounds), None


class SubstructurePotential(HarmonicPotential):
    """Potential for constraining a substructure to reference coordinates.

    This potential applies a flat-bottom harmonic restraint between the current
    coordinates and reference coordinates for a subset of atoms.
    """

    def compute_args(
        self, feats: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Compute the arguments for the substructure potential.

        Parameters
        ----------
        feats : Dict[str, torch.Tensor]
            Dictionary of features from the network output.
        parameters : Dict[str, Any]
            Dictionary of parameters including denoising selection, reference coordinates, scale factor, and buffer distance.

        Returns
        -------
        Tuple
            Tuple containing index, args, and com_args.
        index : torch.Tensor
            Indices of the selected atoms for the potential.
        args : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing the scaling factor, lower bounds, and upper bounds for the potential.
        com_args : Optional[Tuple[torch.Tensor, torch.Tensor]]
            Optional tuple containing the center of mass indices and atom pad mask.
        """
        if (
            self.parameters["denoising_selection"] is None
            or self.parameters["reference_coords"] is None
        ):
            return (
                torch.zeros((2, 0), device=feats["atom_pad_mask"].device),
                (torch.tensor([]), None, None),
                None,
            )

        reference_coords = self.parameters["reference_coords"]  # [..., n_atoms, 3]
        selection = self.parameters["denoising_selection"]  # [n_segment]

        if not isinstance(selection, torch.Tensor):
            selection = torch.from_numpy(selection).to(feats["atom_pad_mask"].device)

        inverse_selector = torch.ones(
            reference_coords.shape[-2], device=feats["atom_pad_mask"].device
        ).bool()

        if selection.shape[0] > 0:
            inverse_selector[selection] = False

        index = torch.where(inverse_selector)[0].unsqueeze(0)
        n_selected = index.shape[1]

        lower_bounds = None
        upper_bounds = torch.full(
            (n_selected,), parameters["buffer"], device=index.device
        )

        k = torch.ones_like(upper_bounds)

        return index, (k, lower_bounds, upper_bounds), None

    def compute_variable(
        self, coords: torch.Tensor, index: torch.Tensor, compute_gradient: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute distances between current coordinates and reference coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Current atomic coordinates, shape [batch, n_atoms, 3]
        index : torch.Tensor
            Paired atom indices, shape [2, n_pairs]
        compute_gradient : bool, optional
            Whether to compute gradients, by default False

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Distances between current and reference coordinates, and optionally gradients
        """
        ref_coords = torch.zeros_like(coords)
        reference_coords = self.parameters["reference_coords"]

        if reference_coords.dim() == 2:
            ref_coords_source = (
                reference_coords[index[0], :]
                .unsqueeze(0)
                .expand(coords.shape[0], -1, -1)
            )
        elif reference_coords.dim() == 3:
            if reference_coords.shape[0] == coords.shape[0]:
                ref_coords_source = reference_coords[:, index[0], :]
            else:
                ref_coords_source = (
                    reference_coords[0, index[0], :]
                    .unsqueeze(0)
                    .expand(coords.shape[0], -1, -1)
                )
        elif reference_coords.dim() == 4:
            if coords.dim() == 4:
                # 4D reference coords with 4D coords
                if (
                    reference_coords.shape[0] == coords.shape[0]
                    and reference_coords.shape[1] == coords.shape[1]
                ):
                    ref_coords_source = reference_coords[:, :, index[0], :]
                else:
                    ref_coords_source = (
                        reference_coords.expand(
                            coords.shape[0], coords.shape[1], -1, -1
                        )
                    )[:, :, index[0], :]
            elif coords.dim() == 3:
                # 4D reference coords with 3D flattened coords (ensemble case)
                # Flatten the reference coordinates to match the flattened ensemble coords
                ref_flattened = reference_coords.reshape(
                    -1, reference_coords.shape[-2], reference_coords.shape[-1]
                )
                if ref_flattened.shape[0] == coords.shape[0]:
                    ref_coords_source = ref_flattened[:, index[0], :]
                else:
                    # Use first reference for all structures if shapes don't match
                    ref_coords_source = (
                        ref_flattened[0, index[0], :]
                        .unsqueeze(0)
                        .expand(coords.shape[0], -1, -1)
                    )
            else:
                raise ValueError(
                    f"Unsupported combination: {reference_coords.dim()}D reference with {coords.dim()}D coords"
                )
        else:
            raise ValueError(
                f"Reference coordinates must be 2D, 3D, or 4D, got {reference_coords.dim()}D."
            )

        ref_coords_source = ref_coords_source.to(
            dtype=coords.dtype, device=coords.device
        )
        if torch.any(index[0] >= ref_coords.shape[-2]):
            raise IndexError(
                f"Index (total size: {len(index[0])}) exceeds the number of atoms in reference coordinates {ref_coords.shape[-2]}."
            )
        ref_coords[..., index[0], :] = ref_coords_source

        r_ij = coords.index_select(-2, index[0]) - ref_coords.index_select(-2, index[0])
        r_ij_norm = torch.linalg.norm(r_ij, dim=-1)

        if not compute_gradient:
            return r_ij_norm

        r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)
        r_hat_ij = torch.where(
            torch.isnan(r_hat_ij), torch.zeros_like(r_hat_ij), r_hat_ij
        ).unsqueeze(
            1
        )  # must add this dimension to match the rest of the potential gradient shapes

        return r_ij_norm, r_hat_ij

    def compute_ensemble(self, coords, feats, parameters):
        """Compute the potential for an ensemble of structures.

        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates, shape [batch, n_ensembles, n_atoms, 3]
        feats : Dict[str, Any]
            Dictionary of features from network
        parameters : Dict[str, Any]
            Dictionary of parameters
        Returns
        -------
        torch.Tensor
            Energy values for the ensemble of structures.
        """
        num_ensembles, ensemble_size = coords.shape[:2]
        original_ref_coords = self.parameters["reference_coords"]

        if (
            original_ref_coords.dim() == 3
            and original_ref_coords.shape[0] == num_ensembles * ensemble_size
        ):
            self.parameters["reference_coords"] = original_ref_coords
        elif original_ref_coords.dim() == 2:
            self.parameters["reference_coords"] = original_ref_coords.unsqueeze(
                0
            ).expand(num_ensembles * ensemble_size, -1, -1)

        result = super().compute_ensemble(coords, feats, parameters)
        self.parameters["reference_coords"] = original_ref_coords

        return result


class DensityPotential(Potential):
    """Potential for density-guided optimization.

    This potential computes an energy based on the agreement between the model
    and the experimental density map. Lower energy corresponds to better agreement.
    Uses the real_space_refine potential Tdata from Phenix (Afonine, et al.)
    T_data = -∑_G ⍴_calc(g) * ⍴_map(g)

    where ρ(g) is the map density at grid position g. ρ(g) is a function of atom positions, occupancies, and B-factors.
    Lower energy corresponds to atoms positioned in regions of higher experimental density.
    """

    def __init__(
        self,
        xmap: XMap_torch,
        parameters: Optional[
            Dict[str, Union[ParameterSchedule, float, int, bool, torch.Tensor]]
        ] = None,
        atom_selection: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None,
    ) -> None:
        """Initialize the density potential.

        Parameters
        ----------
        xmap : XMap_torch
            XMap_torch object containing grid parameters and the experimental map array.
        parameters : Optional[Dict[str, Union[ParameterSchedule, float, int, bool]]], optional
            Dictionary of parameters, by default None
        atom_selection : Optional[Union[torch.Tensor, np.ndarray, List[int]]], optional
            Indices of atoms to apply density guidance to. If None, applies to all atoms, by default None
        """
        super().__init__(parameters)
        self.xmap = xmap
        self.atom_selection = atom_selection
        self._setup_transforms()

    def _setup_transforms(self) -> None:
        """Initialize transformation matrices for coordinate conversions."""
        self.dtype = torch.float32  # need to set this here or else doubles start popping up and ruining operations
        self.device = self.xmap.array.device

        lattice_to_cartesian = (
            self.xmap.unit_cell.frac_to_orth / self.xmap.unit_cell.abc
        )
        cartesian_to_lattice = (
            self.xmap.unit_cell.orth_to_frac * self.xmap.unit_cell.abc.reshape(3, 1)
        )
        grid_to_cartesian = lattice_to_cartesian * self.xmap.voxelspacing.cpu().numpy()

        self.lattice_to_cartesian = torch.tensor(lattice_to_cartesian).to(
            dtype=self.dtype, device=self.device
        )
        self.cartesian_to_lattice = torch.tensor(cartesian_to_lattice).to(
            dtype=self.dtype, device=self.device
        )
        self.grid_to_cartesian = torch.tensor(grid_to_cartesian).to(
            dtype=self.dtype, device=self.device
        )

    def compute_variable(
        self,
        coords: torch.Tensor,
        density_params: Dict[str, torch.Tensor],
        index: torch.Tensor,
        compute_gradient: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute density to be used for energy computation.

        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates, shape [batch, n_atoms, 3]
        density_params : Dict[str, torch.Tensor]
            Dictionary of parameters for density calculation (occupancies, B factors, etc.)
        index : torch.Tensor
            Indices of the atoms to compute density value for, shape [batch, n_atoms] # TODO: extend this to allow multiple regions to change?
        compute_gradient : bool, optional
            Whether to compute gradients, by default False

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            grid coordinates and optionally gradients
        """
        coords_selected = coords[..., index[0], :]  # [..., n_active, 3]

        if not compute_gradient:
            with torch.no_grad():
                rho_calc = self.density_calculator(coords_selected, **density_params)
                return rho_calc

        rho_calc = self.density_calculator(coords_selected, **density_params)

        rho_calc.backward(torch.ones_like(rho_calc), retain_graph=True)

        # Get gradients with respect to coordinates
        grad_coords = coords.grad.clone()
        coords.grad.zero_()

        return rho_calc, grad_coords

    def compute_args(
        self, feats: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Tuple[
        Dict[str, torch.Tensor], Tuple, Optional[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Prepare arguments for density potential calculation.

        Parameters
        ----------
        feats : Dict[str, Any]
            Dictionary of features from network
        parameters : Dict[str, Any]
            Dictionary of parameters (occupancies and B factors are required for this potential)

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Tuple, Optional[Tuple[torch.Tensor, torch.Tensor]]]
            Tuple containing (index_dict, args, com_args)
        """
        # Get all non-padded atoms first
        all_indices = torch.where(feats["atom_pad_mask"][0].bool())[0]

        # Apply atom selection if provided
        if self.atom_selection is not None:
            # Convert selection to tensor if needed
            if isinstance(self.atom_selection, (list, np.ndarray)):
                selection = torch.tensor(
                    self.atom_selection, device=feats["atom_pad_mask"].device
                )
            else:
                selection = self.atom_selection.to(device=feats["atom_pad_mask"].device)

            # Filter to only include atoms that are both in selection and not padded
            mask = torch.isin(all_indices, selection)
            indices = all_indices[mask].unsqueeze(0)  # needs to be dim=2
        else:
            # Use all non-padded atoms (original behavior)
            indices = all_indices.unsqueeze(0)  # needs to be dim=2
        elements = parameters["elements"]
        occupancies = parameters["occupancies"]
        b_factors = parameters["b_factors"]

        density_params = {
            "elements": elements,
            "b_factors": b_factors,
            "occupancies": occupancies,
        }

        if parameters["resolution"] == self.xmap.resolution.high:
            self.density_calculator = DifferentiableTransformer(
                xmap=self.xmap,
                scattering_params=parameters["scattering_params"],
                em=parameters["em"],
                device=self.device,
            )
        else:
            try:
                res_high = self.density_calculator.xmap.resolution.high
            except:  # noqa: E722
                res_high = 0.0

            if parameters["resolution"] != res_high:
                xmap = self.xmap.downsample_to_resolution(parameters["resolution"])
                self.density_calculator = DifferentiableTransformer(
                    xmap=xmap,
                    scattering_params=parameters["scattering_params"],
                    em=parameters["em"],
                    device=self.device,
                )

        return indices, density_params, None

    def compute_function(
        self,
        value: torch.Tensor,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        occupancies: torch.Tensor,
        k: float = 1.0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the energy function from the density.

        Parameters
        ----------
        value : torch.Tensor
            Quantity that the energy is calculated from, here of shape [batch, *value_shape] (here it is the density)
        elements : torch.Tensor
            Atomic elements, shape [batch, n_atoms]
        b_factors : torch.Tensor
            B-factors for the atoms, shape [batch, n_atoms]
        occupancies : torch.Tensor
            Occupancies for the atoms, shape [batch, n_atoms]
        k : float, optional
            Scaling factor for the energy, by default 1.0
        compute_derivative : bool, optional
            Whether to compute derivatives, by default False

        Returns
        -------
        torch.Tensor
            Energy values
        """
        # target = -(value * self.density_calculator.xmap.array.to(torch.float32)).sum(
        #     dim=[-3, -2, -1]
        # )
        # target = (
        #     (value - self.density_calculator.xmap.array.to(torch.float32)) ** 2
        # ).sum(dim=[-3, -2, -1])
        target = self.density_calculator.xmap.array.float().expand(
            value.shape[0], -1, -1, -1
        )
        # energy = batched_hybrid_loss(
        #     value, target, alpha=0.7
        # ).squeeze()
        energy = batched_squared_error(value, target)

        return energy

    def compute_function_ensemble(
        self,
        value: torch.Tensor,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        occupancies: torch.Tensor,
        k: float = 1.0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the energy function from the interpolated density.

        Parameters
        ----------
        value : torch.Tensor
            Quantity that the energy is calculated from, here of shape [batch, ensemble_size, *value_shape] (here it is the density)
        elements : torch.Tensor
            Atomic elements, shape [batch, n_atoms]
        b_factors : torch.Tensor
            B-factors for the atoms, shape [batch, n_atoms]
        occupancies : torch.Tensor
            Occupancies for the atoms, shape [batch, n_atoms]
        k : float, optional
            Scaling factor for the energy, by default 1.0

        Returns
        -------
        torch.Tensor
            Energy values
        """
        value = value.sum(
            dim=1
        )  # [batch, ensemble_size, *value_shape] -> [batch, *value_shape]
        # target = -(value * self.density_calculator.xmap.array.to(torch.float32)).sum(
        #     dim=[-3, -2, -1]
        # )
        target = self.density_calculator.xmap.array.float().expand(
            value.shape[0], -1, -1, -1
        )
        # energy = batched_hybrid_loss(
        #     value, target, alpha=0.7
        # )
        energy = batched_squared_error(value, target)

        return energy

    def compute_gradient(
        self,
        coords: torch.Tensor,
        feats: Dict[str, Any],
        parameters: Dict[str, Any],
    ):
        """Compute the gradient of the density potential.

        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates, shape [batch, n_atoms, 3]
        feats : Dict[str, Any]
            Dictionary of features from network
        parameters : Dict[str, Any]
            Dictionary of parameters (occupancies and B factors are required for this potential)

        Returns
        -------
        torch.Tensor
            Gradient of the density potential with respect to atomic coordinates.
        """
        coords = coords.clone().detach().requires_grad_()
        index, args, com_args = self.compute_args(feats, parameters)
        value, _ = self.compute_variable(coords, args, index, compute_gradient=True)

        energy = self.compute_function(
            value,
            **args,
        )

        energy.backward(torch.ones_like(energy))
        grad_atom = coords.grad.clone()
        coords.grad.zero_()

        return grad_atom

    def compute(self, coords, feats, parameters):
        index, args, com_args = self.compute_args(feats, parameters)

        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[..., atom_pad_mask, :]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
        value = self.compute_variable(coords, args, index, compute_gradient=False)
        energy = self.compute_function(value, *args)
        return energy

    def compute_ensemble(self, coords, feats, parameters):
        num_ensembles, ensemble_size = coords.shape[:2]
        original_shape = coords.shape

        coords_batched = coords.reshape(
            num_ensembles * ensemble_size, *coords.shape[2:]
        )

        index, args, com_args = self.compute_args(feats, parameters)

        if index.shape[1] == 0:
            return torch.zeros((num_ensembles), device=coords.device)

        expanded_args = {}
        for key, value in args.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                expanded_args[key] = value.expand(
                    num_ensembles * ensemble_size, *value.shape[1:]
                )
            else:
                expanded_args[key] = value

        value = self.compute_variable(
            coords_batched, expanded_args, index, compute_gradient=False
        )
        value = value.reshape(num_ensembles, ensemble_size, *value.shape[1:])
        energy = self.compute_function_ensemble(value, **expanded_args)

        return energy

    def compute_gradient_ensemble(self, coords, feats, parameters):
        num_ensembles, ensemble_size = coords.shape[:2]
        coords_batched = coords.reshape(
            num_ensembles * ensemble_size, *coords.shape[2:]
        )

        coords_batched = coords_batched.clone().detach().requires_grad_()
        index, args, com_args = self.compute_args(feats, parameters)

        expanded_args = {}
        for key, value in args.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                expanded_args[key] = value.expand(
                    num_ensembles * ensemble_size, *value.shape[1:]
                )
            else:
                expanded_args[key] = value

        value, _ = self.compute_variable(
            coords_batched, expanded_args, index, compute_gradient=True
        )
        value = value.reshape(num_ensembles, ensemble_size, *value.shape[1:])
        energy = self.compute_function_ensemble(value, **expanded_args)

        energy.backward(torch.ones_like(energy))
        grad_atom = coords_batched.grad.clone()
        coords_batched.grad.zero_()

        return grad_atom.reshape(num_ensembles, ensemble_size, *grad_atom.shape[1:])

    def interpolate_density_at_positions(
        self, positions: torch.Tensor, mode: str = "tricubic"
    ) -> torch.Tensor:
        """Interpolate experimental density values at given atomic positions with symmetry.

        This method applies all symmetry operations to the atomic positions and interpolates
        the density at each symmetric position, then returns the average.

        Parameters
        ----------
        positions : torch.Tensor
            Atomic positions in Cartesian coordinates, shape [batch, n_atoms, 3]

        Returns
        -------
        torch.Tensor
            Interpolated density values at each position, shape [batch, n_atoms]
        """
        if self.xmap is None:
            raise ValueError("XMap_torch object and experimental map must be provided")

        batch_size, n_atoms, _ = positions.shape

        grid_coords = self._compute_grid_coordinates(positions)  # [batch, n_atoms, 3]

        n_ops = self.xmap.R_matrices.shape[0]

        # [batch, n_atoms, 3] @ [n_ops, 3, 3] -> [n_ops, batch, n_atoms, 3]
        grid_coords_rot = torch.einsum(
            "oij,bnj->obni", self.xmap.R_matrices, grid_coords
        )

        grid_shape = torch.tensor(self.xmap.array.shape, device=positions.device)
        grid_shape_xyz = torch.tensor(
            self.xmap.array.shape[::-1], device=positions.device
        )
        grid_coords_rot_trans = grid_coords_rot + self.xmap.t_vectors.unsqueeze(
            1
        ).unsqueeze(2) * grid_shape_xyz.view(1, 1, 1, 3)  # [n_ops, batch, n_atoms, 3]

        # to zyx
        grid_coords_rot_trans = grid_coords_rot_trans[
            ..., [2, 1, 0]
        ]  # [n_ops, batch, n_atoms, 3]

        # apply periodic boundary
        grid_coords_rot_trans = grid_coords_rot_trans % grid_shape.view(
            1, 1, 1, 3
        )  # [n_ops, batch, n_atoms, 3]

        # Use grid_sample to interpolate density values at all symmetric positions
        interp_func = (
            trilinear_interpolation_torch
            if mode == "trilinear"
            else tricubic_interpolation_torch
        )
        interpolated = interp_func(self.xmap.array.float(), grid_coords_rot_trans)

        # Reshape and sum over symmetry operations
        interpolated = interpolated.view(
            batch_size, n_atoms, n_ops
        )  # [batch, n_atoms, n_ops]
        interpolated_mean = interpolated.mean(dim=2)  # [batch, n_atoms]

        return interpolated_mean

    def _compute_grid_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Transform Cartesian coordinates to grid coordinates.

        Parameters
        ----------
        coordinates : torch.Tensor
            Cartesian coordinates of shape (batch_size, n_atoms, 3).

        Returns
        -------
        torch.Tensor
            Grid coordinates of shape (batch_size, n_atoms, 3).
        """
        if hasattr(self.xmap, "origin") and not torch.allclose(
            torch.tensor(self.xmap.origin, device=self.device, dtype=coordinates.dtype),
            torch.zeros(3, device=self.device),
        ):
            coordinates = coordinates - torch.tensor(
                self.xmap.origin, device=self.device, dtype=coordinates.dtype
            )

        grid_coordinates = torch.matmul(coordinates, self.cartesian_to_lattice.T)
        grid_coordinates /= self.xmap.voxelspacing.to(self.device)

        if hasattr(self.xmap, "offset"):
            grid_coordinates -= self.xmap.offset.to(
                device=self.device, dtype=coordinates.dtype
            )

        return grid_coordinates


def get_potentials():
    potentials = [
        SymmetricChainCOMPotential(
            parameters={
                "guidance_interval": 4,
                "guidance_weight": 0.5,
                "resampling_weight": 0.5,
                "buffer": ExponentialInterpolation(start=1.0, end=5.0, alpha=-2.0),
            }
        ),
        VDWOverlapPotential(
            parameters={
                "guidance_interval": 5,
                # "guidance_weight": 0.125, # testing
                "guidance_weight": PiecewiseStepFunction(
                    thresholds=[0.4], values=[0.125, 0.0]
                ),
                "resampling_weight": PiecewiseStepFunction(
                    thresholds=[0.6], values=[0.01, 0.0]
                ),
                "buffer": 0.225,
            }
        ),
        ConnectionsPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.15,
                "resampling_weight": 1.0,
                "buffer": 2.0,
            }
        ),
        PoseBustersPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": PiecewiseSchedule(
                    thresholds=[1 - 175 / 200],
                    values=[0.05, 0.0],
                ),
                # "guidance_weight": Ramp(
                #     base=0.05,
                #     start_t=0.00,
                #     end_t=0.25,
                #     ramps=[
                #         {"target": 0.3, "alpha": -2},
                #     ]
                #     * 3,
                # ),
                "resampling_weight": 0.1,
                "bond_buffer": 0.01,
                "angle_buffer": 0.10,
                "clash_buffer": 0.05,
            }
        ),
        ChiralAtomPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.10,
                "resampling_weight": 1.0,
                "buffer": 0.52360,
            }
        ),
        StereoBondPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.05,
                "resampling_weight": 1.0,
                "buffer": 0.52360,
            }
        ),
        PlanarBondPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.05,
                "resampling_weight": 1.0,
                "buffer": 0.26180,
            }
        ),
        BondPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": PiecewiseSchedule(
                    thresholds=[1 - 175 / 200],
                    values=[0.05, 0.0],
                ),
                # "guidance_weight": Ramp(
                #     base=0.05,
                #     start_t=0.00,
                #     end_t=0.25,
                #     ramps=[
                #         {"target": 0.3, "alpha": -2},
                #     ]
                #     * 3,
                # ),
                "resampling_weight": 0.1,
                "buffer": 0.01,
                "aa_bond_length": 1.32,  # Angstroms
                "nucleotide_bond_length": 1.60,  # Angstroms
            }
        ),
    ]
    return potentials
