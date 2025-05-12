"""Copy of Boltz-1x potentials used for FK steering. Added my own density potential here.

Date: 6 May 2025
Author: Karson Chrispens (karson.chrispens@ucsf.edu)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Set, List, Union, Tuple, cast
from copy import deepcopy
import torch
import torch.nn.functional as F

from boltz.data import const
from boltz.model.potentials.schedules import *
from adp3d.utils.interpolation import trilinear_interpolation_torch

from .density import XMap_torch


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
            -2 * k.expand_as(neg_overflow_mask)[neg_overflow_mask] * (lower_bounds - value)[neg_overflow_mask]
        )
        dEnergy[pos_overflow_mask] = (
            2 * k.expand_as(pos_overflow_mask)[pos_overflow_mask] * (value - upper_bounds)[pos_overflow_mask]
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


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
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
            # If no selection is provided, return empty indices
            return (
                torch.zeros((2, 0), device=feats["atom_pad_mask"].device),
                (torch.tensor([]), None, None),
                None,
            )

        reference_coords = self.parameters["reference_coords"]  # [n_atoms, 3]
        selection = self.parameters["denoising_selection"]  # [n_segment]

        if not isinstance(selection, torch.Tensor):
            selection = torch.from_numpy(selection).to(feats["atom_pad_mask"].device)

        inverse_selector = torch.ones(
            reference_coords.shape[1], device=feats["atom_pad_mask"].device
        ).bool()

        if selection.shape[0] > 0:
            inverse_selector[selection] = False

        index = torch.where(inverse_selector)[0].unsqueeze(
            0
        )  # The atoms outside the denoising region
        n_selected = index.shape[0]

        lower_bounds = None
        # Upper bounds based on the buffer parameter
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
        ref_coords[..., index[0], :] = self.parameters["reference_coords"][
            ..., index[0], :
        ].to(dtype=coords.dtype, device=coords.device)

        r_ij = coords.index_select(-2, index[0]) - ref_coords.index_select(-2, index[0])
        r_ij_norm = torch.linalg.norm(r_ij, dim=-1)

        if not compute_gradient:
            return r_ij_norm

        r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)

        return r_ij_norm, r_hat_ij


class DensityPotential(Potential):
    """Potential for density-guided optimization.

    This potential computes an energy based on the agreement between the model
    and the experimental density map. Lower energy corresponds to better agreement.
    Uses the real_space_refine potential Tdata from Phenix (Afonine, et al. )
    T_data = -∑_A ρ_interp(r_A)

    where ρ_interp(r_A) is the experimental map density interpolated at atomic position r_A.
    Lower energy corresponds to atoms positioned in regions of higher experimental density.
    """

    def __init__(
        self,
        xmap: XMap_torch,
        parameters: Optional[
            Dict[str, Union[ParameterSchedule, float, int, bool, torch.Tensor]]
        ] = None,
    ) -> None:
        """Initialize the density potential.

        Parameters
        ----------
        xmap : XMap_torch
            XMap_torch object containing grid parameters and the experimental map array.
        parameters : Optional[Dict[str, Union[ParameterSchedule, float, int, bool]]], optional
            Dictionary of parameters, by default None
        """
        super().__init__(parameters)
        self.xmap = xmap
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
        index: torch.Tensor,
        compute_gradient: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute grid coordinates to be used for density interpolation.

        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates, shape [batch, n_atoms, 3]
        index : torch.Tensor
            Indices of the atoms to compute density value for, shape [batch, n_atoms]
        compute_gradient : bool, optional
            Whether to compute gradients, by default False

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            grid coordinates and optionally gradients
        """
        # TODO: think of how to use index creatively
        coords_translated = coords[..., index[0], :]  # [batch, n_active, 3]

        if self.parameters["initial_centroid"] is not None:
            coords_translated = coords_translated + self.parameters[
                "initial_centroid"
            ].repeat(
                coords.shape[0] // self.parameters["initial_centroid"].shape[0], 1, 1
            )

        if not compute_gradient:
            interpolated_density = self.interpolate_density_at_positions(
                coords_translated
            )
            return interpolated_density

        # If we need gradients, we use autograd
        coords_translated = coords_translated.clone().detach().requires_grad_(True)

        interpolated_density = self.interpolate_density_at_positions(
            coords_translated
        )  # [batch, n_active]

        to_grad = interpolated_density.sum(dim=-1)
        to_grad.backward(torch.ones_like(to_grad))

        # Get gradients with respect to coordinates
        grad_coords = coords_translated.grad.clone()
        coords_translated.grad.zero_()

        return interpolated_density, grad_coords

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
        indices = torch.where(feats["atom_pad_mask"][0].bool())[0].unsqueeze(
            0
        )  # needs to be dim=2
        elements = torch.where(feats["ref_element"])[0]
        occupancies = parameters["occupancies"]
        b_factors = parameters["b_factors"]

        return indices, (elements, b_factors, occupancies), None

    def compute_function(
        self,
        value: torch.Tensor,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        occupancies: torch.Tensor,
        k: float = 1.0,
        compute_derivative: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the energy function from the interpolated density.

        Parameters
        ----------
        value : torch.Tensor
            Quantity that the energy is calculated from, here of shape [batch, n_atoms] (value is interpolated density)
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
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Energy values, and optionally derivatives
        """

        # TODO: Apply weights based on B-factors and occupancies
        # This is a placeholder for future implementation
        weights = torch.ones_like(b_factors)
        if elements is not None:
            # Placeholder: scale weight by element
            pass
        if occupancies is not None:
            # Placeholder: scale weight by occupancy
            pass
        if b_factors is not None:
            # Placeholder: weights *= torch.exp(-b_factors / (8 * np.pi**2)) ??
            pass

        weighted_density = k * value * weights  # [batch, n_atoms]
        negative_sum = -torch.sum(weighted_density, dim=-1)  # [batch]

        if not compute_derivative:
            return negative_sum

        dEnergy = -k * weights * torch.ones_like(value)  # [batch, n_atoms]

        return negative_sum.detach(), dEnergy

    def interpolate_density_at_positions(self, positions: torch.Tensor) -> torch.Tensor:
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
        interpolated = trilinear_interpolation_torch(
            self.xmap.array.float(), grid_coords_rot_trans
        )

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
                "guidance_weight": 0.05,
                "resampling_weight": 0.1,
                "bond_buffer": 0.20,
                "angle_buffer": 0.20,
                "clash_buffer": 0.15,
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
    ]
    return potentials
