import torch
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import torch.nn.functional as F
from torchquad import (
    Simpson,
    Trapezoid,
    GaussLegendre,
    set_up_backend,
)  # backend should be inferred from integration domain...
import numpy as np

from adp3d.qfit.volume import XMap, EMMap, GridParameters, Resolution
from adp3d.qfit.unitcell import UnitCell
from adp3d.qfit.spacegroups import GetSpaceGroup
from adp3d.data.sf import ATOM_STRUCTURE_FACTORS, ELECTRON_SCATTERING_FACTORS


@dataclass
class DensityParameters:
    """Parameters for electron density calculation.

    Controls the numerical parameters used in density calculations and
    integration procedures.

    Parameters
    ----------
    rmax : float
        Maximum radius for density calculation in Angstroms.
    rstep : float
        Step size for radial grid in Angstroms.
    smin : float
        Minimum scattering vector magnitude in inverse Angstroms.
    smax : float
        Maximum scattering vector magnitude in inverse Angstroms.
    quad_points : int
        Number of quadrature points for numerical integration.
    integration_method : str
        Integration method to use ('gausslegendre' or 'simpson' or 'trapezoid').
    """

    rmax: float = 3.0
    rstep: float = 0.01
    smin: float = 0.0
    smax: float = 0.5
    quad_points: int = 50
    integration_method: str = "gausslegendre"

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.rmax <= 0 or self.rstep <= 0:
            raise ValueError("rmax and rstep must be positive")
        if self.smin >= self.smax:
            raise ValueError("smin must be less than smax")
        if self.quad_points < 2:
            raise ValueError("quad_points must be at least 2")
        if self.integration_method not in ["gausslegendre", "simpson", "trapezoid"]:
            raise ValueError(
                "integration_method must be 'gausslegendre', 'simpson' or 'trapezoid'"
            )


class ScatteringIntegrand(torch.nn.Module):
    """Computes the scattering integrand for radial density calculation.

    Handles both electron and X-ray scattering modes with appropriate
    parameter validation and efficient computation.
    """

    def __init__(
        self,
        asf: torch.Tensor,
        bfactor: torch.Tensor,
        em: bool = False,
    ) -> None:
        """Initialize the scattering integrand calculator.

        Parameters
        ----------
        asf : torch.Tensor
            Atomic scattering factors with shape (n_atoms, n_coeffs, 2).
            The second dimension contains coefficients [a_i, b_i].
        bfactor : torch.Tensor
            B-factors for each atom with shape (n_atoms,).
        em : bool, optional
            Whether to use electron microscopy mode, by default False.
        r : Optional[torch.Tensor], optional
            Radial distances, required for torchquad integration.
        """
        super().__init__()

        self.register_buffer("asf", asf)
        self.register_buffer("bfactor", bfactor.view(-1, 1, 1))
        self.em = em

        self._validate_inputs(asf, bfactor)

        self.n_atoms = asf.shape[0]
        self.asf_range = 5

        if not em:
            self.register_buffer(
                "constant_term", self.asf[:, 5, 0].view(self.n_atoms, 1, 1)
            )
        else:
            self.register_buffer(
                "constant_term", self.asf[:, 4, 0].new_zeros(self.n_atoms, 1, 1)
            )

    def _validate_inputs(self, asf: torch.Tensor, bfactor: torch.Tensor) -> None:
        """Validate input tensors for shape and content.

        Parameters
        ----------
        asf : torch.Tensor
            Atomic scattering factors tensor to validate.
        bfactor : torch.Tensor
            B-factors tensor to validate.

        Raises
        ------
        ValueError
            If tensor shapes or content are invalid.
        """
        if asf.dim() != 3 or asf.shape[2] != 2:
            raise ValueError(
                f"asf must have shape (n_atoms, n_coeffs, 2), got {asf.shape}"
            )

        if bfactor.dim() != 1:
            raise ValueError(
                f"bfactor must be 1-dimensional, got {bfactor.dim()}-dimensional"
            )

        if not self.em and asf.shape[1] < 6:
            raise ValueError(
                f"For X-ray scattering, exactly 6 coefficients required, got {asf.shape[1]}"
            )

        if self.em and asf.shape[1] < 5:
            raise ValueError(
                f"For electron scattering, at least 5 coefficients required, got {asf.shape[1]}"
            )

    def compute_scattering_factors(self, s2: torch.Tensor) -> torch.Tensor:
        """Compute atomic scattering factors for given squared scattering vectors.

        Parameters
        ----------
        s2 : torch.Tensor
            Squared scattering vector magnitudes with shape [n_s, 1].

        Returns
        -------
        torch.Tensor
            Computed scattering factors [n_atoms, n_s, 1].
        """
        n_s = s2.shape[0]

        s2_broadcast = s2.unsqueeze(0)  # [1, n_s, 1]

        if self.em:
            # initialize with zeros for EM mode
            f = self.constant_term.new_zeros(self.n_atoms, n_s, 1)
        else:
            # initialize with constant term for X-ray mode (shape: [n_atoms, n_s, 1])
            f = self.constant_term.expand(self.n_atoms, n_s, 1).clone()

        for i in range(self.asf_range):
            a_coeff = self.asf[:, i, 0].view(self.n_atoms, 1, 1)  # [n_atoms, 1, 1]
            b_coeff = self.asf[:, i, 1].view(self.n_atoms, 1, 1)  # [n_atoms, 1, 1]

            # [n_atoms, 1, 1] * exp(-[n_atoms, 1, 1] * [1, n_s, 1]) -> [n_atoms, n_s, 1]
            f += a_coeff * torch.exp(-b_coeff * s2_broadcast)

        return f  # [n_atoms, n_s, 1]

    def forward(
        self,
        s: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the scattering integrand for torchquad integration.

        Parameters
        ----------
        s : torch.Tensor
            Scattering vector magnitudes .
        r : torch.Tensor
            Radial distances to use for computing the density.

        Returns
        -------
        torch.Tensor
            Computed integrand values.
        """
        s_flat = s.reshape(-1)

        s_expanded = s_flat.unsqueeze(-1)  # (n_s, 1)
        r_expanded = r.view(1, -1)  # (1, n_r)

        s2 = s_expanded * s_expanded

        f = self.compute_scattering_factors(s2)  # (n_atoms, n_s, 1)

        four_pi_s = 4 * torch.pi * s_expanded  # (n_s, 1)
        w = 8 * f * torch.exp(-self.bfactor * s2) * s_expanded  # (n_atoms, n_s, 1)

        eps = 1e-4
        r_small_mask = r_expanded < eps

        result = w.new_zeros(
            (self.n_atoms, s.shape[0], r.shape[0])
        )

        # prevent singularity with 4th order Taylor expansion
        if r_small_mask.any():
            ar_small = four_pi_s * torch.where(r_small_mask[0], r_expanded, 0)
            ar2_small = ar_small * ar_small
            taylor_term = 1.0 - ar2_small / 6.0
            small_r_result = w * four_pi_s * taylor_term
            result += small_r_result

        r_large_mask = ~r_small_mask
        if r_large_mask.any():
            ar_large = four_pi_s * torch.where(r_large_mask[0], r_expanded, 0)
            sin_term = torch.sin(ar_large)
            large_r_result = w * sin_term / torch.where(r_large_mask[0], r_expanded, 1)
            result += large_r_result

        # (n_atoms, n_s, n_r) to (n_s, n_atoms, n_r)
        result = result.permute(1, 0, 2)

        return result


class XMap_torch:
    """Torch version of qFit XMap for handling crystallographic symmetry."""

    def __init__(
        self,
        xmap: XMap = None,
        array: Optional[torch.Tensor] = None,
        grid_parameters: Optional[GridParameters] = None,
        unit_cell: Optional[UnitCell] = None,
        resolution: Optional[Union[Resolution, float]] = None,
        hkl: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize differentiable XMap.

        Parameters
        ----------
        xmap : XMap, optional
            Initialize from qFit XMap (preferred).
        array : Optional[torch.Tensor], optional
            Map array, by default None.
        grid_parameters : Optional[GridParameters], optional
            Grid parameters for the map, meaning voxel spacing (voxelspacing) and offset.
        unit_cell : Optional[UnitCell], optional
            Crystallographic unit cell information.
        resolution : Optional[Union[Resolution, float]], optional
            Map resolution in Angstroms, by default None.
        hkl : Optional[torch.Tensor], optional
            Miller indices for the map, by default None.
        device : torch.device, optional
            Device to use for computations, by default 'cpu'.
        """
        if xmap is not None:
            self.unit_cell = xmap.unit_cell
            self.resolution = xmap.resolution
            self.hkl = xmap.hkl
            self.origin = xmap.origin
            self.shape = xmap.shape

            self.voxelspacing = torch.tensor(xmap.voxelspacing, device=device)
            self.offset = torch.tensor(xmap.offset, device=device)
        else:
            self.unit_cell = unit_cell
            self.resolution = resolution
            self.hkl = hkl
            self.origin = origin
            self.array = array
            self.shape = self.array.shape

            self.voxelspacing = torch.tensor(
                grid_parameters.voxelspacing, device=device
            )
            self.offset = torch.tensor(grid_parameters.offset, device=device)

        self._validate_input(xmap)

        self._setup_symmetry_matrices(device)

    def _validate_input(self, xmap) -> None:
        """Validate input parameters."""
        if xmap is not None:
            if not isinstance(xmap, XMap):
                raise ValueError(
                    "xmap must be an instance of qFit's XMap (adp3d.qfit.volume.XMap)"
                )
        else:
            if self.array is None:
                raise ValueError("array must be provided")
            if self.unit_cell is None:
                raise ValueError("unit_cell must be provided")
            if self.resolution is None:
                raise ValueError("resolution must be provided")
            if self.hkl is None:
                raise ValueError("hkl must be provided")
            if self.voxelspacing is None:
                raise ValueError("grid_parameters must be provided")
            if self.offset is None:
                raise ValueError("grid_parameters must be provided")

    def _setup_symmetry_matrices(self, device: torch.device) -> None:
        """Precompute symmetry operation matrices for efficient application."""
        symops = self.unit_cell.space_group.symop_list
        n_ops = len(symops)

        R_matrices = torch.zeros((n_ops, 3, 3), device=device)
        t_vectors = torch.zeros((n_ops, 3), device=device)

        for i, symop in enumerate(symops):
            R_matrices[i] = torch.tensor(symop.R, device=device, dtype=torch.float32)
            t_vectors[i] = torch.tensor(symop.t, device=device, dtype=torch.float32)

        self.R_matrices = R_matrices
        self.t_vectors = t_vectors

    def _apply_symmetry_chunk(
        self,
        density: torch.Tensor,
        output: torch.Tensor,
        chunk_start: int,
        chunk_end: int,
        grid_shape: Tuple[int, ...],
        batch_size: int,
    ) -> None:
        """Apply a chunk of symmetry operations to improve memory efficiency.

        Parameters
        ----------
        density : torch.Tensor
            Input density grid of shape (batch_size, *grid_shape).
        output : torch.Tensor
            Output density grid to be updated.
        chunk_start : int
            Start index of symmetry operations chunk.
        chunk_end : int
            End index of symmetry operations chunk.
        grid_shape : Tuple[int, ...]
            Shape of the density grid.
        batch_size : int
            Number of batches in the input.
        """
        device = density.device
        chunk_size = chunk_end - chunk_start
        grid_shape_tensor = torch.tensor(grid_shape, device=device)

        base_coords = torch.stack(
            torch.meshgrid(
                *[torch.arange(s, device=device) for s in grid_shape], indexing="ij"
            ),
            dim=-1,
        ).float()

        base_coords = base_coords + self.offset
        base_coords = base_coords.view(1, *grid_shape, 3)

        R_chunk = self.R_matrices[chunk_start:chunk_end]
        t_chunk = self.t_vectors[chunk_start:chunk_end]

        coords_expanded = base_coords.expand(chunk_size, *grid_shape, 3)

        transformed_coords = torch.einsum("nij,b...j->n...i", R_chunk, coords_expanded)

        transformed_coords = transformed_coords + (
            t_chunk.view(chunk_size, 1, 1, 1, 3) * grid_shape_tensor.view(1, 1, 1, 1, 3)
        )

        transformed_coords = transformed_coords % grid_shape_tensor.view(1, 1, 1, 1, 3)

        normalized_coords = (
            transformed_coords / (grid_shape_tensor.view(1, 1, 1, 1, 3) - 1)
        ) * 2 - 1

        for b in range(batch_size):
            normalized_coords_batch = normalized_coords.view(
                chunk_size, -1, grid_shape[1], grid_shape[2], 3
            )

            transformed_density = F.grid_sample(
                density[b : b + 1, None].expand(chunk_size, 1, *grid_shape),
                normalized_coords_batch,
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )

            output[b] += transformed_density.sum(dim=0)[0]

    def apply_symmetry(
        self,
        density: torch.Tensor,
        normalize: bool = True,
        chunk_size: Optional[int] = None,
        memory_efficient: bool = True,
    ) -> torch.Tensor:
        """Apply crystallographic symmetry operations to density maps in batch.

        Parameters
        ----------
        density : torch.Tensor
            Input density grid of shape (batch_size, *grid_shape).
        normalize : bool, optional
            Whether to normalize the output density, by default True.
        chunk_size : Optional[int], optional
            Number of symmetry operations to process at once, by default None.
        memory_efficient : bool, optional
            Whether to use memory-efficient implementation, by default True.

        Returns
        -------
        torch.Tensor
            Symmetry-expanded density grid.
        """
        batch_size = density.shape[0]
        grid_shape = density.shape[1:]
        device = density.device
        n_ops = len(self.R_matrices)

        output = density.clone()

        if memory_efficient or chunk_size is not None:
            chunk_size = chunk_size or max(1, n_ops // 4)

            for chunk_start in range(0, n_ops, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_ops)
                self._apply_symmetry_chunk(
                    density, output, chunk_start, chunk_end, grid_shape, batch_size
                )
        else:
            base_coords = torch.stack(
                torch.meshgrid(
                    *[torch.arange(s, device=device) for s in grid_shape], indexing="ij"
                ),
                dim=-1,
            ).float()

            base_coords = base_coords + self.offset
            base_coords = base_coords.view(1, *grid_shape, 3)
            grid_shape_tensor = torch.tensor(grid_shape, device=device)

            coords_expanded = base_coords.expand(n_ops, *grid_shape, 3)
            transformed_coords = torch.einsum(
                "nij,b...j->n...i", self.R_matrices, coords_expanded
            )

            transformed_coords = transformed_coords + (
                self.t_vectors.view(n_ops, 1, 1, 1, 3)
                * grid_shape_tensor.view(1, 1, 1, 1, 3)
            )

            transformed_coords = transformed_coords % grid_shape_tensor.view(
                1, 1, 1, 1, 3
            )
            normalized_coords = (
                transformed_coords / (grid_shape_tensor.view(1, 1, 1, 1, 3) - 1)
            ) * 2 - 1

            for b in range(batch_size):
                normalized_coords_batch = normalized_coords.view(
                    n_ops, -1, grid_shape[1], grid_shape[2], 3
                )

                transformed_density = F.grid_sample(
                    density[b : b + 1, None].expand(n_ops, 1, *grid_shape),
                    normalized_coords_batch,
                    mode="bilinear",
                    align_corners=True,
                    padding_mode="border",
                )

                output[b] += transformed_density.sum(dim=0)[0]

        if normalize:
            output = output / n_ops

        return output


class DifferentiableTransformer(torch.nn.Module):
    """Differentiable transformation of atomic coordinates to electron density.

    Implements a fully differentiable pipeline for converting atomic coordinates
    to electron density maps with crystallographic symmetry operations. Supports
    both X-ray and electron microscopy modes with flexible parameter configuration.
    """

    def __init__(
        self,
        xmap: XMap_torch,
        scattering_params: torch.Tensor,
        density_params: Optional[DensityParameters] = None,
        em: bool = False,
        space_group: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize differentiable transformer.

        Parameters
        ----------
        xmap : DifferentiableXMap
            Differentiable XMap object.
        scattering_params : torch.Tensor
            Atomic scattering parameters for each element, of shape [n_elem, n_coeffs, 2].
        density_params : Optional[DensityParameters], optional
            Parameters for density calculation, by default None.
        em : bool, optional
            Whether to use electron microscopy mode, by default False.
        space_group : Optional[int], optional
            Space group number, by default None.
        device : torch.device, optional
            Device to use for computations, by default 'cpu'.
        """
        super().__init__()
        self.device = device
        self.xmap = xmap
        self.unit_cell = xmap.unit_cell
        if space_group is not None:
            self.unit_cell.space_group = GetSpaceGroup(space_group)

        self.register_buffer("voxelspacing", xmap.voxelspacing)
        self.grid_shape = xmap.shape
        self.scattering_params = scattering_params.to(device)
        self.density_params = density_params or DensityParameters()
        self.em = em

        self.xmap = xmap

        self._setup_transforms()
        self._setup_integrator()

    def _setup_transforms(self) -> None:
        """Initialize transformation matrices for coordinate conversions."""
        self.dtype = (
            torch.float32
        )  # need to set this here or else doubles start popping up and ruining operations

        abc = torch.tensor(self.unit_cell.abc, dtype=self.dtype, device=self.device)

        lattice_to_cartesian = self._unit_cell_to_cartesian_matrix()
        self.register_buffer(
            "lattice_to_cartesian",
            (lattice_to_cartesian / abc.reshape(3, 1)).to(dtype=self.dtype),
        )

        self.register_buffer(
            "cartesian_to_lattice",
            torch.inverse(self.lattice_to_cartesian).to(dtype=self.dtype),
        )

        voxelspacing = self.voxelspacing.to(dtype=self.dtype, device=self.device)
        self.register_buffer(
            "grid_to_cartesian",
            (self.lattice_to_cartesian * voxelspacing.reshape(3, 1)).to(
                dtype=self.dtype
            ),
        )

    def _setup_integrator(self) -> None:
        """Set up numerical integrator based on density parameters."""
        # TODO: get torchquad intermediates on the right device for sure, might require contributing to torchquad
        set_up_backend("torch", "float32", True)
        if self.density_params.integration_method.lower() == "gausslegendre":
            self.integrator = GaussLegendre()
        elif self.density_params.integration_method.lower() == "simpson":
            self.integrator = Simpson()
        elif self.density_params.integration_method.lower() == "trapezoid":
            self.integrator = Trapezoid()
        else:
            raise ValueError(
                f"Unsupported integration method: {self.density_params.integration_method}"
            )

    def forward(
        self,
        coordinates: torch.Tensor,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        occupancies: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing electron density with symmetry.

        Parameters
        ----------
        coordinates : torch.Tensor
            Atomic coordinates of shape (batch_size, n_atoms, 3).
        elements : torch.Tensor
            Element indices of shape (batch_size, n_atoms).
        b_factors : torch.Tensor
            B-factors of shape (batch_size, n_atoms).
        occupancies : torch.Tensor
            Occupancies of shape (batch_size, n_atoms).

        Returns
        -------
        torch.Tensor
            Symmetry-expanded density grid of shape (batch_size, *grid_shape).
        """
        if not (
            coordinates.shape[0]
            == elements.shape[0]
            == b_factors.shape[0]
            == occupancies.shape[0]
        ):
            raise ValueError("Batch sizes must match for all inputs")

        # convert to device
        coordinates = coordinates.to(self.device)
        elements = elements.to(self.device)
        b_factors = b_factors.to(self.device)
        occupancies = occupancies.to(self.device)

        batch_size = coordinates.shape[0]

        radial_densities = self._compute_radial_densities(elements, b_factors)

        density = torch.zeros(
            (batch_size,) + self.grid_shape, device=self.device, dtype=coordinates.dtype
        )

        grid_coordinates = self._compute_grid_coordinates(coordinates).to(
            dtype=torch.float32
        )

        lmax = torch.tensor(
            [self.density_params.rmax / vs for vs in self.voxelspacing],
            device=self.device,
        )
        active = torch.ones_like(elements, dtype=torch.bool)

        for i in range(len(self.xmap.R_matrices)):
            R = self.xmap.R_matrices[i]
            t = self.xmap.t_vectors[i]

            grid_coordinates_rot = torch.matmul(grid_coordinates, R.transpose(-2, -1))
            grid_coordinates_rot = grid_coordinates_rot + t.view(
                1, 1, 3
            ) * torch.tensor(
                self.grid_shape, device=self.device, dtype=coordinates.dtype
            ).view(
                1, 1, 3
            )

            density += dilate_points_torch(
                grid_coordinates_rot,
                active,
                occupancies,
                lmax,
                radial_densities,
                self.density_params.rstep,
                self.density_params.rmax,
                self.grid_to_cartesian,
                self.grid_shape,
            )

        return density

    def _compute_radial_densities(
        self, elements: torch.Tensor, b_factors: torch.Tensor
    ) -> torch.Tensor:
        """Compute radial densities using numerical integration.

        Parameters
        ----------
        elements : torch.Tensor
            Element indices of shape (batch_size, n_atoms).
        b_factors : torch.Tensor
            B-factors of shape (batch_size, n_atoms).

        Returns
        -------
        torch.Tensor
            Radial densities of shape (batch_size, n_atoms, n_radial).
        """
        r = torch.arange(
            0,
            self.density_params.rmax + self.density_params.rstep,
            self.density_params.rstep,
            device=self.device,
        )

        batch_size, n_atoms = elements.shape
        n_radial = r.shape[0]

        elements_flat = elements.reshape(-1)
        b_factors_flat = b_factors.reshape(-1)

        combined = torch.stack([elements_flat, b_factors_flat], dim=1)
        unique_combinations, inverse_indices = torch.unique(
            combined, dim=0, return_inverse=True
        )

        unique_elements = unique_combinations[:, 0].long()
        element_params = self.scattering_params[
            unique_elements
        ]  # Shape: [n_unique, n_coeffs, 2]
        unique_bfactors = unique_combinations[:, 1]

        def compute_batched_density(
            params: torch.Tensor, bfac: torch.Tensor
        ) -> torch.Tensor:
            """Compute density for a batch of parameters and b-factors."""
            integrand = ScatteringIntegrand(
                params.unsqueeze(0), bfac.reshape(-1), em=self.em
            )

            result = self.integrator.integrate(
                lambda s: integrand(s, r),
                dim=1,
                N=self.density_params.quad_points,
                integration_domain=torch.tensor(
                    [[self.density_params.smin, self.density_params.smax]],
                    device=self.device,
                ),
                backend="torch",
            )

            return result[0]  # Shape is (n_radial,)

        # Create a vmap that processes all unique combinations at once
        batched_compute = torch.vmap(compute_batched_density)
        all_unique_densities = batched_compute(element_params, unique_bfactors)

        densities = all_unique_densities[inverse_indices].reshape(
            batch_size, n_atoms, n_radial
        )

        return densities

    def _compute_radial_derivatives(
        self, elements: torch.Tensor, b_factors: torch.Tensor
    ) -> torch.Tensor:
        """Compute radial density derivatives efficiently.

        Parameters
        ----------
        elements : torch.Tensor
            Element indices of shape (batch_size, n_atoms).
        b_factors : torch.Tensor
            B-factors of shape (batch_size, n_atoms).

        Returns
        -------
        torch.Tensor
            Radial density derivatives of shape (batch_size, n_atoms, n_radial).
        """
        densities = self._compute_radial_densities(elements, b_factors)

        # Calculate gradients using finite differences for better efficiency
        # This computes gradients for all atoms in all batches at once
        padding = 1  # For central difference approximation
        padded = F.pad(densities, (padding, padding), mode="replicate")

        # Central difference formula: (f(x+h) - f(x-h))/(2h)
        derivatives = (padded[:, :, 2:] - padded[:, :, :-2]) / (
            2 * self.density_params.rstep
        )

        return derivatives

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
        grid_coordinates /= self.voxelspacing.to(self.device)

        if hasattr(self.xmap, "offset"):
            grid_coordinates -= self.xmap.offset.to(
                device=self.device, dtype=coordinates.dtype
            )

        return grid_coordinates

    def _unit_cell_to_cartesian_matrix(self) -> torch.Tensor:
        """Compute transformation matrix from unit cell to Cartesian coordinates.

        Returns
        -------
        torch.Tensor
            Transformation matrix of shape (3, 3).
        """
        a, b, c = self.unit_cell.abc
        alpha, beta, gamma = map(
            np.deg2rad,
            [self.unit_cell.alpha, self.unit_cell.beta, self.unit_cell.gamma],
        )

        a = torch.tensor(a, device=self.device)
        b = torch.tensor(b, device=self.device)
        c = torch.tensor(c, device=self.device)
        alpha = torch.tensor(alpha, device=self.device)
        beta = torch.tensor(beta, device=self.device)
        gamma = torch.tensor(gamma, device=self.device)

        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        volume_term = torch.sqrt(
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )

        matrix = torch.zeros((3, 3), device=self.device)
        matrix[0, 0] = a
        matrix[0, 1] = b * cos_gamma
        matrix[0, 2] = c * cos_beta
        matrix[1, 1] = b * sin_gamma
        matrix[1, 2] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        matrix[2, 2] = (a * b * c * volume_term) / (a * b * sin_gamma)

        return matrix


def dilate_points_torch(
    coordinates: torch.Tensor,
    active: torch.Tensor,
    occupancies: torch.Tensor,
    lmax: torch.Tensor,
    radial_densities: torch.Tensor,
    rstep: float,
    rmax: float,
    grid_to_cartesian: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    chunk_size: int = 256,
) -> torch.Tensor:
    """Point dilation onto a grid across batches.

    Parameters
    ----------
    coordinates : torch.Tensor
        Atomic coordinates of shape (batch_size, n_atoms, 3).
    active : torch.Tensor
        Boolean mask of active atoms of shape (batch_size, n_atoms).
    occupancies : torch.Tensor
        Occupancies of shape (batch_size, n_atoms).
    lmax : torch.Tensor
        Maximum distances in grid units of shape (3,).
    radial_densities : torch.Tensor
        Precomputed radial densities of shape (batch_size, n_atoms, n_radial).
    rstep : float
        Step size for radial grid.
    rmax : float
        Maximum radius for density calculation.
    grid_to_cartesian : torch.Tensor
        Transformation matrix from grid to Cartesian of shape (3, 3).
    grid_shape : Tuple[int, int, int]
        Output grid shape (int, int, int).
    chunk_size : int, optional
        Number of atoms to process in each chunk, by default 256.

    Returns
    -------
    torch.Tensor
        Density grid.
    """
    device = coordinates.device
    dtype = coordinates.dtype
    batch_size, n_atoms = coordinates.shape[:2]
    rmax2 = rmax * rmax

    g00, g01, g02 = (
        grid_to_cartesian[0, 0],
        grid_to_cartesian[0, 1],
        grid_to_cartesian[0, 2],
    )
    g11, g12 = grid_to_cartesian[1, 1], grid_to_cartesian[1, 2]
    g22 = grid_to_cartesian[2, 2]

    result = torch.zeros((batch_size,) + grid_shape, device=device, dtype=dtype)

    for b in range(batch_size):
        for atom_idx in range(n_atoms):
            if not active[b, atom_idx]:
                continue

            q = occupancies[b, atom_idx]
            center_a, center_b, center_c = coordinates[b, atom_idx]

            cmin = torch.ceil(center_c - lmax[2]).long()
            cmax = torch.floor(center_c + lmax[2]).long()
            bmin = torch.ceil(center_b - lmax[1]).long()
            bmax = torch.floor(center_b + lmax[1]).long()
            amin = torch.ceil(center_a - lmax[0]).long()
            amax = torch.floor(center_a + lmax[0]).long()

            if cmax < cmin or bmax < bmin or amax < amin:
                continue

            c_coords = torch.arange(cmin, cmax + 1, device=device, dtype=dtype)
            b_coords = torch.arange(bmin, bmax + 1, device=device, dtype=dtype)
            a_coords = torch.arange(amin, amax + 1, device=device, dtype=dtype)

            grid_c, grid_b, grid_a = torch.meshgrid(
                c_coords, b_coords, a_coords, indexing="ij"
            )

            dc = center_c - grid_c
            db = center_b - grid_b
            da = center_a - grid_a

            dz = g22 * dc
            dy = g12 * dc + g11 * db
            dx = g02 * dc + g01 * db + g00 * da

            d2_zyx = dx * dx + dy * dy + dz * dz

            mask = d2_zyx <= rmax2
            if not torch.any(mask):
                continue

            r = torch.sqrt(d2_zyx[mask])

            # Differentiable interpolation into radial_densities
            rad_continuous = r / rstep
            rad_indices_low = torch.floor(rad_continuous).long()
            rad_indices_high = rad_indices_low + 1

            max_idx = radial_densities.shape[2] - 1
            rad_indices_low = torch.clamp(rad_indices_low, 0, max_idx)
            rad_indices_high = torch.clamp(rad_indices_high, 0, max_idx)

            weights_high = rad_continuous - rad_indices_low.float()
            weights_low = 1.0 - weights_high

            atom_density = radial_densities[b, atom_idx]
            values = q * (
                weights_low * atom_density[rad_indices_low]
                + weights_high * atom_density[rad_indices_high]
            )

            # Apply periodic boundary conditions
            c_indices = torch.remainder(grid_c[mask], grid_shape[0]).long()
            b_indices = torch.remainder(grid_b[mask], grid_shape[1]).long()
            a_indices = torch.remainder(grid_a[mask], grid_shape[2]).long()

            result[
                b, c_indices, b_indices, a_indices
            ] += values  # += is automatically scatter_add_

    return result


def normalize(t: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to a Gaussian with mean 0 and std dev 1."""
    if t.numel() == 0 or t.ndim == 0:
        return t

    if t.dtype not in [torch.complex32, torch.complex64, torch.complex128]:
        return (t - t.mean()) / (t.std(unbiased=False) + 1e-8)

    real_part = (t.real - t.real.mean()) / (t.real.std(unbiased=False) + 1e-8)
    imag_part = (t.imag - t.imag.mean()) / (t.imag.std(unbiased=False) + 1e-8)

    return torch.view_as_complex(
        torch.cat([real_part[..., None], imag_part[..., None]], -1)
    )


def to_f_density(map: torch.Tensor) -> torch.Tensor:
    """FFT a density map."""
    # f_density
    return torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(map, dim=(-3, -2, -1)), dim=(-3, -2, -1), norm="ortho"
        ),
        dim=(-3, -2, -1),
    )
