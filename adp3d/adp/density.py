import torch
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import torch.nn.functional as F
from torchquad import Simpson, Trapezoid
import numpy as np

from adp3d.qfit.volume import XMap, EMMap, GridParameters
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
        Integration method to use ('simpson' or 'trapezoid').
    """

    rmax: float = 3.0
    rstep: float = 0.01
    smin: float = 0.0
    smax: float = 0.5
    quad_points: int = 50
    integration_method: str = "simpson"

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.rmax <= 0 or self.rstep <= 0:
            raise ValueError("rmax and rstep must be positive")
        if self.smin >= self.smax:
            raise ValueError("smin must be less than smax")
        if self.quad_points < 2:
            raise ValueError("quad_points must be at least 2")
        if self.integration_method not in ["simpson", "trapezoid"]:
            raise ValueError("integration_method must be 'simpson' or 'trapezoid'")


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
                "constant_term", 
                self.asf[:, 5, 0].view(self.n_atoms, 1, 1)
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
            raise ValueError(f"asf must have shape (n_atoms, n_coeffs, 2), got {asf.shape}")
        
        if bfactor.dim() != 1:
            raise ValueError(f"bfactor must be 1-dimensional, got {bfactor.dim()}-dimensional")
            
        if not self.em and asf.shape[1] < 6:
            raise ValueError(f"For X-ray scattering, exactly 6 coefficients required, got {asf.shape[1]}")
        
        if self.em and asf.shape[1] < 5:
            raise ValueError(f"For electron scattering, at least 5 coefficients required, got {asf.shape[1]}")

    def compute_scattering_factors(
        self, 
        s2: torch.Tensor
    ) -> torch.Tensor:
        """Compute atomic scattering factors for given squared scattering vectors.
        
        Parameters
        ----------
        s2 : torch.Tensor
            Squared scattering vector magnitudes.
            
        Returns
        -------
        torch.Tensor
            Computed scattering factors.
        """
        batch_shape = s2.shape
        
        if self.em:
            f = torch.zeros(
                (self.n_atoms,) + batch_shape, 
                device=s2.device, 
                dtype=s2.dtype
            )
        else:
            f = self.constant_term.expand((self.n_atoms,) + batch_shape).clone()
        
        for i in range(self.asf_range):
            a_coeff = self.asf[:, i, 0].view(-1, *(1,)*len(batch_shape))
            b_coeff = self.asf[:, i, 1].view(-1, *(1,)*len(batch_shape))
            f += a_coeff * torch.exp(-b_coeff * s2.view((1,) + batch_shape))
            
        return f

    def forward(
        self, 
        s: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the scattering integrand for torchquad integration.
        
        Parameters
        ----------
        s : torch.Tensor
            Scattering vector magnitudes from torchquad.
        r : torch.Tensor
            Radial distances to use for computing the density.
            
        Returns
        -------
        torch.Tensor
            Computed integrand values.
        """
        
        s_expanded = s.unsqueeze(-1) # (n_s, 1)
        r_expanded = r.view(1, -1)  # (1, n_r)
        
        s2 = s_expanded * s_expanded
        
        f = self.compute_scattering_factors(s2)  # (n_atoms, n_s, 1)
        
        four_pi_s = 4 * torch.pi * s_expanded  # (n_s, 1)
        w = 8 * f * torch.exp(-self.bfactor * s2) * s_expanded  # (n_atoms, n_s, 1)
        
        eps = 1e-4
        r_small_mask = r_expanded < eps
        
        result = torch.zeros(
            (self.n_atoms, s.shape[0], r.shape[0]), 
            device=s.device, 
            dtype=s.dtype
        )
        
        # prevent singularity with 4th order Taylor expansion
        if r_small_mask.any():
            ar_small = four_pi_s * r_expanded[:, r_small_mask[0]]
            ar2_small = ar_small * ar_small
            taylor_term = 1.0 - ar2_small / 6.0
            result[:, :, r_small_mask[0]] = w * four_pi_s * taylor_term
        
        r_large_mask = ~r_small_mask
        if r_large_mask.any():
            ar_large = four_pi_s * r_expanded[:, r_large_mask[0]]
            sin_term = torch.sin(ar_large)
            result[:, :, r_large_mask[0]] = w * sin_term / r_expanded[:, r_large_mask[0]]
        
        # (n_atoms, n_s, n_r) to (n_s, n_atoms, n_r)
        result = result.permute(1, 0, 2)
        
        return result


class DifferentiableXMap(torch.nn.Module):
    """Differentiable version of XMap for handling crystallographic symmetry.

    Implements a vectorized approach to applying symmetry operations across batches
    of electron density maps while maintaining differentiability.
    """

    def __init__(
        self,
        xmap: XMap = None,
        grid_parameters: GridParameters = None,
        unit_cell: Optional[UnitCell] = None,
        resolution: Optional[float] = None,
        hkl: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize differentiable XMap.

        Parameters
        ----------
        xmap : XMap
            Initialize from qFit XMap (preferred).
        grid_parameters : GridParameters
            Grid parameters for the map, meaning voxel spacing (voxelspacing) and offset.
        unit_cell : UnitCell
            Crystallographic unit cell information.
        resolution : Optional[float], optional
            Map resolution in Angstroms, by default None.
        hkl : Optional[torch.Tensor], optional
            Miller indices for the map, by default None.
        device : torch.device, optional
            Device to use for computations, by default 'cpu'.
        """
        super().__init__()
        if xmap is not None:
            self.unit_cell = xmap.unit_cell
            self.resolution = xmap.resolution
            self.hkl = xmap.hkl

            self.register_buffer(
                "voxel_spacing", torch.tensor(xmap.voxelspacing, device=device)
            )
            self.register_buffer("offset", torch.tensor(xmap.offset, device=device))
        else:
            self.unit_cell = unit_cell
            self.resolution = resolution
            self.hkl = hkl

            self.register_buffer(
                "voxel_spacing",
                torch.tensor(grid_parameters.voxelspacing, device=device),
            )
            self.register_buffer(
                "offset", torch.tensor(grid_parameters.offset, device=device)
            )

        self._setup_symmetry_matrices(device)

    def _setup_symmetry_matrices(self, device: torch.device) -> None:
        """Precompute symmetry operation matrices for efficient application."""
        symops = self.unit_cell.space_group.symop_list
        n_ops = len(symops)

        R_matrices = torch.zeros((n_ops, 3, 3), device=device)
        t_vectors = torch.zeros((n_ops, 3), device=device)

        for i, symop in enumerate(symops):
            R_matrices[i] = torch.tensor(symop.R, device=device, dtype=torch.float32)
            t_vectors[i] = torch.tensor(symop.t, device=device, dtype=torch.float32)

        self.register_buffer("R_matrices", R_matrices)
        self.register_buffer("t_vectors", t_vectors)

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

        # Create output tensor
        output = density.clone()

        # Memory-efficient implementation
        if memory_efficient or chunk_size is not None:
            # Process in chunks to save memory
            chunk_size = chunk_size or max(1, n_ops // 4)

            for chunk_start in range(0, n_ops, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_ops)
                self._apply_symmetry_chunk(
                    density, output, chunk_start, chunk_end, grid_shape, batch_size
                )
        else:
            # Original implementation (process all at once)
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
        unit_cell: UnitCell,
        voxel_spacing: Union[torch.Tensor, np.ndarray],
        grid_shape: Tuple[int, int, int],
        scattering_params: Dict[str, torch.Tensor],
        density_params: Optional[DensityParameters] = None,
        em: bool = False,
        space_group: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize differentiable transformer.

        Parameters
        ----------
        unit_cell : UnitCell
            Crystallographic unit cell information.
        voxel_spacing : Union[torch.Tensor, np.ndarray]
            Grid spacing in each dimension.
        grid_shape : Tuple[int, int, int]
            Shape of the density grid.
        scattering_params : Dict[str, torch.Tensor]
            Atomic scattering parameters for each element.
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
        self.unit_cell = unit_cell
        if space_group is not None:
            self.unit_cell.space_group = GetSpaceGroup(space_group)

        if not isinstance(voxel_spacing, torch.Tensor):
            voxel_spacing = torch.tensor(voxel_spacing, device=device)

        self.register_buffer("voxel_spacing", voxel_spacing)
        self.grid_shape = grid_shape
        self.scattering_params = {k: v.to(device) for k, v in scattering_params.items()}
        self.density_params = density_params or DensityParameters()
        self.em = em

        grid_parameters = GridParameters(
            voxelspacing=voxel_spacing.cpu().numpy(), offset=(0, 0, 0)
        )
        self.xmap = DifferentiableXMap(
            grid_parameters=grid_parameters, unit_cell=unit_cell, device=device
        )

        self._setup_transforms()
        self._setup_integrator()

    def _setup_transforms(self) -> None:
        """Initialize transformation matrices for coordinate conversions."""
        abc = self.unit_cell.abc
        self.register_buffer(
            "lattice_to_cartesian",
            self._unit_cell_to_cartesian_matrix() / torch.tensor(abc).reshape(3, 1),
        )
        self.register_buffer(
            "cartesian_to_lattice", torch.inverse(self.lattice_to_cartesian)
        )
        self.register_buffer(
            "grid_to_cartesian",
            self.lattice_to_cartesian * self.voxel_spacing.reshape(3, 1),
        )

    def _setup_integrator(self) -> None:
        """Set up numerical integrator based on density parameters."""
        if self.density_params.integration_method.lower() == "simpson": # TODO: check if these are sufficient, compare to normal qFit
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
        normalize: bool = True,
        chunk_size: Optional[int] = None,
        memory_efficient: bool = True,
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
        normalize : bool, optional
            Whether to normalize the output density, by default True.
        chunk_size : Optional[int], optional
            Number of symmetry operations to process at once, by default None.
        memory_efficient : bool, optional
            Whether to use memory-efficient implementation, by default True.

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

        initial_density = self._compute_density(
            coordinates, elements, b_factors, occupancies
        )

        symmetrized_density = self.xmap.apply_symmetry(
            initial_density,
            normalize=normalize,
            chunk_size=chunk_size,
            memory_efficient=memory_efficient,
        )

        return symmetrized_density

    def _compute_density(
        self,
        coordinates: torch.Tensor,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        occupancies: torch.Tensor,
    ) -> torch.Tensor:
        """Compute electron density without symmetry operations.

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
            Electron density grid of shape (batch_size, *grid_shape).
        """
        batch_size = coordinates.shape[0]
        radial_densities = self._compute_radial_densities(elements, b_factors)
        grid_coordinates = self._compute_grid_coordinates(coordinates)

        density = torch.zeros(
            (batch_size,) + self.grid_shape,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

        lmax = torch.tensor(
            [self.density_params.rmax / vs for vs in self.voxel_spacing],
            device=coordinates.device,
        )

        # Process all atoms in all batches at once for better parallelism
        active = torch.ones_like(elements, dtype=torch.bool)
        dilate_points_torch(
            grid_coordinates,
            active,
            occupancies,
            lmax,
            radial_densities,
            self.density_params.rstep,
            self.density_params.rmax,
            self.grid_to_cartesian,
            density,
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
            device=elements.device,
        )

        batch_size, n_atoms = elements.shape
        n_radial = r.shape[0]

        densities = torch.zeros((batch_size, n_atoms, n_radial), device=elements.device)

        unique_elements = torch.unique(elements)
        for elem in unique_elements:
            batch_indices, atom_indices = torch.where(elements == elem)
            if batch_indices.numel() == 0:
                continue

            b_factor_values = b_factors[batch_indices, atom_indices]
            unique_b_factors = torch.unique(b_factor_values)

            for b_factor in unique_b_factors:
                mask = (elements == elem) & (b_factors == b_factor)
                if not mask.any():
                    continue

                b_batch_indices, b_atom_indices = torch.where(mask)

                asf = self.scattering_params[elem.item()]
                asf_expanded = asf.expand(b_batch_indices.size(0), -1, -1)

                b_factor_tensor = torch.full(
                    (b_batch_indices.size(0),), b_factor, device=elements.device
                )

                integrand = ScatteringIntegrand(
                    asf_expanded, b_factor_tensor, em=self.em
                )

                result = self.integrator.integrate( # FIXME: This isn't working right
                    lambda s: integrand(s, r),
                    dim=1,
                    N=self.density_params.quad_points,
                    integration_domain=[
                        [self.density_params.smin, self.density_params.smax]
                    ],
                )

                # result has shape (n_atoms_in_group, n_radial)
                for i, (b_idx, a_idx) in enumerate(
                    zip(b_batch_indices, b_atom_indices)
                ):
                    densities[b_idx, a_idx] = result[i]

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
        grid_coordinates = torch.matmul(coordinates, self.cartesian_to_lattice.T)
        grid_coordinates /= self.voxel_spacing
        return grid_coordinates

    def _unit_cell_to_cartesian_matrix(self) -> torch.Tensor:
        """Compute transformation matrix from unit cell to Cartesian coordinates.

        Returns
        -------
        torch.Tensor
            Transformation matrix of shape (3, 3).
        """
        a, b, c = self.unit_cell.abc
        alpha, beta, gamma = map(np.deg2rad, [self.unit_cell.alpha, self.unit_cell.beta, self.unit_cell.gamma])

        # Convert to tensors
        a = torch.tensor(a, device=self.voxel_spacing.device)
        b = torch.tensor(b, device=self.voxel_spacing.device)
        c = torch.tensor(c, device=self.voxel_spacing.device)
        alpha = torch.tensor(alpha, device=self.voxel_spacing.device)
        beta = torch.tensor(beta, device=self.voxel_spacing.device)
        gamma = torch.tensor(gamma, device=self.voxel_spacing.device)

        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        # Calculate volume term
        volume_term = torch.sqrt(
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )

        # Create transformation matrix
        matrix = torch.zeros((3, 3), device=self.voxel_spacing.device)
        matrix[0, 0] = a
        matrix[0, 1] = b * cos_gamma
        matrix[0, 2] = c * cos_beta
        matrix[1, 1] = b * sin_gamma
        matrix[1, 2] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        matrix[2, 2] = (a * b * c * volume_term) / (a * b * sin_gamma)

        return matrix

    def _unit_cell_to_cartesian_matrix_batched(
        self, unit_cells: List[UnitCell]
    ) -> torch.Tensor:
        """Compute transformation matrices for multiple unit cells.

        Parameters
        ----------
        unit_cells : List[UnitCell]
            List of unit cells for each batch item.

        Returns
        -------
        torch.Tensor
            Batch of transformation matrices of shape (batch_size, 3, 3).
        """
        batch_size = len(unit_cells)
        matrices = torch.zeros((batch_size, 3, 3), device=self.voxel_spacing.device)

        for i, uc in enumerate(unit_cells):
            a, b, c = uc.abc
            alpha, beta, gamma = map(np.deg2rad, uc.angles)

            # Convert to tensors
            a = torch.tensor(a, device=self.voxel_spacing.device)
            b = torch.tensor(b, device=self.voxel_spacing.device)
            c = torch.tensor(c, device=self.voxel_spacing.device)
            alpha = torch.tensor(alpha, device=self.voxel_spacing.device)
            beta = torch.tensor(beta, device=self.voxel_spacing.device)
            gamma = torch.tensor(gamma, device=self.voxel_spacing.device)

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

            matrix = torch.zeros((3, 3), device=self.voxel_spacing.device)
            matrix[0, 0] = a
            matrix[0, 1] = b * cos_gamma
            matrix[0, 2] = c * cos_beta
            matrix[1, 1] = b * sin_gamma
            matrix[1, 2] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
            matrix[2, 2] = (a * b * c * volume_term) / (a * b * sin_gamma)

            matrices[i] = matrix

        return matrices


def dilate_points_torch(
    coordinates: torch.Tensor,
    active: torch.Tensor,
    occupancies: torch.Tensor,
    lmax: torch.Tensor,
    radial_densities: torch.Tensor,
    rstep: float,
    rmax: float,
    grid_to_cartesian: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Point dilation across batches.

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
    out : torch.Tensor
        Output density grid of shape (batch_size, *grid_shape).

    Returns
    -------
    torch.Tensor
        Updated density grid.
    """
    device = coordinates.device
    batch_size, n_atoms = coordinates.shape[:2]
    rmax2 = rmax * rmax
    grid_shape = out.shape[1:]  # Assuming out shape is (batch, z, y, x)

    # Flatten batch and atom dimensions for vectorized processing
    # Process only active atoms
    flat_mask = active.view(-1)
    active_indices = torch.nonzero(flat_mask, as_tuple=True)[0]

    if len(active_indices) == 0:
        return out

    # Get batch and atom indices
    batch_indices = active_indices // n_atoms
    atom_indices = active_indices % n_atoms

    # Get coordinates, occupancies, and radial densities for active atoms
    active_coords = coordinates[batch_indices, atom_indices]
    active_occupancies = occupancies[batch_indices, atom_indices]
    active_radial_densities = radial_densities[batch_indices, atom_indices]

    # Create voxel grid for each active atom
    bounds_min = torch.ceil(active_coords - lmax.unsqueeze(0)).long()
    bounds_max = torch.floor(active_coords + lmax.unsqueeze(0)).long()

    # Process each active atom (this loop is unavoidable due to varying grid sizes)
    for i, (b_idx, a_idx) in enumerate(zip(batch_indices, atom_indices)):
        center = active_coords[i]
        b_min, b_max = bounds_min[i], bounds_max[i]

        # Skip if any dimension is empty
        if any((b_max - b_min + 1) <= 0):
            continue

        # Create grid coordinates
        z_range = torch.arange(b_min[0], b_max[0] + 1, device=device)
        y_range = torch.arange(b_min[1], b_max[1] + 1, device=device)
        x_range = torch.arange(b_min[2], b_max[2] + 1, device=device)

        grid_z, grid_y, grid_x = torch.meshgrid(
            z_range, y_range, x_range, indexing="ij"
        )
        grid_points = torch.stack([grid_z, grid_y, grid_x], dim=-1).reshape(-1, 3)

        # Compute distances in one vectorized operation
        rel_vectors = grid_points - center
        cart_vectors = torch.matmul(rel_vectors, grid_to_cartesian.T)
        distances2 = torch.sum(cart_vectors**2, dim=1)

        # Apply density only to points within range
        mask = distances2 <= rmax2
        if not mask.any():
            continue

        # Get indices and values in one operation
        grid_points_in_range = grid_points[mask]
        distances = torch.sqrt(distances2[mask])
        rad_indices = torch.clamp(
            (distances / rstep).long(), 0, active_radial_densities.shape[1] - 1
        )

        # Compute density values
        density_values = active_radial_densities[i, rad_indices] * active_occupancies[i]

        # Map to output grid with proper periodic boundary conditions
        z_indices = torch.remainder(grid_points_in_range[:, 0], grid_shape[0])
        y_indices = torch.remainder(grid_points_in_range[:, 1], grid_shape[1])
        x_indices = torch.remainder(grid_points_in_range[:, 2], grid_shape[2])

        # Use scatter_add for parallel updates
        out[b_idx, z_indices, y_indices, x_indices] += density_values

    return out


def to_f_density(map: torch.Tensor) -> torch.Tensor:
    """FFT a density map."""
    # f_density
    return torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(map, dim=(-3, -2, -1)), dim=(-3, -2, -1), norm="ortho"
        ),
        dim=(-3, -2, -1),
    )


def to_density(f_map: torch.Tensor) -> torch.Tensor:
    """Inverse FFT a FFTed density map."""
    # density
    return torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(f_map, dim=(-3, -2, -1)), dim=(-3, -2, -1), norm="ortho"
        ),
        dim=(-3, -2, -1),
    )


def radial_hamming_3d(f_mag: torch.Tensor, cutoff_radius: float) -> torch.Tensor:
    """3D radial Hamming filter in Fourier space

    Args:
        f_mag: Frequency magnitudes from FFT
        cutoff_radius: Frequency cutoff in same units as frequency coordinates

    Returns:
        3D tensor containing the Hamming filter
    """
    filter = torch.zeros_like(f_mag)

    mask = f_mag <= cutoff_radius

    r_scaled = f_mag[mask] / cutoff_radius  # Scale to [0,1]
    hamming_vals = 0.54 + 0.46 * torch.cos(torch.pi * r_scaled)

    filter[mask] = hamming_vals

    return filter


def downsample_fft(
    fft: torch.Tensor,
    original_pixel_size: Union[torch.Tensor, tuple],
    target_pixel_size: Union[torch.Tensor, tuple],
) -> torch.Tensor:
    """
    Downsample density map in Fourier space # TODO: MAKE BATCHABLE

    Args:
        fft: torch.Tensor
            Fourier transform (assumed to be fftshifted) of density map
        original_pixel_size: torch.Tensor or Tuple
            Current sampling rate in each dimension (e.g. 1/2 Å)
        target_pixel_size: torch.Tensor or Tuple
            Desired sampling rate in each dimension (e.g. 2.5 Å)
    """
    if isinstance(original_pixel_size, torch.Tensor) and isinstance(
        target_pixel_size, torch.Tensor
    ):
        downsample_factor = target_pixel_size / original_pixel_size
    elif isinstance(original_pixel_size, tuple) and isinstance(
        target_pixel_size, tuple
    ):
        downsample_factor = torch.tensor(
            tuple(t / o for t, o in zip(target_pixel_size, original_pixel_size))
        )
    else:
        raise ValueError(
            "original_pixel_size and target_pixel_size must be either both torch.Tensor or both tuple"
        )

    original_shape = torch.tensor(fft.shape)
    new_shape = (original_shape / downsample_factor).long()

    if torch.any(new_shape <= 0):
        raise ValueError("Downsampling would result in invalid dimensions")

    crops = []
    for size, orig_size in zip(new_shape, original_shape):
        start = ((orig_size - size) // 2).item()
        end = start + size.item()
        crops.append(slice(start, end))

    downsampled_fft = fft[tuple(crops)]

    return downsampled_fft


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
