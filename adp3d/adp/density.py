import torch
from typing import Optional, Tuple, Union
from copy import deepcopy
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
import warnings
from einops import rearrange

from adp3d.qfit.volume import XMap, GridParameters, Resolution
from adp3d.qfit.unitcell import UnitCell
from adp3d.qfit.spacegroups import GetSpaceGroup
from adp3d.utils.quadrature import GaussLegendreQuadrature


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
        Maximum scattering vector magnitude in inverse Angstroms. Default is based on 1.0 Å.
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
            self.array = torch.tensor(xmap.array, device=device)
            self.shape = xmap.shape

            self.voxelspacing = torch.tensor(xmap.voxelspacing, device=device)
            self.offset = torch.tensor(xmap.offset, device=device)
            self.xmap = xmap
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

            self.xmap = XMap(array, grid_parameters, unit_cell, resolution, hkl, origin)

        self._validate_input(xmap)

        self._setup_symmetry_matrices(device)

    def _validate_input(self, xmap) -> None:
        """Validate input parameters."""
        if xmap is not None:
            if not isinstance(xmap, XMap):
                raise ValueError(
                    "xmap must be an instance of qFit's XMap (adp3d.qfit.volume.XMap)"
                )
            if not hasattr(self.resolution, "high"):
                warnings.warn(
                    f"resolution is not a Resolution object (is {type(self.resolution)}), using provided resolution as high and 1000 Å as low"
                )
                self.resolution = Resolution(high=self.resolution, low=1000.0)
            if self.resolution.low is None:
                warnings.warn(
                    f"resolution does not have low limit set, using 1000 Å as low"
                )
                self.resolution.low = 1000.0
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
            if not hasattr(self.resolution, "high"):
                warnings.warn(
                    f"resolution is not a Resolution object (is {type(self.resolution)}), using provided resolution as high and 1000 Å as low"
                )
                self.resolution = Resolution(high=self.resolution, low=1000.0)
            if not hasattr(self.resolution, "low") and isinstance(
                self.resolution, Resolution
            ):
                warnings.warn(
                    f"resolution does not have a low attribute, using 1000 Å as low"
                )
                self.resolution.low = 1000.0

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

    def tofile(
        self, filename: str, density: Union[torch.Tensor, NDArray] = None
    ) -> None:
        """Save the map to a file.

        Parameters
        ----------
        filename : str
            Output filename.
        density : Union[torch.Tensor, NDArray], optional
            Density grid to save, by default None.
            If provided, it will be used to update the map array.
        """
        if density is not None:
            if density.shape != self.array.shape:
                raise ValueError(
                    f"Density shape {density.shape} does not match map shape {self.array.shape}"
                )
            if isinstance(density, torch.Tensor):
                density = density.cpu().numpy()
        else:
            density = self.array.cpu().numpy()

        xmap_writer = deepcopy(self.xmap)
        xmap_writer.array = density
        xmap_writer.tofile(filename)

    def apply_symmetry(
        self,
        density: torch.Tensor,
    ) -> torch.Tensor:
        """Apply crystallographic symmetry operations to density maps in batch.

        Parameters
        ----------
        density : torch.Tensor
            Input density grid of shape (batch_size, *grid_shape).

        Returns
        -------
        torch.Tensor
            Symmetry-expanded density grid.
        """
        batch_size = density.shape[0]
        grid_shape = density.shape[1:]
        device = density.device
        n_ops = len(self.R_matrices)

        base_coords = torch.stack(
            torch.meshgrid(
                *[torch.arange(s, device=device) for s in grid_shape], indexing="ij"
            ),
            dim=-1,
        ).float()

        base_coords = base_coords + self.offset
        base_coords = base_coords.reshape(1, -1, 3)  # [1, z*y*x, 3]
        grid_shape_tensor = torch.tensor(grid_shape, device=device)

        # R_matrices: [n_ops, 3, 3], base_coords: [1, z*y*x, 3]
        # Result: [n_ops, z*y*x, 3]
        rotated_coords = torch.einsum("nij,bkj->nki", self.R_matrices, base_coords)

        translated_coords = rotated_coords + self.t_vectors.unsqueeze(
            1
        ) * grid_shape_tensor.unsqueeze(0).unsqueeze(0)

        translated_coords = translated_coords % grid_shape_tensor.unsqueeze(
            0
        ).unsqueeze(0)

        transformed_coords = translated_coords.reshape(n_ops, *grid_shape, 3)

        normalized_coords = (
            transformed_coords
            / (grid_shape_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0) - 1)
        ) * 2 - 1

        transposed_density = density.unsqueeze(0)  # [1, batch_size, z, y, x]

        expanded_density = transposed_density.expand(n_ops, batch_size, *grid_shape)

        transformed_density = F.grid_sample(
            expanded_density,  # [n_ops, batch_size, *grid_shape]
            normalized_coords,  # [n_ops, z, y, x, 3]
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )  # Result: [n_ops, batch_size, *grid_shape]

        summed_density = transformed_density.sum(dim=0)  # [batch_size, *grid_shape]

        return summed_density


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
            NOTE: The indexing on the elements MUST match the element indices in the input to forward!
        density_params : Optional[DensityParameters], optional
            Parameters for density calculation, by default None.
            Min and max scattering vector magnitudes will be updated by resolutions in the XMap.
        em : bool, optional
            Whether to use electron microscopy mode, by default False.
        space_group : Optional[int], optional
            Space group number, by default None (in which case it is expected from the XMap unit cell).
        device : torch.device, optional
            Device to use for computations, by default 'cpu'.
        """
        super().__init__()
        self.device = device
        self.xmap = xmap
        self.unit_cell = xmap.unit_cell
        if space_group is not None:
            self.unit_cell.space_group = GetSpaceGroup(space_group)

        self.grid_shape = xmap.shape

        # TODO: currently this means we're essentially forced to use the existing B-factors
        self.scattering_params = scattering_params.to(device)
        self.density_params = density_params or DensityParameters()
        self.em = em

        self.xmap = xmap

        self.density_params.smax = 1 / (2 * self.xmap.resolution.high)
        self.density_params.smin = 1 / (2 * self.xmap.resolution.low)

        self.integrator = GaussLegendreQuadrature(
            num_points=self.density_params.quad_points,
            device=self.device,
            dtype=self.scattering_params.dtype,
        )

        self._setup_transforms()

    def _setup_transforms(self) -> None:
        """Initialize transformation matrices for coordinate conversions."""
        self.dtype = (
            torch.float32
        )  # need to set this here or else doubles start popping up and ruining operations

        lattice_to_cartesian = (
            self.xmap.unit_cell.frac_to_orth / self.xmap.unit_cell.abc
        )
        cartesian_to_lattice = (
            self.xmap.unit_cell.orth_to_frac * self.xmap.unit_cell.abc.reshape(3, 1)
        )
        grid_to_cartesian = lattice_to_cartesian * self.xmap.voxelspacing.cpu().numpy()
        self.register_buffer(
            "lattice_to_cartesian",
            torch.tensor(lattice_to_cartesian).to(dtype=self.dtype, device=self.device),
        )

        self.register_buffer(
            "cartesian_to_lattice",
            torch.tensor(cartesian_to_lattice).to(dtype=self.dtype, device=self.device),
        )

        self.register_buffer(
            "grid_to_cartesian",
            torch.tensor(grid_to_cartesian).to(dtype=self.dtype, device=self.device),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        occupancies: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing electron density with symmetry.

        Parameters
        ----------
        coordinates : torch.Tensor
            Atomic coordinates of shape (batch_size, n_atoms, 3).
        elements : torch.Tensor
            Element indices of shape (batch_size, n_atoms).
            NOTE: The indexing on the elements MUST match the element indices in the input to forward!
        b_factors : torch.Tensor
            B-factors of shape (batch_size, n_atoms).
        occupancies : torch.Tensor
            Occupancies of shape (batch_size, n_atoms).
        active : torch.Tensor
            Boolean mask of active atoms of shape (batch_size, n_atoms).

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

        coordinates = coordinates.to(self.device)
        elements = elements.to(self.device)
        b_factors = b_factors.to(self.device)
        occupancies = occupancies.to(self.device)
        active = (
            torch.ones_like(elements, dtype=torch.bool, device=self.device)
            if active is None
            else active.to(dtype=torch.bool, device=self.device)
        )

        batch_size = coordinates.shape[0]

        radial_densities = self._compute_radial_densities(elements, b_factors).to(
            self.dtype
        )  # integrator seems to be promoting to float64?

        density = torch.zeros(
            (batch_size,) + self.grid_shape, device=self.device, dtype=coordinates.dtype
        )

        grid_coordinates = self._compute_grid_coordinates(coordinates).to(
            dtype=torch.float32
        )

        lmax = torch.tensor(
            [self.density_params.rmax / vs for vs in self.xmap.voxelspacing],
            device=self.device,
        )

        for i in range(len(self.xmap.R_matrices)):
            R = self.xmap.R_matrices[i]
            t = self.xmap.t_vectors[i]

            grid_coordinates_rot = torch.matmul(grid_coordinates, R.T)
            grid_coordinates_rot = grid_coordinates_rot + t.view(
                1, 1, 3
            ) * torch.tensor(
                self.grid_shape[::-1], device=self.device, dtype=coordinates.dtype
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
        element_asf = self.scattering_params[
            unique_elements
        ]  # Shape: [n_unique, n_coeffs, 2]
        unique_bfactors = unique_combinations[:, 1]

        def integrate_single_element(
            asf: torch.Tensor, bfac: torch.Tensor
        ) -> torch.Tensor:
            """Compute density for a batch of parameters and b-factors."""

            def integrand_fn(s):
                return scattering_integrand(
                    s,
                    r,
                    asf,
                    bfac,
                    em=self.em,
                )

            result = self.integrator(
                integrand_fn,
                integration_limits=torch.tensor(
                    [[self.density_params.smin, self.density_params.smax]],
                    device=self.device,
                ),
                dim=1,
            )

            return result  # Shape: [batch, n_radial]

        # Create a vmap that processes all unique combinations at once
        integrate_elements = torch.vmap(integrate_single_element)
        all_unique_densities = integrate_elements(element_asf, unique_bfactors)

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
        grid_coordinates /= self.xmap.voxelspacing.to(self.device)

        if hasattr(self.xmap, "offset"):
            grid_coordinates -= self.xmap.offset.to(
                device=self.device, dtype=coordinates.dtype
            )

        return grid_coordinates


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
) -> torch.Tensor:
    """Point dilation onto a grid across batches.
    Batch could be either many structures for many density maps,
    or an ensemble, where you would need to sum across the output batch to get a map.

    Parameters
    ----------
    coordinates : torch.Tensor
        Grid coordinates corresponding to atom positions of shape (batch_size, n_atoms, 3).
    active : torch.Tensor
        Boolean mask of active atoms of shape (batch_size, n_atoms).
    occupancies : torch.Tensor
        Occupancies of shape (batch_size, n_atoms).
    lmax : torch.Tensor
        Maximum distances from atom to consider in grid units of shape (3,).
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

    Returns
    -------
    torch.Tensor
        Density grid (batch_size, *grid_shape).
    """
    device = coordinates.device
    dtype = coordinates.dtype
    batch_size, n_atoms = coordinates.shape[:2]
    rmax2 = rmax * rmax

    result = torch.zeros((batch_size,) + grid_shape, device=device, dtype=dtype)

    max_extents = [int(torch.ceil(lmax[i]).item()) for i in range(3)]
    nearby_grid = torch.stack(
        torch.meshgrid(
            [  # c, b, a ordered
                torch.arange(-max_extents[2], max_extents[2] + 1, device=device),
                torch.arange(-max_extents[1], max_extents[1] + 1, device=device),
                torch.arange(-max_extents[0], max_extents[0] + 1, device=device),
            ],
            indexing="ij",
        ),
        dim=-1,
    ).reshape(
        -1, 3
    )  # [n_nearby, 3]

    # Compute the a, b, c distance from the grid voxels for each atom
    grid_difference = coordinates - torch.floor(coordinates)  # [batch_size, n_atoms, 3]

    # Transform grid to a, b, c for subtraction
    nearby_grid_abc = nearby_grid[:, [2, 1, 0]]  # [n_nearby, 3]

    # Compute the distance from the grid for each atom to each nearby grid point in the offset
    delta_to_nearby = grid_difference.view(
        batch_size, 1, n_atoms, 3
    ) - nearby_grid_abc.view(
        1, -1, 1, 3
    )  # [batch_size, n_nearby, n_atoms, 3]
    cartesian_delta_to_nearby = torch.matmul(
        delta_to_nearby, grid_to_cartesian.T
    )  # [batch_size, n_nearby, n_atoms, 3]
    distances_to_nearby = torch.linalg.norm(
        cartesian_delta_to_nearby, dim=-1
    )  # [batch_size, n_nearby, n_atoms]

    rad_continuous = distances_to_nearby / rstep  # [batch_size, n_nearby, n_atoms]
    rad_indices_low = torch.floor(
        rad_continuous
    ).long()  # [batch_size, n_nearby, n_atoms]
    weights_high = (
        rad_continuous - rad_indices_low.float()
    )  # [batch_size, n_nearby, n_atoms]
    weights_low = 1.0 - weights_high  # [batch_size, n_nearby, n_atoms]

    # Clamp to valid range
    max_rad_idx = radial_densities.shape[-1] - 1
    rad_indices_low = torch.clamp(
        rad_indices_low, 0, max_rad_idx
    )  # [batch_size, n_nearby, n_atoms]
    rad_indices_high = torch.clamp(
        rad_indices_low + 1, 0, max_rad_idx
    )  # [batch_size, n_nearby, n_atoms]

    active_mask = (
        active.reshape(batch_size * n_atoms).nonzero().squeeze(-1)
    )  # [n_nearby * n_atoms].nonzero() -> [n_active_atoms]
    batch_idx = active_mask // n_atoms  # [n_active_atoms]
    atom_idx = active_mask % n_atoms  # [n_active_atoms]

    # Calculate grid points to interpolate onto
    n_active_atoms = len(active_mask)
    n_nearby = nearby_grid.shape[0]
    coord_floored = torch.floor(
        coordinates[batch_idx, atom_idx]
    ).long()  # [n_active_atoms, 3]

    # modulo for periodic boundary
    grid_points = (
        coord_floored.view(-1, 1, 3) + nearby_grid_abc.view(1, -1, 3)
    ) % torch.tensor(
        grid_shape[::-1], device=device
    )  # [n_active_atoms, n_nearby, 3]

    atom_indices = torch.arange(n_active_atoms, device=device).repeat_interleave(
        n_nearby
    )
    offset_indices = torch.arange(n_nearby, device=device).repeat(n_active_atoms)
    final_batch_indices = batch_idx[atom_indices]
    final_atom_indices = atom_idx[atom_indices]  # all [n_active_atoms * n_nearby]

    rad_indices_low_final = rad_indices_low[
        final_batch_indices, offset_indices, final_atom_indices
    ]
    rad_indices_high_final = rad_indices_high[
        final_batch_indices, offset_indices, final_atom_indices
    ]

    # Interpolate radial densities onto grid and scale by occupancy
    densities = (
        weights_low[final_batch_indices, offset_indices, final_atom_indices]
        * radial_densities[
            final_batch_indices, final_atom_indices, rad_indices_low_final
        ]
        + weights_high[final_batch_indices, offset_indices, final_atom_indices]
        * radial_densities[
            final_batch_indices, final_atom_indices, rad_indices_high_final
        ]
    ) * occupancies[
        final_batch_indices, final_atom_indices
    ]  # [n_active_atoms * n_nearby]

    # scatter_add_ onto the grid
    grid_points_flat = grid_points.reshape(
        -1, 3
    ).long()  # [n_active_atoms * n_nearby, 3]
    grid_strides = [
        grid_shape[1] * grid_shape[2],
        grid_shape[2],
        1,
    ]  # [grid_y * grid_x, grid_x, 1]

    # Calculate strided indices for proper scatter_add_
    flat_grid_indices = (
        final_batch_indices * (grid_shape[0] * grid_shape[1] * grid_shape[2])  # batch
        + grid_points_flat[:, 2] * grid_strides[0]  # z
        + grid_points_flat[:, 1] * grid_strides[1]  # y
        + grid_points_flat[:, 0]  # x
    ).long()  # [n_active_atoms * n_nearby]

    result.view(-1).scatter_add_(
        0, flat_grid_indices, densities
    )  # Add onto [batch_size * grid_z * grid_y * grid_x] shape

    return result


def scattering_integrand(
    s: torch.Tensor,
    r: torch.Tensor,
    asf: torch.Tensor,
    bfactor: torch.Tensor,
    em: bool = False,
) -> torch.Tensor:
    """Compute the scattering integrand for radial density calculation.

    Parameters
    ----------
    s : torch.Tensor
        Scattering vector magnitudes, shape (..., n_s).
    r : torch.Tensor
        Radial distances, shape (n_r,).
    asf : torch.Tensor
        Atomic scattering factors with shape (..., n_coeffs, 2).
        The coefficients are [a_i, b_i] pairs for the scattering model.
    bfactor : torch.Tensor
        B-factors, shape (...,).
    em : bool, optional
        Whether to use electron microscopy mode, by default False.

    Returns
    -------
    torch.Tensor
        Computed integrand values with shape (..., n_s, n_r).
    """
    s_expanded = s.reshape(*s.shape, 1)  # [..., n_s, 1]
    r_expanded = r.reshape(1, -1)  # [1, n_r]

    s2 = s_expanded * s_expanded  # [..., n_s, 1]

    bfactor_expanded = bfactor.reshape(*bfactor.shape, 1, 1)  # [..., 1, 1]

    if em:
        # electron microscopy mode (no constant term)
        a_coeffs = asf[..., :5, 0]  # [..., 5]
        b_coeffs = asf[..., :5, 1]  # [..., 5]

        a_coeffs = a_coeffs.reshape(*a_coeffs.shape, 1, 1)  # [..., 5, 1, 1]
        b_coeffs = b_coeffs.reshape(*b_coeffs.shape, 1, 1)  # [..., 5, 1, 1]

        exp_terms = torch.exp(-b_coeffs * s2.unsqueeze(-3))  # [..., 5, n_s, 1]

        f = torch.sum(a_coeffs * exp_terms, dim=-3)  # [..., n_s, 1]
    else:
        # X-ray scattering mode (includes constant term)
        a_coeffs = asf[..., :5, 0]  # [..., 5]
        b_coeffs = asf[..., :5, 1]  # [..., 5]
        constant_term = asf[..., 5, 0].reshape(*asf.shape[:-2], 1, 1)  # [..., 1, 1]

        a_coeffs = a_coeffs.reshape(*a_coeffs.shape, 1, 1)  # [..., 5, 1, 1]
        b_coeffs = b_coeffs.reshape(*b_coeffs.shape, 1, 1)  # [..., 5, 1, 1]

        exp_terms = torch.exp(-b_coeffs * s2.unsqueeze(-3))  # [..., 5, n_s, 1]

        f = torch.sum(a_coeffs * exp_terms, dim=-3) + constant_term  # [..., n_s, 1]

    four_pi_s = 4 * torch.pi * s_expanded  # [..., n_s, 1]
    w = 8 * f * torch.exp(-bfactor_expanded * s2) * s_expanded  # [..., n_s, 1]

    eps = 1e-4
    r_small_mask = (r_expanded < eps).expand(
        s_expanded.shape[:-1] + r_expanded.shape[-1:]
    )  # [..., n_s, n_r]

    ar = four_pi_s * r_expanded  # [..., n_s, n_r]
    ar2 = ar * ar

    # prevent singularity with 4th order Taylor expansion
    taylor_term = 1.0 - ar2 / 6.0
    small_r_values = w * four_pi_s * taylor_term  # [..., n_s, n_r]

    sin_term = torch.sin(ar)

    safe_r = torch.where(r_expanded > 0, r_expanded, torch.ones_like(r_expanded))
    large_r_values = w * sin_term / safe_r  # [..., n_s, n_r]

    result = torch.where(r_small_mask, small_r_values, large_r_values)

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
