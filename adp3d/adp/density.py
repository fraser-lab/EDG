import torch
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import torch.nn.functional as F
from torchquad import Simpson, Trapezoid
import numpy as np

from adp3d.qfit.volume import XMap, EMMap, GridParameters

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
    integration_method: str = 'simpson'

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.rmax <= 0 or self.rstep <= 0:
            raise ValueError("rmax and rstep must be positive")
        if self.smin >= self.smax:
            raise ValueError("smin must be less than smax")
        if self.quad_points < 2:
            raise ValueError("quad_points must be at least 2")
        if self.integration_method not in ['simpson', 'trapezoid']:
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
        em: bool = False
    ) -> None:
        """Initialize scattering integrand calculator.

        Parameters
        ----------
        asf : torch.Tensor
            Atomic scattering factors of shape (n_atoms, n_coeffs, 2).
        bfactor : torch.Tensor
            B-factors of shape (n_atoms,).
        em : bool, optional
            Whether to use electron microscopy mode, by default False.
        """
        super().__init__()
        if asf.dim() != 3 or asf.shape[2] != 2:
            raise ValueError("asf must have shape (n_atoms, n_coeffs, 2)")
        if bfactor.dim() != 1:
            raise ValueError("bfactor must be 1-dimensional")
        
        self.register_buffer("asf", asf)
        self.register_buffer("bfactor", bfactor)
        self.em = em

    def forward(
        self,
        s: torch.Tensor,
        r: torch.Tensor
    ) -> torch.Tensor:
        """Compute scattering integrand.

        Parameters
        ----------
        s : torch.Tensor
            Scattering vector magnitudes of shape (n_quad,).
        r : torch.Tensor
            Radial distances of shape (n_points,).

        Returns
        -------
        torch.Tensor
            Integrand values of shape (n_atoms, n_quad, n_points).
        """
        s = s.unsqueeze(0).unsqueeze(-1)
        r = r.unsqueeze(0).unsqueeze(1)
        s2 = s * s
        
        if self.em:
            if self.asf.shape[1] < 5:
                raise ValueError("For electron scattering, at least 5 coefficients required")
            loop_range = 5
            f = torch.zeros(
                (self.asf.shape[0], 1, 1),
                device=self.asf.device,
                dtype=self.asf.dtype
            )
        else:
            if self.asf.shape[1] != 6:
                raise ValueError("For X-ray scattering, exactly 6 coefficients required")
            loop_range = 5
            f = self.asf[:, -1:, 0].unsqueeze(1).unsqueeze(2)
        
        for i in range(loop_range):
            f += (self.asf[:, i, 0].unsqueeze(1).unsqueeze(2) *
                  torch.exp(-self.asf[:, i, 1].unsqueeze(1).unsqueeze(2) * s2))
            
        w = 8 * f * torch.exp(-self.bfactor.unsqueeze(1).unsqueeze(2) * s2) * s
        a = 4 * torch.pi * s
        ar = a * r
        
        eps = 1e-4
        small_r = r < eps
        large_r = ~small_r
        
        result = torch.zeros_like(ar)
        result[..., large_r] = w[..., large_r] * torch.sin(ar[..., large_r]) / r[..., large_r]
        result[..., small_r] = w[..., small_r] * a[..., small_r] * (
            1 - (a[..., small_r]**2 * r[..., small_r]**2) / 6.0
        )
        
        return result

class DifferentiableXMap(torch.nn.Module):
    """Differentiable version of XMap for handling crystallographic symmetry.
    
    Implements a vectorized approach to applying symmetry operations across batches
    of electron density maps while maintaining differentiability.
    """
    
    def __init__(
        self,
        xmap: XMap,
        resolution: Optional[float] = None,
        hkl: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cpu')
    ) -> None:
        """Initialize differentiable XMap.

        Parameters
        ----------
        grid_parameters : GridParameters
            Grid parameters for the map.
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
        self.grid_parameters = grid_parameters
        self.unit_cell = unit_cell
        self.resolution = resolution
        self.hkl = hkl
        
        self.register_buffer(
            "voxel_spacing",
            torch.tensor(grid_parameters.voxelspacing, device=device)
        )
        self.register_buffer(
            "offset",
            torch.tensor(grid_parameters.offset, device=device)
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
        
    def apply_symmetry(
        self,
        density: torch.Tensor,
        normalize: bool = True,
        chunk_size: Optional[int] = None
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
            If None, process all operations simultaneously.

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
                *[torch.arange(s, device=device) for s in grid_shape],
                indexing='ij'
            ),
            dim=-1
        ).float()
        
        base_coords = base_coords + self.offset
        base_coords = base_coords.view(1, *grid_shape, 3)
        grid_shape_tensor = torch.tensor(grid_shape, device=device)
        
        output = density.clone()
        
        if chunk_size is None:
            coords_expanded = base_coords.expand(n_ops, *grid_shape, 3)
            transformed_coords = torch.einsum(
                'nij,b...j->n...i',
                self.R_matrices,
                coords_expanded
            )
            
            transformed_coords = transformed_coords + (
                self.t_vectors.view(n_ops, 1, 1, 1, 3) * 
                grid_shape_tensor.view(1, 1, 1, 1, 3)
            )
            
            transformed_coords = transformed_coords % grid_shape_tensor.view(1, 1, 1, 1, 3)
            normalized_coords = (
                transformed_coords / 
                (grid_shape_tensor.view(1, 1, 1, 1, 3) - 1)
            ) * 2 - 1
            
            for b in range(batch_size):
                normalized_coords_batch = normalized_coords.view(
                    n_ops, -1, grid_shape[1], grid_shape[2], 3
                )
                
                transformed_density = F.grid_sample(
                    density[b:b+1, None].expand(n_ops, 1, *grid_shape),
                    normalized_coords_batch,
                    mode='bilinear',
                    align_corners=True,
                    padding_mode='border'
                )
                
                output[b] += transformed_density.sum(dim=0)[0]
                
        else:
            for chunk_start in range(0, n_ops, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_ops)
                chunk_size_actual = chunk_end - chunk_start
                
                R_chunk = self.R_matrices[chunk_start:chunk_end]
                t_chunk = self.t_vectors[chunk_start:chunk_end]
                
                coords_expanded = base_coords.expand(chunk_size_actual, *grid_shape, 3)
                transformed_coords = torch.einsum(
                    'nij,b...j->n...i',
                    R_chunk,
                    coords_expanded
                )
                
                transformed_coords = transformed_coords + (
                    t_chunk.view(chunk_size_actual, 1, 1, 1, 3) * 
                    grid_shape_tensor.view(1, 1, 1, 1, 3)
                )
                
                transformed_coords = transformed_coords % grid_shape_tensor.view(1, 1, 1, 1, 3)
                normalized_coords = (
                    transformed_coords / 
                    (grid_shape_tensor.view(1, 1, 1, 1, 3) - 1)
                ) * 2 - 1
                
                for b in range(batch_size):
                    normalized_coords_batch = normalized_coords.view(
                        chunk_size_actual, -1, grid_shape[1], grid_shape[2], 3
                    )
                    
                    transformed_density = F.grid_sample(
                        density[b:b+1, None].expand(chunk_size_actual, 1, *grid_shape),
                        normalized_coords_batch,
                        mode='bilinear',
                        align_corners=True,
                        padding_mode='border'
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
        device: torch.device = torch.device('cpu')
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
        self.scattering_params = {
            k: v.to(device) for k, v in scattering_params.items()
        }
        self.density_params = density_params or DensityParameters()
        self.em = em
        
        grid_parameters = GridParameters(
            voxelspacing=voxel_spacing.cpu().numpy(),
            offset=(0, 0, 0)
        )
        self.xmap = DifferentiableXMap(
            grid_parameters=grid_parameters,
            unit_cell=unit_cell,
            device=device
        )
        
        self._setup_transforms()
        self._setup_integrator()

    def _setup_transforms(self) -> None:
        """Initialize transformation matrices for coordinate conversions."""
        abc = self.unit_cell.abc
        self.register_buffer(
            "lattice_to_cartesian",
            self._unit_cell_to_cartesian_matrix() / torch.tensor(abc).reshape(3, 1)
        )
        self.register_buffer(
            "cartesian_to_lattice",
            torch.inverse(self.lattice_to_cartesian)
        )
        self.register_buffer(
            "grid_to_cartesian",
            self.lattice_to_cartesian * self.voxel_spacing.reshape(3, 1)
        )

    def _setup_integrator(self) -> None:
        """Set up numerical integrator based on density parameters."""
        if self.density_params.integration_method.lower() == 'simpson':
            self.integrator = Simpson(dim=1)
        elif self.density_params.integration_method.lower() == 'trapezoid':
            self.integrator = Trapezoid(dim=1)
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
        chunk_size: Optional[int] = None
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

        Returns
        -------
        torch.Tensor
            Symmetry-expanded density grid of shape (batch_size, *grid_shape).
        """
        if not (coordinates.shape[0] == elements.shape[0] == 
                b_factors.shape[0] == occupancies.shape[0]):
            raise ValueError("Batch sizes must match for all inputs")
            
        initial_density = self._compute_density(
            coordinates, elements, b_factors, occupancies
        )
        
        symmetrized_density = self.xmap.apply_symmetry(
            initial_density,
            normalize=normalize,
            chunk_size=chunk_size
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
            dtype=coordinates.dtype
        )
        
        lmax = torch.tensor([
            self.density_params.rmax / vs for vs in self.voxel_spacing
        ], device=coordinates.device)
        
        for b in range(batch_size):
            dilate_points_torch(
                grid_coordinates[b],
                torch.ones_like(elements[b], dtype=torch.bool),
                occupancies[b],
                lmax,
                radial_densities[b],
                self.density_params.rstep,
                self.density_params.rmax,
                self.grid_to_cartesian,
                density[b],
            )
            
        return density

    def _compute_radial_densities(
        self,
        elements: torch.Tensor,
        b_factors: torch.Tensor
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
            device=elements.device
        )
        
        batch_size, n_atoms = elements.shape
        n_radial = r.shape[0]
        
        densities = torch.zeros(
            (batch_size, n_atoms, n_radial),
            device=elements.device
        )
        
        unique_elements = torch.unique(elements)
        for elem in unique_elements:
            mask = elements == elem
            if not mask.any():
                continue
                
            asf = self.scattering_params[elem.item()]
            integrand = ScatteringIntegrand(
                asf.expand(mask.sum(), -1, -1),
                b_factors[mask],
                em=self.em
            )
            
            result = self.integrator.integrate(
                lambda s: integrand(s, r),
                dim=1,
                N=self.density_params.quad_points,
                integration_domain=[[self.density_params.smin, self.density_params.smax]]
            )
            
            densities[mask] = result
            
        return densities

    def _compute_radial_derivatives(
        self,
        elements: torch.Tensor,
        b_factors: torch.Tensor
    ) -> torch.Tensor:
        """Compute radial density derivatives.

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
        return torch.gradient(densities, self.density_params.rstep, edge_order=2)[0]

    def _compute_grid_coordinates(
        self,
        coordinates: torch.Tensor
    ) -> torch.Tensor:
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
        a, b, c, alpha, beta, gamma = self.unit_cell.abc + self.unit_cell.angles
        alpha, beta, gamma = map(torch.deg2rad, [alpha, beta, gamma])
        
        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)
        
        volume = (a * b * c * torch.sqrt(
            1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 +
            2 * cos_alpha * cos_beta * cos_gamma
        ))
        
        matrix = torch.zeros((3, 3), device=self.voxel_spacing.device)
        matrix[0, 0] = a
        matrix[0, 1] = b * cos_gamma
        matrix[0, 2] = c * cos_beta
        matrix[1, 1] = b * sin_gamma
        matrix[1, 2] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        matrix[2, 2] = volume / (a * b * sin_gamma)
        
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
    out: torch.Tensor,
) -> torch.Tensor:
    """PyTorch implementation of point dilation."""
    device = coordinates.device
    grid_shape = torch.tensor(out.shape, device=device)
    rmax2 = rmax * rmax
    
    for n in range(len(coordinates)):
        if not active[n]:
            continue
            
        center = coordinates[n]
        bounds_min = torch.ceil(center - lmax).long()
        bounds_max = torch.floor(center + lmax).long()
        
        z_range = torch.arange(bounds_min[0], bounds_max[0] + 1, device=device)
        y_range = torch.arange(bounds_min[1], bounds_max[1] + 1, device=device)
        x_range = torch.arange(bounds_min[2], bounds_max[2] + 1, device=device)
        
        Z, Y, X = torch.meshgrid(z_range, y_range, x_range, indexing='ij')
        grid_points = torch.stack([Z, Y, X], dim=-1)
        
        rel_coords = grid_points - center
        cart_coords = torch.einsum('ij,...j->...i', grid_to_cartesian, rel_coords)
        distances2 = torch.sum(cart_coords * cart_coords, dim=-1)
        
        mask = distances2 <= rmax2
        if not mask.any():
            continue
        
        distances = torch.sqrt(distances2[mask])
        indices = (distances / rstep).long().clamp(0, radial_densities.shape[1] - 1)
        
        density_values = radial_densities[n, indices] * occupancies[n]
        
        Z_idx = torch.remainder(Z[mask], out.shape[0])
        Y_idx = torch.remainder(Y[mask], out.shape[1])
        X_idx = torch.remainder(X[mask], out.shape[2])
        
        out[Z_idx, Y_idx, X_idx] += density_values
        
    return out