import torch
from . import dilate_points_cuda, CUDA_AVAILABLE
from typing import Optional, Tuple, Union


class DilateAtomCentricCUDA(torch.autograd.Function):
    """Custom CUDA-accelerated atom-centric density dilation operation.

    This implementation provides efficient forward and backward passes for computing
    density maps from atomic positions and radial profiles.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        atom_coords_grid: torch.Tensor,  # [batch_size, N_atoms, 3] grid units
        atom_occupancies: torch.Tensor,  # [batch_size, N_atoms]
        radial_profiles: torch.Tensor,  # [batch_size, N_atoms, N_radial_points]
        radial_profiles_derivatives: torch.Tensor,  # [batch_size, N_atoms, N_radial_points]
        r_step: float,  # Scalar step size for radial_profiles
        rmax_cartesian: float,  # Max radius in Cartesian
        lmax_grid_units: torch.Tensor,  # [3] tensor with [lx, ly, lz]
        grid_dims: torch.Tensor,  # [3] tensor with [Dz, Dy, Dx]
        grid_to_cartesian_matrix: torch.Tensor,  # [3, 3] transformation matrix
    ) -> torch.Tensor:
        """Forward pass: computes density grid from atomic positions and radial profiles.

        Parameters
        ----------
        atom_coords_grid : torch.Tensor
            Atomic coordinates in grid units, shape [batch_size, symmetry_ops, N_atoms, 3]
        atom_occupancies : torch.Tensor
            Atomic occupancies, shape [batch_size, N_atoms]
        radial_profiles : torch.Tensor
            Pre-calculated radial density values P(r), shape [batch_size, N_atoms, N_radial_points]
        radial_profiles_derivatives : torch.Tensor
            Pre-calculated derivatives of radial density P'(r), shape [batch_size, N_atoms, N_radial_points]
        r_step : float
            Step size for radial_profiles sampling
        rmax_cartesian : float
            Maximum radius for an atom's influence in Cartesian space
        lmax_grid_units : torch.Tensor
            Maximum extent in grid units along each axis, shape [3]
        grid_dims : torch.Tensor
            Dimensions of output grid [Dz, Dy, Dx], shape [3]
        grid_to_cartesian_matrix : torch.Tensor
            Transformation matrix from grid to Cartesian coordinates, shape [3, 3]

        Returns
        -------
        torch.Tensor
            Output density grid, shape [batch_size, Dz, Dy, Dx]
        """
        atom_coords_grid = atom_coords_grid.contiguous()
        atom_occupancies = atom_occupancies.contiguous()
        radial_profiles = radial_profiles.contiguous()
        radial_profiles_derivatives = radial_profiles_derivatives.contiguous()
        lmax_grid_units = lmax_grid_units.contiguous()
        grid_dims = grid_dims.contiguous()
        grid_to_cartesian_matrix = grid_to_cartesian_matrix.contiguous()

        if CUDA_AVAILABLE:
            output_density_grid = dilate_points_cuda.forward(
                atom_coords_grid,
                atom_occupancies,
                radial_profiles,
                r_step,
                rmax_cartesian,
                lmax_grid_units,
                grid_dims,
                grid_to_cartesian_matrix,
            )
        else:
            raise RuntimeError("CUDA is not available.")

        ctx.save_for_backward(
            atom_coords_grid,
            atom_occupancies,
            radial_profiles,
            radial_profiles_derivatives,
            lmax_grid_units,
            grid_dims,
            grid_to_cartesian_matrix,
        )

        ctx.r_step = r_step
        ctx.rmax_cartesian = rmax_cartesian

        return output_density_grid

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass: computes gradients with respect to inputs.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of loss with respect to output density grid

        Returns
        -------
        Tuple[Optional[torch.Tensor], ...]
            Gradients with respect to each input from the forward pass
        """
        (
            atom_coords_grid,
            atom_occupancies,
            radial_profiles,
            radial_profiles_derivatives,
            lmax_grid_units,
            grid_dims,
            grid_to_cartesian_matrix,
        ) = ctx.saved_tensors

        r_step = ctx.r_step
        rmax_cartesian = ctx.rmax_cartesian

        grad_output = grad_output.contiguous()

        if CUDA_AVAILABLE:
            grad_atom_coords_grid, grad_atom_occupancies, grad_radial_profiles = (
                dilate_points_cuda.backward(
                    grad_output,
                    atom_coords_grid,
                    atom_occupancies,
                    radial_profiles,
                    radial_profiles_derivatives,
                    r_step,
                    rmax_cartesian,
                    lmax_grid_units,
                    grid_dims,
                    grid_to_cartesian_matrix,
                )
            )
        else:
            raise RuntimeError("Backward failed: CUDA is not available.")

        return (
            grad_atom_coords_grid,  # atom_coords_grid
            grad_atom_occupancies,  # atom_occupancies
            grad_radial_profiles,  # radial_profiles
            None,  # radial_profiles_derivatives
            None,  # r_step
            None,  # rmax_cartesian
            None,  # lmax_grid_units
            None,  # grid_dims
            None,  # grid_to_cartesian_matrix
        )


def dilate_atom_centric(
    atom_coords_grid: torch.Tensor,
    atom_occupancies: torch.Tensor,
    radial_profiles: torch.Tensor,
    radial_profiles_derivatives: torch.Tensor,
    r_step: float,
    rmax_cartesian: float,
    lmax_grid_units: torch.Tensor,
    grid_dims: Union[torch.Tensor, tuple],
    grid_to_cartesian_matrix: torch.Tensor,
) -> torch.Tensor:
    """High-level interface to atom-centric density dilation.

    This function handles data type and device consistency, and calls the
    appropriate implementation based on availability.

    Parameters
    ----------
    atom_coords_grid : torch.Tensor
        Atomic coordinates in grid units, shape [batch_size, symmetry_ops, N_atoms, 3]
    atom_occupancies : torch.Tensor
        Atomic occupancies, shape [batch_size, N_atoms]
    radial_profiles : torch.Tensor
        Pre-calculated radial density values P(r), shape [batch_size, N_atoms, N_radial_points]
    radial_profiles_derivatives : torch.Tensor
        Pre-calculated derivatives of radial density P'(r), shape [batch_size, N_atoms, N_radial_points]
    r_step : float
        Step size for radial_profiles sampling
    rmax_cartesian : float
        Maximum radius for an atom's influence in Cartesian space
    lmax_grid_units : torch.Tensor
        Maximum extent in grid units along each axis, shape [3]
    grid_dims : torch.Tensor or tuple
        Dimensions of output grid [Dz, Dy, Dx], shape [3]
    grid_to_cartesian_matrix : torch.Tensor
        Transformation matrix from grid to Cartesian coordinates, shape [3, 3]

    Returns
    -------
    torch.Tensor
        Output density grid, shape [batch_size, Dz, Dy, Dx]
    """

    device = atom_coords_grid.device
    dtype = atom_coords_grid.dtype

    lmax_grid_units = lmax_grid_units.to(torch.int32)
    grid_dims = (
        grid_dims.to(torch.int32)
        if isinstance(grid_dims, torch.Tensor)
        else torch.tensor(grid_dims, dtype=torch.int32, device=device)
    )

    atom_coords_grid = atom_coords_grid.to(dtype)
    atom_occupancies = atom_occupancies.to(dtype)
    radial_profiles = radial_profiles.to(dtype)
    radial_profiles_derivatives = radial_profiles_derivatives.to(dtype)
    grid_to_cartesian_matrix = grid_to_cartesian_matrix.to(dtype)

    return DilateAtomCentricCUDA.apply(
        atom_coords_grid,
        atom_occupancies,
        radial_profiles,
        radial_profiles_derivatives,
        r_step,
        rmax_cartesian,
        lmax_grid_units,
        grid_dims,
        grid_to_cartesian_matrix,
    )
