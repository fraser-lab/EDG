import torch
from typing import Callable, Tuple


class GaussLegendreQuadrature(torch.nn.Module):
    """Gaussian-Legendre quadrature implementation in PyTorch."""

    def __init__(
        self,
        num_points: int = 5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the Gauss-Legendre quadrature module.

        Parameters
        ----------
        num_points : int, optional
            Number of quadrature points to use, by default 5
        device : torch.device, optional
            Device to place the tensors on, by default torch.device("cpu")
        dtype : torch.dtype, optional
            Data type for the tensors, by default torch.float32
        """
        super().__init__()
        self.num_points = num_points
        self.roots, self.weights = self._get_legendre_roots_and_weights(
            num_points, device, dtype
        )

    def forward(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        integration_limits: torch.Tensor,
        dim: int = -1,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """Compute the definite integral using Gauss-Legendre quadrature.

        Parameters
        ----------
        func : Callable[[torch.Tensor], torch.Tensor]
            Function to integrate. Should accept and return tensors.
        integration_limits : torch.Tensor
            Tensor of shape [..., 2] containing lower and upper integration limits.
        dim : int, optional
            Dimension of the function output to integrate over, by default -1
        keepdim : bool, optional
            Whether to keep the integrated dimension, by default False

        Returns
        -------
        torch.Tensor
            Approximated definite integral
        """

        lower_limits = integration_limits[..., 0]
        upper_limits = integration_limits[..., 1]

        scale_factor = (upper_limits - lower_limits) / 2
        midpoint = (upper_limits + lower_limits) / 2

        shape = [1] * max(lower_limits.dim() + 1, dim + 1)
        shape[dim] = self.num_points

        roots_reshaped = self.roots.reshape(shape).to(integration_limits.device)
        weights_reshaped = self.weights.reshape(shape).to(integration_limits.device)

        scale_shape = list(lower_limits.shape)
        dim_pos = lower_limits.dim() + dim + 1 if dim < 0 else dim
        scale_shape.insert(dim_pos, 1)

        scale_factor_reshaped = scale_factor.reshape(scale_shape)
        midpoint_reshaped = midpoint.reshape(scale_shape)

        scaled_points = scale_factor_reshaped * roots_reshaped + midpoint_reshaped
        function_values = func(scaled_points)

        diff = function_values.ndim - weights_reshaped.ndim
        weighted_values = function_values * weights_reshaped.reshape(*weights_reshaped.shape, *(diff * [1]))
        integral = scale_factor_reshaped * weighted_values.sum(dim=dim, keepdim=keepdim)

        return integral

    def _get_legendre_roots_and_weights(
        self, num_points: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Numerically compute Legendre polynomial roots and weights. Generally taken from SciPy:
        https://github.com/scipy/scipy/blob/main/scipy/special/_orthogonal.py#L160

        Parameters
        ----------
        num_points : int
            Number of quadrature points
        device : torch.device
            Device to place the tensors on
        dtype : torch.dtype
            Data type for the tensors

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Roots and weights of the Legendre polynomial
        """
        diagonal = torch.zeros(num_points, device=device, dtype=dtype)

        k_values = torch.arange(1, num_points, device=device, dtype=dtype)
        off_diagonal = k_values / torch.sqrt(4 * k_values.pow(2) - 1)

        tridiag = torch.diag(diagonal)
        if num_points > 1:
            tridiag = (
                tridiag + torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)
            )
        eigenvalues, eigenvectors = torch.linalg.eigh(tridiag)

        roots = eigenvalues
        weights = 2.0 * eigenvectors[0, :].pow(2)

        return roots, weights
