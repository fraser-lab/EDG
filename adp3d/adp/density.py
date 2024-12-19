"""Calculate density map from atomic coordinates.

Author: Karson Chrispens
Created: 11 Nov 2024
Updated: 3 Dec 2024"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gemmi
from einops import rearrange
from adp3d.data.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
)
import warnings
from typing import Union, Tuple, List


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
    Downsample density map in Fourier space # TODO: MAKE BATCHABLE (and this whole file for that matter)

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


class DensityCalculator(nn.Module):
    """
    Module to calculate electron density map from atomic coordinates.

    Parameters
    ----------
    grid : gemmi.FloatGrid
        Grid object from gemmi library.
    center_shift : torch.Tensor
        Shift to center the grid.
    device : torch.device
        Device to run calculations on.
    resolution : float
        Resolution to compute the map to, in Angstroms.
    em : bool
        If True, use electron scattering factors instead of atomic structure factors.
    """

    def __init__(
        self,
        grid: gemmi.FloatGrid,
        center_shift: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        resolution: float = 2.0,
        em: bool = False,
    ):
        super(DensityCalculator, self).__init__()
        self.grid = grid
        self.center_shift = center_shift
        self.device = device
        self.em = em

        # constants
        self.pi2 = (
            torch.pi**2
        )  # TODO: change to spherical coordinates maybe? should be 4 pi^2 in that case
        self._range = 5 if em else 6
        self.sf = ELECTRON_SCATTERING_FACTORS if em else ATOM_STRUCTURE_FACTORS

        # element dictionaries
        if torch.jit.is_scripting():
            self.element_symbols = torch.jit.Attribute(ATOMIC_NUM_TO_ELEMENT, List[str])
        else:
            self.element_symbols = ATOMIC_NUM_TO_ELEMENT

        self._setup_coeffs()

        # grid spacing
        self.nx, self.ny, self.nz = self.grid.shape
        if any([s == 0 for s in self.grid.spacing]):
            self.spacing = (
                self.grid.unit_cell.a / (self.nx - 1),
                self.grid.unit_cell.b / (self.ny - 1),
                self.grid.unit_cell.c / (self.nz - 1),
            )
        else:
            self.spacing = self.grid.spacing

        # pre-compute grid coordinates
        self._setup_real_grid()
        self._setup_fourier_grid()

        # pre-compute filter and mask
        self.set_filter_and_mask(resolution)

    def _setup_coeffs(self):
        """Setup coefficients for density calculation."""
        # for Fourier space
        self.a_dict = {}
        self.b_dict = {}

        # for real space
        self.aw_dict = {}
        self.bw_dict = {}

        for e in self.element_symbols[1:]: # skip ? element
            self.a_dict[e] = torch.tensor(
                self.sf[e][0][: self._range],
                device=self.device,
            )
            self.b_dict[e] = torch.tensor(
                self.sf[e][1][: self._range],
                device=self.device,
            )
            self.bw_dict[e] = torch.tensor(
                [
                    -self.pi2 / b if b > 1e-4 else 0
                    for b in self.sf[e][1][: self._range]
                ],
                device=self.device,
            )
            self.aw_dict[e] = (
                torch.tensor(self.sf[e][0][: self._range], device=self.device)
                * (-self.bw_dict[e] / torch.pi) ** 1.5
            )

    def _setup_real_grid(self):
        """Setup flattened real space grid."""
        origin = (
            torch.tensor([*self.grid.get_position(0, 0, 0)], device=self.device)
            - self.center_shift
        )

        if all([s > 0 for s in self.spacing]):
            x = torch.arange(
                origin[0],
                origin[0] + self.spacing[0] * self.nx,
                self.spacing[0],
                device=self.device,
            )[: self.nx]
            y = torch.arange(
                origin[1],
                origin[1] + self.spacing[1] * self.ny,
                self.spacing[1],
                device=self.device,
            )[: self.ny]
            z = torch.arange(
                origin[2],
                origin[2] + self.spacing[2] * self.nz,
                self.spacing[2],
                device=self.device,
            )[: self.nz]
        else:
            x = torch.linspace(
                origin[0], self.grid.unit_cell.a, self.nx, device=self.device
            )
            y = torch.linspace(
                origin[1], self.grid.unit_cell.b, self.ny, device=self.device
            )
            z = torch.linspace(
                origin[2], self.grid.unit_cell.c, self.nz, device=self.device
            )

        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
        self.real_grid_flat = rearrange(
            torch.stack([grid_x, grid_y, grid_z], dim=-1), "x y z c -> (x y z) c"
        )

    def _setup_fourier_grid(self):
        """Setup Fourier space frequency grid."""

        # Compute frequency axes
        fx = torch.fft.fftshift(
            torch.fft.fftfreq(self.nx, self.spacing[0], device=self.device)
        )
        fy = torch.fft.fftshift(
            torch.fft.fftfreq(self.ny, self.spacing[1], device=self.device)
        )
        fz = torch.fft.fftshift(
            torch.fft.fftfreq(self.nz, self.spacing[2], device=self.device)
        )

        # Create frequency grid
        fxx, fyy, fzz = torch.meshgrid(fx, fy, fz, indexing="ij")
        self.freq_grid = torch.stack([fxx, fyy, fzz], dim=-1)
        self.freq_norm = torch.linalg.norm(self.freq_grid, dim=-1)

    def _compute_density_chunk(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
        chunk_size: int = 1000000,
    ) -> torch.Tensor:
        """Compute density in chunks to reduce memory usage.

        Parameters
        ----------

        X : torch.Tensor
            Coordinates of atoms. Shape (n_atoms, 3).
        elements : torch.Tensor
            Element indices of atoms.
        chunk_size : int
            Number of atoms to process in a single chunk. Lower values reduce memory usage.

        Returns
        -------
        torch.Tensor
            Flattened density map with all chunks combined.
        """

        n_chunks = (self.real_grid_flat.shape[0] + chunk_size - 1) // chunk_size

        atom_symbols = [
            self.element_symbols[e.item()] for e in elements
        ]  # TODO: handle elements better

        aw_coeffs = torch.stack(
            [self.aw_dict[s] for s in atom_symbols], dim=0
        )  # (n_active_atoms, _range)
        bw_coeffs = torch.stack(
            [self.bw_dict[s] for s in atom_symbols], dim=0
        )  # (n_active_atoms, _range)

        density = torch.zeros_like(self.real_grid_flat[:, 0], device=self.device)

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, self.real_grid_flat.shape[0])
            grid_chunk = self.real_grid_flat[start_idx:end_idx]

            # chunk distances
            distances = torch.cdist(
                grid_chunk, X, p=2.0
            )  # (n_grid_points, n_active_atoms)

            # flattened chunk grid
            density_chunk = torch.zeros(
                grid_chunk.shape[0], device=self.device
            )  # (n_grid_points, )

            distances_expanded = distances.unsqueeze(-1).expand(
                -1, -1, self._range
            )  # (n_grid_points, n_active_atoms, _range)

            exp_term = torch.exp(
                bw_coeffs.unsqueeze(0) * distances_expanded**2
            )  # (n_grid_points, n_active_atoms, _range)

            density_chunk = torch.sum(
                aw_coeffs.unsqueeze(0) * exp_term, dim=-1
            )  # (n_grid_points, n_active_atoms)

            density_chunk = torch.sum(density_chunk, dim=-1)  # (n_grid_points, )

            density[start_idx:end_idx] = density_chunk

        return density

    def _compute_f_density_chunk(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
        chunk_size: int = 1000000,
    ) -> torch.Tensor:
        """Compute Fourier transform of density.

        Parameters
        ----------

        X : torch.Tensor
            Coordinates of atoms. Shape (n_atoms, 3).
        elements : torch.Tensor
            Element indices of atoms.
        chunk_size : int
            Number of atoms to process in a single chunk. Lower values reduce memory usage.

        Returns
        -------
        torch.Tensor
            Fourier coefficients in 3D array.
        """

        atom_symbols = [
            self.element_symbols[e.item()] for e in elements
        ] # TODO: handle elements better

        a_coeffs = torch.stack(
            [self.a_dict[s] for s in atom_symbols], dim=0
        )  # (n_active_atoms, _range)
        b_coeffs = torch.stack(
            [self.b_dict[s] for s in atom_symbols], dim=0
        )  # (n_active_atoms, _range)

        n_freq_points = self.nx * self.ny * self.nz
        n_chunks = (n_freq_points + chunk_size - 1) // chunk_size

        f_density = torch.zeros(
            n_freq_points, dtype=torch.complex64, device=self.device
        )

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_freq_points)

            freq_chunk = self.freq_grid.reshape(-1, 3)[
                start_idx:end_idx
            ]  # (n_freq_points, 3)
            freq_norm_chunk = self.freq_norm.reshape(-1)[
                start_idx:end_idx
            ]  # (n_freq_points, )

            f_density_chunk = torch.zeros(
                freq_chunk.shape[0], dtype=torch.complex64, device=self.device
            )

            phase = (
                -2j * torch.pi * torch.einsum("fc,ac->fa", freq_chunk, X)
            )  # (n_freq_points, n_active_atoms)

            freq_norm_sq = freq_norm_chunk[:, None, None] ** 2  # (n_freq_points, 1, 1)

            # expand b_coeffs and freq_norm_sq to match: (1, n_active_atoms, _range) and (n_freq_points, 1, 1)
            gaussian_terms = torch.exp(
                -b_coeffs[None, :, :] * freq_norm_sq  # scale to match frequency
            )  # (n_freq_points, n_active_atoms, _range)

            # expand a_coeffs to match: (1, n_active_atoms, _range)
            form_factors = torch.sum(
                a_coeffs[None, :, :] * gaussian_terms, dim=-1
            )  # (n_freq_points, n_active_atoms)

            f_density_chunk = torch.sum(
                form_factors * torch.exp(phase), dim=-1
            )  # (n_freq_points, )

            f_density[start_idx:end_idx] = f_density_chunk

        return f_density  # flattened

    def compute_density_real(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
    ) -> torch.Tensor:
        """Compute electron density map with optimized memory usage.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of atoms. Shape (n_atoms, 3).
        elements : torch.Tensor
            Element indices of atoms.

        Returns
        -------
        torch.Tensor
            3D density map."""

        density_flat = self._compute_density_chunk(
            X, elements, chunk_size=100000
        )

        # Reshape back to 3D
        return density_flat.reshape(self.nx, self.ny, self.nz)

    def compute_density_fourier(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
    ) -> torch.Tensor:
        """Compute electron density map in Fourier space."""

        f_density = self._compute_f_density_chunk(X, elements, chunk_size=100000)

        f_density = f_density.reshape(self.nx, self.ny, self.nz)
        return f_density

    def _resolution_filter(
        self, resolution: float = 2.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hamming filter in 3D Fourier space to limit resolution."""
        if resolution <= 0:
            raise ValueError("Resolution must be greater than 0")

        # we want cutoff frequency to correspond to resolution
        cutoff_radius = 1 / resolution

        nyquist_d = 1 / (2 * self.spacing[0])
        nyquist_h = 1 / (2 * self.spacing[1])
        nyquist_w = 1 / (2 * self.spacing[2])

        if any(
            cutoff_radius > nyquist_x for nyquist_x in [nyquist_d, nyquist_h, nyquist_w]
        ):
            warnings.warn("Cutoff radius exceeds the Nyquist frequency.")

        mask = self.freq_norm < cutoff_radius

        filter = radial_hamming_3d(self.freq_norm, cutoff_radius)

        return filter, mask

    def set_filter_and_mask(self, resolution: float = 2.0) -> None:
        self.filter, self.mask = self._resolution_filter(resolution)

    def apply_filter_and_mask(
        self, f_density: torch.Tensor, shape_back: bool = False
    ) -> torch.Tensor:
        """Apply resolution filter and mask to Fourier density.
        NOTE: If shape_back is True, the returned tensor will NOT be comparable to a non-shape_back density.

        Parameters
        ----------
        f_density : torch.Tensor
            3D Fourier coefficients of density map.
        shape_back : bool
            If True, return the reshaped density map.

        Returns
        -------
        torch.Tensor
            Filtered and masked Fourier coefficients, shape (n_masked, ).
            If shape_back is True, returns the Fourier coefficients reshaped to (mask_x, mask_y, mask_z).
        """
        if self.filter is None:
            raise ValueError("Filter not set. Run set_filter_and_mask() first.")

        f_density *= self.filter

        if shape_back:

            indices = torch.nonzero(self.mask)
            mins = indices.min(dim=0).values
            maxs = indices.max(dim=0).values

            reduced_f_density = f_density[
                mins[0] : maxs[0] + 1, mins[1] : maxs[1] + 1, mins[2] : maxs[2] + 1
            ]

            return reduced_f_density

        return f_density[self.mask]

    def forward(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
        resolution: float = 2.0,
        real: bool = False,
        to_normalize: bool = False,
    ) -> torch.Tensor:
        """Compute density map up to given resolution.

        Parameters
        ----------

        X : torch.Tensor
            Coordinates of atoms, shape (n_atoms, 3).
        elements : torch.Tensor
            Element indices of atoms.
        target_density : torch.Tensor
            Target density map.
        resolution : float
            Resolution to compute the map to, in Angstroms.
        real : bool
            If True, compute real space density map. If False, compute Fourier coefficients.
        to_normalize : bool
            If True, normalize the density map.

        Returns
        -------
        torch.Tensor
            If real, returns the real space density map (shape: (nx, ny, nz)).
            If not, returns the Fourier coefficients (shape: (n_masked, )).
        """
        if resolution <= 0.0:
            raise ValueError("Resolution must be greater than 0")

        self.set_filter_and_mask(resolution)

        if real:
            density = self.compute_density_real(X, elements)
            f_density = to_f_density(density)
            density = torch.abs(
                to_density(self.apply_filter_and_mask(f_density, shape_back=True))
            )
            return normalize(density) if to_normalize else density
        else:
            f_density = self.compute_density_fourier(X, elements)
            f_density = self.apply_filter_and_mask(f_density)
            return normalize(f_density) if to_normalize else f_density
