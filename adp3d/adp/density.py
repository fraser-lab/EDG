"""Calculate density map from atomic coordinates.

Author: Karson Chrispens
Created: 11 Nov 2024
Updated: 20 Nov 2024"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gemmi
from einops import rearrange, repeat
from adp3d.data.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ELEMENTS,
    IDX_TO_ELEMENT,
)
import warnings

def to_f_density(map):
    """FFT a density map."""
    # f_density
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(map, dim=(-3,-2,-1)), dim=(-3,-2,-1)), dim=(-3,-2,-1))

def to_density(f_map):
    """Inverse FFT a FFTed density map."""
    # density
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(f_map, dim=(-3,-2,-1)), dim=(-3,-2,-1)), dim=(-3,-2,-1))

class DensityCalculator(nn.Module):
    def __init__(
        self,
        grid: gemmi.FloatGrid,
        center_shift: torch.Tensor,
        device: torch.device,
        em: bool = False,
        unpad: int = 0,
    ):
        super(DensityCalculator, self).__init__()
        self.grid = grid
        self.center_shift = center_shift
        self.device = device
        self.em = em
        self.unpad = unpad

        # constants
        self.four_pi2 = 4 * torch.pi**2
        self._range = 5 if em else 6
        self.sf = ELECTRON_SCATTERING_FACTORS if em else ATOM_STRUCTURE_FACTORS

        # element dictionaries
        self.aw_dict = {}
        self.bw_dict = {}
        for e in set(IDX_TO_ELEMENT.values()):
            if e == "nan":
                continue
            self.bw_dict[e] = torch.tensor(
                [
                    -self.four_pi2 / b if b > 1e-4 else 0
                    for b in self.sf[e][1][: self._range]
                ],
                device=device,
            )
            self.aw_dict[e] = (
                torch.tensor(self.sf[e][0][: self._range], device=device)
                * (-self.bw_dict[e] / torch.pi) ** 1.5
            )

        # pre-compute grid coordinates
        self._setup_real_grid()
        self._setup_fourier_grid()

        # pre-compute filter
        self.filter = None
        self.set_filter_and_mask()

    def _setup_real_grid(self):
        """Setup flattened real space grid."""
        origin = (
            torch.tensor([*self.grid.get_position(0, 0, 0)], device=self.device)
            - self.center_shift
        )

        if self.grid.spacing[0] > 0:
            x = torch.arange(
                origin[0],
                origin[0] + self.grid.spacing[0] * self.grid.shape[0],
                self.grid.spacing[0],
                device=self.device,
            )[: self.grid.shape[0]]
            y = torch.arange(
                origin[1],
                origin[1] + self.grid.spacing[1] * self.grid.shape[1],
                self.grid.spacing[1],
                device=self.device,
            )[: self.grid.shape[1]]
            z = torch.arange(
                origin[2],
                origin[2] + self.grid.spacing[2] * self.grid.shape[2],
                self.grid.spacing[2],
                device=self.device,
            )[: self.grid.shape[2]]
        else:
            x = torch.linspace(
                origin[0], self.grid.unit_cell.a, self.grid.shape[0], device=self.device
            )
            y = torch.linspace(
                origin[1], self.grid.unit_cell.b, self.grid.shape[1], device=self.device
            )
            z = torch.linspace(
                origin[2], self.grid.unit_cell.c, self.grid.shape[2], device=self.device
            )

        if (
            x.size()[0] != self.grid.shape[0]
            or y.size()[0] != self.grid.shape[1]
            or z.size()[0] != self.grid.shape[2]
        ):
            raise ValueError(
                f"Size of the density map is not matching the input size: ({x.size()[0]}, {y.size()[0]}, {z.size()[0]}) is not {self.grid.shape}"
            )

        grid = torch.meshgrid(x, y, z, indexing="ij")
        self.real_grid_flat = rearrange(torch.stack(grid, dim=-1), "x y z c -> (x y z) c")

    def _setup_fourier_grid(self):
        """Setup Fourier space frequency grid."""
        nx, ny, nz = self.grid.shape
        
        # Compute frequency axes
        fx = torch.fft.fftshift(torch.fft.fftfreq(nx, self.grid.spacing[0], device=self.device))
        fy = torch.fft.fftshift(torch.fft.fftfreq(ny, self.grid.spacing[1], device=self.device))
        fz = torch.fft.fftshift(torch.fft.fftfreq(nz, self.grid.spacing[2], device=self.device))
        
        # Create frequency grid
        fxx, fyy, fzz = torch.meshgrid(fx, fy, fz, indexing='ij')
        self.freq_grid = torch.stack([fxx, fyy, fzz], dim=-1)
        self.freq_norm = torch.sqrt(torch.sum(self.freq_grid ** 2, dim=-1))

    def _compute_density_chunk(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
        C_expand: torch.Tensor,
        chunk_size: int = 1000000,
    ) -> torch.Tensor:
        """Compute density in chunks to reduce memory usage.

        Parameters
        ----------

        X : torch.Tensor
            Coordinates of atoms. Shape (n_atoms, 3).
        elements : torch.Tensor
            Element indices of atoms.
        C_expand : torch.Tensor
            Mask for atoms to include in density calculation depending on chains in protein.
        chunk_size : int
            Number of atoms to process in a single chunk. Lower values reduce memory usage.

        Returns
        -------
        torch.Tensor
            Flattened density map with all chunks combined.
        """

        n_chunks = (self.grid_flat.shape[0] + chunk_size - 1) // chunk_size
        density_chunks = []

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, self.grid_flat.shape[0])
            grid_chunk = self.grid_flat[start_idx:end_idx]

            # chunk distances
            distances = torch.cdist(grid_chunk, X) # this is expensive!!

            # flattened chunk grid
            density_chunk = torch.zeros(grid_chunk.shape[0], device=self.device)

            active_atoms = (elements != 5) & (C_expand == 1)
            for atom_idx in torch.where(active_atoms)[0]:  # TODO: vectorize
                e = IDX_TO_ELEMENT[elements[atom_idx].item()]
                r = distances[:, atom_idx]

                r_expanded = repeat(r, "r -> r n", n=self._range)
                density_contribution = torch.sum(
                    self.aw_dict[e].unsqueeze(0)
                    * torch.exp(
                        self.bw_dict[e].unsqueeze(0) * r_expanded**2
                    ),
                    dim=1,
                )
                density_chunk += density_contribution

            density_chunks.append(density_chunk)

        return torch.cat(density_chunks)

    def _compute_f_density_chunk(self,
        X: torch.Tensor,
        elements: torch.Tensor,
        C_expand: torch.Tensor,
        chunk_size: int = 1000000,
    ) -> torch.Tensor:
        """Compute Fourier transform of density.

        Parameters
        ----------

        X : torch.Tensor
            Coordinates of atoms. Shape (n_atoms, 3).
        elements : torch.Tensor
            Element indices of atoms.
        C_expand : torch.Tensor
            Mask for atoms to include in density calculation depending on chains in protein.
        chunk_size : int
            Number of atoms to process in a single chunk. Lower values reduce memory usage.

        Returns
        -------
        torch.Tensor
            Fourier coefficients in 3D array.
        """
        active_atoms = (elements != 5) & (C_expand == 1) # (n_active_atoms, )
        active_X = X[active_atoms] # (n_active_atoms, 3)
        active_elements = elements[active_atoms] # (n_active_atoms, )

        atom_symbols = [IDX_TO_ELEMENT[e.item()] for e in active_elements] # TODO: handle elements better

        a_coeffs = torch.stack([self.aw_dict[s] for s in atom_symbols], dim=0) # (n_active_atoms, _range)
        b_coeffs = torch.stack([self.bw_dict[s] for s in atom_symbols], dim=0) # (n_active_atoms, _range)

        nx, ny, nz, _ = self.freq_grid.shape
        n_freq_points = nx * ny * nz
        n_chunks = (n_freq_points + chunk_size - 1) // chunk_size

        f_density = torch.zeros(n_freq_points, dtype=torch.complex64, device=self.device)

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_freq_points)

            freq_chunk = rearrange(self.freq_grid, "x y z c -> (x y z) c")[start_idx:end_idx] # (n_freq_points, 3)
            freq_norm_chunk = rearrange(self.freq_norm, "x y z -> (x y z)")[start_idx:end_idx] # (n_freq_points, )

            f_density_chunk = torch.zeros(freq_chunk.shape[0], dtype=torch.complex64, device=self.device)

            phase = -2j * torch.pi * torch.einsum('fc,ac->fa', freq_chunk, active_X) # (n_freq_points, n_active_atoms)

            freq_norm_sq = freq_norm_chunk[:, None] ** 2 # (n_freq_points, 1)

            # expand b_coeffs and freq_norm_sq to match: (1, n_active_atoms, _range) and (n_freq_points, 1, 1)
            gaussian_terms = torch.exp(b_coeffs[None, :, :] * freq_norm_sq[:, None, :]) # (n_freq_points, n_active_atoms, _range)

            # expand a_coeffs to match: (1, n_active_atoms, _range)
            form_factors = torch.sum(a_coeffs[None, :, :] * gaussian_terms, dim=-1) # (n_freq_points, n_active_atoms)

            f_density_chunk = torch.sum(form_factors * torch.exp(phase), dim=-1) # (n_freq_points, )

            f_density[start_idx:end_idx] = f_density_chunk

        return rearrange(f_density, "(x y z) -> x y z", x=nx, y=ny, z=nz)

    def compute_density_real(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
        C_expand: torch.Tensor,
    ) -> torch.Tensor:
        """Compute electron density map with optimized memory usage.
        
        Parameters
        ----------
        X : torch.Tensor
            Coordinates of atoms. Shape (n_atoms, 3).
        elements : torch.Tensor
            Element indices of atoms.
        C_expand : torch.Tensor
            Mask for atoms to include in density calculation depending on chains in protein.
            
        Returns
        -------
        torch.Tensor
            3D density map."""

        density_flat = self._compute_density_chunk(
            X, elements, C_expand 
        )

        # Reshape back to 3D # TODO: are things like this a major slowdown?
        return rearrange(
            density_flat,
            "(x y z) -> x y z",
            x=self.grid.shape[0],
            y=self.grid.shape[1],
            z=self.grid.shape[2],
        )

    def compute_density_fourier(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
        C_expand: torch.Tensor,
    ) -> torch.Tensor:
        """Compute electron density map in Fourier space."""

        f_density = self._compute_f_density_chunk(
            X, elements, C_expand
        )

        return f_density

    def _resolution_filter(self, resolution: float = 2.0) -> torch.Tensor:
        """Gaussian filter with cosine cutoff for density map."""
        if resolution <= 0:
            raise ValueError("Resolution must be greater than 0")

        # we want cutoff frequency to correspond to resolution
        sigma = resolution / (2 * torch.pi)
        cutoff_radius = 2 / (torch.pi * sigma)

        d, h, w = self.grid.shape
        fd = torch.fft.fftshift(torch.fft.fftfreq(d, self.grid.spacing[0], device=self.device))
        fh = torch.fft.fftfreq(torch.fft.fftfreq(h, self.grid.spacing[1], device=self.device))
        fw = torch.fft.fftshift(torch.fft.fftfreq(w, self.grid.spacing[2], device=self.device))

        nyquist_d = torch.abs(fd).max()
        nyquist_h = torch.abs(fh).max()
        nyquist_w = torch.abs(fw).max()
        min_nyquist = min(nyquist_d, nyquist_h, nyquist_w)

        if cutoff_radius > min_nyquist:
            warnings.warn(
                f"Cutoff radius ({cutoff_radius:.3f}) exceeds the minimum Nyquist frequency ({min_nyquist:.3f}). "
                f"This may lead to aliasing. Consider using a larger resolution value (current: {resolution})."
            )

        f_xx, f_yy, f_zz = torch.meshgrid(fd, fh, fw, indexing="ij")
        f_sq = (f_xx**2 + f_yy**2 + f_zz**2)
        f_mag = torch.sqrt(f_sq)

        gauss = torch.exp(-2 * torch.pi**2 * f_sq * sigma ** 2)

        # cosine falloff
        falloff_width = cutoff_radius * 0.2 # can tweak this
        falloff_mask = (f_mag >= cutoff_radius) & (f_mag < cutoff_radius + falloff_width)
        t = (f_mag[falloff_mask] - cutoff_radius) / falloff_width
        gauss[falloff_mask] *= 0.5 * (1 + torch.cos(t * torch.pi))

        # zero freq above nyquist
        mask = torch.ones_like(gauss)
        mask[f_mag >= cutoff_radius + falloff_width] = 0
        gauss *= mask

        return gauss, mask

    def set_filter_and_mask(self, resolution: float = 2.0):
        self.filter, self.mask = self._resolution_filter(resolution)

    def apply_filter_and_mask(self, f_density: torch.Tensor) -> torch.Tensor:
        """Apply resolution filter and mask to Fourier density.
        
        Parameters
        ----------
        f_density : torch.Tensor
            3D Fourier coefficients of density map.
            
        Returns
        -------
        torch.Tensor
            Filtered and masked Fourier coefficients."""
        if self.filter is None:
            raise ValueError("Filter not set. Run set_filter() first.")

        f_density *= self.filter

        return f_density[self.mask]

    def forward(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
        C_expand: torch.Tensor,
        resolution: float = None,
        real: bool = False,
    ) -> torch.Tensor:
        """Compute density map up to given resolution.

        Parameters
        ----------

        X : torch.Tensor
            Coordinates of atoms.
        elements : torch.Tensor
            Element indices of atoms.
        C_expand : torch.Tensor
            Mask for atoms to include in density calculation depending on chains in protein.
        target_density : torch.Tensor
            Target density map.
        resolution : float
            Resolution to compute the norm over, in Angstroms.
        """
        if resolution <= 0:
            raise ValueError("Resolution must be greater than 0")

        if self.filter is None and resolution is not None:
            warnings.warn("Setting filter based on resolution provided.")
            self.set_filter_and_mask(resolution)
        elif self.filter is None:
            raise ValueError("Filter not set and resolution not provided. Run set_filter() first.")

        if real:
            density = self.compute_density_real(X, elements, C_expand)
            f_density = to_f_density(density)
            density = to_density(self.apply_filter_and_mask(f_density))
            return density
        else:
            f_density = self.compute_density_fourier(X, elements, C_expand)
            f_density = self.apply_filter_and_mask(f_density)
            return f_density
