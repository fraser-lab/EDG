"""Calculate density map from atomic coordinates.

Author: Karson Chrispens
Created: 11 Nov 2024
Updated: 11 Nov 2024"""

import torch
import gemmi
from einops import rearrange, repeat
from adp3d.data.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ELEMENTS,
    IDX_TO_ELEMENT,
)


class DensityCalculator:
    def __init__(
        self,
        grid: gemmi.FloatGrid,
        center_shift: torch.Tensor,
        device: torch.device,
        em: bool = False,
    ):
        self.grid = grid
        self.center_shift = center_shift
        self.device = device
        self.em = em

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

        # Pre-compute grid coordinates
        self._setup_grid()

        # Precompute filter
        self.filter = None
        self.set_filter()

    def _setup_grid(self):
        """Store grid coordinates in a flat tensor for efficient computation."""
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
        self.grid_flat = rearrange(torch.stack(grid, dim=-1), "x y z c -> (x y z) c")

    def _compute_density_chunk(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
        C_expand: torch.Tensor,
        variance_scale: float = 1.0,
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
        variance_scale : float
            Multiplier on the Gaussian kernel variance. Use >1 for this to "smear" the
            density out. A value of 10 will give an approximately 10 Angstrom resolution map.
        chunk_size : int
            Number of atoms to process in a single chunk. Lower values reduce memory usage.

        Returns
        -------
        torch.Tensor
            Flattened density map with all chunks combined.
        """
        variance_scale = torch.tensor(variance_scale, device=self.device)

        n_chunks = (self.grid_flat.shape[0] + chunk_size - 1) // chunk_size
        density_chunks = []

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, self.grid_flat.shape[0])
            grid_chunk = self.grid_flat[start_idx:end_idx]

            # chunk distances
            distances = torch.cdist(grid_chunk, X)

            # flattened chunk grid
            density_chunk = torch.zeros(grid_chunk.shape[0], device=self.device)

            active_atoms = (elements != 5) & (C_expand == 1)
            for atom_idx in torch.where(active_atoms)[0]:
                e = IDX_TO_ELEMENT[elements[atom_idx].item()]
                r = distances[:, atom_idx]

                r_expanded = repeat(r, "r -> r n", n=self._range)
                density_contribution = torch.sum(
                    self.aw_dict[e].unsqueeze(0)
                    * torch.exp(
                        self.bw_dict[e].unsqueeze(0) * r_expanded**2 / variance_scale
                    ),
                    dim=1,
                )
                density_chunk += density_contribution

            density_chunks.append(density_chunk)

        return torch.cat(density_chunks)

    def compute_density(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
        C_expand: torch.Tensor,
        variance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute electron density map with optimized memory usage."""

        density_flat = self._compute_density_chunk(
            X, elements, C_expand, variance_scale
        )

        # Reshape back to 3D # TODO: are things like this a major slowdown?
        return rearrange(
            density_flat,
            "(x y z) -> x y z",
            x=self.grid.shape[0],
            y=self.grid.shape[1],
            z=self.grid.shape[2],
        )

    def _resolution_filter(self, resolution: float = 2.0) -> torch.Tensor:
        """Compute simple Gaussian resolution filter for density map."""
        if resolution <= 0:
            raise ValueError("Resolution must be greater than 0")

        # we want cutoff frequency to correspond to resolution
        sigma = resolution / (2 * torch.pi)

        d, h, w = self.grid.shape
        fd = torch.fft.fftfreq(d, self.grid.spacing[0], device=self.device)
        fh = torch.fft.fftfreq(h, self.grid.spacing[1], device=self.device)
        # using rfftn for density, so only need half the frequencies
        fw = torch.fft.rfftfreq(w, self.grid.spacing[2], device=self.device)
        f_xx, f_yy, f_zz = torch.meshgrid(fd, fh, fw, indexing="ij")
        f_sq = (f_xx**2 + f_yy**2 + f_zz**2) / sigma

        return torch.exp(-2 * torch.pi**2 * f_sq)
    
    def set_filter(self, resolution: float = 2.0):
        self.filter = self._resolution_filter(resolution)

    def compute_ll_density(
        self,
        X: torch.Tensor,
        elements: torch.Tensor,
        C_expand: torch.Tensor,
        target_density: torch.Tensor,
        variance_scale: float = 1.0,
        resolution: float = 2.0,
    ) -> torch.Tensor:
        """Compute log likelihood in Fourier space with minimal memory usage.

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
        variance_scale : float
            Multiplier on the Gaussian kernel variance. Use >1 for this to "smear" the
            density out. A value of 10 will give an approximately 10 Angstrom resolution map.
        resolution : float
            Resolution to compute the norm over, in Angstroms.
        """
        if resolution <= 0:
            raise ValueError("Resolution must be greater than 0")

        density = self.compute_density(X, elements, C_expand, variance_scale)
        diff = density - target_density

        # Compute FFT in chunks if needed
        diff_fft = torch.fft.rfftn( # rfftn is faster for real input
            diff, norm="forward"
        )  # "forward" does 1/N normalization, which we need with DFT

        return torch.sum(torch.real(diff_fft * torch.conj(diff_fft)))
