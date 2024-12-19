"""Implement the ADP-3D algorithm (Levy et al. 2024) for protein structure refinement using Chroma as a plug-and-play prior for reconstruction.

Author: Karson Chrispens (karson.chrispens@ucsf.edu)
Created: 6 Aug 2024
Updated: 20 Nov 2024
"""

from typing import Tuple, Union, List, Dict
import torch
import torch.nn.functional as F
import numpy as np
import os
import gemmi
import warnings
from einops import rearrange, reduce, repeat
from chroma import Chroma, Protein
from chroma.layers.structure.mvn import BackboneMVNGlobular

# from chroma.layers.structure.sidechain import SideChainBuilder
from chroma.constants import AA_GEOMETRY, AA20_3
from tqdm import tqdm
from adp3d.adp.density import DensityCalculator, to_density, to_f_density, normalize
from adp3d.utils.utility import try_gpu
from adp3d.data.sf import ATOMIC_NUM_TO_ELEMENT 
from adp3d.data.io import export_density_map, ma_cif_to_XCS


def identity_submatrices(N) -> Tuple[torch.Tensor]:
    """Generate all identity submatrices of size N x 3.

    Parameters
    ----------
    N : int
        number of rows in the identity submatrices, must be > 3.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor]
        (num_submatrices, N, 3) tensor of identity submatrices if N % 3 == 0, else
        (num_submatrices, 3, N) tensor of identity submatrices and the
        overhang submatrix of size (N, 3).

    Raises
    ------
    ValueError
        if N is less than 3.
    """
    # Ensure that N is at least 3
    if N < 3:
        raise ValueError(
            "N must be at least 3 to create identity submatrices of size N x 3."
        )
    overhang = N % 3
    if overhang != 0:
        base_submatrices = F.pad(
            torch.eye(N - overhang)
            .reshape((N - overhang) // 3, 3, N - overhang)
            .transpose(1, 2),
            (0, 0, 0, overhang),
            "constant",
            0,
        )
        extra_submatrix = torch.cat(
            [torch.zeros(N - overhang, 3), torch.eye(overhang, 3)], dim=0
        )
        return base_submatrices, extra_submatrix
    else:
        return torch.eye(N).reshape(N // 3, 3, N).transpose(1, 2), None


def _t(epoch: int, total_epochs: int) -> torch.Tensor:
    """Time schedule for the model refinement task.

    Parameters
    ----------
    epoch : int
        Current epoch.

    Returns
    -------
    torch.Tensor
        Current time.
    """
    return torch.tensor(
        (1.0 - 0.001) * (1.0 - np.sqrt(epoch / total_epochs)) + 0.001
    ).float()


def _resolution_schedule(
    epoch: int,
    total_epochs: int,
    start: float = 2.0,
    end: float = 2.0,
    activate_epoch: int = 0,
) -> float:
    """Resolution schedule for the model refinement task.

    Parameters
    ----------
    epoch : int
        Current epoch.
    total_epochs : int
        Total number of epochs.
    start : float, optional
        Starting resolution, by default 2.0.
    end : float, optional
        Ending resolution, by default 2.0.

    Returns
    -------
    float
        Current resolution.
    """
    if epoch < activate_epoch:
        return start
    a = (end - start) / (total_epochs - activate_epoch)
    b = start - activate_epoch * a
    return a * epoch + b


@torch.jit.script
def cos_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes the cosine similarity between two tensors.

    Parameters
    ----------
    a : torch.Tensor
    b : torch.Tensor

    Returns
    -------
    torch.Tensor
        Scaled scalar product of a and b.
    """
    return torch.real(
        torch.sum(a * torch.conj(b)) / torch.linalg.norm(a) / torch.linalg.norm(b)
    )


class ADP3D:
    def __init__(
        self,
        y: str,
        seq: torch.Tensor,
        structure: str,
        protein: Protein = None,
        all_atom: bool = False,
        em: bool = False,
        device: Union[str, torch.device] = None,
    ):
        """Initialize the ADP3D class.

        This class implements the Atomic Denoising Prior framework from Levy et al. 2024.

        Creating this class with a structure and map implies that the structure's atomic model is fit into the map.

        Parameters
        ----------
        y : str
            Input density measurement for incorporating density information, as a CCP4, MRC, or MTZ file.
        seq : torch.Tensor
            Sequence Tensor for incorporating sequence information, defined in Chroma's XC**S** format
        structure : str
            Input file path for an incomplete structure
        protein : Protein, optional
            Chroma Protein object for initializing with a given protein structure, by default None
        all_atom : bool, optional
            Whether to use all-atom coordinates, by default False
        em : bool, optional
            Whether to use electron scattering factors, by default False
        gpu : bool, optional
            Whether to use GPU, by default True
        """

        if device is None:
            self.device = try_gpu()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if self.device == torch.device("cpu"):
            warnings.warn(
                "Running on CPU. Consider using a GPU for faster computation."
            )

        self.em = em
        self.all_atom = all_atom
        # enforce input args
        check = [x is not None for x in [y, seq, structure]]
        if not any(check):
            raise ValueError("Arguments y, seq, and structure must be defined.")

        self.protein = protein

        self.seq = seq.to(self.device)
        self.x_bar, self.C_bar, _ = ma_cif_to_XCS(
            structure, all_atom=self.all_atom
        )
        self.x_bar = self.x_bar.to(self.device)
        self.C_bar = self.C_bar.to(self.device)

        # processing coordinates
        # TODO: (LATER PROCESS CHROMA I/O SO UNMODELED ATOMS ARE nan)
        flat_x_bar = rearrange(self.x_bar, "b r a c -> b (r a) c").squeeze()
        mask = flat_x_bar != 0
        values = torch.where(mask, flat_x_bar, torch.tensor(float("nan")))
        self.center_shift = torch.nanmean(
            values, dim=0
        )  # this will need to be applied to the map too so map and model are aligned
        self.x_bar -= self.center_shift  # centering
        self.x_bar = self.x_bar[torch.abs(self.C_bar) == 1].unsqueeze(
            0
        )  # get only chain 1

        # TODO: consider later maintaining only one sequence tensor
        self.seq = self.seq[torch.abs(self.C_bar) == 1].unsqueeze(0)  # get only chain 1
        self.C_bar = self.C_bar[torch.abs(self.C_bar) == 1].unsqueeze(
            0
        )  # get only chain 1

        # set x_bar to backbone
        if not self.all_atom:
            self.x_bar = self.x_bar[:, :, :4, :]

        self.density_extension = os.path.splitext(y)[1]
        if self.density_extension not in (".ccp4", ".map", ".mtz", ".cif"):
            warnings.warn("Density map must be a CCP4, MRC, SF-CIF, or MTZ file.")

        if self.density_extension in (
            ".mtz",
            ".cif",
        ):  # NOTE: MTZ or SF-CIF aren't used currently
            raise NotImplementedError(
                "MTZ and SF-CIF files are not yet supported for density input."
            )
        elif self.density_extension in (".ccp4", ".map", ".mrc"):
            map = gemmi.read_ccp4_map(y)
            map.setup(np.nan, gemmi.MapSetup.ReorderOnly) # necessary to get the proper spacing
            self.grid = map.grid
            if map.grid.spacing == (0.0, 0.0, 0.0):
                raise ValueError("Spacing of the density map is zero. Make sure your input map is properly processed.")
            self.y = normalize(
                torch.from_numpy(np.ascontiguousarray(self.grid.array)).to(
                    self.device, dtype=torch.float32
                )
            )
            self.f_y = normalize(to_f_density(self.y))
        else:
            raise ValueError("Density map must be a CCP4, MRC, SF-CIF, or MTZ file.")

        self.density_calculator = torch.jit.script(
            DensityCalculator(self.grid, self.center_shift, self.device, em=em)
        )

        # Importing sqrt(covariance matrix) from Chroma
        # These take in Z and C (for _multiply_R) or X and C (for _multiply_R_inverse)
        self.mvn = BackboneMVNGlobular(covariance_model="globular")
        self.multiply_corr = self.mvn._multiply_R
        self.multiply_inverse_corr = self.mvn._multiply_R_inverse

        # Build correlation matrix for incomplete structure log likelihood
        r = self.seq.size()[
            1
        ]  # Number of residues (from sequence, since we want total number in the correlation matrix)
        a = (
            4 # Chroma only diffuses on backbone
        )  # Number of atoms in one residue (backbone only)
        N = r * a  # Number of atoms in the protein backbone
        identity_matrices, overhang_matrix = identity_submatrices(N)
        overhang = N % 3

        Z = rearrange(identity_matrices, "b (r a) c -> b r a c", r=r, a=a).to(
            self.device
        )
        C = torch.ones(
            identity_matrices.size()[0], r, device=self.device
        )  # all residues in sequence should be active for this part

        self.R = rearrange(self.multiply_corr(Z, C), "b r a c -> (r a) (b c)")

        if overhang_matrix is not None:
            overhang_matrix = overhang_matrix.unsqueeze(0).to(self.device)
            overhang_matrix = rearrange(
                overhang_matrix, "b (r a) c -> b r a c", r=r, a=a
            )
            last_R_columns = rearrange(
                self.multiply_corr(overhang_matrix, self.C_bar).squeeze(),
                "r a c -> (r a) c",
            )[
                :, :overhang
            ]  # should be shape N x overhang for concat with identity_matrices

            self.R = torch.cat([self.R, last_R_columns], dim=1)

        # Build measurement matrix for incomplete structure log likelihood
        # Going to only account for backbone atoms, so 4 atoms per residue. Could extend to all atom later, I think.
        mask = torch.ones(r, device=self.device).float()
        if self.C_bar.size()[1] != r:
            raise ValueError(
                "Number of residues in input coordinates must match number of residues in sequence."
            )
        mask[(self.C_bar != 1).reshape(-1)] = 0.0  # TODO: this requires CIF right now
        mask = repeat(mask, "m -> (m repeat)", repeat=4)
        self.A = torch.diag(mask)
        self.A = self.A[self.A.any(dim=1)]

        # Precompute SVD for incomplete structure log likelihood
        AR = self.A @ self.R
        # U shape (m, m), S shape (m, n), V_T shape (n, n)
        self.U, S_, self.V_T = torch.linalg.svd(AR)
        if len(S_.size()) == 1:
            S_ = torch.diag(S_)
            # pad columns of S_ with zeros to match shape of U
            S_ = F.pad(S_, (0, self.U.size()[0] - S_.size()[0]), "constant", 0)
        self.S_plus = torch.linalg.pinv(S_)

        # Initialize Chroma
        self.chroma = Chroma(device=self.device)
        self.denoiser = self.chroma.backbone_network.denoise
        self.noise_scheduler = (
            self.chroma.backbone_network.noise_perturb._schedule_coefficients
        )
        self.sequence_sampler = None  # NOTE: Placeholder for sequence sampling. This will need to be implemented if doing model refinement with a denoiser other than Chroma
        self.chi_sampler = None  # NOTE: Placeholder for chi angle sampling. This will need to be implemented if doing model refinement with a denoiser other than Chroma
        self.sequence_chi_sampler = (
            self.chroma.design_network.sample
        )  # In Chroma, this gets all atom coordinates. Has nograd decorator.
        self.sequence_loss = (
            self.chroma.design_network.forward
        )  # .loss() runs forward anyway, might as well just use forward
        # self.chi_to_X = (
        #     SideChainBuilder()
        # )  # Chroma's sidechain builder for making all-atom coordinates from backbone and chi angles.

    def ll_incomplete_structure(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the log likelihood of the incomplete structure given the denoised coordinates.

        Parameters
        ----------
        z : torch.Tensor
            Denoised coordinates.

        Returns
        -------
        torch.Tensor
            Log likelihood of the incomplete structure.
        """
        z = rearrange(z, "b r a c -> b (r a) c").squeeze()
        x_bar = rearrange(
            self.x_bar[self.C_bar == 1].unsqueeze(0)[:, :, :4, :],
            "b r a c -> b (r a) c",
        ).squeeze()  # TODO: take only modeled residues, requires CIF
        # model backbone only for this
        denoised = self.A @ self.V_T @ z
        denoised = denoised[denoised.any(dim=1)]
        measured = self.S_plus @ self.U.T @ x_bar
        return -torch.linalg.vector_norm(torch.flatten(denoised - measured)) ** 2

    def grad_ll_incomplete_structure(self, z: torch.Tensor) -> torch.Tensor:

        with torch.enable_grad():
            if z.requires_grad == False:
                z.requires_grad_(True)

            result = self.ll_incomplete_structure(z)
            result.backward()

        return z.grad, result

    def ll_sequence(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute the log likelihood of the sequence given the denoised coordinates.

        Parameters
        ----------
        s : torch.Tensor
            Sequence.
        z : torch.Tensor
            Denoised coordinates.

        Returns
        -------
        torch.Tensor
            Log likelihood of the sequence.
        """
        raise NotImplementedError(
            "Sequence log likelihood not implemented as Chroma has built in log likelihood."
        )

    def grad_ll_sequence(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of the log likelihood of the sequence given the denoised coordinates.

        Parameters
        ----------
        s : torch.Tensor
            Sequence.
        z : torch.Tensor
            Denoised coordinates.

        Returns
        -------
        torch.Tensor
            Gradient of the log likelihood of the sequence w.r.t. denoised coordinates z.
        """
        # s.requires_grad = False
        # result = self.ll_sequence(s, z)

        # result.backward(inputs=z)
        # return z.grad
        raise NotImplementedError(
            "Sequence gradient not implemented as Chroma has built in log likelihood."
        )

    def _extract_elements(self, all_atom: bool = False) -> torch.Tensor:
        """Extract elements from the denoised protein coordinates and, if all atom, sequence.

        Parameters
        ----------
        all_atom : bool, optional
            Whether the coordinates are backbone or all-atom, by default False

        Returns
        -------
        torch.Tensor
            Extracted elements for each residue, shape (residues, (4 or 14) elements), in order of Chroma's AA_GEOMETRY definition.
        """

        pass # TODO: DELETE AS NO LONGER NEEDED W/O CHROMA

    def _gamma(
        self,
        X: torch.Tensor,
        all_atom: bool = False,
        resolution: float = 2.0,
        real: bool = False,
    ) -> torch.Tensor:
        """Compute electron density map of the coordinates X.

        Args:
            X (torch.Tensor): The input all-atom protein coordinates in shape (batch, residues, atoms, 3).
            size (Tuple[int]): The desired size of the output density map.
            all_atom (bool): Whether the input coordinates are all-atom or backbone only.
            resolution (float): The resolution of the density map.
            real (bool): Whether to return the real space density map or the Fourier space density map.
        Returns:
            torch.Tensor: The density map.
        """

        if X.size()[2] == 4 and all_atom:
            raise ValueError(
                "Input coordinates are backbone only, all_atom must be False."
            )
        if len(X.size()) == 4:
            X = rearrange(
                X, "b r a c -> b (r a) c"
            ).squeeze()  # if b = 1, should result in (r a, c)
        else:
            X = X.squeeze()  # assuming already in shape (r a, c)

        if len(X.size()) != 2:
            raise ValueError(
                "Input coordinates must be of shape (residues * atoms, 3). Batched X is not yet supported in _gamma."
            )

        if all_atom and X.size()[0] != self.x_bar.size()[1] * self.x_bar.size()[2]:
            print(X.size(), self.x_bar.size())
            raise ValueError(
                "Number of atoms in input coordinates must match number of atoms in structure."
            )
        elif not all_atom and X.size()[0] != self.x_bar.size()[1] * 4:
            print(X.size(), self.x_bar.size())
            raise ValueError(
                "Number of atoms in input coordinates must match number of backbone atoms in structure."
            )

        elements = rearrange(self._extract_elements(all_atom), "r a -> (r a)")
        C_expand = repeat(
            self.C_bar, "b r -> b (r a)", a=14 if all_atom else 4
        ).squeeze()  # expand to all atoms, squeeze batch dim out

        density = self.density_calculator.forward(
            X, elements, C_expand, resolution=resolution, real=real, to_normalize=True
        )

        return density

    def grad_ll_density(
        self,
        X: torch.Tensor,
        all_atom: bool = False,
        resolution: float = 2.0,
        real: bool = False,
        loss: str = "l2",
    ) -> torch.Tensor:
        """Compute the gradient of the log likelihood of the density given the atomic coordinates and side chain angles.

        Parameters
        ----------

        X : torch.Tensor
            Denoised coordinates.
        all_atom : bool, optional
            Whether the coordinates are all-atom or backbone only, by default False.
        resolution : float, optional
            The resolution of the density map, by default 2.0.
        real : bool, optional
            Whether to return the real space density map or the Fourier space density map, by default False.
        loss : str, optional
            Loss function to use for the density map, by default "l2". Options are "l2" and "cosine" for L2 norm and cosine similarity, respectively.

        Returns
        -------
        torch.Tensor
            Gradient of the log likelihood of the density.
        """

        # setup resolution filter
        self.density_calculator.set_filter_and_mask(resolution)

        # take elements and C_expand out of gradient computation
        if X.size()[2] == 4 and all_atom:
            raise ValueError(
                "Input coordinates are backbone only, all_atom must be False."
            )
        if len(X.size()) == 4:
            X = rearrange(
                X, "b r a c -> b (r a) c"
            ).squeeze()  # if b = 1, should result in (r a, c)
        else:
            X = X.squeeze()  # assuming already in shape (r a, c)

        if len(X.size()) != 2:
            raise ValueError(
                "Input coordinates must be of shape (residues * atoms, 3). Batched X is not yet supported in _gamma."
            )

        if all_atom and X.size()[0] != self.x_bar.size()[1] * self.x_bar.size()[2]:
            print(X.size(), self.x_bar.size())
            raise ValueError(
                "Number of atoms in input coordinates must match number of atoms in structure."
            )
        elif not all_atom and X.size()[0] != self.x_bar.size()[1] * 4:
            print(X.size(), self.x_bar.size())
            raise ValueError(
                "Number of atoms in input coordinates must match number of backbone atoms in structure."
            )

        elements = rearrange(self._extract_elements(all_atom), "r a -> (r a)")
        C_expand = repeat(
            self.C_bar, "b r -> b (r a)", a=14 if all_atom else 4
        ).squeeze()  # expand to all atoms, squeeze batch dim out

        if real:
            y = normalize(
                torch.abs(
                    to_density(
                        self.density_calculator.apply_filter_and_mask(
                            self.f_y, shape_back=True
                        )
                    )
                )
            )
        else:
            f_y = self.density_calculator.apply_filter_and_mask(self.f_y)

        with torch.enable_grad():
            X = X.clone().detach().requires_grad_(True)
            if X.is_cuda:  # use autocast for mixed precision
                with torch.autocast(device_type="cuda"):
                    density = normalize(
                        self.density_calculator.forward(
                            X, elements, C_expand, resolution=resolution, real=real
                        )
                    )
            else:
                density = self.density_calculator.forward(
                    X, elements, C_expand, resolution=resolution, real=real
                )
            if loss == "l2":
                result = (
                    -torch.sum((density - y) ** 2)
                    if real
                    else -torch.sum(torch.abs(density - f_y) ** 2)
                )
            elif loss == "cosine":
                result = (
                    cos_similarity(density, y)
                    if real
                    else cos_similarity(
                        density, f_y
                    )  # this may break due to torch.abs on the cos_similarity when in Fourier space
                )
            result.backward()

            if self.all_atom:
                grad = rearrange(
                    X.grad, "(b r a) c -> b r a c", r=X.size()[0] // 14, a=14, b=1
                )  # FIXME: not batchable ATM
            else:
                grad = rearrange(
                    X.grad, "(b r a) c -> b r a c", r=X.size()[0] // 4, a=4, b=1
                )

        return grad, result

    def model_refinement_optimizer(
        self,
        epochs: int = 4000,
        lr_m_s_d: List[float] = [1e-2, 1e-5, 1e-2],
        momenta: List[float] = [9e-1] * 3,
        map_resolution: float = 2.0,
        output_dir: str = "output",
    ) -> Tuple[Protein, List[float], List[float], List[float]]:
        """Use gradient descent with momentum to optimize log likelihood functions for model refinement task along with denoised coordinates.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs, by default 4000
        lr_m_s_d : List[float], optional
            Learning rates for each log likelihood function,
            first for incomplete *m*odel, then *s*equence, then *d*ensity. By default [1e-1, 1e-5, 3e-5]
        momenta : List[float], optional
            Momenta for gradient descent updates, by default [9e-1]*3
        map_resolution : float, optional
            Resolution to compare the density map at by the end of the resolution schedule, by default 2.0.
        output_dir : str, optional
            Output directory for saving refined protein structures, by default "output

        Returns
        -------
        Protein, List[float], List[float], List[float]
            Denoised and data matched protein structure.
            List of log likelihoods. Order is [incomplete structure, density, sequence].
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.protein is not None:
            X, C, _ = self.protein.to_XCS(
                device=self.device
            )  # handle GT protein initialization
        else:
            Z = torch.randn_like(self.x_bar[:, :, :4, :], device=self.device)
            X = self.multiply_corr(Z, self.C_bar)
            C = torch.ones_like(
                self.C_bar, device=self.device
            )  # NOTE: self.C_bar provides correct chains and masking
        S = self.seq

        loss_d = []
        loss_s = []
        loss_m = []

        # FIXME fix chroma CIF I/O
        Protein.from_XCS(X + self.center_shift, C, S).to_PDB(
            os.path.join(output_dir, "init.pdb")
        )

        v_i_m = torch.zeros(X.size(), device=self.device)
        v_i_s = torch.zeros(X.size(), device=self.device)
        v_i_d = torch.zeros(X.size(), device=self.device)

        pbar = tqdm(range(epochs), desc="Model Refinement")
        for epoch in pbar:
            # Denoise coordinates
            t = _t(epoch, epochs).to(self.device)

            with torch.no_grad():
                X_0 = self.denoiser(X.detach(), C, t)

            # transform denoised coordinates to whitened space
            z_0 = self.multiply_inverse_corr(X_0, C)

            # sample chi angles
            if epoch % 100 == 0:
                X_aa, _, _ = self.sequence_chi_sampler(
                    X_0, C, S, t=0.0, return_scores=False, resample_chi=True
                )

            # sequence matching
            if epoch >= 3000:
                with torch.enable_grad():
                    X_aa.requires_grad_(True)
                    ll_sequence = torch.sum(
                        self.sequence_loss(X_aa, C, S, t=0.0)["logp_S"]
                    )  # Get sequence log probability
                    ll_sequence.backward()
                    grad_ll_sequence = X_aa.grad

                grad_ll_sequence = self.multiply_inverse_corr(
                    grad_ll_sequence[:, :, :4, :], C
                )  # Get gradient of log likelihood of sequence in whitened space
            else:
                ll_sequence = torch.tensor(0.0, device=self.device)
                grad_ll_sequence = torch.zeros_like(z_0, device=self.device)

            ### Accumulate gradients

            # incomplete structure
            incomplete_structure_grad, ll_incomplete_structure = (
                self.grad_ll_incomplete_structure(z_0)
            )
            v_i_m = momenta[0] * v_i_m + lr_m_s_d[0] * incomplete_structure_grad

            # sequence
            v_i_s = (
                momenta[1] * v_i_s + lr_m_s_d[1] * grad_ll_sequence
            )  # NOTE: This should change if a model other than Chroma is used.

            current_resolution = _resolution_schedule(
                epoch, epochs, start=15.0, end=map_resolution, activate_epoch=0
            )

            # density
            if self.all_atom:
                density_grad, density_loss = self.grad_ll_density(
                    X_aa,
                    all_atom=True,
                    resolution=current_resolution,
                    real=True,  # TODO: change to fourier space eventually
                    loss="cosine",
                )
                density_grad_transformed = self.multiply_inverse_corr(
                    density_grad[:, :, :4, :],
                    C,  # get the backbone coords to update
                )
            else:
                density_grad, density_loss = self.grad_ll_density(
                    X_0,
                    all_atom=False,
                    resolution=current_resolution,
                    real=True,  # TODO: change to fourier space eventually
                    loss="cosine",
                )
                density_grad_transformed = self.multiply_inverse_corr(
                    density_grad[:, :, :4, :],
                    C,  # get the backbone coords to update
                )

            lr_density = lr_m_s_d[2] * (current_resolution / map_resolution) ** 3
            v_i_d = momenta[2] * v_i_d + lr_density * density_grad_transformed

            # Update denoised coordinates
            z_t_1 = z_0 + v_i_m + v_i_s + v_i_d

            # Add noise (this is the same as what OG ADP3D does)
            t1 = _t(epoch + 1, epochs).to(self.device)
            alpha, sigma, _, _, _, _ = self.noise_scheduler(t1)
            X = self.multiply_corr(alpha * z_t_1 + sigma * torch.randn_like(z_t_1), C)

            if (
                epoch % 50 == 0
            ):  # ~40 structure is minimum for a good diffusion trajectory
                Protein.from_XCS(X + self.center_shift, C, S).to_PDB(
                    os.path.join(output_dir, f"output_epoch_{epoch}.pdb")
                )
            pbar.set_postfix(
                {
                    "d": density_loss.item(),
                    "s": ll_sequence.item(),
                    "m": ll_incomplete_structure.item(),
                }
            )
            loss_m.append(ll_incomplete_structure.item())
            loss_d.append(density_loss.item())
            loss_s.append(ll_sequence.item())

        X_aa = self.sequence_chi_sampler(
            X, C, S, t=0.0, return_scores=False, resample_chi=True
        )[0]
        final_density = torch.real(
            self._gamma(X_aa, all_atom=True, resolution=map_resolution, real=True)
        )
        X_aa += self.center_shift
        MAP_protein = Protein.from_XCS(X_aa, C, S)

        export_density_map(
            final_density,
            self.grid,
            os.path.join(output_dir, "final_density.ccp4"),
        )

        return MAP_protein, loss_m, loss_d, loss_s
