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
from adp3d.adp.density import DensityCalculator
from adp3d.utils.utility import DotDict, try_gpu
from adp3d.data.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ELEMENTS,
    IDX_TO_ELEMENT,
)


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
        self.x_bar, self.C_bar, S = Protein(structure).to_XCS(
            all_atom=True, device=self.device
        )

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
        S = S[torch.abs(self.C_bar) == 1].unsqueeze(0)  # get only chain 1

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
            self.grid = map.grid
            self.y = torch.from_numpy(np.ascontiguousarray(self.grid.array)).to(
                self.device, dtype=torch.float32
            )
        else:
            raise ValueError("Density map must be a CCP4, MRC, SF-CIF, or MTZ file.")

        self.density_calculator = DensityCalculator(
            self.grid, self.center_shift, self.device, em
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
            self.x_bar.size()[2] if not self.all_atom else 4
        )  # Number of atoms in one residue
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

        # Precompute SVD for incomplete structure log likelihood
        AR = self.A @ self.R
        AR = AR[AR.any(dim=1)]
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

        return z.grad

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

        bb_elements = [ELEMENTS[a] for a in ["N", "C", "C", "O"]]
        if all_atom:
            elements = torch.zeros(
                self.seq.size()[1], 14, device=self.device, dtype=torch.int8
            )  # 14 is max number of atoms in residue from chroma
            # assign each residue elements to all atoms
            seq = self.seq[
                0, :
            ]  # TODO: put this in init eventually, going to need to change all instances where dim 1 of seq is accessed
            for i in range(seq.size()[0]):
                atoms = AA_GEOMETRY[AA20_3[seq[i]]]["atoms"]
                residue_elements = bb_elements + [ELEMENTS[a[:1]] for a in atoms]
                # pad up to 14
                residue_elements += [5] * (
                    14 - len(residue_elements)
                )  # 5 is the "nan" element # TODO: this can be done better I think
                elements[i] = torch.tensor(residue_elements, device=self.device)
        else:
            elements = torch.zeros(
                self.seq.size()[1], 4, device=self.device, dtype=torch.int8
            )  # backbone elements

            # assign each residue elements to backbone atoms
            for i in range(self.seq.size()[1]):
                elements[i] = torch.tensor(bb_elements, device=self.device)
        return elements

    def _gamma(
        self,
        X: torch.Tensor,
        all_atom: bool = False,
        variance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute electron density map of the all-atom coordinates X.

        Args:
            X (torch.Tensor): The input all-atom protein coordinates in shape (batch, residues, atoms, 3).
            size (Tuple[int]): The desired size of the output density map.
            all_atom (bool): Whether the input coordinates are all-atom or backbone only.
            variance_scale (float): Multiplier on the Gaussian kernel variance. Use >1 for this to "smear" the
            density out. A value of 10 will give an approximately 10 Angstrom resolution map.

        Returns:
            torch.Tensor: The density map.
        """

        variance_scale = torch.tensor(variance_scale, device=self.device)

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

        density = self.density_calculator.compute_density(
            X, elements, C_expand, variance_scale
        )

        return density

    def ll_density_real(self, X: torch.Tensor) -> torch.Tensor:
        """OLD: Real space analog of ll_density.
        Compute the log likelihood of the density given the atomic coordinates and side chain angles.

        Parameters
        ----------
        X : torch.Tensor
            Denoised coordinates.

        Returns
        -------
        torch.Tensor
            Log likelihood of the density.
        """

        density = self._gamma(
            X, all_atom=self.all_atom
        )  # Get density map from denoised coordinates

        if density.size() != self.y.size():
            raise ValueError("Density map and input density map must be the same size.")

        diff = density - self.y

        return -torch.linalg.norm(diff) ** 2

    def ll_density(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the log likelihood of the density given the atomic coordinates and side chain angles.

        Parameters
        ----------
        X : torch.Tensor
            Denoised coordinates.

        Returns
        -------
        torch.Tensor
            Log likelihood of the density.
        """
        if len(X.size()) == 4:
            X = rearrange(X, "b r a c -> b (r a) c").squeeze()  # b, N, 3

        # FIXME: MTZ and SF-CIF not implemented

        elements = rearrange(self._extract_elements(self.all_atom), "r a -> (r a)")
        C_expand = repeat(
            self.C_bar, "b r -> b (r a)", a=14 if self.all_atom else 4
        ).squeeze()

        if X.is_cuda:  # use autocast for mixed precision, should be a cheaper operation
            with torch.autocast(device_type="cuda"):
                result = self.density_calculator.compute_ll_density(
                    X, elements, C_expand, self.y
                )
        else:
            result = self.density_calculator.compute_ll_density(
                X, elements, C_expand, self.y
            )

        return result

    def grad_ll_density(self, X: torch.Tensor, resolution: float = 2.0) -> torch.Tensor:
        """Compute the gradient of the log likelihood of the density given the atomic coordinates and side chain angles.

        Parameters
        ----------

        X : torch.Tensor
            Denoised coordinates.

        Returns
        -------
        torch.Tensor
            Gradient of the log likelihood of the density.
        """

        # setup resolution filter
        self.density_calculator.set_filter(resolution)

        # take elements and C_expand out of gradient computation
        elements = rearrange(self._extract_elements(self.all_atom), "r a -> (r a)")
        C_expand = repeat(
            self.C_bar, "b r -> b (r a)", a=14 if self.all_atom else 4
        ).squeeze()

        if len(X.size()) == 4:
            X = rearrange(X, "b r a c -> b (r a) c").squeeze()  # b, N, 3

        with torch.enable_grad():
            X.requires_grad_(True)
            if X.is_cuda:  # use autocast for mixed precision
                with torch.autocast(device_type="cuda"):
                    result = self.density_calculator.compute_ll_density(
                        X, elements, C_expand, self.y
                    )
            else:
                result = self.density_calculator.compute_ll_density(
                    X, elements, C_expand, self.y
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
        return grad

    def model_refinement_optimizer(
        self,
        epochs: int = 4000,
        lr_m_s_d: List[float] = [1e-1, 1e-5, 3e-5],
        momenta: List[float] = [9e-1] * 3,
        map_resolution: float = 2.0,
        output_dir: str = "output",
    ) -> Protein:
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
        output_dir : str, optional
            Output directory for saving refined protein structures, by default "output

        Returns
        -------
        Protein
            Denoised and data matched protein structure.
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

        # FIXME fix chroma CIF I/O
        Protein.from_XCS(X, C, S).to_PDB(os.path.join(output_dir, "init.pdb"))

        v_i_m = torch.zeros(X.size(), device=self.device)
        v_i_s = torch.zeros(X.size(), device=self.device)
        v_i_d = torch.zeros(X.size(), device=self.device)

        for epoch in tqdm(range(epochs), desc="Model Refinement"):
            # Denoise coordinates
            t = _t(epoch, epochs).to(self.device)

            with torch.no_grad():
                X_0 = self.denoiser(X.detach(), C, t)

            # transform denoised coordinates to whitened space
            z_0 = self.multiply_inverse_corr(X_0, C)

            # sample chi angles
            if epoch % 100 == 0:  # TODO: implement all atom
                X_aa, _, _, _ = self.sequence_chi_sampler(
                    X_0, C, S, t=0.0, return_scores=True, resample_chi=True
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
                )  # Get gradient of log likelihood of sequence in whitened space # TODO: remove :4 once all atom is implemented
            else:
                grad_ll_sequence = torch.zeros_like(z_0, device=self.device)

            ### Accumulate gradients

            # incomplete structure
            v_i_m = momenta[0] * v_i_m + lr_m_s_d[
                0
            ] * self.grad_ll_incomplete_structure(z_0)

            # sequence
            v_i_s = (
                momenta[1] * v_i_s + lr_m_s_d[1] * grad_ll_sequence
            )  # NOTE: This should change if a model other than Chroma is used.

            current_resolution = (
                15 if epoch < 3000 else 15 - (epoch - 3000) / 100
            )  # TODO: generalize to different epoch counts

            # density
            if self.all_atom:
                density_grad_transformed = self.multiply_inverse_corr(
                    self.grad_ll_density(X_aa, current_resolution)[:, :, :4, :],
                    C,  # get the backbone coords to update
                )
            else:
                density_grad_transformed = self.multiply_inverse_corr(
                    self.grad_ll_density(X_0, current_resolution), C
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
                epoch % 100 == 0
            ):  # ~40 structure is minimum for a good diffusion trajectory
                Protein.from_XCS(X, C, S).to_PDB(
                    os.path.join(output_dir, f"output_epoch_{epoch}.pdb")
                )

        MAP_protein = Protein.from_XCS(X, C, S)

        return MAP_protein
