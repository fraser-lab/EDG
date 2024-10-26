"""Implement the ADP-3D algorithm (Levy et al. 2024) for protein structure refinement using Chroma as a plug-and-play prior for reconstruction.

Author: Karson Chrispens (karson.chrispens@ucsf.edu)
Created: 6 Aug 2024
Updated: 20 Aug 2024
"""

from typing import Tuple, Union, List, Dict
import torch
import torch.nn.functional as F
import numpy as np
import os
import gemmi
import reciprocalspaceship as rs
import warnings
from einops import rearrange, reduce, repeat
from chroma import Chroma, Protein
from chroma.layers.structure.mvn import BackboneMVNGlobular

# from chroma.layers.structure.sidechain import SideChainBuilder
from chroma.constants import AA_GEOMETRY, AA20_3
from tqdm import tqdm
from adp3d.utils.utility import DotDict
from adp3d.data.sf import ATOM_STRUCTURE_FACTORS, ELECTRON_SCATTERING_FACTORS
from SFC_Torch.Fmodel import SFcalculator
from SFC_Torch.io import PDBParser


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
    ):
        """Initialize the ADP3D class.

        This class implements the Atomic Denoising Prior framework from Levy et al. 2024.

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
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # enforce input args
        check = [x is not None for x in [y, seq, structure]]
        if not any(check):
            raise ValueError("Arguments y, seq, and structure must be defined.")

        self.protein = protein

        self.seq = seq
        self.x_bar, self.C_bar, S = Protein(structure).to_XCS(device=self.device)

        # processing coordinates
        self.x_bar -= self.x_bar.mean(dim=(0, 1, 2))  # centering
        self.x_bar = self.x_bar[torch.abs(self.C_bar) == 1].unsqueeze(
            0
        )  # take only chain 1
        self.C_bar = self.C_bar[torch.abs(self.C_bar) == 1].unsqueeze(
            0
        )  # get only chain 1

        # save backbone only PDB model NOTE: Get rid of this once all atom works
        Protein.from_XCS(
            self.x_bar,
            torch.ones(self.x_bar.size()[1], device=self.device).unsqueeze(0),
            S,
        ).to_PDB(f"{os.path.splitext(structure)[0]}_bb.pdb")

        self.density_extension = os.path.splitext(y)[1]
        if self.density_extension not in (".ccp4", ".map", ".mtz", ".cif"):
            warnings.warn("Density map must be a CCP4, MRC, SF-CIF, or MTZ file.")

        # deal with SFcalculator
        if self.density_extension in (
            ".mtz",
            ".cif",
        ):  # NOTE: MTZ or SF-CIF aren't used currently, but will be once all atom or chi sampler is used.
            structure_bb = gemmi.read_pdb(
                f"{os.path.splitext(structure)[0]}_bb.pdb"
            )  # NOTE: Get rid of this once all atom works, and build the density beforehand

            if os.path.splitext(structure)[1] == ".cif":
                structure = gemmi.make_structure_from_block(
                    gemmi.cif.read(structure)[0]
                )
            else:  # assume PDB
                structure = gemmi.read_pdb(structure)

            structure_bb.spacegroup_hm = (
                structure.spacegroup_hm
            )  # NOTE: Get rid of this once all atom works
            structure_bb.cell = (
                structure.cell
            )  # NOTE: Get rid of this once all atom works
            self.input_sfcalculator = (
                SFcalculator(  # NOTE: Get rid of this once all atom works
                    pdbmodel=PDBParser(structure_bb),
                    dmin=2.0,
                    set_experiment=False,
                    device=self.device,
                )
            )
        else:
            map = gemmi.read_ccp4_map(y)
            map_array = map.grid.array
            self.y = torch.from_numpy(np.ascontiguousarray(map_array)).to(
                self.device, dtype=torch.float32
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
        a = self.x_bar.size()[2]  # Number of atoms in one residue
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
        self.chroma = Chroma()
        self.denoiser = self.chroma.backbone_network.denoise
        self.noise_scheduler = (
            self.chroma.backbone_network.noise_perturb._schedule_coefficients
        )
        self.sequence_sampler = None  # NOTE: Placeholder for sequence sampling. This will need to be implemented if doing model refinement with a denoiser other than Chroma
        self.chi_sampler = None  # NOTE: Placeholder for chi angle sampling. This will need to be implemented if doing model refinement with a denoiser other than Chroma
        self.sequence_chi_sampler = (
            self.chroma.design_network.sample  # TODO: test this
        )  # In Chroma, this can get log_p sequence and all atom coordinates.
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
            self.x_bar[self.C_bar == 1].unsqueeze(0), "b r a c -> b (r a) c"
        ).squeeze()  # TODO: take only modeled residues, requires CIF
        denoised = self.A @ self.V_T @ z
        denoised = denoised[denoised.any(dim=1)]
        measured = self.S_plus @ self.U.T @ x_bar
        return -torch.linalg.vector_norm(torch.flatten(denoised - measured)) ** 2

    def grad_ll_incomplete_structure(self, z: torch.Tensor) -> torch.Tensor:

        if z.requires_grad == False:
            z = z.clone().detach().requires_grad_(True)  # reset graph history

        result = self.ll_incomplete_structure(z)

        return torch.autograd.grad(result, z)[0]

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

    def gamma(
        self, X: torch.Tensor, size: tuple = (100, 100, 100)
    ) -> torch.Tensor:  # TODO: Test here
        """Compute electron density map of the all-atom coordinates X.

        Args:
            X (torch.Tensor): The input all-atom protein coordinates.
            size (torch.Tensor): The desired size of the output density map.

        Returns:
            torch.Tensor: The structure factors.
        """
        # TODO: account for time dependent resolution
        if len(X.size()) == 4:
            X = rearrange(X, "b r a c -> b (r a) c").squeeze()

    def ll_density(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the log likelihood of the density given the atomic coordinates and side chain angles.

        Parameters
        ----------
        z : torch.Tensor
            Denoised coordinates.

        Returns
        -------
        torch.Tensor
            Log likelihood of the density.
        """
        if len(X.size()) == 4:
            X = rearrange(X, "b r a c -> b (r a) c")  # b, N, 3

        if self.density_extension in (".mtz", ".cif"):

            if X.size()[1] != self.input_sfcalculator.n_atoms:
                raise ValueError(
                    "Number of atoms in input coordinates must match number of atoms in structure factor calculator."
                )
            Fprotein = self.input_sfcalculator.calc_fprotein(Return=True)
            Fmodel = self.input_sfcalculator.calc_fprotein_batch(
                X, Return=True
            ).squeeze()

            return -torch.linalg.norm(torch.abs(Fprotein) - torch.abs(Fmodel)) ** 2
        else:  # TODO: fix and move this to Fourier space
            density = self.gamma(
                X, size=self.y.size()
            )  # Get density map from denoised coordinates

            if density.size() != self.y.size():
                raise ValueError(
                    "Density map and input density map must be the same size."
                )

            return (
                -torch.linalg.vector_norm(
                    torch.flatten(density) - torch.flatten(self.y)
                )
                ** 2
            )

    def grad_ll_density(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of the log likelihood of the density given the atomic coordinates and side chain angles.

        Parameters
        ----------

        z : torch.Tensor
            Denoised coordinates.

        Returns
        -------
        torch.Tensor
            Gradient of the log likelihood of the density.
        """
        if X.requires_grad == False:
            X = X.clone().detach().requires_grad_(True)  # reset graph history

        result = self.ll_density(X)

        return torch.autograd.grad(result, X)[0]

    def model_refinement_optimizer(
        self,
        epochs: int = 4000,
        lr_m_s_d: List[float] = [1e-1, 1e-5, 3e-5],
        momenta: List[float] = [9e-1] * 3,
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

        Returns
        -------
        Protein
            Denoised and data matched protein structure.
        """

        if self.protein is not None:
            X, C, _ = self.protein.to_XCS(
                device=self.device
            )  # handle GT protein initialization
        else:
            Z = torch.randn_like(self.x_bar, device=self.device)
            X = self.multiply_corr(Z, self.C_bar)
            C = torch.ones_like(
                self.C_bar, device=self.device
            )  # NOTE: self.C_bar provides correct chains and masking
        S = self.seq

        # FIXME
        Protein.from_XCS(X, C, S).to_CIF(os.path.join(output_dir, "init.cif"))

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

            if epoch >= 3000:
                with torch.enable_grad():
                    X_0.requires_grad_(True)
                    if epoch % 100 == 0:  # TODO: implement all atom
                        X_aa, _, _, scores = self.sequence_chi_sampler(
                            X_0, C, S, t=0.0, return_scores=True
                        )

                    ll_sequence = scores["logp_S"]  # Get sequence log probability
                    grad_ll_sequence = torch.autograd.grad(ll_sequence, X_aa)[
                        0
                    ]  # Backpropagate to get gradients

                grad_ll_sequence = self.multiply_inverse_corr(
                    grad_ll_sequence[:, :, :4, :], C
                )  # Get gradient of log likelihood of sequence in whitened space # TODO: remove :4 once all atom is implemented
            else:
                grad_ll_sequence = torch.zeros_like(z_0, device=self.device)

            # Accumulate gradients
            v_i_m = momenta[0] * v_i_m + lr_m_s_d[
                0
            ] * self.grad_ll_incomplete_structure(z_0)

            v_i_s = (
                momenta[1] * v_i_s + lr_m_s_d[1] * grad_ll_sequence
            )  # NOTE: This should change if a model other than Chroma is used.

            # TODO: update below with all atom once done
            # z_0_aa = self.multiply_inverse_corr(X_aa, self.C_bar)  # Transform to whitened space
            density_grad_tf = torch.clamp(
                self.multiply_inverse_corr(self.grad_ll_density(X_0), C),
                -1000,
                1000,
            )
            v_i_d = (
                momenta[2] * v_i_d + lr_m_s_d[2] * density_grad_tf
            )  # FIXME try gradient clipping?

            # Update denoised coordinates
            z_t_1 = z_0 + v_i_m + v_i_s + v_i_d
            # z_t_1 = z_0 + v_i_d

            # Add noise (this is the same as what OG ADP3D does)
            t1 = _t(epoch + 1, epochs).to(self.device)
            alpha, sigma, _, _, _, _ = self.noise_scheduler(t1)
            X = self.multiply_corr(alpha * z_t_1 + sigma * torch.randn_like(z_t_1), C)

            if epoch % 500 == 0:  # FIXME
                Protein.from_XCS(X, C, S).to_CIF(
                    os.path.join(output_dir, f"output_epoch_{epoch}.cif")
                )

        MAP_protein = Protein.from_XCS(X, C, S)

        return MAP_protein
