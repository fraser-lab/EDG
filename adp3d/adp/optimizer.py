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
from adp3d.data.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ELEMENTS,
    IDX_TO_ELEMENT,
)
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
        all_atom: bool = False,
        em: bool = False,
        gpu: bool = True,
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
        """

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if gpu
            else torch.device("cpu")
        )
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
        self.center_shift = torch.nanmean(values, dim=0) # this will need to be applied to the map too so map and model are aligned
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

        # save backbone only PDB model NOTE: Get rid of this once all atom works
        Protein.from_XCS(
            self.x_bar[:, :, :4, :],
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
            # self.input_sfcalculator = ( # FIXME
            #     SFcalculator(  # NOTE: Get rid of this once all atom works
            #         pdbmodel=PDBParser(structure_bb),
            #         dmin=2.0,
            #         set_experiment=False,
            #         device=self.device,
            #     )
            # )
        elif self.density_extension in (".ccp4", ".map", ".mrc"):
            map = gemmi.read_ccp4_map(y)
            self.grid = map.grid
            self.y = torch.from_numpy(np.ascontiguousarray(self.grid.array)).to(
                self.device, dtype=torch.float32
            )
        else:
            raise ValueError("Density map must be a CCP4, MRC, SF-CIF, or MTZ file.")

        # Importing sqrt(covariance matrix) from Chroma
        # These take in Z and C (for _multiply_R) or X and C (for _multiply_R_inverse)
        self.mvn = BackboneMVNGlobular(covariance_model="globular")
        self.multiply_corr = self.mvn._multiply_R
        self.multiply_inverse_corr = self.mvn._multiply_R_inverse

        # Build correlation matrix for incomplete structure log likelihood
        r = self.seq.size()[
            1
        ]  # Number of residues (from sequence, since we want total number in the correlation matrix)
        a = self.x_bar.size()[2] if not self.all_atom else 4  # Number of atoms in one residue
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
            self.x_bar[self.C_bar == 1].unsqueeze(0)[:, :, :4, :], "b r a c -> b (r a) c"
        ).squeeze()  # TODO: take only modeled residues, requires CIF
        # model backbone only for this
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
        size: Tuple[int] = (100, 100, 100),
        all_atom: bool = False,
    ) -> torch.Tensor:
        """Compute electron density map of the all-atom coordinates X.

        Args:
            X (torch.Tensor): The input all-atom protein coordinates.
            size (Tuple[int]): The desired size of the output density map.

        Returns:
            torch.Tensor: The structure factors.
        """
        # TODO: account for time dependent resolution
        if X.size()[2] == 4 and all_atom:
            raise ValueError(
                "Input coordinates are backbone only, all_atom must be False."
            )
        if len(X.size()) == 4:
            X = rearrange(
                X, "b r a c -> b (r a) c"
            ).squeeze()  # if b = 1, should result in (r a, c)
        else:
            X = X.squeeze() # assuming already in shape (r a, c)

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

        origin = (
            torch.tensor([*self.grid.get_position(0, 0, 0)], device=self.device)
            - self.center_shift
        )
        if self.grid.spacing[0] > 0:  # if one is > 0, all are > 0 from GEMMI
            x = torch.arange(
                origin[0],
                origin[0] + self.grid.spacing[0] * size[0],
                self.grid.spacing[0],
            )[
                : size[0]
            ]  # This gets right size. Size is sometimes off due to precision errors
            y = torch.arange(
                origin[1],
                origin[1] + self.grid.spacing[1] * size[1],
                self.grid.spacing[1],
            )[: size[1]]
            z = torch.arange(
                origin[2],
                origin[2] + self.grid.spacing[2] * size[2],
                self.grid.spacing[2],
            )[: size[2]]
        else:
            x = torch.linspace(origin[0], self.grid.unit_cell.a, size[0])
            y = torch.linspace(origin[1], self.grid.unit_cell.b, size[1])
            z = torch.linspace(origin[2], self.grid.unit_cell.c, size[2])

        if x.size()[0] != size[0] or y.size()[0] != size[1] or z.size()[0] != size[2]:
            raise ValueError(
                f"Size of the density map is not matching the input size: ({x.size()[0]}, {y.size()[0]}, {z.size()[0]}) is not {size}"
            )

        grid = torch.meshgrid(x, y, z, indexing="ij")
        grid = torch.stack(grid, dim=-1).to(self.device)

        four_pi2 = 4 * torch.pi**2

        if self.em:
            sf = ELECTRON_SCATTERING_FACTORS
            _range = 5
        else:
            sf = ATOM_STRUCTURE_FACTORS
            _range = 6

        bw_dict = lambda x: torch.tensor(
            [-four_pi2 / b if b > 1e-4 else 0 for b in sf[x][1][:_range]],
            device=self.device,
        )
        aw_dict = (
            lambda x: torch.tensor(sf[x][0][:_range], device=self.device)
            * (-bw_dict(x) / torch.pi) ** 1.5
        )

        # flatten to shape (x*y*z, 3) to match X shape (residues * atoms, 3)
        grid_flat = rearrange(grid, "x y z c -> (x y z) c")  # Shape: (x*y*z, 3)

        # pairwise distances between each voxel (grid point) and each atom
        distances = torch.cdist(grid_flat, X)  # Shape: (x*y*z, residues * atoms)

        density_flat = torch.zeros(
            grid_flat.size(0), device=self.device
        )  # Shape: (x*y*z,)

        # Iterate over each element in the protein
        for i, atom in enumerate(elements):
            if atom.item() == 5 or C_expand[i].item() != 1:
                continue

            e = IDX_TO_ELEMENT[atom.item()]

            aw_values = aw_dict(e)  # Shape: (_range,)
            bw_values = bw_dict(e)  # Shape: (_range,)

            # precomputed distances for this atom across all grid points
            # (this could be a bit more efficient by not computing distances that are too far away?)
            # seems fine for now
            r = distances[:, i]  # Shape: (x*y*z,)
            aw_expanded = repeat(
                aw_values, "n -> r n", r=r.shape[0]
            )  # Shape: (x*y*z, _range)
            bw_expanded = repeat(
                bw_values, "n -> r n", r=r.shape[0]
            )  # Shape: (x*y*z, _range)
            r_expanded = repeat(r, "r -> r n", n=_range)  # Shape: (x*y*z, _range)

            density_contribution = torch.sum(
                aw_expanded * torch.exp(bw_expanded * r_expanded**2), dim=1
            )

            # accumulate contributions to the density
            density_flat += density_contribution

        # back to volume
        density = rearrange(
            density_flat, "(x y z) -> x y z", x=size[0], y=size[1], z=size[2]
        )

        return density

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
            density = self._gamma(
                X, size=self.y.size(), all_atom=self.all_atom
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

            if epoch % 100 == 0:  # TODO: implement all atom
                X_aa, _, _, _ = self.sequence_chi_sampler(
                    X_0, C, S, t=0.0, return_scores=True, resample_chi=True
                )

            if epoch >= 3000:
                with torch.enable_grad():
                    X_aa = X_aa.clone().detach().requires_grad_(True)
                    ll_sequence = torch.sum(
                        self.sequence_loss(X_aa, C, S, t=0.0)["logp_S"]
                    )  # Get sequence log probability
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

            if self.all_atom:
                density_grad_transformed = self.multiply_inverse_corr(
                    self.grad_ll_density(X_aa)[:, :, :4, :], C # get the backbone coords to update
                )
            else:
                density_grad_transformed = self.multiply_inverse_corr(
                    self.grad_ll_density(X_0), C
                )
            v_i_d = momenta[2] * v_i_d + lr_m_s_d[2] * density_grad_transformed

            # Update denoised coordinates 
            z_t_1 = z_0 + v_i_m + v_i_s + v_i_d # FIXME
            # z_t_1 = z_0 + v_i_m + v_i_s

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
