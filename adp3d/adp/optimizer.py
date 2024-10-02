"""Implement the ADP-3D algorithm (Levy et al. 2024) for protein structure refinement using Chroma as a plug-and-play prior for reconstruction.

Author: Karson Chrispens (karson.chrispens@ucsf.edu)
Created: 6 Aug 2024
Updated: 20 Aug 2024
"""

from typing import Tuple, Union, List, Dict
import torch
import numpy as np
from einops import rearrange, reduce, repeat
from chroma import Chroma, Protein
from chroma.layers.structure.mvn import BackboneMVNGlobular

# from chroma.layers.structure.sidechain import SideChainBuilder
from chroma.constants import AA_GEOMETRY, AA20_3
from tqdm import tqdm
from adp3d.utils.utility import DotDict
from qfit.transformer import Transformer
from qfit.unitcell import UnitCell
from qfit.volume import XMap, Resolution
from qfit.structure.structure import Structure


def get_elements_from_XCS(X: torch.Tensor, S: torch.Tensor) -> List[str]:
    """Get element names from an XCS tensor representation of a protein.

    Parameters
    ----------
    X : torch.Tensor
        Protein coordinates in XCS format.
    S : torch.Tensor
        Protein sequence in XCS format.

    Returns
    -------
    torch.Tensor
        Elements for the protein atoms compatible with qFit.
    """
    if X.size()[2] == 4:
        # All residues are backbone atoms = GLY
        # repeat [N, C, C, O] for each residue
        return ["N", "C", "C", "O"] * X.size()[1]
    else:
        elements = []

        # Get 3 letter code from one letter in sequence.
        index_to_code = {i: code for i, code in enumerate(AA20_3)}
        S = [index_to_code[int(aa)] for aa in S.squeeze()]

        # Get elements from XCS
        for residue in S:
            backbone_elements = ["N", "C", "C", "O"]
            if residue == "GLY":
                elements.extend(backbone_elements)
            else:
                backbone_elements.extend(
                    [atom[0] for atom in AA_GEOMETRY[residue]["atoms"]]
                )
                elements.extend(backbone_elements)

        return elements


def minimal_XCS_to_Structure(
    X: torch.Tensor, S: torch.Tensor
) -> Dict:  # TODO: Only works for single chains.
    """Transform a minimal XCS tensor representation of a protein to a qFit Structure object.
    (but secretly just a Dict with the necessary keys for a qFit Transformer as qFit does not type check for Structure)

    Parameters
    ----------
    X : torch.Tensor
        Protein coordinates in XCS format.
    S : torch.Tensor
        Sequence in XCS format.

    Returns
    -------
    Dict
        Dictionary with the necessary keys for a qFit Transformer.
    """
    if X.size()[0] > 1:
        raise ValueError(
            "Currently only one protein structure can be converted at a time."
        )
    e = get_elements_from_XCS(X, S)
    coor = (
        rearrange(X, "b r a c -> b (r a) c").squeeze().cpu().numpy()
    )  # FIXME: does qFit work without this as numpy?
    coor = np.ascontiguousarray(coor).astype(np.float64)

    natoms = coor.shape[0]
    active = np.ones(natoms, dtype=bool)
    b = np.array([10] * natoms, dtype=np.float64)
    q = np.array([1] * natoms, dtype=np.float64)
    structure = {
        "coor": coor,
        "b": b,
        "q": q,
        "e": e,
        "active": active,
        "natoms": natoms,
    }
    return structure


def XCS_to_Structure(X: torch.Tensor, C: torch.Tensor, S: torch.Tensor) -> Structure:
    """Transform an XCS tensor representation of a protein to a qFit Structure object.

    Parameters
    ----------
    X : torch.Tensor
        Protein coordinates in XCS format.
    C : torch.Tensor
        Chain map in XCS format.
    S : torch.Tensor
        Sequence in XCS format.

    Returns
    -------
    Structure
        qFit Structure object.
    """
    pass


class ADP3D:
    def __init__(
        self,
        y: torch.Tensor,
        seq: torch.Tensor,
        structure: str,
        protein: Protein = None,
    ):
        """Initialize the ADP3D class.

        This class implements the Atomic Denoising Prior framework from Levy et al. 2024.

        Parameters
        ----------
        y : torch.Tensor
            Input density measurement in torch.Tensor format with 3 axes. Ideally loaded by gemmi or mrcfile.
        seq : torch.Tensor
            Sequence Tensor for incorporating sequence information, defined in Chroma's XC**S** format
        structure : str
            Input CIF file path for an incomplete structure
        protein : Protein, optional
            Chroma Protein object for initializing with a given protein structure, by default None
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # enforce input args # TODO: check if this is needed
        check = [x is not None for x in [y, seq, structure]]
        if not any(check):
            raise ValueError("Arguments y, seq, and structure must be defined.")

        self.protein = protein
        self.y = y
        self.seq = seq
        self.x_bar, self.C_bar, _ = Protein.from_CIF(
            structure, device=self.device
        ).to_XCS()

        # Importing sqrt(covariance matrix) from Chroma
        # These take in Z and C (for _multiply_R) or X and C (for _multiply_R_inverse)
        self.mvn = BackboneMVNGlobular(covariance_model="globular")
        self.multiply_corr = self.mvn._multiply_R
        self.multiply_inverse_corr = self.mvn._multiply_R_inverse

        # Build correlation matrix for incomplete structure log likelihood
        z_bar = self.multiply_inverse_corr(self.x_bar, self.C_bar)

        # fix dimensions for _expand_per_chain
        if z_bar.dim() == 4:
            z_bar = rearrange(z_bar, "b r a c -> b (r a) c")

        C_bar_mask_all, _, _ = self.mvn._expand_per_chain(z_bar, self.C_bar)
        B, _, _ = self.mvn._globular_parameters(
            C_bar_mask_all
        )  # shape will be (B, C), so likely (1, 1)

        # NOTE: Only on single chains for now
        B = B.squeeze()  # B is now a scalar
        N = self.x_bar.size()[1]  # Number of residues
        A = 4 * N  # Number of atoms in backbone
        R = torch.zeros(A, A)  # Initialize correlation matrix on all atoms
        for i in range(A):
            R[i:, i] = self.mvn._nu * B ** torch.arange(i, A, device=self.device)
            if i > 0:
                R[i:, i] = B ** torch.arange(0, A - i, device=self.device)

        R = (self.mvn._scale * R).to(self.device)

        # Build measurement matrix for incomplete structure log likelihood
        # Going to only account for backbone atoms, so 4 atoms per residue. Could extend to all atom later, I think.
        n = self.seq.size()[1] * 4
        m = self.x_bar.size()[1] * 4
        self.A = torch.eye(
            m, n, device=self.device
        )  # measurement matrix to transform z into the form comparable to x_bar. m is the measurement dimension # FIXME: I think this doesn't work, test it out

        # Precompute SVD for incomplete structure log likelihood

        AR = self.A @ R
        # U shape (m, m), S shape (m, n), V_T shape (n, n)
        self.U, S, self.V_T = torch.linalg.svd(AR)
        if len(S.size()) == 1:
            S = torch.diag(S)
        # S_plus shape (n, m)
        self.S_plus = torch.linalg.pinv(S)

        # Initialize Chroma
        self.chroma = Chroma()
        self.denoiser = self.chroma.backbone_network.sample_sde
        self.sequence_sampler = None  # NOTE: Placeholder for sequence sampling. This will need to be implemented if doing model refinement with a denoiser other than Chroma
        self.chi_sampler = None  # NOTE: Placeholder for chi angle sampling. This will need to be implemented if doing model refinement with a denoiser other than Chroma
        self.sequence_chi_sampler = (
            self.chroma.design_network.sample
        )  # In Chroma, this can get log_p sequence and all atom coordinates.
        # self.chi_to_X = (
        #     SideChainBuilder()
        # )  # Chroma's sidechain builder for making all-atom coordinates from backbone and chi angles.

    def ll_incomplete_structure(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the log likelihood of the incomplete structure given the denoised coordinates.

        Parameters
        ----------
        x_bar : torch.Tensor
            Incomplete structure coordinates.
        z : torch.Tensor
            Denoised coordinates.
        C : torch.Tensor
            Chain map (needed for multiplying correlation matrix).

        Returns
        -------
        torch.Tensor
            Log likelihood of the incomplete structure.
        """
        z = rearrange(z, "b r a c -> b (r a) c").squeeze()
        x_bar = rearrange(self.x_bar, "b r a c -> b (r a) c").squeeze()
        to_norm = self.A @ self.V_T @ z - self.S_plus @ self.U.T @ x_bar
        breakpoint()
        return (
            -torch.linalg.vector_norm(torch.flatten(to_norm)) ** 2
        )

    def grad_ll_incomplete_structure(  # TODO: Test here
        self, x_bar: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        x_bar.requires_grad = False
        result = self.ll_incomplete_structure(x_bar, z)

        return torch.autograd.grad(result, z)

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
        cls, X: torch.Tensor, S: torch.Tensor, size: tuple = (100, 100, 100)
    ) -> torch.Tensor:  # TODO: Test here
        """Compute a density map of the all-atom coordinates X.

        Args:
            X (torch.Tensor): The input all-atom protein coordinates.
            S (torch.Tensor): The sequence tensor.
            size (torch.Tensor): The desired size of the output density map.

        Returns:
            torch.Tensor: The density map (shape (100, 100, 100)).
        """
        # NOTE: NO TYPECHECKS IN QFIT so I can do this
        structure = DotDict(minimal_XCS_to_Structure(X, S))
        # NOTE: EMMap in qFit doesn't seem to be interoperable with the Transformer, so I'm going to use XMap
        map = XMap(
            array=np.zeros(size, dtype=np.float64),
            unit_cell=UnitCell(),
            resolution=Resolution(high=2.0, low=2.0),
        )
        transformer = Transformer(
            structure, map, em=True, simple=True
        )  # TODO: not sure what simple means here exactly, check this later
        transformer.density()
        return (
            transformer.xmap.array
        )  # TODO: Test, make sure densities are aligned. In this case, the density has origin at 0 and size 100, 100, 100. Make sure the observed density is compatible with this.

    def ll_density(  # TODO: Test here
        self, y: torch.Tensor, z: torch.Tensor, C: torch.Tensor, S: torch.Tensor
    ) -> torch.Tensor:
        """Compute the log likelihood of the density given the atomic coordinates and side chain angles.

        Parameters
        ----------
        y : torch.Tensor
            Density map.
        z : torch.Tensor
            Denoised coordinates.
        C : torch.Tensor
            Chain map (needed for multiplying correlation matrix).
        S : torch.Tensor
            Sequence tensor.

        Returns
        -------
        torch.Tensor
            Log likelihood of the density.
        """

        X = self.multiply_corr(
            z, C
        )  # Transform denoised coordinates to Cartesian space
        density = self.gamma(
            X, S, size=y.size()
        )  # Get density map from denoised coordinates

        if density.size() != y.size():
            raise ValueError("Density map and input density map must be the same size.")

        return -torch.linalg.vector_norm(torch.flatten(density) - torch.flatten(y)) ** 2

    def grad_ll_density(self, z: torch.Tensor) -> torch.Tensor:  # TODO: Test here
        """Compute the gradient of the log likelihood of the density given the atomic coordinates and side chain angles.

        Parameters
        ----------

        y : torch.Tensor
            Density map.
        z : torch.Tensor
            Denoised coordinates.
        C : torch.Tensor
            Chain map (needed for multiplying correlation matrix).

        Returns
        -------
        torch.Tensor
            Gradient of the log likelihood of the density.
        """
        self.y.requires_grad = False  # TODO: move these to init?
        self.C_bar.requires_grad = False
        result = self.ll_density(self.y, z, self.C_bar)

        return torch.autograd.grad(result, z)

    def _t(epoch: int) -> torch.Tensor:
        """Time schedule for the model refinement task.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        torch.Tensor
            Current temperature. # FIXME: Not sure?
        """
        return 1 - torch.sqrt(torch.tensor(epoch) / 4000)

    def model_refinement_optimizer(
        self,
        epochs: float = 4000,
        lr_m_s_d: List[float] = [1e-1, 1e-5, 3e-5],
        momenta: List[float] = [9e-1] * 3,
    ) -> Protein:
        """Use gradient descent with momentum to optimize log likelihood functions for model refinement task along with denoised coordinates.

        Parameters
        ----------
        epochs : float, optional
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
        v_i_m = torch.zeros(X.size())
        v_i_s = torch.zeros(X.size())
        v_i_d = torch.zeros(X.size())
        X, C = self.x_bar, self.C_bar
        S = self.seq

        for epoch in tqdm(range(epochs), desc="Model Refinement"):
            # Denoise coordinates
            output = self.denoiser(C, X_init=X, N=1)  # TODO: Test here
            X = output["X_sample"]  # Get denoised coordinates
            C = output["C"]  # Get denoised chain map (required for Chroma functions)

            z_0 = self.multiply_inverse_corr(
                X, C
            )  # transform denoised coordinates to whitened space # TODO: Test here

            if epoch % 100 == 0:
                X_aa, _, _, scores = self.sequence_chi_sampler(X, C, S, t=0.0)

            ll_sequence = scores["logp_S"]  # Get sequence log probability
            ll_sequence.backward(
                input=X_aa
            )  # Backpropagate to get gradients # TODO: Test here
            grad_ll_sequence = self.multiply_corr(
                X_aa.grad[:, :, :4, :], C
            )  # Get gradient of log likelihood of sequence # TODO: Test here

            # Accumulate gradients
            v_i_m = momenta[0] * v_i_m + lr_m_s_d[
                0
            ] * self.grad_ll_incomplete_structure(z_0)
            v_i_s = momenta[1] * v_i_s + lr_m_s_d[
                1
            ] * grad_ll_sequence(  # NOTE: This should change if a model other than Chroma is used.
                self.seq, z_0
            )
            z_0_aa = self.multiply_inverse_corr(X_aa, C)  # Transform to whitened space
            v_i_d = momenta[2] * v_i_d + lr_m_s_d[2] * self.grad_ll_density(z_0_aa)

            # Update denoised coordinates
            z_t_1 = z_0 + v_i_m + v_i_s + v_i_d
            X = self.multiply_corr(z_t_1, C)

            # Add noise (currently just assuming the default addition of noise by chroma.backbone_network.sample_sde, not what is defined in the ADP paper)
            # https://github.com/search?q=repo%3Ageneratebio%2Fchroma%20noise&type=code

        MAP_protein = Protein.from_XCS(X, C, S)

        return MAP_protein
