"""Implement the ADP-3D algorithm (Levy et al. 2024) for protein structure refinement using Chroma as a plug-and-play prior for reconstruction.

Author: Karson Chrispens (karson.chrispens@ucsf.edu)
Created: 6 Aug 2024
Updated: 20 Aug 2024
"""

from typing import Tuple, Union, List
import torch
from chroma import Chroma, Protein
from chroma.layers.structure.mvn import BackboneMVNGlobular
from chroma.layers.structure.sidechain import SideChainBuilder
from tqdm import tqdm


class ADP3D:
    def __init__(
        self,
        protein: Protein = None,
        y: torch.Tensor = None,
        seq: torch.Tensor = None,
        structure: torch.Tensor = None,
    ):
        """Initialize the ADP3D class.

        This class implements the Atomic Denoising Prior framework from Levy et al. 2024.

        Parameters
        ----------
        protein : Protein, optional
            Chroma Protein object for initializing with a given protein structure, by default None
        y : torch.Tensor, optional
            Input density measurement in torch.Tensor format with 3 axes, by default None
        seq : torch.Tensor, optional
            Sequence Tensor for incorporating sequence information, defined in Chroma's XC*S* format, by default None
        structure : torch.Tensor, optional
            Input atomic coordinates for an incomplete structure, size (num_residues, num_atoms, 3), by default None
        """
        self.protein = protein
        self.y = y
        self.seq = seq
        self.x_bar = structure

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Importing sqrt(covariance matrix) from Chroma
        # These take in Z and C (for _multiply_R) or X and C (for _multiply_R_inverse)
        mvn = BackboneMVNGlobular(covariance_model="globular")
        self.multiply_corr = mvn._multiply_R
        self.multiply_inverse_corr = mvn._multiply_R_inverse

        # Initialize Chroma
        chroma = Chroma()
        self.denoiser = chroma.backbone_network.sample_sde
        self.sequence_sampler = None  # NOTE: Placeholder for sequence sampling. This will need to be implemented if doing model refinement with a denoiser other than Chroma
        self.chi_sampler = None  # NOTE: Placeholder for chi angle sampling. This will need to be implemented if doing model refinement with a denoiser other than Chroma
        self.sequence_chi_sampler = (
            chroma.design_network.sample
        )  # In Chroma, this can get log_p sequence and all atom coordinates.
        # self.chi_to_X = (
        #     SideChainBuilder()
        # )  # Chroma's sidechain builder for making all-atom coordinates from backbone and chi angles.

    def gamma(
        self, X: torch.Tensor, size: tuple = (100, 100, 100)
    ) -> torch.Tensor:  # TODO: Test here
        """Compute a density map of the atomic coordinates X and side chain angles chi.

        Args:
            X (torch.Tensor): The input all-atom protein coordinates.
            size (torch.Tensor): The desired size of the output density map.

        Returns:
            torch.Tensor: The density map (shape (100, 100, 100)).
        """
        density_map = torch.zeros(*size)
        for i in range(X.size()[2]):  # Iterate over atoms
            for j in range(5):
                return  # FIXME: something like density_map += torch.exp(-torch.norm(X[i] - chi[j])) / (2 * torch.tensor([1, 1, 1])) ** 2
        return density_map

    def ll_incomplete_structure(  # TODO: Test here
        self, x_bar: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Compute the log likelihood of the incomplete structure given the denoised coordinates.

        Parameters
        ----------
        x_bar : torch.Tensor
            Incomplete structure.
        z : torch.Tensor
            Denoised coordinates.

        Returns
        -------
        torch.Tensor
            Log likelihood of the incomplete structure.
        """
        return torch.sum(
            (x_bar - z) ** 2
        )  # FIXME: This is a placeholder, need to implement the actual log likelihood function

    def grad_ll_incomplete_structure(  # TODO: Test here
        self, x_bar: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        x_bar.requires_grad = False
        result = self.ll_incomplete_structure(x_bar, z)

        result.backward(inputs=z)
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

    def ll_density(  # TODO: Test here
        self, y: torch.Tensor, z: torch.Tensor, C: torch.Tensor
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

        Returns
        -------
        torch.Tensor
            Log likelihood of the density.
        """

        X = self.multiply_corr(
            z, C
        )  # Transform denoised coordinates to Cartesian space

        density = self.gamma(X)

    def grad_ll_density(  # TODO: Test here
        self, y: torch.Tensor, z: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
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
        y.requires_grad = False
        C.requires_grad = False
        result = self.ll_density(y, z, C)

        result.backward(inputs=z)
        return z.grad

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
        protein: Protein,
        x_bar: torch.Tensor,
        seq: torch.Tensor,
        y: torch.Tensor,
        epochs: float = 4000,
        lr_m_s_d: List[float] = [1e-1, 1e-5, 3e-5],
        momenta: List[float] = [9e-1] * 3,
    ) -> Protein:
        """Use gradient descent with momentum to optimize log likelihood functions for model refinement task along with denoised coordinates.

        Parameters
        ----------
        protein : Protein
            Input incomplete protein structure as a Chroma Protein object. Most often this should be a
        y : torch.Tensor
            Measured density map data in 3D space.
        epochs : float, optional
            Number of epochs, by default 4000
        lr_m_s_d : List[float], optional
            Learning rates for optimization on each gradient of each log likelihood function,
            first for incomplete *m*odel, then *s*equence, then *d*ensity. By default [1e-1, 1e-5, 3e-5]
        momenta : List[float], optional
            Momenta for optimization updates, by default [9e-1]*3

        Returns
        -------
        Protein
            Denoised and data matched protein structure.
        """
        v_i_m = torch.zeros(X.size())
        v_i_s = torch.zeros(X.size())
        v_i_d = torch.zeros(X.size())
        X, C, _ = protein.to_XCS()
        S = self.seq

        for _ in tqdm(range(epochs), desc="Model Refinement"):
            # Denoise coordinates
            output = self.denoiser(C, X_init=X, N=1)  # TODO: Test here
            X = output["X_sample"]  # Get denoised coordinates
            C = output["C"]  # Get denoised chain map (required for Chroma functions)

            z_0 = self.multiply_inverse_corr(
                X, C
            )  # transform denoised coordinates to whitened space # TODO: Test here

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
            ] * self.grad_ll_incomplete_structure(x_bar, z_0)
            v_i_s = momenta[1] * v_i_s + lr_m_s_d[
                1
            ] * grad_ll_sequence(  # NOTE: This should change if a model other than Chroma is used.
                seq, z_0
            )
            z_0_aa = self.multiply_inverse_corr(X_aa, C)  # Transform to whitened space
            v_i_d = momenta[2] * v_i_d + lr_m_s_d[2] * self.grad_ll_density(y, z_0_aa)

            # Update denoised coordinates
            z_t_1 = z_0 + v_i_m + v_i_s + v_i_d
            X = self.multiply_corr(z_t_1, C)

            # Add noise (currently just assuming the default addition of noise by chroma.backbone_network.sample_sde, not what is defined in the ADP paper)
            # https://github.com/search?q=repo%3Ageneratebio%2Fchroma%20noise&type=code

        MAP_protein = Protein.from_XCS(X, C, S)

        return MAP_protein
