"""Implement the ADP-3D algorithm (Levy et al. 2024) for protein structure refinement using Chroma as a plug-and-play prior for reconstruction.

Author: Karson Chrispens (karson.chrispens@ucsf.edu)
Created: 6 Aug 2024
Updated: 20 Aug 2024
"""

from typing import Tuple, Union, List
import torch
from chroma import Chroma, Protein, conditioners
from chroma.layers.structure.mvn import BackboneMVNGlobular
from tqdm import tqdm
from utils.register_api import register_api
from utils.key_location import (
    KEY_LOCATION,
)  # Define your own key_location.py file with the path to your Chroma API key as KEY_LOCATION

# Importing sqrt(covariance matrix) from Chroma
# These take in Z and C (for _multiply_R) or X and C (for _multiply_R_inverse)
mvn = BackboneMVNGlobular(covariance_model="globular")
multiply_corr = mvn._multiply_R
multiply_inverse_corr = mvn._multiply_R_inverse

def gamma(X: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    """Compute a density map of the atomic coordinates X and side chain angles chi.

    Args:
        X (torch.Tensor): The input protein coordinates.

    Returns:
        torch.Tensor: The density map (shape (x, y, z)).
    """
    density_map = torch.zeros(100, 100, 100)
    for i in range(X.size()[2]): # Iterate over atoms
        for j in range(5):
            return # FIXME: something like density_map += torch.exp(-torch.norm(X[i] - chi[j])) / (2 * torch.tensor([1, 1, 1])) ** 2
    return density_map

def _ll_incomplete_structure(x_bar: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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
    return torch.sum((x_bar - z) ** 2) # FIXME: This is a placeholder, need to implement the actual log likelihood function

def grad_ll_incomplete_structure(x_bar: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    x_bar.requires_grad = False
    result = _ll_incomplete_structure(x_bar, z)

    result.backward()
    return z.grad

def _ll_sequence(s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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
    return torch.sum((s - z) ** 2) # FIXME: This is a placeholder, need to implement the actual log likelihood function

def grad_ll_sequence(s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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
        Gradient of the log likelihood of the sequence.
    """
    s.requires_grad = False
    result = _ll_sequence(s, z)

    result.backward()
    return z.grad

def _ll_density(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compute the log likelihood of the density given the atomic coordinates and side chain angles.

    Parameters
    ----------
    z : torch.Tensor
        Denoised coordinates.
    y : torch.Tensor
        Density map.

    Returns
    -------
    torch.Tensor
        Log likelihood of the density.
    """
    density = gamma(z)

def grad_ll_density(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compute the gradient of the log likelihood of the density given the atomic coordinates and side chain angles.

    Parameters
    ----------
    z : torch.Tensor
        Denoised coordinates.
    y : torch.Tensor
        Density map.

    Returns
    -------
    torch.Tensor
        Gradient of the log likelihood of the density.
    """
    y.requires_grad = False
    result = _ll_density(y, z)

    result.backward()
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

def model_refinement_optimizer(protein: Protein, x_bar: torch.Tensor, seq: torch.Tensor, y: torch.Tensor, epochs: float = 4000, lr_m_s_d: List[float] = [1e-1, 1e-5, 3e-5], momenta: List[float] = [9e-1] * 3) -> Protein:
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
    v_i_m = torch.zeros(X.size()[1:]) # Get shape of (num_residues, num_atoms, 3), as first dimension is batch size
    v_i_s = torch.zeros(X.size()[1:])
    v_i_d = torch.zeros(X.size()[1:])
    X, C, S = protein.to_XCS()
    chroma = Chroma()

    for _ in tqdm(range(epochs), desc="Model Refinement"):
        # Denoise coordinates
        output = chroma.backbone_network.sample_sde(C, X_init=X, N=1)
        print(type(output))
        X = output["X_sample"] # Get denoised coordinates
        C = output["C"] # Get denoised chain map (required for Chroma functions)
        
        z_0 = multiply_inverse_corr(X, C) # transform denoised coordinates to whitened space

        # Compute gradient of log likelihood functions
        for i in range(X.size()[1]): # Iterate over residues
            v_i_m = momenta[0] * v_i_m + lr_m_s_d[0] * grad_ll_incomplete_structure(x_bar, z_0)
            v_i_s = momenta[1] * v_i_s + lr_m_s_d[1] * grad_ll_sequence(seq, z_0)
            v_i_d = momenta[2] * v_i_d + lr_m_s_d[2] * grad_ll_density(y, z_0)

        # Update denoised coordinates
        z_t_1 = z_0 + v_i_m + v_i_s + v_i_d
        X = multiply_corr(z_t_1, C)

        # Add noise (currently just assuming the default addition of noise by chroma.backbone_network.sample_sde, not what is defined in the ADP paper)
        
    denoised_protein = Protein.from_XCS(X, C, S)

    return denoised_protein

def main():
    register_api(KEY_LOCATION)

if __name__ == "__main__":
    main()
