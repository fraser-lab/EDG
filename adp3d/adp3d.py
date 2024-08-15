"""Implement the ADP-3D algorithm (Levy et al. 2024) for protein structure refinement using Chroma as a plug-and-play prior for reconstruction.

Author: Karson Chrispens (karson.chrispens@ucsf.edu)
Created: 6 Aug 2024
Updated: 15 Aug 2024
"""

from typing import Tuple, Union
import torch
import torch.nn.functional as F
from chroma import Chroma, Protein, conditioners
from utils.register_api import register_api
from utils.key_location import KEY_LOCATION # Define your own key_location.py file with the path to your Chroma API key as KEY_LOCATION

# Chroma API registration

register_api(KEY_LOCATION)

# Attempting to implement with Chroma's conditioners, alternative is using chroma.sample(steps=1) to get an individual SDE integration step
class ADP3DConditioner(conditioners.Conditioner):
    def __init__(self, density, model):
        super().__init__()
        self.density = density # figure out a good way to load this
        self.model = model

    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> Tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, float],
    ]:
        return X_out, C_out, O, U_out, t
