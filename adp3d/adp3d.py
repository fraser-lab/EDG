"""Implement the ADP-3D algorithm (Levy et al. 2024) for protein structure refinement using Chroma as a plug-and-play prior for reconstruction.

Author: Karson Chrispens (karson.chrispens@ucsf.edu)
Date: 6 Aug 2024
"""

from typing import Tuple, Union
import torch
import torch.nn.functional as F
from chroma import Chroma, Protein, conditioners
from utils import register_api, key_location



# Chroma API registration

# Attempting to implement with Chroma's conditioner, alternative is using chroma.sample(steps=1) to get an individual denoising step
# class ADP3DConditioner(conditioners.Conditioner):
#     def __init__(self, density, model):
#         super().__init__()
#         self.density = density
#         self.model = model

#     def sequence_

#     def forward(
#         self,
#         X: torch.Tensor,
#         C: torch.LongTensor,
#         O: torch.Tensor,
#         U: torch.Tensor,
#         t: Union[torch.Tensor, float],
#     ) -> Tuple[
#         torch.Tensor,
#         torch.LongTensor,
#         torch.Tensor,
#         torch.Tensor,
#         Union[torch.Tensor, float],
#     ]:
#         return X_out, C_out, O, U_out, t


chroma = Chroma()
