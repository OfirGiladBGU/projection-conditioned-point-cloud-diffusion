import sys
from pathlib import Path

# Ensure parent directory is in path for absolute imports
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from model.simple.simple_model_utils import FeedForward, BasePointModel


class SimplePointModel(BasePointModel):
    """
    A simple model that processes a point cloud by applying a series of MLPs to each point
    individually, along with some pooled global features.
    """

    def get_layers(self):
        return nn.ModuleList([FeedForward(
            d_in=(3 * self.dim), d_hidden=(4 * self.dim), d_out=self.dim,
            activation=nn.SiLU(), is_gated=True, bias1=False, bias2=False, bias_gate=False, use_layernorm=True
        ) for _ in range(self.num_layers)])

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):

        # Prepare inputs
        x, coords = self.prepare_inputs(inputs, t)

        # Model
        for layer in self.layers:
            x_pool_max, x_pool_std = self.get_global_tensors(x)
            x_input = torch.cat((x, x_pool_max, x_pool_std), dim=-1)  # (B, N, 3 * D)
            x = x + layer(x_input)  # (B, N, D_model)

        # Project
        x = self.output_projection(x)  # (B, N, D_out)
        x = torch.transpose(x, -2, -1)  # -> (B, D_out, N)

        return x


class SimpleNearestNeighborsPointModel(BasePointModel):
    """ 
    A simple model that processes a point cloud by applying a series of MLPs to each point
    individually, along with some pooled global features, and the features of its nearest
    neighbors.
    """

    def __init__(self, num_neighbors: int = 4, **kwargs):
        self.num_neighbors = num_neighbors
        super().__init__(**kwargs)
        self.knn_points = self._knn_points_torch

    @staticmethod
    def _knn_points_torch(p1: torch.Tensor, p2: torch.Tensor, K: int, return_nn: bool = False):
        """Naive KNN in pure PyTorch (CUDA-capable via torch.cdist)."""
        dists = torch.cdist(p1, p2)  # (B, N, N)
        dists, idx = torch.topk(dists, k=K, dim=-1, largest=False)
        neighbors = None
        if return_nn:
            neighbors = torch.gather(
                p2.unsqueeze(1).expand(-1, p1.size(1), -1, p2.size(-1)),
                2,
                idx.unsqueeze(-1).expand(-1, -1, -1, p2.size(-1)),
            )
        return dists, idx, neighbors

    def get_layers(self):
        return nn.ModuleList([FeedForward(
            d_in=((3 + self.num_neighbors) * self.dim), d_hidden=(4 * self.dim), d_out=self.dim,
            activation=nn.SiLU(), is_gated=True, bias1=False, bias2=False, bias_gate=False, use_layernorm=True
        ) for _ in range(self.num_layers)])

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):

        # Prepare inputs
        x, coords = self.prepare_inputs(inputs, t)  # (B, N, D), (B, N, 3)

        # Get nearest neighbors. Note that the first neighbor is the identity, which is convenient
        _dists, indices, _neighbors = self.knn_points(
            p1=coords, p2=coords, K=(self.num_neighbors + 1),
            return_nn=False)  # (B, N, K), (B, N, K)
        (B, N, D), (_B, _N, K) = x.shape, indices.shape

        # Model
        for layer in self.layers:
            # Gather feature neighbors using the precomputed KNN indices
            expand_idx = indices.unsqueeze(-1).expand(-1, -1, -1, D)
            neighbor_feats = torch.gather(
                x.unsqueeze(1).expand(-1, N, -1, D),
                2,
                expand_idx,
            )  # (B, N, K, D)
            x_neighbor = neighbor_feats.reshape(B, N, K * D)
            x_pool_max, x_pool_std = self.get_global_tensors(x)
            x_input = torch.cat((x_neighbor, x_pool_max, x_pool_std), dim=-1)  # (B, N, (3+K)*D)
            x = x + layer(x_input)  # (B, N, D_model)

        # Project
        x = self.output_projection(x)  # (B, N, D_out)
        x = torch.transpose(x, -2, -1)  # -> (B, D_out, N)

        return x
