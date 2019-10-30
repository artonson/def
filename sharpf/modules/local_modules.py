from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sharpf.modules.base import ParameterizedModule, load_with_spec


class LocalTransformBase(ParameterizedModule):
    """
    LocalTransformBase - Abstract class for Local transformation methods for ParameterizedPointNet.
    Current class deals with local point neigbourhood transformation.

    ...

    Attributes
    ----------

    Methods
    -------
    forward(x)
       Performs a local transformation operation

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: x: batch of points for convolution, shape = (B, C_in, N_in, M_in)
                  B - batch size,
                  C_in - number of features
                  N_in - number of points
                  M_in - number of patches
        output: out: (B, C_out, N_out, M_out) tensor
        """
        out = x
        return out


class LocalDynamicGraph(LocalTransformBase):
    """
    LocalDynamicGraph - class for Local transformation methods for ParameterizedPointNet.
    Current class deals with local dynamic graph construction for DGCNN.

    ...

    Attributes
    ----------

    Methods
    -------
    forward(x)
       Performs a local transformation operation

    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        """
        input: x: batch of points for local transformation, shape = (B, N, C, M_in)
                  B - batch size,
                  N_in - number of points,
                  C_in - number of features,
                  M_in - number of patches
        output: out: (B, N, C, M_out) tensor
        """
        idx = x[1]
        x = x[0]
        batch_size = x.size(0)
        num_points = x.size(1)
        num_dims = x.size(2)
        k = idx.size(2)
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
        idx += idx_base
        idx = idx.view(-1)
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)        
        out = torch.cat((feature-x, x), dim=3).permute(0, 1, 3, 2)
        return out


local_module_by_kind = {
    'local_base': LocalTransformBase,
    'local_dynamic_graph': LocalDynamicGraph
}
