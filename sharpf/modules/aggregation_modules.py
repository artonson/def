from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sharpf.modules.base import ParameterizedModule, load_with_spec


class AggregationBase(ParameterizedModule):
    """
    AggregationBase - Abstract class for Aggregation methods for ParameterizedPointNet.
    Current class deals with point feature aggregation from neigbourhood.

    ...

    Attributes
    ----------

    Methods
    -------
    forward(x)
       Performs an aggregation operation

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: x: batch of points for interpolation, shape = (B, C_in, N_in, M_in)
                  B - batch size,
                  C_in - number of features
                  N_in - number of points
                  M_in - number of patches
        output: out: (B, C_out, N_out, M_out) tensor
        """
        out = x
        return out


class AggregationMax(AggregationBase):
    """
    AggregationMax - class for max aggregation method for ParameterizedPointNet.
    Current class deals with point feature aggregation from neigbourhood.

    ...

    Parameters
    ----------
    keepdim: bool, whether to keep number of dimensions or not
    dim: int, over which dimension to take max 

    Methods
    -------
    forward(x)
       Performs an aggregation operation

    """

    def __init__(self, keepdim, dim, **kwargs):
        super().__init__(**kwargs)
        self.keepdim = keepdim
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: x: batch of features for aggregation, shape = (B, N_in, C_in, M_in)
                  B - batch size,
                  N_in - number of points,
                  C_in - number of features,
                  M_in - number of points in the neighbourhood
        output: out: (B, N_out, C_out, M_out) tensor
        """
        out = x.max(dim=self.dim, keepdim=self.keepdim)[0]
        return out
    
    
class AggregationMaxPooling(AggregationBase):
    """
    AggregationMaxPooling - class for max pooling aggregation method for ParameterizedPointNet.
    Current class deals with point feature aggregation from neigbourhood.

    ...

    Parameters
    ----------
    kernel_size: list, size of the window to take max over

    Methods
    -------
    forward(x)
       Performs an aggregation operation

    """

    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.max_pool = nn.MaxPool2d(self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: x: batch of features for aggregation, shape = (B, N_in, C_in, M_in)
                  B - batch size,
                  N_in - number of points,
                  C_in - number of features,
                  M_in - number of points in the neighbourhood
        output: out: (B, N_out, C_out, M_out) tensor
        """
        x = x.transpose(2,1).contiguous()
        out = self.max_pool(x)
        out = out.transpose(2,1).contiguous()
        return out


aggregation_module_by_kind = {
    'aggregation_base': AggregationBase,
    'aggregation_max': AggregationMax,
    'aggregation_max_pooling': AggregationMaxPooling
}
