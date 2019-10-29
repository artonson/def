from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sharpf.modules.base import ParameterizedModule, load_with_spec


class AggregationBase(ParameterizedModule):
    """
    ConvBase - Abstract class for Convolution methods for ParameterizedPointNet.
    Current class deals with point projection from one dimensionality to another.

    ...

    Attributes
    ----------

    Methods
    -------
    forward(x)
       Performs a convolution operation

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: x: batch of points for convolution, shape = (B, C_in, N, M)
                  B - batch size,
                  C_in - number of features
                  N_in - number of points
                  M_in - number of patches
        output: out: (B, C_out, N_out, M_out) tensor
        """
        out = x
        return out


aggregation_module_by_kind = {
    'aggregation_base': AggregationBase
}
