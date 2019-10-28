import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.base import ParameterizedModule, load_with_spec

class TransformsBase(ParameterizedModule):
    '''
    TransformsBase - Abstract class for transformation methods for ParameterizedPointNet.
    Current class deals with performing transforms between convolution layers such as pooling, etc.

    ...

    Attributes
    ----------  
 
    Methods
    -------
    forward(x)
       Performs the transformation 
 
    '''
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    '''
    input: x: batch of points for convolution, shape = (B, C_in, N_in, M_in)
              B - batch size,
              C_in - number of features
              N_in - number of points
              M_in - number of patches
    output: out: (B, C_out, N_out, M_out) tensor
    '''
        out = x
        return out

