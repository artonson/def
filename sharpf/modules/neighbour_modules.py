from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sharpf.modules.base import ParameterizedModule, load_with_spec
from contrib.pointweb.lib.pointops.functions import pointops


class NeighbourBase(ParameterizedModule):
    """
    NeighbourBase - Abstract class for patches preparation in ParameterizedPointNet
    Current class deals with: - sampling, aka centroids determination
                              - grouping, aka local region set construction

    ...

    Attributes
    ----------

    Methods
    -------
    forward(xyz, centroids, features)
       Prepares patches from xyz-data and concats additional features
    """

    def forward(self, xyz: torch.Tensor(), centroids: torch.Tensor = None, features: torch.Tensor = None) -> torch.Tensor:
        """
        input: xyz: (B, 3, N) coordinates of the features,
               features: (B, C, N) descriptors of the features (normals, etc)
                         None by default
        output: new_features: (B, C+3, npoint, nsample) tensor
        """
        new_features = xyz
        return new_features


class PointNet2SamplingAndGrouping(NeighbourBase):
    """
    Groups with a ball query of radius
    parameters:
        radius: float32, Radius of ball
        nsample: int32, Maximum number of features to gather in the ball
    """
    def __init__(self, radius=None, nsample=32, use_xyz=True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.op = pointops.QueryAndGroup(radius, nsample, use_xyz)

    def forward(self,
                xyz: torch.Tensor,
                new_xyz: torch.Tensor = None,
                features: torch.Tensor = None,
                idx: torch.Tensor = None) -> torch.Tensor:
        """
        input: xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
               features: (b, c, n)
               idx: idx of neighbors
        output: new_features: (b, c+3, m, nsample)
        """
        return self.op(xyz, new_xyz, features, idx)


class NeighourKNN(NeighbourBase):
    """
    NeighbourBase - Abstract class for patches preparation in ParameterizedPointNet
    Current class deals with: - sampling, aka centroids determination
                              - grouping, aka local region set construction

    ...

    Attributes
    ----------

    Methods
    -------
    forward(x)
       Prepares kNN indices for each point
    """

    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def forward(self, x):
        """
        input: x: batch of points for local transformation, shape = (B, N, C, M_in)
                  B - batch size,
                  N_in - number of points,
                  C_in - number of features,
                  M_in - number of patches
        output: x: (B, N, C, M_in) tensor
                idx: indices of nearest neighbour points for each point, shape = (B, N, M_out) tensor
        """
        if x.size(-1) == 1:
            x = x.squeeze(-1)
        inner = -2*torch.matmul(x, x.transpose(2, 1))
        xx = torch.sum(x**2, dim=2, keepdim=True)
        pairwise_distance = -xx.transpose(2, 1) - inner - xx
        idx = pairwise_distance.topk(k=self.k, dim=-1)[1] 
        return x, idx


neighbour_module_by_kind = {
    'neighbour_base': NeighbourBase,
    'pointnet2_sampling_grouping': PointNet2SamplingAndGrouping,
    'neighbour_knn': NeighourKNN
}
