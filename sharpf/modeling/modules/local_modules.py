import torch
import torch.nn as nn


class LocalDynamicGraph(nn.Module):
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

    def forward(self, x):
        """
        input: x[0]: batch of points for local transformation, shape = (B, N, C, M_in)
                  B - batch size,
                  N_in - number of points,
                  C_in - number of features,
                  M_in - number of patches
               x[1]: indices of nearest neighbour points for each point, shape = (B, N, M_out) tensor
        output: out: (B, N, C, M_out) tensor
        """
        idx = x[1]
        x = x[0]

        batch_size, num_points, num_dims = x.shape
        k = idx.shape[2]

        x = x.transpose(2, 1)
        x = x.view(batch_size, -1, num_points)
        idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1)
        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)

        feature = x.view(batch_size * num_points, -1)[idx, :]

        feature = feature.view(batch_size, num_points, k, num_dims)
        # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        x = x.view(batch_size, num_points, 1, num_dims).expand(batch_size, num_points, k, num_dims)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 1, 3, 2)
        return feature
