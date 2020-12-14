import torch
import torch.nn as nn


class LocalDynamicGraph(nn.Module):
    """
    LocalDynamicGraph - class for Local transformation methods for ParameterizedPointNet.
    Current class deals with local dynamic graph construction for DGCNN.
    """

    def forward(self, x):
        """
        Args:
            x (tuple): of elements
                x[0] (Tensor): of shape (B, N, C). Batch of points
                x[1] (Tensor): of shape (B, N, k). Indices of nearest neighbour points for each point
        Returns:
            Tensor: of shape (B, N, C, k).
        """
        return local_dynamic_graph(x)


def local_dynamic_graph(x):
    """
    Args:
        x (tuple): of elements
            x[0] (Tensor): of shape (B, N, C). Batch of points
            x[1] (Tensor): of shape (B, N, k). Indices of nearest neighbour points for each point
    Returns:
        Tensor: of shape (B, N, C, k).
    """
    idx = x[1]
    x = x[0]

    batch_size, num_points, num_dims = x.shape

    k = idx.shape[2]

    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points  # (B, 1, 1)
    idx = idx + idx_base  # (B, N, k)
    idx = idx.view(-1)  # (B * N * k,)

    feature = x.view(batch_size * num_points, num_dims)[idx, :]  # (B * N * k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (B, N, k, num_dims)

    x = x.unsqueeze(2).expand(batch_size, num_points, k, num_dims)  # (B, N, k, num_dims)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 1, 3, 2)  # (B, N, 2 * num_dims, k)
    return feature
