import torch
import torch.nn as nn


class NeighbourKNN(nn.Module):

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        """
        Args:
            x (Tensor): of shape (B, N, C)
        Returns:
            Tensor: of shape (B, N, C). Same as input
            Tensor: of shape (B, N, k) . Indices of nearest neighbour points for each point
        """
        inner = torch.matmul(x, x.transpose(2, 1))  # (B, N, N)
        xx = torch.sum(x ** 2, dim=2, keepdim=True)  # (B, N, 1)
        pairwise_distance = xx - 2 * inner + xx.transpose(2, 1)  # (B, N, N)
        idx = pairwise_distance.topk(self.k, dim=2, largest=False)[1]  # (B, N, k)
        return x, idx
