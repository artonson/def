import torch
import torch.nn as nn


class NeighbourKNN(nn.Module):

    def __init__(self, k, channels_last=True):
        super().__init__()
        self.k = k
        self.channels_last = channels_last

    def forward(self, x):
        """
        Args:
            x (Tensor):
                of shape (B, N, C) if channel_last=true
                of shape (B, C, N) if channel_last=false
        Returns:
            Tensor: Same as input
            Tensor: of shape (B, N, k) . Indices of nearest neighbour points for each point
        """
        return neighbour_knn(x, self.k, self.channels_last)


def neighbour_knn(x, k, channels_last):
    """
    Args:
        x (Tensor):
            of shape (B, N, C) if channels_last=true
            of shape (B, C, N) if channels_last=false
    Returns:
        Tensor: Same as input
        Tensor: of shape (B, N, k) . Indices of nearest neighbour points for each point
    """
    with torch.no_grad():
        if channels_last:
            inner = torch.matmul(x, x.transpose(2, 1))  # (B, N, N)
            xx = torch.sum(x ** 2, dim=2, keepdim=True)  # (B, N, 1)
        else:
            inner = torch.matmul(x.transpose(1, 2), x)  # (B, N, N)
            xx = torch.sum(x ** 2, dim=1).unsqueeze(2)  # (B, N, 1)
        pairwise_distance = xx - 2 * inner + xx.transpose(2, 1)  # (B, N, N)
        idx = pairwise_distance.topk(k, dim=2, largest=False)[1]  # (B, N, k)

    return x, idx
