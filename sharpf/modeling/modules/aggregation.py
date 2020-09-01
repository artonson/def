import torch
import torch.nn as nn


class AggregationMax(nn.Module):
    """
    AggregationMax - class for max aggregation method for ParameterizedPointNet.
    Current class deals with point feature aggregation from neigbourhood.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): of shape (B, N, C, k). Batch of points where k is number of points in the neighbourhood
        Returns:
            Tensor: of shape (B, N, C).
        """
        out = x.max(dim=3)[0]
        return out


class GlobalMaxPooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): of shape (B, N, C, 1)
        Returns:
            Tensor: of shape (B, N, C).
        """
        x = x.squeeze(3)
        return x.max(dim=1, keepdim=True)[0].expand_as(x)
