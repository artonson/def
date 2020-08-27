from typing import List

import torch
import torch.nn as nn


class StackedConv(nn.Module):
    """
    StackedConv - class for Convolution method for ParameterizedPointNet.
    Current class deals with point projection from one dimensionality to another.
    """

    def __init__(
            self,
            channels: List[int],
            kernel_size=1,
            bn=True,
            relu=True,
            dropout_prob=None,
            conv_bias=False,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.bn = bn
        self.relu = relu
        self.dropout_prob = dropout_prob
        self.conv = conv_model_creator(
            channels=self.channels,
            kernel_size=self.kernel_size,
            bn=self.bn,
            relu=self.relu,
            dropout_prob=self.dropout_prob,
            conv_bias=conv_bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): of shape (B, N, C_in, k) or (B, N, C_in).
                        Batch of points where k is number of points in the neighbourhood
        Returns:
            Tensor: of shape (B, N, C_out, k) or (B, N, C_out, 1).
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        out = self.conv(x.transpose(2, 1))
        return out.transpose(2, 1)


def conv_model_creator(channels=(6, 64), kernel_size=1, bn=True, relu=True, dropout_prob=None,
                       conv_bias=False):
    layers = []

    for i in range(len(channels) - 1):
        layer = nn.Sequential(
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, bias=False if bn else conv_bias),
            nn.BatchNorm2d(channels[i + 1]) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        layers.append(layer)
    if dropout_prob:
        layers.append(nn.Dropout(p=dropout_prob))

    return nn.Sequential(*layers)
