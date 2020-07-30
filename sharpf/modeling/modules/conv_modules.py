from typing import List

import torch
import torch.nn as nn


class StackedConv(nn.Module):
    """
    StackedConv - class for Convolution method for ParameterizedPointNet.
    Current class deals with point projection from one dimensionality to another.

    ...

    Attributes
    ----------
    channels: list, list of channels for convolutions; length of the list is blocks+1
    kernel_size: int, size of the window to perform convolution over
    bn: bool, whether to add BatchNormalization layer or not
    relu: bool, whether to add activation for the output or not
    blocks: int, number of convolution blocks to stack

    Methods
    -------
    forward(x)
       Performs a convolution operation

    """

    def __init__(
            self,
            channels: List[int],
            kernel_size=1,
            bn=True,
            relu=True,
            dropout_prob=None,
            blocks=1,
            conv_bias=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        channels = list(channels)
        self.channels = channels
        self.kernel_size = kernel_size
        self.bn = bn
        self.relu = relu
        self.blocks = blocks
        self.dropout_prob = dropout_prob
        self.conv = conv_model_creator(
            channels=self.channels,
            kernel_size=self.kernel_size,
            blocks=self.blocks,
            bn=self.bn,
            relu=self.relu,
            dropout_prob=self.dropout_prob,
            conv_bias=conv_bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: x: batch of points for convolution, shape = (B, C_in, N_in, M_in)
                  B - batch size,
                  N_in - number of points
                  C_in - number of features
                  M_in - number of patches
        output: out: (B, N_out, C_out, M_out) tensor
        """
        out = self.conv(x.transpose(2, 1))
        return out.transpose(2, 1)


def conv_model_creator(channels=(6, 64), kernel_size=1, blocks=1, bn=True, relu=True, dropout_prob=None,
                       conv_bias=False):
    assert blocks == len(channels) - 1, 'wrong number of blocks'
    layers = []

    for i in range(blocks):
        layer = nn.Sequential(
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, bias=False if bn else conv_bias),
            nn.BatchNorm2d(channels[i + 1]) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        layers.append(layer)
    if dropout_prob:
        layers.append(nn.Dropout(p=dropout_prob))

    return nn.Sequential(*layers)
