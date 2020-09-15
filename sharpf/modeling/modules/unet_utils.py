import torch
import torch.nn as nn


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, dim):
        # todo fixme
        conv1 = ConvReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dim=dim,
        )
        conv2 = ConvReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dim=dim,
        )
        super().__init__(conv1, conv2)
