from typing import List

import torch
import torch.nn as nn

from ..modules.point_resnet import PointResNet, Bottleneck, Bottleneck2
from ..modules.pt_utils import MaskedUpsample
from ..modules.unet_utils import CenterBlock

__all__ = ['Unet1D']


class DecoderBlock1D(nn.Module):
    def __init__(
            self,
            num_blocks,
            in_channels,
            skip_channels,
            out_channels,
            radius,
            nsample,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm1d
    ):
        super().__init__()
        assert num_blocks >= 1
        self.upsample = MaskedUpsample(radius=radius, nsample=nsample, mode='nearest')
        modules = [nn.Conv1d(in_channels + skip_channels, out_channels, 1, bias=False),
                   norm_layer(out_channels),
                   act_layer(inplace=True)]
        for i in range(1, num_blocks):
            modules.append(nn.Conv1d(out_channels, out_channels, 1, bias=False))
            modules.append(norm_layer(out_channels))
            modules.append(act_layer(inplace=True))
        self.block = nn.Sequential(*modules)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                if act_layer == nn.LeakyReLU:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif act_layer == nn.ReLU:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x, skip=None, **interpolation_kwargs):
        x = self.upsample(features=x, **interpolation_kwargs)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class UnetDecoder1D(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            decoder_layers,
            radius,
            nsample,
            center=False,
            act_layer: nn.Module = nn.ReLU,
            norm_layer: nn.Module = nn.BatchNorm1d
    ):
        super().__init__()

        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        # commented because the first features have stride=1
        skip_channels = list(encoder_channels[1:])  # + [0]
        out_channels = decoder_channels
        assert len(in_channels) == len(skip_channels) == len(out_channels) == len(decoder_layers)

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, dim=1
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock1D(num_blocks, in_ch, skip_ch, out_ch, r, n, act_layer=act_layer, norm_layer=norm_layer)
            for num_blocks, in_ch, skip_ch, out_ch, r, n in
            zip(decoder_layers, in_channels, skip_channels, out_channels, radius, nsample)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features, interpolation_kwargs_list):

        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip, **interpolation_kwargs_list[i])

        return x


class Unet1D(nn.Module):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        decoder_channels: list of numbers of ``Conv1D`` layer filters in decoder blocks
        in_channels: number of input channels for model, default is 3.

    Returns:
        ``torch.nn.Module``: **Unet1D**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
            self,
            encoder_name: str = "resnet26",
            decoder_channels: List[int] = (512, 256, 128, 64),
            decoder_layers: List[int] = (2, 2, 2, 2),
            in_channels: int = 3,
            act_layer: nn.Module = nn.ReLU,
            norm_layer: nn.Module = nn.BatchNorm1d,
            expansion: int = 4,
            **encoder_kwargs
    ):
        super().__init__()
        decoder_channels = list(decoder_channels)
        self.out_channels = decoder_channels[-1]

        if expansion == 4:
            block = Bottleneck
        elif expansion == 2:
            block = Bottleneck2
        else:
            raise ValueError

        if encoder_name == "resnet26":
            self.encoder = PointResNet(block, [2, 2, 2, 2], in_chans=in_channels,
                                       act_layer=act_layer, norm_layer=norm_layer,
                                       **encoder_kwargs)
        elif encoder_name == "resnet50":
            self.encoder = PointResNet(block, [3, 4, 6, 3], in_chans=in_channels,
                                       act_layer=act_layer, norm_layer=norm_layer,
                                       **encoder_kwargs)
        else:
            raise NotImplementedError

        encoder_channels = [item['num_chs'] for item in self.encoder.feature_info]

        self.decoder = UnetDecoder1D(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            decoder_layers=decoder_layers,
            radius=[0.05, 0.05, 0.05, 0.05],
            nsample=[30, 30, 30, 30],
            center=True if encoder_name.startswith("vgg") else False,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

    def forward(self, features: torch.Tensor):
        """
        Args:
            features (Tensor) of shape (B, C, N) where first 3 channels out of C correspond to x,y,z
        Returns:
        """
        xyz = features[:, :3, :].transpose(1, 2).contiguous()
        mask = torch.ones(features.size(0), features.size(2), dtype=torch.int, device=features.device)

        end_points = self.encoder(xyz, mask, features)
        features = [end_points['res1_features'],
                    end_points['res2_features'],
                    end_points['res3_features'],
                    end_points['res4_features'],
                    end_points['res5_features']]
        interpolation_kwargs_list = [
            dict(up_xyz=end_points['res4_xyz'], xyz=end_points['res5_xyz'], up_mask=end_points['res4_mask'],
                 mask=end_points['res5_mask']),
            dict(up_xyz=end_points['res3_xyz'], xyz=end_points['res4_xyz'], up_mask=end_points['res3_mask'],
                 mask=end_points['res4_mask']),
            dict(up_xyz=end_points['res2_xyz'], xyz=end_points['res3_xyz'], up_mask=end_points['res2_mask'],
                 mask=end_points['res3_mask']),
            dict(up_xyz=end_points['res1_xyz'], xyz=end_points['res2_xyz'], up_mask=end_points['res1_mask'],
                 mask=end_points['res2_mask']),
        ]
        decoder_output = self.decoder(*features, interpolation_kwargs_list=interpolation_kwargs_list)
        return decoder_output
