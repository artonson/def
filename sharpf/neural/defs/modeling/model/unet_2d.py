from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Unet2D']


class DecoderBlock2D(nn.Module):
    def __init__(
            self,
            num_blocks,
            in_channels,
            skip_channels,
            out_channels,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            kernel_size=3
    ):
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        modules = [
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            norm_layer(out_channels),
            act_layer(inplace=True)]
        for i in range(1, num_blocks):
            modules.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False))
            modules.append(norm_layer(out_channels))
            modules.append(act_layer(inplace=True))
        self.block = nn.Sequential(*modules)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if act_layer == nn.LeakyReLU:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif act_layer == nn.ReLU:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class UnetDecoder2D(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            decoder_layers,
            center=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            decoder_kernel_size=3
    ):
        super().__init__()

        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])

        # commented because the first features have stride=1
        skip_channels = list(encoder_channels[1:])  # + [0]
        out_channels = decoder_channels
        assert len(in_channels) == len(skip_channels) == len(out_channels)

        if center:
            raise NotImplementedError
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock2D(num_blocks, in_ch, skip_ch, out_ch, act_layer=act_layer, norm_layer=norm_layer,
                           kernel_size=decoder_kernel_size)
            for num_blocks, in_ch, skip_ch, out_ch in zip(decoder_layers, in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]

        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class Unet2D(nn.Module):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        in_channels: number of input channels for model, default is 3.

    Returns:
        ``torch.nn.Module``: **Unet2D**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
            self,
            encoder_name: str = "resnet50",
            decoder_channels: List[int] = (512, 256, 128, 64),
            decoder_layers: List[int] = (2, 2, 2, 2),
            in_channels: int = 3,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            decoder_kernel_size=3,
            **encoder_kwargs
    ):
        super().__init__()
        decoder_channels = list(decoder_channels)
        self.out_channels = decoder_channels[-1]
        self.encoder = timm.create_model(encoder_name, features_only=True, pretrained=False, **encoder_kwargs)
        self.encoder.conv1.stride = 1
        _patch_first_conv2d(self.encoder, in_channels)
        self.output_stride = 2 ** len(decoder_channels)

        for n, m in self.encoder.named_modules():
            if isinstance(m, nn.Conv1d):
                if act_layer == nn.LeakyReLU:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif act_layer == nn.ReLU:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

        for m in self.encoder.modules():
            if hasattr(m, 'zero_init_last_bn'):
                m.zero_init_last_bn()

        self.decoder = UnetDecoder2D(
            encoder_channels=self.encoder.feature_info.channels(),
            decoder_channels=decoder_channels,
            decoder_layers=decoder_layers,
            center=True if encoder_name.startswith("vgg") else False,
            act_layer=act_layer,
            norm_layer=norm_layer,
            decoder_kernel_size=decoder_kernel_size
        )

    def forward(self, x):
        # pad
        b, c, h, w = x.size()
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
        if h % self.output_stride != 0:
            pad_height = (h // self.output_stride + 1) * self.output_stride - h
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
        if w % self.output_stride != 0:
            pad_width = (w // self.output_stride + 1) * self.output_stride - w
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        # unpad
        decoder_output = decoder_output[:, :, pad_top:, pad_left:]
        if pad_bottom:
            decoder_output = decoder_output[:, :, :-pad_bottom, :]
        if pad_right:
            decoder_output = decoder_output[:, :, :, :-pad_right]

        return decoder_output


def _patch_first_conv2d(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight, requires_grad=True)
    if reset:
        module.reset_parameters()
