from typing import Optional, List

import timm
import torch
import torch.nn as nn

from ..modules import UnetDecoder
from ...utils.init import initialize_decoder


class Unet(nn.Module):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        in_channels: number of input channels for model, default is 3.

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
            self,
            encoder_name: str = "resnet50",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
    ):
        super().__init__()
        decoder_channels = list(decoder_channels)
        self.out_channels = decoder_channels[-1]
        self.encoder = timm.create_model(encoder_name, features_only=True, pretrained=False)
        _patch_first_conv(self.encoder, in_channels)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.feature_info.channels(),
            decoder_channels=decoder_channels,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.initialize()

    def initialize(self):
        initialize_decoder(self.decoder)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        return decoder_output


def _patch_first_conv(model, in_channels):
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

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()
