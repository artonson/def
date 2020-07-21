import torch
from torch import nn
import json
import numpy as np
from .decoder import UnetDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder

class Classifier(EncoderDecoder):
    #
    # predict has sharp / no sharp
    #

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_channels=[4096, 2048, 1024],
            classes=1,
            activation='sigmoid',
            attention=False,
            attention_type=None
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        if activation == 'None':
            activation = None

        # if attention:
        #         #     if attention_type is None or attention_type == "None":
        #         #         self.attention = nn.Identity()

        decoder = nn.Sequential(nn.Identity(),
                                nn.Conv1d(512, decoder_channels[0], 1),#131072, decoder_channels[0], 1),
                                nn.BatchNorm1d(decoder_channels[0]),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Conv1d(decoder_channels[0], decoder_channels[1], 1),
                                nn.BatchNorm1d(decoder_channels[1]),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Conv1d(decoder_channels[1], decoder_channels[2], 1),
                                nn.BatchNorm1d(decoder_channels[2]),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Conv1d(decoder_channels[2], classes, 1))

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        batch_sz = x.shape[0]
        mask = torch.reshape(x[:, -1, :, :], (batch_sz, 1, x.shape[-2], x.shape[-1]))
        x = x[:, :-1, :, :]
        x = self.encoder.features[0](x, mask)
        x = self.encoder.features[1:](x)
        x = nn.AdaptiveAvgPool2d((7, 7))(x)
        #x = torch.flatten(x, 1)

        x = self.decoder(x.reshape(batch_sz, 512, 7*7))
        pred = self.activation(x[:, :, 0])
       # print('pred', pred)
        return pred

class classifier_unet(EncoderDecoder):
    #
    # classifier predicts sharp / no sharp and adds category to feature vector
    #
    
    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
            attention_type=None
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = Classifier_UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            attention_type=attention_type
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)

class two_stage_unet(EncoderDecoder):
    def __init__(self,
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 decoder_use_batchnorm=True,
                 decoder_channels=(256, 128, 64, 32, 16),
                 classes=1,
                 activation='sigmoid',
                 classifier=None,
                 center=False,  # useful for VGG models
                 attention_type=None
                 ):

        self.classes = classes
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            attention_type=attention_type
        )

        super().__init__(encoder, decoder, activation)
        self.classifier = classifier.eval()


    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        # classifier desision
        class_res = self.classifier.forward(x)
        self.class_res_binary = []
        for c in class_res:
            self.class_res_binary.append(int(c > 0.5))
        # segmentator decision
        #print(class_res)
        x = self.encoder(x)
        x = self.decoder(x)


        from torch import erf
        def calculate_density(mean, sigma, discretization, a=0, b=1, margin=2):
            # compute how many bins should be (heuristic)
            #             sigma = 1 / discretization

            # allow margins from interval ranges for better accuracy
            a = a - margin * (1 / discretization)
            b = b + margin * (1 / discretization)

            num_bins = discretization + margin * 2

            # calculate bin limits
            tmp_bins = np.tile(np.linspace(a, b, num_bins + 1), (mean.shape[0], mean.shape[1], 1))
            bin_limits = torch.FloatTensor(tmp_bins).to('cuda')

            # normalizing coefficient
            Z = 0.5 * (erf((b - mean) / (np.sqrt(2) * sigma)) - erf((a - mean) / (np.sqrt(2) * sigma)))

            # compute discretized densities
            # c = torch.zeros((mean.shape[0], mean.shape[1], 240))

            # for i, bin in enumerate(bin_limits):
            tmp = erf((bin_limits[:, :, 1:] - mean.unsqueeze(-1)) / (np.sqrt(2) * sigma)) - erf(
                (bin_limits[:, :, :-1] - mean.unsqueeze(-1)) / (np.sqrt(2) * sigma))
            c = tmp / (2 * Z.unsqueeze(-1))

            # return the computed density and set of centers of the bins
            return c.permute(1, 0, 2, ).type(torch.float32), (bin_limits[:, :, :-1] + (1 / discretization / 2)).type(
                torch.float32), num_bins

        discretization = 240  # heuristic
        sigma = 1 / discretization
        batch_sz = x.shape[0]
        for i,cl in enumerate(self.class_res_binary):

            if cl: #sharp
                print('sharp')
                continue

            else: #not sharp
                print('not sharp')
                #x[i] = torch.ones([1, x.shape[1], x.shape[-1]]).to('cuda')
                tmp_ones = torch.ones([1, 64, 64]).to('cuda')
                #print(x[i].shape)
                density, linspace, num_bins = calculate_density(torch.t(tmp_ones.reshape(1, -1)), sigma, discretization)
                x[i] = density.reshape(1, 244, 64, 64)

        return x

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            for i,cl in enumerate(self.class_res_binary):
                if self.activation:
                    if cl:
                        x[i] = self.activation(x[i])

        return x


class Unet(EncoderDecoder):
    """Unet is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        enter: if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)
        attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
            attention_type=None
    ):
        self.classes = classes
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            attention_type=attention_type
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)
