import torch
import torch.nn as nn
from .pd_vgg import PDVGG as VGG
from .pd_vgg import make_layers_pd as make_layers
from pretrainedmodels.models.torchvision_models import pretrained_settings


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGEncoder(VGG):

    def __init__(self, config, batch_norm=False, *args, **kwargs):
        super().__init__(
            make_layers(config, batch_norm=batch_norm), 
            *args, 
            **kwargs
        )
        self.pretrained = False
        del self.classifier

    def forward(self, x, mask_in=None):
        features = []
        batch_sz = x.shape[0]
        mask = torch.reshape(x[:,-1, :, :], (batch_sz,1, x.shape[-2], x.shape[-1]))
        x = x[:, :-1, :, :]
        for i,module in enumerate(self.features):
            if isinstance(module, nn.MaxPool2d):
                features.append(x)
            #if i == 0:
            if isinstance(module, nn.MaxPool2d):
                x = module(x,mask)
            else:
                x = module(x, mask)
            #else:
            #    x = module(x)

        features.append(x)
        features = features[1:]
        features = features[::-1]
       
        return features#x.view(x.size(0), -1, 1)

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith('classifier'):
                state_dict.pop(k)
        super().load_state_dict(state_dict, **kwargs)


vgg_encoders_partconv = {

    'vgg11_partconv': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'pretrained_settings': pretrained_settings['vgg11'],
        'params': {
            'config': cfg['A'],
            'batch_norm': False,
        },
    },

    'vgg11_partconv_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'pretrained_settings': pretrained_settings['vgg11_bn'],
        'params': {
            'config': cfg['A'],
            'batch_norm': True,
        },
    },

    'vgg13_partconv': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'pretrained_settings': pretrained_settings['vgg13'],
        'params': {
            'config': cfg['B'],
            'batch_norm': False,
        },
    },

    'vgg13_partconv_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'pretrained_settings': pretrained_settings['vgg13_bn'],
        'params': {
            'config': cfg['B'],
            'batch_norm': True,
        },
    },

    'vgg16_partconv': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'pretrained_settings': pretrained_settings['vgg16'],
        'params': {
            'config': cfg['D'],
            'batch_norm': False,
        },
    },

    'vgg16_partconv_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'pretrained_settings': pretrained_settings['vgg16_bn'],
        'params': {
            'config': cfg['D'],
            'batch_norm': True,
        },
    },

    'vgg19_partconv': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'pretrained_settings': pretrained_settings['vgg19'],
        'params': {
            'config': cfg['E'],
            'batch_norm': False,
        },
    },

    'vgg19_partconv_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'pretrained_settings': pretrained_settings['vgg19_bn'],
        'params': {
            'config': cfg['E'],
            'batch_norm': True,
        },
    },
}
