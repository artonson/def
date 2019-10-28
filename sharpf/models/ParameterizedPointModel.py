import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sharpf.modules.base import ParameterizedModule, load_with_spec


PATCHES = {'default': 'some default module', '': }
CONV = {'default': 'some default module', '': }
TRANSFORMS = {'default': 'some default module', '': }
SCALING = {'default': 'some default module', '': }

MODULES_ZOO = {'patches': PATCHES, 'conv': CONV, 'transforms': TRANSFORMS, 'scaling': SCALING }


class ParameterizedPointModel(ParameterizedModule):
    def __init__(self, patches, conv, transforms, scaling, **kwargs):
        super(ParameterizedPointModel, self).__init__()
        self.patches = patches
        self.conv = conv
        self.transforms = transforms
        self.scaling = scaling

        self.modules_list = nn.ModuleList([
                                           self.patches,
                                           self.conv,
                                           self.transforms,
                                           self.scaling
                                          ])

    def forward(self, x):
        out = [x]
        for module in self.module_list:
            out.append(module(out[-1]))
        return out[-1]

    @classmethod
    def from_spec(cls, spec):
        patches = load_with_spec(spec['pathes'], PATCHES)
        conv = load_with_spec(spec['conv'], CONV)
        transforms = load_with_spec(spec['transforms'], TRANSFORMS)
        scaling = load_with_spec(spec['scaling'], SCALING)

        return cls(patches, conv, transforms, scaling)
