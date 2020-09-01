from typing import List

import torch.nn as nn


class Sequential(nn.Sequential):

    def __init__(self, modules: List[nn.Module]):
        super().__init__(*modules)
