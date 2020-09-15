import logging

import numpy as np
import torch

from .evaluator import DatasetEvaluator
from ..utils.comm import all_gather, synchronize

log = logging.getLogger(__name__)


class QuantileEvaluator(DatasetEvaluator):
    """
    Evaluate 0.95 quantile
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self._maxs = None

    def process(self, inputs, outputs):
        """
        Args:
            inputs (dict): the inputs to a model.
            outputs (dict): the outputs of a model.
        """
        x = inputs['points']
        batch_maxs = x.view(x.size(0), -1).max(dim=1).values
        self._maxs = torch.cat([self._maxs, batch_maxs]) if self._maxs is not None else batch_maxs

    def evaluate(self):
        # gather results across gpus
        synchronize()
        maxs = torch.cat(all_gather(self._maxs)).numpy()
        print(f"0.95 quantile: {np.quantile(maxs, 0.95)}")
        print(f"max: {maxs.max()}")
        return {'scalars': {}, 'images': {}}
