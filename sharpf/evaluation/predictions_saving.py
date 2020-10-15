import logging
import os

import numpy as np
import torch

from .evaluator import DatasetEvaluator

log = logging.getLogger(__name__)


class PredictionsSaving(DatasetEvaluator):

    def __init__(self, filter_expressions=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_expressions = filter_expressions if filter_expressions is not None else []

    def reset(self):
        pass

    def process(self, inputs, outputs):
        """
        Args:
            inputs (dict): the inputs to a model.
            outputs (dict): the outputs of a model.
        """
        # apply filters
        final_mask = torch.ones_like(inputs['index'], device=inputs['index'].device, dtype=torch.bool)
        for key, op, value in self.filter_expressions:
            if op == '=':
                mask = inputs[key] == value
            elif op == '>':
                mask = inputs[key] > value
            elif op == '>=':
                mask = inputs[key] >= value
            else:
                raise ValueError
            final_mask = final_mask * mask

        if final_mask.sum() == 0:
            return

        inds = final_mask.nonzero(as_tuple=True)[0]
        dataset_indexes = torch.index_select(inputs['index'], 0, inds)
        preds = torch.index_select(outputs['distances'], 0, inds)

        for i, index in enumerate(dataset_indexes):
            self.save_tensor(preds[i], f"{self.dataset_name}_{index.item()}.npy")

    def save_tensor(self, tensor, name):
        current_dir = os.getcwd()
        save_dir = os.path.join(current_dir, 'predictions')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.save(os.path.join(save_dir, name), tensor.cpu().numpy())

    def evaluate(self):
        return {'scalars': {}, 'images': {}}
