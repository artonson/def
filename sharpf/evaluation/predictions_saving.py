import logging
import os

import numpy as np

from .evaluator import DatasetEvaluator

log = logging.getLogger(__name__)


class PredictionsSaving(DatasetEvaluator):

    def reset(self):
        pass

    def process(self, inputs, outputs):
        """
        Args:
            inputs (dict): the inputs to a model.
            outputs (dict): the outputs of a model.
        """
        for i, index in enumerate(inputs['index']):
            self.save_tensor(outputs['distances'][i], f"{self.dataset_name}_{index.item()}.npy")

    def save_tensor(self, tensor, name):
        print(tensor.shape, name)
        current_dir = os.getcwd()
        save_dir = os.path.join(current_dir, 'predictions')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.save(os.path.join(save_dir, name), tensor.cpu().numpy())

    def evaluate(self):
        return {'scalars': {}, 'images': {}}
