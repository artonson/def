import logging

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from ..utils.comm import all_gather, synchronize
from ..utils.image import plot_to_image

log = logging.getLogger(__name__)


class RegressionEvaluator(DatasetEvaluator):
    """
    Evaluate regression metrics.
    """

    def __init__(self, input_key, output_key, dataset_name):
        super().__init__(dataset_name)
        self.input_key = input_key
        self.output_key = output_key
        self.reset()

    def reset(self):
        self._rmse_sum = 0
        self._size = 0
        self._rmse_hist = 0

    def process(self, inputs, outputs):
        """
        Args:
            inputs (dict): the inputs to a model.
            outputs (dict): the outputs of a model.
        """
        target = inputs[self.input_key]
        preds = outputs[self.output_key]

        batch_size = target.size(0)
        mean_squared_errors = F.mse_loss(preds, target, reduction='none').view(batch_size, -1).mean(dim=1)  # (batch)
        root_mean_squared_errors = torch.sqrt(mean_squared_errors)  # (batch)

        self._rmse_sum += root_mean_squared_errors.sum().cpu()
        self._size += torch.tensor(batch_size)
        self._rmse_hist += torch.histc(root_mean_squared_errors, bins=100, min=0.0, max=1.0).cpu()

    def evaluate(self):

        # gather results across gpus
        synchronize()
        rmse_sum = torch.cat(all_gather(self._rmse_sum.view(1))).sum()
        size = torch.cat(all_gather(self._size.view(1))).sum()
        rmse_hist = torch.stack(all_gather(self._rmse_hist), dim=0).sum(dim=0)

        # calculate metrics
        mean_rmse = rmse_sum / size
        mean_rmse = mean_rmse.item()

        scalars = {f'mean_rmse/{self.dataset_name}': mean_rmse}
        images = {f'rmse_hist/{self.dataset_name}': plot_to_image(
            self._plot_rmse_hist(rmse_hist, f"{self.dataset_name}, {mean_rmse}"))}

        # todo write rmse_hist to disk or try to use pl.evalresult.write

        return {'scalars': scalars, 'images': images}

    def _plot_rmse_hist(self, rmse_hist, label):
        figure = plt.figure(figsize=(4.8, 3.6))
        prototype_hist = rmse_hist[0] if isinstance(rmse_hist, list) else rmse_hist
        bins = torch.linspace(0, 1, len(prototype_hist) + 1)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        if isinstance(rmse_hist, list):
            for label_i, hist in zip(label, rmse_hist):
                plt.bar(center, hist, label=label_i, align='center', width=width, alpha=0.5)
        else:
            plt.bar(center, rmse_hist, label=label, align='center', width=width)
        plt.xlabel('RMSE per patch value')
        plt.ylabel('# of patches')
        plt.xlim((0.0, 1.0))
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.tight_layout()
        return figure
