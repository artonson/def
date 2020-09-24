import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from ..utils.comm import all_gather, synchronize, is_main_process
from ..utils.image import plot_to_image

log = logging.getLogger(__name__)


class RegressionEvaluator(DatasetEvaluator):
    """
    Evaluate regression metrics and plot histograms.
    """

    def __init__(self, input_key: str, output_key: str, mask_expressions=None, baseline_hist_dir_path=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.mask_expressions = mask_expressions if mask_expressions is not None else []
        self.mask_key_set = set([key for key, op, value in self.mask_expressions])
        self.baseline_hist_dir_path = baseline_hist_dir_path
        self.reset()

    def reset(self):
        self.rmses = []
        self.masks = {key: [] for key in self.mask_key_set}

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
        root_mean_squared_errors = torch.sqrt(mean_squared_errors)
        self.rmses.append(root_mean_squared_errors.detach().cpu())

        for key in self.mask_key_set:
            try:
                self.masks[key].append(inputs[key].detach().cpu())
            except KeyError as e:
                raise KeyError(
                    f"Can't find {key} among mask keys: {self.masks.keys()} or input keys: {inputs.keys()}") from e

    def evaluate(self):
        self.rmses = torch.cat(self.rmses)
        for key in self.mask_key_set:
            self.masks[key] = torch.cat(self.masks[key])

        # gather results across gpus
        synchronize()
        rmses = torch.cat(all_gather(self.rmses))
        masks = {}
        for key in self.mask_key_set:
            masks[key] = torch.cat(all_gather(self.masks[key]))

        # calculate metrics for whole dataset
        mean_rmse = rmses.mean().item()
        quantile_90 = np.quantile(rmses, 0.90)
        quantile_95 = np.quantile(rmses, 0.95)
        quantile_99 = np.quantile(rmses, 0.99)
        rmse_hist = torch.histc(rmses, bins=100, min=0.0, max=1.0)

        scalars = {
            f'count/{self.dataset_name}': rmses.size(0),
            f'mean_rmse/{self.dataset_name}': mean_rmse,
            f'q90_rmse/{self.dataset_name}': quantile_90,
            f'q95_rmse/{self.dataset_name}': quantile_95,
            f'q99_rmse/{self.dataset_name}': quantile_99,
        }
        hist_name = f'rmse_hist/{self.dataset_name}'
        images = {hist_name: plot_to_image(self._plot_rmse_hist(rmse_hist, hist_name))}

        if is_main_process():
            self.save_hist(rmse_hist, f'rmse_hist_{self.dataset_name}.pt')

        # calculate metrics for subsets
        for key, op, value in self.mask_expressions:
            if op == '=':
                mask = masks[key] == value
            elif op == '>':
                mask = masks[key] > value
            elif op == '>=':
                mask = masks[key] >= value
            else:
                raise ValueError

            subset_count = mask.sum().item()
            scalars[f'count_{key}{op}{value}/{self.dataset_name}'] = subset_count
            if subset_count >= 1:
                rmse_subset = torch.masked_select(rmses, mask)
                mean_rmse_subset = rmse_subset.mean().item()
                quantile_90_subset = np.quantile(rmse_subset, 0.90)
                quantile_95_subset = np.quantile(rmse_subset, 0.95)
                quantile_99_subset = np.quantile(rmse_subset, 0.99)
                rmse_hist_subset = torch.histc(rmse_subset, bins=100, min=0.0, max=1.0)
                scalars[f'mean_rmse_{key}{op}{value}/{self.dataset_name}'] = mean_rmse_subset
                scalars[f'q90_rmse_{key}{op}{value}/{self.dataset_name}'] = quantile_90_subset
                scalars[f'q95_rmse_{key}{op}{value}/{self.dataset_name}'] = quantile_95_subset
                scalars[f'q99_rmse_{key}{op}{value}/{self.dataset_name}'] = quantile_99_subset

                hist_name = f'rmse_hist_{key}{op}{value}/{self.dataset_name}'
                images[hist_name] = plot_to_image(self._plot_rmse_hist(rmse_hist_subset, hist_name))

                if is_main_process():
                    save_hist_name = hist_name.replace('/', '_')
                    self.save_hist(rmse_hist_subset, f'{save_hist_name}.pt')

        return {'scalars': scalars, 'images': images}

    def _plot_rmse_hist(self, hist: torch.Tensor, hist_name: str):
        figure = plt.figure(figsize=(4.8, 3.6))
        prototype_hist = hist[0] if isinstance(hist, list) else hist
        bins = torch.linspace(0, 1, len(prototype_hist) + 1)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        alpha = 1.0
        if self.baseline_hist_dir_path is not None:
            search_hist_name = hist_name.replace('/', '_')
            path = os.path.join(self.baseline_hist_dir_path, f'{search_hist_name}.pt')
            if os.path.exists(path):
                try:
                    baseline_hist = torch.load(path)
                except RuntimeError:
                    baseline_hist = np.load(path.replace('.pt', '.npy'))
                    hist = hist.cpu().numpy()
                if baseline_hist.sum() == hist.sum() and baseline_hist.shape == hist.shape:
                    alpha = 0.5
                    plt.bar(center, baseline_hist, align='center', label='baseline', width=width, alpha=alpha)
                    plt.legend(loc='upper right')
                else:
                    log.info(f"Baseline histogram ({path}) shape does not match.")
            else:
                log.info(f"Can't find baseline histogram: {path}")
        plt.bar(center, hist, align='center', width=width, alpha=alpha)
        plt.xlabel('RMSE per patch value')
        plt.ylabel('# of patches')
        plt.xlim((0.0, 1.0))
        plt.yscale('log')
        plt.tight_layout()
        return figure

    def save_hist(self, hist, name):
        current_dir = os.getcwd()
        hist_dir = os.path.join(current_dir, 'hist')
        if not os.path.exists(hist_dir):
            os.mkdir(hist_dir)
        torch.save(hist, os.path.join(hist_dir, name))
        np.save(os.path.join(hist_dir, name).replace('.pt', '.npy'), hist.cpu().numpy())
