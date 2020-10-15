import logging
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from ..utils.comm import all_gather, synchronize, is_main_process
from ..utils.image import plot_to_image

log = logging.getLogger(__name__)


class SharpnessEvaluator(DatasetEvaluator):
    """
    Evaluate metrics and plot histograms.
    """

    def __init__(self, bp_ts_per_resolution: Optional[List[int]] = None, resolution: Optional[float] = None,
                 calculate_rmse: bool = True, calculate_bad_points: bool = False, calculate_iou: bool = False,
                 calculate_ba: bool = False, calculate_rmse_dl1: bool = False, calculate_bad_points_dl1: bool = False,
                 mask_expressions=None, reference_hist_dirs=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = resolution
        self.mask_expressions = mask_expressions if mask_expressions is not None else []
        self.mask_key_set = set([key for key, op, value in self.mask_expressions])
        self.reference_hist_dirs = [['', None]]
        if reference_hist_dirs is not None and len(reference_hist_dirs) > 0:
            self.reference_hist_dirs = reference_hist_dirs
        self.bp_ts = bp_ts_per_resolution if bp_ts_per_resolution is not None else []
        self.calculate_rmse = calculate_rmse
        self.calculate_bad_points = calculate_bad_points
        self.calculate_iou = calculate_iou
        self.calculate_ba = calculate_ba
        self.calculate_rmse_dl1 = calculate_rmse_dl1
        self.calculate_bad_points_dl1 = calculate_bad_points_dl1
        if self.calculate_bad_points or self.calculate_bad_points:
            assert self.bp_ts is not None and len(self.bp_ts) > 0
            assert resolution is not None
        if self.calculate_iou or self.calculate_ba:
            assert resolution is not None
        self.reset()

    def reset(self):
        self.rmses = []
        self.rmses_dl1 = []
        self.ious = []
        self.bas = []
        self.bp_pct = {t: [] for t in self.bp_ts}
        self.bp_pct_dl1 = {t: [] for t in self.bp_ts}
        self.masks = {key: [] for key in self.mask_key_set}
        self.scalars = {}
        self.images = {}

    def process(self, inputs, outputs):
        """
        Args:
            inputs (dict): the inputs to a model.
            outputs (dict): the outputs of a model.
        """
        target = inputs['distances']
        preds = outputs['distances']

        batch_size = target.size(0)
        batch_size, num_points = target.view(batch_size, -1).size()

        target = target.view(batch_size, num_points)
        preds = preds.view(batch_size, num_points)
        if self.calculate_iou or self.calculate_ba:
            target_sharp = (target < self.resolution).long()
            preds_sharp = (preds < self.resolution).long()

        # calculate IOU
        if self.calculate_iou:
            intersection = target_sharp * preds_sharp
            union = (target_sharp + preds_sharp - intersection).sum(dim=1)
            union_zero_mask = union == 0
            union[union_zero_mask] = 1.0
            intersection = intersection.sum(dim=1)
            self.ious.append(torch.where(union_zero_mask,
                                         torch.zeros_like(intersection, device=intersection.device, dtype=torch.float),
                                         intersection.float() / union.float()).detach().cpu())

        # calculate Balanced Accuracy
        if self.calculate_ba:
            for i in range(batch_size):
                tp = (preds_sharp[i] * target_sharp[i]).sum().float()
                fp = (preds_sharp[i] * (1 - target_sharp[i])).sum().float()
                fn = ((1 - preds_sharp[i]) * target_sharp[i]).sum().float()
                tn = ((1 - preds_sharp[i]) * (1 - target_sharp[i])).sum().float()
                tpr = tp / (tp + fn)
                tnr = tn / (tn + fp)
                tpr = torch.where(torch.isnan(tpr), tnr, tpr)
                tnr = torch.where(torch.isnan(tnr), tpr, tnr)
                ba = 0.5 * (tpr + tnr)
                self.bas.append(ba.view(1).detach().cpu())

        # calculate RMSE
        squared_errors = F.mse_loss(preds, target, reduction='none')
        if self.calculate_rmse:
            self.rmses.append(torch.sqrt(squared_errors.mean(dim=1)).detach().cpu())

        # calculate BadPoints(T)
        if self.calculate_bad_points:
            for t in self.bp_ts:
                bad_points_mask = squared_errors > (t * self.resolution) ** 2
                self.bp_pct[t].append((bad_points_mask.float().sum(dim=1) / num_points).detach().cpu())

        # calculate RMSE and BadPoints(T) over the points with distance < 1.0 only
        if self.calculate_bad_points_dl1 or self.calculate_rmse_dl1:
            for i in range(batch_size):
                mask_dl1 = target[i] < 1.0
                if torch.any(mask_dl1):
                    squared_errors_dl1 = F.mse_loss(preds[i][mask_dl1], target[i][mask_dl1], reduction='none')
                    if self.calculate_rmse_dl1:
                        self.rmses_dl1.append(torch.sqrt(squared_errors.mean()).view(1).detach().cpu())
                    if self.calculate_bad_points_dl1:
                        for t in self.bp_ts:
                            bad_points_mask = squared_errors_dl1 > (t * self.resolution) ** 2
                            self.bp_pct_dl1[t].append(
                                (bad_points_mask.float().sum() / num_points).view(1).detach().cpu())
                else:
                    if self.calculate_rmse_dl1:
                        self.rmses_dl1.append(torch.tensor(0, dtype=torch.float).view(1))
                    if self.calculate_bad_points_dl1:
                        for t in self.bp_ts:
                            self.bp_pct_dl1[t].append(torch.tensor(0, dtype=torch.float).view(1))

        # save masks for further processing
        for key in self.mask_key_set:
            try:
                self.masks[key].append(inputs[key].detach().cpu())
            except KeyError as e:
                raise KeyError(
                    f"Can't find {key} among mask keys: {self.masks.keys()} or input keys: {inputs.keys()}") from e

    def report_rmse(self, mask=None, mask_suffix=''):
        if mask is not None:
            assert mask_suffix != ''
        if isinstance(self.rmses, list):
            self.rmses = torch.cat(self.rmses)
            synchronize()
            self.rmses = torch.cat(all_gather(self.rmses))
        rmses = self.rmses if mask is None else self.rmses[mask]
        self.scalars[f'count{mask_suffix}/{self.dataset_name}'] = rmses.size(0)
        self.scalars[f'mean_rmse{mask_suffix}/{self.dataset_name}'] = rmses.mean().item()
        self.scalars[f'q90_rmse{mask_suffix}/{self.dataset_name}'] = np.quantile(rmses, 0.90)
        self.scalars[f'q95_rmse{mask_suffix}/{self.dataset_name}'] = np.quantile(rmses, 0.95)
        self.scalars[f'q99_rmse{mask_suffix}/{self.dataset_name}'] = np.quantile(rmses, 0.99)
        rmse_hist = torch.histc(rmses, bins=100, min=0.0, max=1.0)
        hist_name = f'rmse_hist{mask_suffix}/{self.dataset_name}'
        for ref_label, ref_dir in self.reference_hist_dirs:
            self.images[hist_name] = plot_to_image(self._plot_hist(rmse_hist, hist_name, ref_dir,
                                                                   xlabel='RMSE per patch value',
                                                                   reference_label=ref_label))
        if is_main_process():
            save_hist_name = hist_name.replace('/', '_') + '.npy'
            self.save_hist(rmse_hist, save_hist_name)

    def report_rmse_dl1(self, mask=None, mask_suffix=''):
        if mask is not None:
            assert mask_suffix != ''
        if isinstance(self.rmses_dl1, list):
            self.rmses_dl1 = torch.cat(self.rmses_dl1)
            synchronize()
            self.rmses_dl1 = torch.cat(all_gather(self.rmses_dl1))
        rmses_dl1 = self.rmses_dl1 if mask is None else self.rmses_dl1[mask]
        self.scalars[f'mean_rmse_dl1{mask_suffix}/{self.dataset_name}'] = rmses_dl1.mean().item()
        self.scalars[f'q90_rmse_dl1{mask_suffix}/{self.dataset_name}'] = np.quantile(rmses_dl1, 0.90)
        self.scalars[f'q95_rmse_dl1{mask_suffix}/{self.dataset_name}'] = np.quantile(rmses_dl1, 0.95)
        self.scalars[f'q99_rmse_dl1{mask_suffix}/{self.dataset_name}'] = np.quantile(rmses_dl1, 0.99)
        rmse_hist = torch.histc(rmses_dl1, bins=100, min=0.0, max=1.0)
        hist_name = f'rmse_dl1_hist{mask_suffix}/{self.dataset_name}'
        for ref_label, ref_dir in self.reference_hist_dirs:
            self.images[hist_name] = plot_to_image(self._plot_hist(rmse_hist, hist_name, ref_dir,
                                                                   xlabel='RMSE(d<1.0) per patch value',
                                                                   reference_label=ref_label))
        if is_main_process():
            save_hist_name = hist_name.replace('/', '_') + '.npy'
            self.save_hist(rmse_hist, save_hist_name)

    def report_iou(self, mask=None, mask_suffix=''):
        if mask is not None:
            assert mask_suffix != ''
        if isinstance(self.bas, list):
            self.ious = torch.cat(self.ious)
            synchronize()
            self.ious = torch.cat(all_gather(self.ious))
        ious = self.ious if mask is None else self.ious[mask]
        self.scalars[f'mean_iou{mask_suffix}/{self.dataset_name}'] = ious.mean().item()
        iou_hist = torch.histc(ious, bins=100, min=0.0, max=1.0)
        hist_name = f'iou_hist{mask_suffix}/{self.dataset_name}'
        for ref_label, ref_dir in self.reference_hist_dirs:
            self.images[hist_name] = plot_to_image(self._plot_hist(iou_hist, hist_name, ref_dir,
                                                                   xlabel='IOU per patch value',
                                                                   reference_label=ref_label))
        if is_main_process():
            save_hist_name = hist_name.replace('/', '_') + '.npy'
            self.save_hist(iou_hist, save_hist_name)

    def report_ba(self, mask=None, mask_suffix=''):
        if mask is not None:
            assert mask_suffix != ''
        if isinstance(self.bas, list):
            self.bas = torch.cat(self.bas)
            synchronize()
            self.bas = torch.cat(all_gather(self.bas))
        bas = self.bas if mask is None else self.bas[mask]
        self.scalars[f'mean_balanced_accuracy{mask_suffix}/{self.dataset_name}'] = bas.mean().item()
        ba_hist = torch.histc(bas, bins=100, min=0.0, max=1.0)
        hist_name = f'balanced_accuracy_hist{mask_suffix}/{self.dataset_name}'
        for ref_label, ref_dir in self.reference_hist_dirs:
            self.images[hist_name] = plot_to_image(self._plot_hist(ba_hist, hist_name, ref_dir,
                                                                   xlabel='Balanced Accuracy per patch value',
                                                                   reference_label=ref_label))
        if is_main_process():
            save_hist_name = hist_name.replace('/', '_') + '.npy'
            self.save_hist(ba_hist, save_hist_name)

    def report_bad_points(self, mask=None, mask_suffix=''):
        if mask is not None:
            assert mask_suffix != ''
        for t in self.bp_ts:
            if isinstance(self.bp_pct[t], list):
                self.bp_pct[t] = torch.cat(self.bp_pct[t])
                synchronize()
                self.bp_pct[t] = torch.cat(all_gather(self.bp_pct[t]))
            bp_pct_t = self.bp_pct[t] if mask is None else self.bp_pct[t][mask]
            self.scalars[f'mean_bad_points_ratio_{t}r{mask_suffix}/{self.dataset_name}'] = bp_pct_t.mean().item()
            bp_pct_t_hist = torch.histc(bp_pct_t, bins=100, min=0.0, max=1.0)
            hist_name = f'bad_points_ratio_{t}r_hist{mask_suffix}/{self.dataset_name}'
            for ref_label, ref_dir in self.reference_hist_dirs:
                self.images[hist_name] = plot_to_image(self._plot_hist(bp_pct_t_hist, hist_name, ref_dir,
                                                                       xlabel=f'BadPoints({t}r) per patch value',
                                                                       reference_label=ref_label))
            if is_main_process():
                save_hist_name = hist_name.replace('/', '_') + '.npy'
                self.save_hist(bp_pct_t_hist, save_hist_name)

    def report_bad_points_dl1(self, mask=None, mask_suffix=''):
        if mask is not None:
            assert mask_suffix != ''
        for t in self.bp_ts:
            if isinstance(self.bp_pct_dl1[t], list):
                self.bp_pct_dl1[t] = torch.cat(self.bp_pct_dl1[t])
                synchronize()
                self.bp_pct_dl1[t] = torch.cat(all_gather(self.bp_pct_dl1[t]))
            bp_pct_dl1_t = self.bp_pct_dl1[t] if mask is None else self.bp_pct_dl1[t][mask]
            self.scalars[
                f'mean_bad_points_ratio_{t}r_dl1{mask_suffix}/{self.dataset_name}'] = bp_pct_dl1_t.mean().item()
            bp_pct_dl1_t_hist = torch.histc(bp_pct_dl1_t, bins=100, min=0.0, max=1.0)
            hist_name = f'bad_points_ratio_{t}r_dl1_hist{mask_suffix}/{self.dataset_name}'
            for ref_label, ref_dir in self.reference_hist_dirs:
                self.images[hist_name] = plot_to_image(self._plot_hist(bp_pct_dl1_t_hist, hist_name, ref_dir,
                                                                       xlabel=f'BadPoints({t}r)(d<1.0) per patch value',
                                                                       reference_label=ref_label))
            if is_main_process():
                save_hist_name = hist_name.replace('/', '_') + '.npy'
                self.save_hist(bp_pct_dl1_t_hist, save_hist_name)

    def evaluate(self):
        # calculate metrics for whole dataset
        if self.calculate_rmse:
            self.report_rmse()
        if self.calculate_rmse_dl1:
            self.report_rmse_dl1()
        if self.calculate_iou:
            self.report_iou()
        if self.calculate_ba:
            self.report_ba()
        if self.calculate_bad_points:
            self.report_bad_points()
        if self.calculate_bad_points_dl1:
            self.report_bad_points_dl1()

        # calculate metrics for subsets
        for key, op, value in self.mask_expressions:
            self.masks[key] = torch.cat(self.masks[key])
            synchronize()
            self.masks[key] = torch.cat(all_gather(self.masks[key]))
            if op == '=':
                mask = self.masks[key] == value
            elif op == '>':
                mask = self.masks[key] > value
            elif op == '>=':
                mask = self.masks[key] >= value
            else:
                raise ValueError

            subset_count = mask.sum().item()
            if subset_count >= 1:
                mask_suffix = f'_{key}{op}{value}'
                if self.calculate_rmse:
                    self.report_rmse(mask, mask_suffix)
                if self.calculate_rmse_dl1:
                    self.report_rmse_dl1(mask, mask_suffix)
                if self.calculate_iou:
                    self.report_iou(mask, mask_suffix)
                if self.calculate_ba:
                    self.report_ba(mask, mask_suffix)
                if self.calculate_bad_points:
                    self.report_bad_points(mask, mask_suffix)
                if self.calculate_bad_points_dl1:
                    self.report_bad_points_dl1(mask, mask_suffix)

        return {'scalars': self.scalars, 'images': self.images}

    def _plot_hist(self, hist: torch.Tensor, hist_name: str,
                   reference_hist_dir_path: str, xlabel: str, reference_label: str,
                   min=0.0, max=1.0):
        figure = plt.figure(figsize=(4.8, 3.6))
        prototype_hist = hist[0] if isinstance(hist, list) else hist
        bins = torch.linspace(min, max, len(prototype_hist) + 1)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        alpha = 1.0
        if reference_hist_dir_path is not None:
            search_hist_name = hist_name.replace('/', '_')
            path = os.path.join(reference_hist_dir_path, f'{search_hist_name}.npy')
            if os.path.exists(path):
                baseline_hist = np.load(path)
                hist = hist.cpu().numpy()
                if baseline_hist.sum() == hist.sum() and baseline_hist.shape == hist.shape:
                    alpha = 0.5
                    plt.bar(center, baseline_hist, align='center', label=reference_label, width=width, alpha=alpha)
                    plt.legend()
                else:
                    log.info(f"Reference {reference_label} histogram ({path}) shape does not match.")
            else:
                log.info(f"Can't find reference {reference_label} histogram: {path}")
        plt.bar(center, hist, align='center', width=width, alpha=alpha)
        plt.xlabel(xlabel)
        plt.ylabel('# of patches')
        plt.xlim((min, max))
        plt.yscale('log')
        plt.tight_layout()
        return figure

    def save_hist(self, hist, name):
        current_dir = os.getcwd()
        hist_dir = os.path.join(current_dir, 'hist')
        if not os.path.exists(hist_dir):
            os.mkdir(hist_dir)
        np.save(os.path.join(hist_dir, name), hist.cpu().numpy())
