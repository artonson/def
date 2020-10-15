import logging
import os
from typing import List

import k3d
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from ..utils.abc_utils.torch import image_to_points
from ..utils.comm import all_gather, synchronize, is_main_process

log = logging.getLogger(__name__)


class IllustratorPoints(DatasetEvaluator):

    def __init__(self, resolution, golden_set_item_ids,
                 bp_ts_per_resolution,
                 k, filter_expressions=None, depth2pointcloud=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = resolution
        self.bp_ts = bp_ts_per_resolution
        self.golden_set_item_ids = golden_set_item_ids if golden_set_item_ids is not None else []
        self.k = k
        self.filter_expressions = filter_expressions if filter_expressions is not None else []
        self.depth2pointcloud = depth2pointcloud
        self.reset()

    def reset(self):
        self.indexes = []
        self.rmses_dl1 = []
        self.bp_pct_dl1 = {t: [] for t in self.bp_ts}
        self.ious = []
        self.bas = []
        self.device = None

    def process(self, inputs, outputs):
        """
        Args:
            inputs (dict): the inputs to a model.
            outputs (dict): the outputs of a model.
        """
        self.device = inputs['index'].device

        # apply filters
        final_mask = torch.ones_like(inputs['index'], device=self.device, dtype=torch.bool)
        for key, op, value in self.filter_expressions:
            if op == '=':
                mask = inputs[key] == value
            elif op == '>':
                mask = inputs[key] > value
            elif op == '>=':
                mask = inputs[key] >= value
            else:
                raise ValueError
            # final_mask = torch.logical_and(final_mask, mask)
            final_mask = final_mask * mask

        if final_mask.sum() == 0:
            return

        inds = final_mask.nonzero(as_tuple=True)[0]
        item_ids = np.array(inputs['item_id'])[inds.cpu()]
        dataset_indexes = torch.index_select(inputs['index'], 0, inds)
        # inputs['distances']: (2, 4096),
        # points: torch.Size([2, 4096, 3])
        target = torch.index_select(inputs['distances'], 0, inds)
        preds = torch.index_select(outputs['distances'], 0, inds)
        points = torch.index_select(inputs['points'], 0, inds)

        # compute metrics per batch
        batch_size = target.size(0)
        batch_size, num_points = target.view(batch_size, -1).size()

        self.indexes.append(dataset_indexes.detach().cpu())
        target = target.view(batch_size, num_points)
        preds = preds.view(batch_size, num_points)
        target_sharp = (target < self.resolution).long()
        preds_sharp = (preds < self.resolution).long()

        # calculate IOU
        intersection = target_sharp * preds_sharp
        union = (target_sharp + preds_sharp - intersection).sum(dim=1)
        union_zero_mask = union == 0
        union[union_zero_mask] = 1.0
        intersection = intersection.sum(dim=1)
        self.ious.append(torch.where(union_zero_mask,
                                     torch.zeros_like(intersection, device=intersection.device, dtype=torch.float),
                                     intersection.float() / union.float()).detach().cpu())

        # calculate Balanced Accuracy
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

        # calculate RMSE and BadPoints(T) over the points with distance < 1.0 only
        for i in range(batch_size):
            mask_dl1 = target[i] < 1.0
            if torch.any(mask_dl1):
                squared_errors_dl1 = F.mse_loss(preds[i][mask_dl1], target[i][mask_dl1], reduction='none')
                self.rmses_dl1.append(torch.sqrt(squared_errors_dl1.mean()).view(1).detach().cpu())
                for t in self.bp_ts:
                    bad_points_mask = squared_errors_dl1 > (t * self.resolution) ** 2
                    self.bp_pct_dl1[t].append(
                        (bad_points_mask.float().sum() / num_points).view(1).detach().cpu())
            else:
                self.rmses_dl1.append(torch.tensor(0, dtype=torch.float).view(1))
                for t in self.bp_ts:
                    self.bp_pct_dl1[t].append(torch.tensor(0, dtype=torch.float).view(1))

        # plot golden set
        for i, item_id in enumerate(item_ids):
            if str(item_id) in self.golden_set_item_ids:
                # dirty hack; slicing because otherwise b'' gets into filename
                str_item_id = str(item_id)[2:-1]

                filename = f'datasetname={self.dataset_name}_golden_datasetidx={dataset_indexes[i]}_itemid={str_item_id}.html'

                # xyz correspond to first 3 channels
                self._illustrate_to_file(points[i].detach().cpu(),
                                         target[i].detach().cpu(),
                                         preds[i].detach().cpu(),
                                         F.l1_loss(preds[i], target[i], reduction='none').detach().cpu(),
                                         name=filename)

    def plot_k_elements(self, metrics: torch.Tensor, metric_name: str, descending=False):
        # metrics should be sorted s.t. the first values are best, the last values are worst

        if is_main_process():
            argsort_inds = torch.argsort(metrics, descending=descending)
            sorted_metrics = metrics[argsort_inds]
            sorted_indexes = self.indexes[argsort_inds]

            def plot(ks: List[int], name: str):
                k_i = 1
                already_plotted = []  # to avoid situation when indexes e.g. -1 and 0 represent the same element
                for k in ks:
                    try:
                        index = int(sorted_indexes[k])
                    except IndexError:
                        continue
                    if index in already_plotted:
                        continue
                    item = self.dataset[index]
                    metric_value = sorted_metrics[k].item()
                    pred = self.model(item['points'].unsqueeze(0).to(self.device))['distances'].detach().cpu().squeeze(
                        0)
                    l1_errors = F.l1_loss(pred, item['distances'], reduction='none')

                    # dirty hack; slicing because otherwise b'' gets into filename
                    str_item_id = str(item['item_id'])[2:-1]
                    filename = f'datasetname={self.dataset_name}_{metric_name}_{name}{k_i}_datasetidx={index}_metricvalue={metric_value}_itemid={str_item_id}.html'

                    # xyz correspond to first 3 channels
                    self._illustrate_to_file(item['points'], item['distances'], pred, l1_errors, name=filename)

                    k_i += 1

                    already_plotted.append(index)

            # plot k worst
            plot(list(range(-self.k, 0)), 'worst')

            # plot k best
            plot(list(range(0, self.k)), 'best')

            # plot k median
            median_idx = int(sorted_metrics.median(dim=0).indices)
            plot(list(range(median_idx - self.k // 2, median_idx + self.k // 2 + self.k % 2)), 'median')

            # plot q10
            q10_idx = int(0.1 * (sorted_metrics.size(0) - 1))
            plot(list(range(q10_idx - self.k // 2, q10_idx + self.k // 2 + self.k % 2)), 'q10best')

            # plot q90
            q90_idx = int(0.9 * (sorted_metrics.size(0) - 1))
            plot(list(range(q90_idx - self.k // 2, q90_idx + self.k // 2 + self.k % 2)), 'q90worst')

    def evaluate(self):
        num_elements = len(self.rmses_dl1)
        self.indexes = torch.cat(self.indexes) if num_elements > 0 else torch.rand(0)
        self.rmses_dl1 = torch.cat(self.rmses_dl1) if num_elements > 0 else torch.rand(0)
        self.ious = torch.cat(self.ious) if num_elements > 0 else torch.rand(0)
        self.bas = torch.cat(self.bas) if num_elements > 0 else torch.rand(0)
        for t in self.bp_ts:
            self.bp_pct_dl1[t] = torch.cat(self.bp_pct_dl1[t]) if num_elements > 0 else torch.rand(0)

        # gather results across gpus
        synchronize()
        self.indexes = torch.cat(all_gather(self.indexes))
        self.rmses_dl1 = torch.cat(all_gather(self.rmses_dl1))
        self.ious = torch.cat(all_gather(self.ious))
        self.bas = torch.cat(all_gather(self.bas))
        for t in self.bp_ts:
            self.bp_pct_dl1[t] = torch.cat(all_gather(self.bp_pct_dl1[t]))

        if num_elements == 0:
            return {'scalars': {}, 'images': {}}

        self.plot_k_elements(self.rmses_dl1, 'rmsedl1', descending=False)
        self.plot_k_elements(self.ious, 'iou', descending=True)
        self.plot_k_elements(self.bas, 'ba', descending=True)
        for t in self.bp_ts:
            self.plot_k_elements(self.bp_pct_dl1[t], f'bpr{t}rdl1', descending=False)

        return {'scalars': {}, 'images': {}}

    def _get_colors(self, pred, cmap):
        hex_colors = []
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        if len(pred.shape) > 1:
            pred_color = pred.reshape(-1)
        else:
            pred_color = pred
        for i in pred_color:
            hex_colors.append(int(mpl.colors.to_hex(m.to_rgba(i)).replace('#', '0x'), 16))
        hex_colors = np.array(hex_colors, 'uint32')
        return hex_colors

    def _illustrate_3d(self, xyz: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, errors: torch.Tensor):

        plot = k3d.plot(grid_visible=False, axes_helper=0)

        seg_target = (target < self.resolution).float()
        seg_pred = (pred < self.resolution).float()

        col_pred = self._get_colors(pred, cm.coolwarm_r)
        col_true = self._get_colors(target, cm.coolwarm_r)
        col_err = self._get_colors(errors, cm.jet)
        col_seg_pred = self._get_colors(seg_pred, cm.coolwarm)
        col_seg_true = self._get_colors(seg_target, cm.coolwarm)

        points_true = k3d.points(xyz, col_true, point_size=0.02, shader='mesh', name='ground truth')
        points_pred = k3d.points(xyz, col_pred, point_size=0.02, shader='mesh', name='prediction')
        points_err = k3d.points(xyz, col_err, point_size=0.02, shader='mesh', name='metric values')
        points_seg_true = k3d.points(xyz, col_seg_true, point_size=0.02, shader='mesh',
                                     name='segmentation ground truth')
        points_seg_pred = k3d.points(xyz, col_seg_pred, point_size=0.02, shader='mesh', name='segmentation prediction')
        colorbar1 = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                             color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.Jet)
        colorbar2 = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                             color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.coolwarm_r)

        plot += points_true + points_pred + points_err + points_seg_true + points_seg_pred + colorbar1 + colorbar2

        return plot

    def _illustrate_to_file(self, points: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, errors: torch.Tensor,
                            name: str):
        if is_main_process():
            assert name is not None

            if self.depth2pointcloud:
                xyz = image_to_points(points[0, :, :].numpy())
                target = target.view(-1)
                pred = pred.view(-1)
                errors = pred.view(-1)
            else:
                xyz = points[:, :3]

            plot_3d = self._illustrate_3d(xyz, target, pred, errors)

            current_dir = os.getcwd()
            visuals_dir = os.path.join(current_dir, 'visuals')
            if not os.path.exists(visuals_dir):
                os.mkdir(visuals_dir)
            with open(os.path.join(visuals_dir, name).lower(), 'w') as f:
                f.write(plot_3d.get_snapshot())
