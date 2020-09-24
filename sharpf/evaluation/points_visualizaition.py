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

    def __init__(self, golden_set_item_ids, k, filter_expressions=None, depth2pointcloud=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.golden_set_item_ids = golden_set_item_ids if golden_set_item_ids is not None else []
        self.k = k
        self.filter_expressions = filter_expressions if filter_expressions is not None else []
        self.depth2pointcloud = depth2pointcloud
        self.reset()

    def reset(self):
        self.rmses = []
        self.indexes = []
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
        preds = torch.index_select(outputs['pred_distances'], 0, inds)
        points = torch.index_select(inputs['points'], 0, inds)

        # print(item_ids.shape, dataset_indexes.shape, target.shape, preds.shape, points.shape)

        # compute metrics per batch
        batch_size = target.size(0)
        mean_squared_errors = F.mse_loss(preds, target, reduction='none').view(batch_size, -1).mean(dim=1)  # (batch)
        root_mean_squared_errors = torch.sqrt(mean_squared_errors)
        self.rmses.append(root_mean_squared_errors.detach().cpu())
        self.indexes.append(dataset_indexes.detach().cpu())

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

    def evaluate(self):
        self.rmses = torch.cat(self.rmses) if len(self.rmses) > 0 else torch.rand(0)
        self.indexes = torch.cat(self.indexes) if len(self.indexes) > 0 else torch.rand(0)
        assert self.rmses.size() == self.indexes.size()

        # gather results across gpus
        synchronize()
        rmses = torch.cat(all_gather(self.rmses))
        indexes = torch.cat(all_gather(self.indexes))

        if rmses.size(0) == 0:
            return {'scalars': {}, 'images': {}}

        if is_main_process():
            argsort_inds = torch.argsort(rmses)
            rmses = rmses[argsort_inds]
            indexes = indexes[argsort_inds]

            def plot(ks: List[int], name: str):
                k_i = 1
                already_plotted = []  # to avoid situation when indexes -1 and 0 represent the same element
                for k in ks:
                    try:
                        index = int(indexes[k])
                    except IndexError:
                        continue
                    if index in already_plotted:
                        continue
                    item = self.dataset[index]
                    rmse = rmses[k].item()
                    pred = self.model(item['points'].unsqueeze(0).to(self.device)).detach().cpu().squeeze(0)
                    l1_errors = F.l1_loss(pred, item['distances'], reduction='none')

                    # dirty hack; slicing because otherwise b'' gets into filename
                    str_item_id = str(item['item_id'])[2:-1]
                    filename = f'datasetname={self.dataset_name}_{name}{k_i}_datasetidx={index}_itemid={str_item_id}_rmse={rmse}.html'

                    # xyz correspond to first 3 channels
                    self._illustrate_to_file(item['points'], item['distances'], pred, l1_errors, name=filename)

                    k_i += 1

                    already_plotted.append(index)

            # plot k worst
            plot(list(range(-self.k, 0)), 'worst')

            # plot k best
            plot(list(range(0, self.k)), 'best')

            # plot k median
            median_idx = int(rmses.median(dim=0).indices)
            plot(list(range(median_idx - self.k // 2, median_idx + self.k // 2 + self.k % 2)), 'median')

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

        col_pred = self._get_colors(pred, cm.coolwarm_r)
        col_true = self._get_colors(target, cm.coolwarm_r)
        col_err = self._get_colors(errors, cm.jet)

        points_true = k3d.points(xyz, col_true, point_size=0.02, shader='mesh', name='ground truth')
        points_pred = k3d.points(xyz, col_pred, point_size=0.02, shader='mesh', name='prediction')
        points_err = k3d.points(xyz, col_err, point_size=0.02, shader='mesh', name='metric values')
        colorbar1 = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                             color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.Jet)
        colorbar2 = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                             color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.coolwarm_r)

        plot += points_true + points_pred + points_err + colorbar1 + colorbar2

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
