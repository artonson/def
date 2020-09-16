import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

import k3d
import matplotlib as mpl
import matplotlib.cm as cm

from .evaluator import DatasetEvaluator
from ..utils.comm import all_gather, synchronize, is_main_process

__all__ = ['IllustratorPoints']


def get_colors(pred, cmap):
    hex_colors = []
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = cmap
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    if len(pred.shape) > 1:
        pred_color = pred.reshape(-1)
    else:
        pred_color = pred
    for i in pred_color:
        hex_colors.append(int(mpl.colors.to_hex(m.to_rgba(i)).replace('#', '0x'), 16))
    hex_colors = np.array(hex_colors, 'uint32')
    return hex_colors


class IllustratorPoints(DatasetEvaluator):

    def __init__(self, golden_set_ids, k=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.golden_set_ids = golden_set_ids
        self.k = k
        self.reset()

    def reset(self):
        self.metric = []
        self.relative_golden_ids = []

    def _illustrate_3d(self, data, target, pred, metric):

        plot = k3d.plot(grid_visible=False, axes_helper=0)

        if type(pred) is np.ndarray:
            col_pred = get_colors(pred, cm.coolwarm_r)
        else:
            col_pred = get_colors(pred.cpu().numpy(), cm.coolwarm_r)
        if type(target) is np.ndarray:
            col_true = get_colors(target, cm.coolwarm_r)
        else:
            col_true = get_colors(target.cpu().numpy(), cm.coolwarm_r)
        if type(metric) is np.ndarray:
            col_err = get_colors(metric, cm.jet)
        else:
            col_err = get_colors(metric.cpu().numpy(), cm.jet)

        if type(data) is np.ndarray:
            data_numpy = data
        else:
            data_numpy = data.cpu().numpy()
        points_true = k3d.points(data_numpy, col_true, point_size=0.02, shader='mesh', name='ground truth')
        points_pred = k3d.points(data_numpy, col_pred, point_size=0.02, shader='mesh', name='prediction')
        points_err = k3d.points(data_numpy, col_err, point_size=0.02, shader='mesh', name='metric values')
        colorbar1 = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                             color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.Jet)
        colorbar2 = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                             color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.coolwarm_r)

        plot += points_true + points_pred + points_err + colorbar1 + colorbar2

        return plot

    def illustrate_to_file(self, data, target, pred, metric, name=None):
        if name is None:
            name = 'illustrate-points'
        dr = os.getcwd()
        plot_3d = self._illustrate_3d(data, target, pred, metric)

        if not os.path.exists(f'{dr}/visuals'):
            os.mkdir(f'{dr}/visuals')
        with open(f'{dr}/visuals/{name}.html', 'w') as f:
            f.write(plot_3d.get_snapshot())

    def process(self, inputs, outputs):
        """
        Args:
            inputs (dict): the inputs to a model.
            outputs (dict): the outputs of a model.
        """
        target = inputs['distances']
        preds = outputs['pred_distances']
        item_ids = inputs['item_id']
        dataset_ind = inputs['index']

        mean_squared_metrics = F.mse_loss(preds, target, reduction='none').reshape(target.shape)
        root_mean_squared_metrics = torch.sqrt(mean_squared_metrics)

        self.metric.append(root_mean_squared_metrics)  # rmse for points
        tmp = [int(dataset_ind[i]) for i, elem in enumerate(item_ids) if str(elem) in self.golden_set_ids]
        if len(tmp) != 0:
            self.relative_golden_ids.append(torch.LongTensor(tmp))

    def evaluate(self):
        # gather results across gpus
        log.info('evaluate visualization')
        synchronize()
        metrics = []
        for elem in all_gather(self.metric):
            for e in elem:
                metrics.append(e)
        metrics = torch.cat(metrics)

        if len(self.relative_golden_ids) == 0:
            return {'scalars': {}, 'images': {}}
        relative_golden_ids = []
        for tmp in all_gather(self.relative_golden_ids):
            relative_golden_ids.append(torch.LongTensor(torch.cat(tmp)))
        relative_golden_ids = torch.LongTensor(torch.cat(relative_golden_ids))

        metrics_golden = metrics[relative_golden_ids].cpu().numpy()
        # calculate quantiles
        k_best_idx = np.argsort(torch.sqrt(metrics.mean(dim=1)).cpu().numpy(), axis=0)[::-1][:self.k]
        k_worst_idx = np.argsort(torch.sqrt(metrics.mean(dim=1)).cpu().numpy(), axis=0)[:self.k]

        log.info('get input data')

        if is_main_process():
            log.info('saving visualization')
            for i in relative_golden_ids:
                item_golden = self.dataset[i]
                pred_golden = self.model(item_golden['points'].unsqueeze(0).to('cuda'))  # predict with model
                self.illustrate_to_file(item_golden['points'], item_golden['distances'], pred_golden, metrics_golden[i],
                                        name=f'illustration-points_golden-set_{i}')

            for i in range(len(k_best_idx)):
                item_best = self.dataset[k_best_idx[i]]
                item_worst = self.dataset[k_worst_idx[i]]
                pred_best = self.model(item_best['points'].unsqueeze(0).to('cuda'))
                self.illustrate_to_file(item_best['points'], item_best['distances'], pred_best, metrics[k_best_idx[i]],
                                        name=f'illustration-points_best-metric_{i}')
                pred_worst = self.model(item_worst['points'].unsqueeze(0).to('cuda'))
                self.illustrate_to_file(item_worst['points'], item_worst['distances'], pred_worst,
                                        metrics[k_worst_idx[i]],
                                        name=f'illustration-points_worst-metric_{i}')
        log.info('!saved visualization')
        return {'scalars': {}, 'images': {}}
