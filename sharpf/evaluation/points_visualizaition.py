import logging
import os

import k3d
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from ..utils.comm import all_gather, synchronize, is_main_process

log = logging.getLogger(__name__)


class IllustratorPoints(DatasetEvaluator):

    def __init__(self, golden_set_item_ids, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.golden_set_item_ids = golden_set_item_ids if golden_set_item_ids is not None else []
        self.k = k
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
        target = inputs['distances']
        preds = outputs['pred_distances']
        item_ids = inputs['item_id']
        dataset_indexes = inputs['index']
        self.device = preds.device

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
                self._illustrate_to_file(inputs['points'][i][:, :3].detach().cpu(),
                                         target[i].detach().cpu(),
                                         preds[i].detach().cpu(),
                                         F.l1_loss(preds[i], target[i], reduction='none').detach().cpu(),
                                         name=filename)

    def evaluate(self):

        self.rmses = torch.cat(self.rmses)
        self.indexes = torch.cat(self.indexes)
        assert self.rmses.size() == self.indexes.size()

        # gather results across gpus
        synchronize()
        rmses = torch.cat(all_gather(self.rmses))
        indexes = torch.cat(all_gather(self.indexes))

        if is_main_process():

            def plot_top_k(type: str):
                assert type in ['best', 'worst']
                rmse_top_k_idxs = indexes[
                    torch.topk(rmses, min(self.k, rmses.size(0)), largest=type == 'worst').indices]
                for k_i, index in enumerate(rmse_top_k_idxs):
                    item = self.dataset[index]
                    pred = self.model(item['points'].unsqueeze(0).to(self.device)).detach().cpu().squeeze(0)
                    l1_errors = F.l1_loss(pred, item['distances'], reduction='none')
                    str_item_id = str(item['item_id'])[
                                  2:-1]  # dirty hack; slicing because otherwise b'' gets into filename
                    filename = f'datasetname={self.dataset_name}_{type}{k_i + 1}_datasetidx={index}_itemid={str_item_id}.html'

                    # xyz correspond to first 3 channels
                    self._illustrate_to_file(item['points'][:, :3], item['distances'], pred, l1_errors, name=filename)

            plot_top_k('best')
            plot_top_k('worst')

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

    def _illustrate_to_file(self, xyz: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, errors: torch.Tensor,
                            name: str):
        assert name is not None

        plot_3d = self._illustrate_3d(xyz, target, pred, errors)

        current_dir = os.getcwd()
        visuals_dir = os.path.join(current_dir, 'visuals')
        if not os.path.exists(visuals_dir):
            os.mkdir(visuals_dir)
        with open(os.path.join(visuals_dir, name), 'w') as f:
            f.write(plot_3d.get_snapshot())
