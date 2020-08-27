import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import AxesGrid
import k3d
from itertools import product
import torch
import numpy as np
import msgpack

# TODO: add saving data dir

def get_colors(pred, cmap):
    hex_colors = []
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = cmap
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    for i in pred:
        hex_colors.append(int(mpl.colors.to_hex(m.to_rgba(i)).replace('#', '0x'), 16))
    hex_colors = np.array(hex_colors, 'uint32')
    return hex_colors

def convert_dist(distance, m):
    rgba_dist = np.zeros((64, 64, 4))
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            rgba_dist[i, j] = m.to_rgba(distance[i, j])
    return rgba_dist


class Illustrator:

    def __init__(self, task):
        self.task = task

    def illustrate_to_file(self, batch_idx, data, preds, targets, metrics, name=None):
        pass

    def _illustrate_3d(self, data, pred, target, metric):

        plot = k3d.plot(grid_visible=False, axes_helper=0)

        col_pred = get_colors(pred.cpu().numpy(), cm.coolwarm_r)
        col_true = get_colors(target.cpu().numpy(), cm.coolwarm_r)
        col_err = get_colors(metric.cpu().numpy(), cm.jet)

        points_true = k3d.points(data, col_true, point_size=0.1, shader='mesh')
        points_pred = k3d.points(data, col_pred, point_size=0.1, shader='mesh')
        points_err = k3d.points(data, col_err, point_size=0.1, shader='mesh')
        colorbar = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                            color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.Jet)

        plot += points_true + points_pred + points_err + colorbar

        return plot


class IllustratorPoints(Illustrator):

    def illustrate_to_file(self, batch_idx, data, preds, targets, metrics, name=None):

        for sample in range(len(preds.size(0))):
            if name is None:
                self.name = f'illustration-points_batch-{batch_idx}_idx-{sample}'
            else:
                self.name = name

            plot_3d = self._illustrate_3d(data[sample], preds[sample], targets[sample], metrics[sample])
            with open(f'experiments/{self.name}.html', 'w') as f:
                f.write(plot_3d.get_snapshot())


class IllustratorDepths(Illustrator):

    def _get_data_3d(self, data):
        depth_pc = np.hstack((np.array(list(product(np.arange(data[0].shape[1]),
                                                    np.arange(data[0].shape[0])))),
                              data.reshape(-1, 1)))
        depth_pc_tensor = torch.Tensor(depth_pc)
        depth_pc_tensor -= depth_pc_tensor.mean(dim=1, keepdim=True)
        points_scales = depth_pc_tensor.norm(dim=1).max(dim=0).values
        depth_pc_tensor /= points_scales

        return np.array(depth_pc_tensor)

    def _illustrate_2d(self, data, pred, target, metric):
        fig = plt.figure(figsize=(10, 10), dpi=200)
        grid = AxesGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(1, 4),
                        axes_pad=0.7,  # pad between axes in inch.
                        label_mode="1",
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="each",
                        cbar_size="11%",
                        cbar_pad="7%",
                        )

        norm_dist = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap_dist = plt.get_cmap('coolwarm_r')
        m_dist = cm.ScalarMappable(norm=norm_dist, cmap=cmap_dist)

        ax_data = grid[0].imshow(data.cpu().numpy())
        cbar = fig.colorbar(ax_data, cax=grid.cbar_axes[0])
        cbar.set_ticks((data.min(), data.max()))
        cbar.ax.tick_params(labelsize=8)
        cbar.locator = ticker.MaxNLocator(6)
        cbar.update_ticks()
        grid[0].set_title('Depth', fontsize=10)

        if self.task == 'regression':
            ax_tg = grid[1].imshow(convert_dist(target.cpu().numpy(), m_dist), cmap=cmap_dist, vmin=0.0, vmax=1.0)
        elif self.task == 'segmentation':
            ax_tg = grid[1].imshow(target.cpu().numpy())
        else:
            # raise error
            ax_tg = None

        cbar = fig.colorbar(ax_tg, cax=grid.cbar_axes[1], norm=norm_dist)
        cbar.ax.tick_params(labelsize=8)
        cbar.locator = ticker.MaxNLocator(5)
        cbar.update_ticks()
        grid[1].set_title('GT', fontsize=10)

        if self.task == 'regression':
            ax_pred = grid[2].imshow(convert_dist(pred.cpu().numpy(), m_dist), cmap=cmap_dist, vmin=0.0, vmax=1.0)
        elif self.task == 'segmentation':
            ax_pred = grid[2].imshow(pred.cpu().numpy(), vmin=0.0, vmax=1.0)
        else:
            # raise error
            ax_pred = None

        cbar = fig.colorbar(ax_pred, cax=grid.cbar_axes[2], norm=norm_dist)
        cbar.ax.tick_params(labelsize=8)
        cbar.locator = ticker.MaxNLocator(5)
        cbar.update_ticks()
        grid[2].set_title('Prediction', fontsize=10)

        norm_metric = mpl.colors.Normalize(vmin=0, vmax=metric.max())
        cmap_metric = plt.get_cmap('viridis')
        m_metric = cm.ScalarMappable(norm=norm_metric, cmap=cmap_metric)
        ax_metric = grid[3].imshow(convert_dist(metric.cpu().numpy(), m_metric), cmap=cmap_metric, vmin='0.0', vmax=metric.max())
        cbar = fig.colorbar(ax_metric, cax=grid.cbar_axes[3], norm=norm_metric)
        cbar.ax.tick_params(labelsize=8)
        cbar.locator = ticker.MaxNLocator(6)
        cbar.update_ticks()
        grid[3].set_title('Metric per pix', fontsize=8.5)

        return fig

    def illustrate_to_file(self, batch_idx, data, preds, targets, metrics, name=None):

        for sample in range(int(preds.size(0))):

            if name is None:
                self.name = f'illustration-depths_task-{self.task}_batch-{batch_idx}_idx-{sample}'
            else:
                self.name = name

            plot_2d = self._illustrate_2d(data[sample][0], preds[sample][0], targets[sample][0], metrics[sample][0])
            plot_2d.savefig(f'./{self.name}.png')

            data_3d = self._get_data_3d(data[sample].cpu().numpy())
            plot_3d = self._illustrate_3d(data_3d, preds[sample].reshape(-1), targets[sample].reshape(-1), metrics[sample].reshape(-1))
            with open(f'./{self.name}.html', 'w') as f:
                f.write(plot_3d.get_snapshot())
