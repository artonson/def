import os
import torch
import msgpack
import logging
import numpy as np
import torch.nn.functional as F

log = logging.getLogger(__name__)

import k3d
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.pyplot as plt
from itertools import product

import trimesh.transformations as tt
from .evaluator import DatasetEvaluator
from ..utils.comm import all_gather, synchronize, is_main_process

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

def convert_dist(distance, m):
    rgba_dist = np.zeros((64, 64, 4))
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            rgba_dist[i, j] = m.to_rgba(distance[i, j])
    return rgba_dist

def image_to_points(image, rays_origins):
    i = np.where(image.ravel() != 0)[0]
    points = np.zeros((len(i), 3))
    points[:, 0] = rays_origins[i, 0]
    points[:, 1] = rays_origins[i, 1]
    points[:, 2] = image.ravel()[i]
    return points, i

def rotate_to_world_origin(camera_origin):
    # construct a 3x3 rotation matrix to a coordinate frame where:
    # Z axis points to world origin aka center of a mesh
    # Y axis points down
    # X axis is computed as Y cross Z

    camera_origin = np.asanyarray(camera_origin)

    e_z = -camera_origin / np.linalg.norm(camera_origin)  # Z axis points to world origin aka center of a mesh
    e_y = np.array([0, 0, -1])  # proxy to Y axis pointing directly down
    # note that real e_y must be
    # 1) orthogonal to e_z;
    # 2) lie in the plane spanned by e_y and e_z;
    # 3) point downwards so <e_y, [0, 0, -1]> >= <e_y, [0, 0, +1]>
    # 4) unit norm
    gamma = np.dot(e_y, e_z)
    e_y = -gamma / (1 + gamma ** 2) * e_z + 1. / (1 + gamma ** 2) * e_y
    if np.dot(e_y, [0, 0, -1]) < np.dot(e_y, [0, 0, 1]):
        e_y *= -1
    e_y /= np.linalg.norm(e_y)
    e_x = np.cross(e_y, e_z)  # X axis
    R = np.array([e_x, e_y, e_z])
    return R

def create_rotation_matrix_z(rz):
    return np.array([[np.cos(rz), -np.sin(rz), 0.0],
                     [np.sin(rz), np.cos(rz), 0.0],
                     [0.0, 0.0, 1.0]])

def camera_to_display(image):
    return image[::-1, ::-1].T


class CameraPose:
    def __init__(self, transform):
        self._camera_to_world_4x4 = transform
        # always store transform from world to camera frame
        self._world_to_camera_4x4 = np.linalg.inv(self._camera_to_world_4x4)

    @classmethod
    def from_camera_to_world(cls, rotation=None, translation=None):
        """Create camera pose from camera to world transform.
        :param rotation: 3x3 rotation matrix of camera frame axes in world frame
        :param translation: 3d location of camera frame origin in world frame
        """
        rotation = np.identity(3) if None is rotation else np.asanyarray(rotation)
        translation = np.zeros(3) if None is translation else np.asanyarray(translation)

        transform = np.identity(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation

        return cls(transform)

    @classmethod
    def from_camera_axes(cls, R=None, t=None):
        """Compute 4x4 camera pose from camera axes given in world frame.
        :param R: a list of 3D basis vectors (cx, cy, cz) defined in world frame
        :param t: 3D vector defining location of camera origin in world frame
        """
        if None is R:
            R = np.identity(3)

        return cls.from_camera_to_world(rotation=R.T, translation=t)

    def world_to_camera(self, points):
        """Transform points from world to camera coordinates.
        Useful for understanding where the objects are, as seen by the camera.
        :param points: either n * 3 array, or a single 3-vector
        """
        points = np.atleast_2d(points)
        return tt.transform_points(points, self._world_to_camera_4x4)

    def camera_to_world(self, points, translate=True):
        """Transform points from camera to world coordinates.
        Useful for understanding where objects bound to camera
        (e.g., image pixels) are in the world.
        :param points: either n * 3 array, or a single 3-vector
        :param translate: if True, also translate the points
        """
        points = np.atleast_2d(points)
        return tt.transform_points(points, self._camera_to_world_4x4, translate=translate)

    @property
    def world_to_camera_4x4(self):
        return self._world_to_camera_4x4

    @property
    def camera_to_world_4x4(self):
        return self._camera_to_world_4x4

    @property
    def frame_origin(self):
        """Return camera frame origin in world coordinates."""
        return self.camera_to_world_4x4[:3, 3]

    @property
    def frame_axes(self):
        """Return camera axes: a list of 3D basis
        vectors (cx, cy, cz) defined in world frame"""
        return self.camera_to_world_4x4[:3, :3].T

    def compose_world_to_camera(self, other_pose):
        """Compose camera poses C_1, C_2, ... (defined relative to each other),
        computing transforms from world frame to an innermost camera frame.
        Equivalent to:
        x_world = <some point>
        other_pose.world_to_camera(
            pose.world_to_camera(
                x_world
            )
        )
        """
        composed_world_to_camera_4x4 = np.dot(other_pose.world_to_camera_4x4, self._world_to_camera_4x4)
        composed_camera_to_world_4x4 = np.linalg.inv(composed_world_to_camera_4x4)
        return CameraPose(composed_camera_to_world_4x4)

    def compose_camera_to_world(self, other_pose):
        """Compose camera poses C_1, C_2, ... (defined relative to each other),
        computing transforms from innermost camera frame to the world frame.
        Equivalent to:
        x_local = <some point>
        pose.camera_to_world(
            pose_local.camera_to_world(
                x_local
            )
        )
        """
        composed_camera_to_world_4x4 = np.dot(self._camera_to_world_4x4, other_pose.camera_to_world_4x4, )
        return CameraPose(composed_camera_to_world_4x4)


class IllustratorPoints(DatasetEvaluator):

    def __init__(self, golden_set_ids, k=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.golden_set_ids = golden_set_ids
        self.k = k
        self.reset()

    def reset(self):
        self.error = []
        self.relative_golden_ids = []

    def _illustrate_3d(self, data, target, pred, error):

        plot = k3d.plot(grid_visible=False, axes_helper=0)

        if type(pred) is np.ndarray:
            col_pred = get_colors(pred, cm.coolwarm_r)
        else:
            col_pred = get_colors(pred.cpu().numpy(), cm.coolwarm_r)
        if type(target) is np.ndarray:
            col_true = get_colors(target, cm.coolwarm_r)
        else:
            col_true = get_colors(target.cpu().numpy(), cm.coolwarm_r)
        if type(error) is np.ndarray:
            col_err = get_colors(error, cm.jet)
        else:
            col_err = get_colors(error.cpu().numpy(), cm.jet)

        if type(data) is np.ndarray:
            data_numpy = data
        else:
            data_numpy = data.cpu().numpy()
        points_true = k3d.points(data_numpy, col_true, point_size=0.02, shader='mesh', name='ground truth')
        points_pred = k3d.points(data_numpy, col_pred, point_size=0.02, shader='mesh', name='prediction')
        points_err = k3d.points(data_numpy, col_err, point_size=0.02, shader='mesh', name='error values')
        colorbar1 = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                             color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.Jet)
        colorbar2 = k3d.line([[], []], shader="mesh",
                             color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.coolwarm_r)

        plot += points_true + points_pred + points_err + colorbar1 + colorbar2

        return plot

    def illustrate_to_file(self, data, target, pred, error, name=None):
        if name is None:
            name = 'illustrate-points'
        dr = os.getcwd()
        log.info('current dir ' + str(dr))
        plot_3d = self._illustrate_3d(data, target, pred, error)

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

        mean_squared_errors = F.mse_loss(preds, target, reduction='none').reshape(target.shape)
        root_mean_squared_errors = torch.sqrt(mean_squared_errors)

        self.error.append(root_mean_squared_errors) # rmse for points
        tmp = [int(dataset_ind[i]) for i, elem in enumerate(item_ids) if str(elem) in self.golden_set_ids]
        if len(tmp) != 0:
            self.relative_golden_ids.append(torch.LongTensor(tmp))

    def evaluate(self):
        # gather results across gpus
        log.info('evaluate visualization')
        synchronize()
        errors = []
        for elem in all_gather(self.error):
            for e in elem:
                errors.append(e)
        errors = torch.cat(errors)

        if len(self.relative_golden_ids) == 0:
            return {'scalars': {}, 'images': {}}
        relative_golden_ids = []
        for tmp in all_gather(self.relative_golden_ids):
            relative_golden_ids.append(torch.LongTensor(torch.cat(tmp)))
        relative_golden_ids = torch.LongTensor(torch.cat(relative_golden_ids))

        errors_golden = errors[relative_golden_ids].cpu().numpy()
        # calculate quantiles
        errors_k_best = np.argsort(torch.sqrt(errors.mean(dim=1)).cpu().numpy(), axis=0)[::-1][:self.k]
        log.info('errors k best ' + str(errors_k_best))
        errors_k_worst = np.argsort(torch.sqrt(errors.mean(dim=1)).cpu().numpy(), axis=0)[:self.k]

        log.info('get input data')

        if is_main_process():
            log.info('saving visualization')
            for i in relative_golden_ids:
                item_golden = self.dataset[i]
                log.info('points shape ' + str(item_golden['points'].shape))
                pred_golden = self.model(item_golden['points'].unsqueeze(0).to('cuda'))  # predict with model
                self.illustrate_to_file(item_golden['points'], item_golden['distances'], pred_golden, errors_golden[i],
                                   name=f'illustration-points_golden-set_{i}')

            for i in range(len(errors_k_best)):
                item_best = self.dataset[errors_k_best[i]]
                item_worst = self.dataset[errors_k_worst[i]]
                pred_best = self.model(item_best['points'].unsqueeze(0).to('cuda'))
                self.illustrate_to_file(item_best['points'], item_best['distances'], pred_best, errors[errors_k_best[i]],
                                   name=f'illustration-points_best-metric_{i}')
                pred_worst = self.model(item_worst['points'].unsqueeze(0).to('cuda'))
                self.illustrate_to_file(item_worst['points'], item_worst['distances'], pred_worst, errors[errors_k_worst[i]],
                                   name=f'illustration-points_worst-metric_{i}')
        log.info('!saved visualization')
        return {'scalars':{}, 'images':{}}

class IllustratorDepths(DatasetEvaluator):

    def __init__(self, dataset, task, golden_set_ids, k=5, dataset_name=""):
        super().__init__(*args, **kwargs)
        self.golden_set_ids = golden_set_ids
        self.k = k
        self.reset()

    def reset(self):
        self.error = []
        self.relative_golden_ids = []
        self.cameras = []

    def _get_data_3d(self, data):
        image_height, image_width = data.shape[1], data.shape[2]
        self.log.info(str(image_height))
        self.log.info(str(image_width))
        resolution_3d = 0.02
        screen_aspect_ratio = 1

        rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(
            2, image_height * image_width).T

        rays_origins = (rays_screen_coords / np.array([[image_height, image_width]]))  # [h, w, 2], in [0, 1]
        factor = image_height / 2 * resolution_3d
        rays_origins[:, 0] = (-2 * rays_origins[:, 0] + 1) * factor  # to [-1, 1] + aspect transform
        rays_origins[:, 1] = (-2 * rays_origins[:, 1] + 1) * factor * screen_aspect_ratio
        rays_origins = np.concatenate([
            rays_origins,
            np.zeros_like(rays_origins[:, [0]])
        ], axis=1)

        data_3d, non_zero_idx = image_to_points(data, rays_origins)

        return data_3d, non_zero_idx

    def _illustrate_3d(self, data, target, pred, error, camera_pose):

        plot = k3d.plot(grid_visible=False, axes_helper=0)

        col_pred = get_colors(pred, cm.coolwarm_r)
        col_true = get_colors(target, cm.coolwarm_r)
        col_err = get_colors(error, cm.jet)

        data_world = camera_pose.camera_to_world(data, translate=False)
        points_true = k3d.points(data_world - np.mean(data_world, axis=0),
                                 col_true, point_size=0.02, shader='mesh', name='ground truth')
        points_pred = k3d.points(data_world - np.mean(data_world, axis=0),
                                 col_pred, point_size=0.02, shader='mesh', name='prediction')
        points_err = k3d.points(data_world - np.mean(data_world, axis=0),
                                col_err, point_size=0.02, shader='mesh', name='error values')
        colorbar = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                            color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.Jet)

        plot += points_true + points_pred + points_err + colorbar

        return plot

    def _illustrate_2d(self, data, target, pred, error):
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

        norm_error = mpl.colors.Normalize(vmin=0, vmax=error.max())
        cmap_error = plt.get_cmap('viridis')
        m_error = cm.ScalarMappable(norm=norm_error, cmap=cmap_error)
        ax_error = grid[3].imshow(convert_dist(error.cpu().numpy(), m_error), cmap=cmap_error, vmin='0.0', vmax=error.max())
        cbar = fig.colorbar(ax_error, cax=grid.cbar_axes[3], norm=norm_error)
        cbar.ax.tick_params(labelsize=8)
        cbar.locator = ticker.MaxNLocator(6)
        cbar.update_ticks()
        grid[3].set_title('error per pix', fontsize=8.5)

        return fig

    def illustrate_to_file(self, data, target, pred, error, camera, name=None):
        if name is None:
            name = f'illustration-depths_task-{self.task}'
        dr = os.getcwd()
        if not os.path.exists(f'{dr}/visuals'):
            os.mkdir(f'{dr}/visuals')
        plot_2d = self._illustrate_2d(data[0], target[0], pred[0], error[0])
        plot_2d.savefig(f'{dr}/visuals/{name}.png')

        camera_pose = CameraPose(camera.cpu().numpy())
        data_3d, non_zero_idx = self._get_data_3d(data[sample].cpu().numpy())
        preds_numpy = pred[0].cpu().numpy()
        target_numpy = target[0].cpu().numpy()
        errors_numpy = error[0].cpu().numpy()

        plot_3d = self._illustrate_3d(data_3d,
                                      target_numpy.ravel()[non_zero_idx],
                                      preds_numpy.ravel()[non_zero_idx],
                                      errors_numpy.ravel()[non_zero_idx],
                                      camera_pose)

        with open(f'{dr}/visuals/{name}.html', 'w') as f:
            f.write(plot_3d.get_snapshot())

    def process(self, inputs, outputs):
        """
        Args:
            inputs (dict): the inputs to a model.
            outputs (dict): the outputs of a model.
        """
        target = inputs[self.input_key]
        preds = outputs[self.output_key]
        item_ids = inputs['item_id']
        dataset_ind = inputs['index']

        batch_size = target.size(0)
        if self.task == 'segmentation':
            error = abs(target - preds) # difference error
        elif self.task == 'regression':
            mean_squared_errors = F.mse_loss(preds, target, reduction='none').view(batch_size, -1).mean(dim=1)  # (batch)
            error = torch.sqrt(mean_squared_errors)  # (batch)

        self.error.append(error)
        self.relative_golden_ids.append([dataset_ind[i] for i, elem in enumerate(item_ids) if elem in self.golden_set_ids])
        self.cameras.append(inputs['camera_pose'])

    def evaluate(self):
        # gather results across gpus
        synchronize()
        errors = torch.cat(all_gather(self.error))
        cameras = torch.cat(all_gather(self.cameras))

        if len(self.relative_golden_ids) == 0:
            log.info('len(relative_golden_ids) == 0')
            return {'scalars': {}, 'images': {}}
        relative_golden_ids = []
        for tmp in all_gather(self.relative_golden_ids):
            relative_golden_ids.append(torch.LongTensor(torch.cat(tmp)))
        relative_golden_ids = torch.LongTensor(torch.cat(relative_golden_ids))

        errors_golden = errors[relative_golden_ids].cpu().numpy()
        # calculate quantiles
        errors_k_best = np.argsort(errors.mean(dim=1).cpu().numpy(), axis=0)[::-1][:self.k]
        log.info('errors k best ' + str(errors_k_best))
        errors_k_worst = np.argsort(errors.mean(dim=1).cpu().numpy(), axis=0)[:self.k]

        # get input data
        items_golden, cameras_golden = self.dataset[relative_golden_ids], cameras[relative_golden_ids]
        items_best, cameras_best = self.dataset[errors_k_best], cameras[errors_k_best]
        items_worst, cameras_worst = self.dataset[errors_k_worst], cameras[errors_k_worst]

        if is_main_process():
            for i in range(len(self.golden_set_ids)):
                preds_golden = self.model(items_golden[i]['image'])  # predict with model
                illustrate_to_file(items_golden[i]['image'], items_golden[i]['distances'], preds_golden, errors_golden[i], cameras_golden[i],
                                   name=f'illustration-depths_task-{self.task}_golden-set_{i}')

            for i in range(self.k):
                preds_best = self.model(items_best[i]['image'])
                illustrate_to_file(items_best[i]['image'], items_best[i]['distances'], preds_best, errors_k_best[i], cameras_best[i],
                                   name=f'illustration-depths_task-{self.task}_best-metric_{i}')
                preds_worst = self.model(items_worst[i]['image'])
                illustrate_to_file(items_worst[i]['image'], items_worst[i]['distances'], preds_worst, errors_k_worst[i], cameras_worst[i],
                                   name=f'illustration-depths_task-{self.task}_worst-metric_{i}')

        return {'scalars': {}, 'images': {}}