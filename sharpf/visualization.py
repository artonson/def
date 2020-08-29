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
import trimesh.transformations as tt

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

def image_to_points(image, rays_origins):
    i = np.where(image.ravel() != 0)[0]
    points = np.zeros((len(i), 3))
    points[:, 0] = rays_origins[i, 0]
    points[:, 1] = rays_origins[i, 1]
    points[:, 2] = image.ravel()[i]
    return points

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


class IllustratorPoints:

    def __init__(self, log):
        self.log = log

    def _illustrate_3d(self, data, pred, target, metric):

        plot = k3d.plot(grid_visible=False, axes_helper=0)

        col_pred = get_colors(pred.cpu().numpy(), cm.coolwarm_r)
        col_true = get_colors(target.cpu().numpy(), cm.coolwarm_r)
        col_err = get_colors(metric.cpu().numpy(), cm.jet)

        points_true = k3d.points(data, col_true, point_size=0.02, shader='mesh', name='ground truth')
        points_pred = k3d.points(data, col_pred, point_size=0.02, shader='mesh', name='prediction')
        points_err = k3d.points(data, col_err, point_size=0.02, shader='mesh', name='metric values')
        colorbar = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                            color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.Jet)

        plot += points_true + points_pred + points_err + colorbar

        return plot

    def illustrate_to_file(self, batch_idx, data, preds, targets, metrics, batch=None, name=None):

        for sample in range(len(preds.size(0))):
            if name is None:
                self.name = f'illustration-points_batch-{batch_idx}_idx-{sample}'
            else:
                self.name = name

            plot_3d = self._illustrate_3d(data[sample], preds[sample], targets[sample], metrics[sample])
            with open(f'experiments/{self.name}.html', 'w') as f:
                f.write(plot_3d.get_snapshot())


class IllustratorDepths:

    def __init__(self, task, log):
        self.task = task
        self.log = log

    def _get_data_3d(self, data):
        image_height, image_width = data.shape[1], data.shape[2]
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

        return image_to_points(data, rays_origins)

    def _illustrate_3d(self, data, pred, target, metric, camera_pose):

        plot = k3d.plot(grid_visible=False, axes_helper=0)

        col_pred = get_colors(pred.cpu().numpy(), cm.coolwarm_r)
        col_true = get_colors(target.cpu().numpy(), cm.coolwarm_r)
        col_err = get_colors(metric.cpu().numpy(), cm.jet)

        points_true = k3d.points(camera_pose.camera_to_world(data, translate=True),
                                 col_true, point_size=0.02, shader='mesh', name='ground truth')
        points_pred = k3d.points(camera_pose.camera_to_world(data, translate=True),
                                 col_pred, point_size=0.02, shader='mesh', name='prediction')
        points_err = k3d.points(camera_pose.camera_to_world(data, translate=True),
                                col_err, point_size=0.02, shader='mesh', name='metric values')
        colorbar = k3d.line([[0, 0, 0], [0, 0, 0]], shader="mesh",
                            color_range=[0, 1], color_map=k3d.colormaps.matplotlib_color_maps.Jet)

        plot += points_true + points_pred + points_err + colorbar

        return plot

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

    def illustrate_to_file(self, batch_idx, data, preds, targets, metrics, batch=None, name=None):

        for sample in range(int(preds.size(0))):
            self.log.info(str(sample))
            if name is None:
                self.name = f'illustration-depths_task-{self.task}_batch-{batch_idx}_idx-{sample}'
            else:
                self.name = name

            plot_2d = self._illustrate_2d(data[sample][0], preds[sample][0], targets[sample][0], metrics[sample][0])
            plot_2d.savefig(f'/trinity/home/g.bobrovskih/sharp_features_pl_hydra_orig/experiments/{self.name}.png')

            camera_pose = CameraPose(batch['camera_pose'][sample].cpu().numpy())
            data_3d = self._get_data_3d(data[sample].cpu().numpy())
            plot_3d = self._illustrate_3d(data_3d,
                                          preds[sample].reshape(-1),
                                          targets[sample].reshape(-1),
                                          metrics[sample].reshape(-1),
                                          camera_pose)
            with open(f'/trinity/home/g.bobrovskih/sharp_features_pl_hydra_orig/experiments/{self.name}.html', 'w') as f:
                f.write(plot_3d.get_snapshot())
