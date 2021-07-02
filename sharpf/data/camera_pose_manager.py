import warnings
from abc import ABC, abstractmethod
from itertools import product

import numpy as np

from sharpf.data import DataGenerationException
from sharpf.utils.camera_utils.camera_pose import CameraPose, create_rotation_matrix_z, rotate_to_world_origin
from sharpf.utils.camera_utils.spherical_spiral_sampling import spherical_spiral_sampling
from sharpf.utils.py_utils.config import load_func_from_config
from sharpf.utils.camera_utils.fibonacci_sphere_sampling import fibonacci_sphere_sampling


class AbstractCameraPoseManager(ABC):
    """Implements obtaining camera poses."""
    def __init__(self, n_images, random_seed=None):
        self.n_images = n_images
        self.camera_poses = []
        self.current_transform_idx = 0
        self.seed = random_seed

    @classmethod
    def from_config(cls, config):
        return cls(config['n_images'], random_seed=config.get('seed'))

    @abstractmethod
    def prepare(self, mesh):
        """Precomputes the transformations specified by class parameters.

        :param mesh: an input mesh
        :type mesh: MeshType (must be present attributes `vertices`, `faces`, and `edges`)
        """
        pass

    def __iter__(self):
        if not self.camera_poses:
            raise DataGenerationException('Raycasting was not prepared')

        for pose in self.camera_poses:
            yield pose


class SphereOrientedToWorldOrigin(AbstractCameraPoseManager):
    def prepare(self, mesh):
        # precompute transformations from world frame to camera frame
        # for all specified camera origins

        # scanning radius is determined from the mesh extent
        scanning_radius = np.max(mesh.bounding_box.extents)

        # XYZ coordinates of camera frame origin in world frame
        camera_origins = fibonacci_sphere_sampling(
            self.n_images, radius=scanning_radius, seed=self.seed)

        # creating transforms matrices
        self.camera_poses = [
            CameraPose.from_camera_axes(
                R=rotate_to_world_origin(camera_origin),
                t=camera_origin)
            for camera_origin in camera_origins
        ]
        self.current_transform_idx = 0


class SphericalSpiralOrientedToWorldOrigin(AbstractCameraPoseManager):
    def __init__(self, n_images, layer_radius, resolution, n_initial_samples, min_arc_length):
        warnings.warn('n_images ignored')
        super().__init__(n_images)
        # parameters needed for the spiral
        self.layer_radius = layer_radius
        self.resolution = resolution
        self.n_initial_samples = n_initial_samples
        self.min_arc_length = min_arc_length

    def prepare(self, mesh):
        # precompute transformations from world frame to camera frame
        # for all specified camera origins

        # scanning radius is determined from the mesh extent
        scanning_radius = np.max(mesh.bounding_box.extents)

        # XYZ coordinates of camera frame origin in world frame
        camera_origins = spherical_spiral_sampling(
            sphere_radius=scanning_radius,
            layer_radius=self.layer_radius,
            resolution=self.resolution,
            n_initial_samples=self.n_initial_samples,
            min_arc_length=self.min_arc_length)

        # creating transforms matrices
        self.camera_poses = [
            CameraPose.from_camera_axes(
                R=rotate_to_world_origin(camera_origin),
                t=camera_origin)
            for camera_origin in camera_origins
        ]
        self.current_transform_idx = 0

    @classmethod
    def from_config(cls, config):
        return cls(
            config['n_images'],
            config['layer_radius'],
            config['resolution'],
            config['n_initial_samples'],
            config['min_arc_length'])


class ZRotationInCameraFrame(AbstractCameraPoseManager):
    """Performs rotation along Z axis in camera frame.
    """
    def prepare(self, mesh):
        self.camera_poses = [
            CameraPose.from_camera_axes(
                R=create_rotation_matrix_z(xy_angle),
            )
            for xy_angle in np.linspace(0, 2 * np.pi, self.n_images)
        ]
        self.current_transform_idx = 0


class XYTranslationInCameraFrame(AbstractCameraPoseManager):
    """Performs translations along X and Y axes in the camera frame.
    """

    def prepare(self, mesh):
        # precompute transformations from world frame to camera frame
        # for all specified camera origins
        n_images_x = int(np.sqrt(self.n_images))
        n_images_y = int(np.sqrt(self.n_images))

        # scanning radius is determined from the mesh extent
        scanning_radius = np.max(mesh.bounding_box.extents) / 2.0 + 1

        # XYZ coordinates of camera frame origin in world frame
        centers = np.mgrid[0:n_images_y, 0:n_images_x].reshape(
            2, n_images_y * n_images_x).T  # [h, w, 2]

        centers = centers / np.array([[n_images_y, n_images_x]])

        centers[:, 0] = (2 * centers[:, 0] - 1) * scanning_radius
        centers[:, 1] = (2 * centers[:, 1] - 1) * scanning_radius
        centers = np.concatenate([
            centers, np.zeros_like(centers[:, [0]])
        ], axis=1)  # [h, w, 3]

        # creating transforms matrices
        self.camera_poses = [
            CameraPose.from_camera_axes(t=center)
            for center in centers
        ]
        self.current_transform_idx = 0


class CompositePoseManager(AbstractCameraPoseManager):
    """Composition of atomic transforms. """

    def __init__(self, sequences):
        self._sequences = sequences

    @classmethod
    def from_config(cls, config):
        sequences = [
            load_func_from_config(POSE_MANAGER_BY_TYPE, sequence_config)
            for sequence_config in config['sequences']
        ]
        return cls(sequences)

    def prepare(self, mesh):
        # precompute transformations from world frame to camera frame
        # for all specified camera origins
        for sequence in self._sequences:
            sequence.prepare(mesh)

    def __iter__(self):
        # iterative over tuples of transformations
        for relative_poses in product(*self._sequences):

            # iterate over components of each transformation, computing the final pose
            pose = CameraPose(np.identity(4))
            for relative_pose in relative_poses:
                pose = pose.compose_world_to_camera(relative_pose)

            yield pose


POSE_MANAGER_BY_TYPE = {
    'composite': CompositePoseManager,
    'sphere_to_origin': SphereOrientedToWorldOrigin,
    'sphere_spiral_to_origin': SphericalSpiralOrientedToWorldOrigin,
    'z_rotation': ZRotationInCameraFrame,
    'xy_translation': XYTranslationInCameraFrame,
}
