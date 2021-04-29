from abc import ABC, abstractmethod
from collections import Mapping

import numpy as np
import trimesh

from sharpf.data import DataGenerationException
from sharpf.utils.camera_utils.camera_pose import CameraPose
from sharpf.utils.camera_utils.raycasting import generate_rays, ray_cast_mesh


class ImagingFunc(ABC):
    """Implements obtaining depth maps from meshes."""
    @abstractmethod
    def get_image_from_pose(self, mesh, pose, *args, **kwargs):
        """Extracts a point cloud.

        :param mesh: an input mesh
        :type mesh: MeshType (must be present attributes `vertices`, `faces`, and `edges`)

        :param pose: camera pose to shoot from
        :type pose: CameraPose
        """
        pass


class RaycastingImaging(ImagingFunc):
    def __init__(self, resolution_image, resolution_3d, projection, validate_image):
        if isinstance(resolution_image, tuple):
            assert len(resolution_image) == 2
        else:
            resolution_image = (resolution_image, resolution_image)
        self.resolution_image = resolution_image
        self.resolution_3d = resolution_3d
        self.projection = projection
        self.validate_image = validate_image
        self.rays_screen_coords, self.rays_origins, self.rays_directions = generate_rays(
            self.resolution_image, self.resolution_3d)

    @classmethod
    def from_config(cls, config):
        return cls(config['resolution_image'],
                   config['resolution_3d'],
                   config['projection'],
                   config['validate_image'])

    def get_image_from_pose(self,
                            mesh: trimesh.base.Trimesh,
                            pose: CameraPose,
                            features: Mapping = None,
                            return_hit_face_indexes=False):
        """Get an image.

        :param mesh: the triangular mesh
        :param pose: object defining camera orientation
        :param features:
        :param return_hit_face_indexes:
        :return: if return_hit_face_indexes is False, return:
            image, points, normals where:
             - image is 2D one-channel image with depth values relative to camera frame
             - points is [n, 3] array with XYZ point cloud coordinates, in world frame
             - normals is [n, 3] array with nX nY nZ normal directions, in world frame
             if return_hit_face_indexes is True, return:
            image, points, normals, mesh_face_indexes where:
            image, points, normals are the same as above
            mesh_face_indexes is [n,] array of face indexes hit by raycasting in the original mesh
        """
        if any(value is None for value in [self.rays_screen_coords,
                                           self.rays_origins,
                                           self.rays_directions]):
            raise DataGenerationException('Raycasting was not prepared')

        # get a point cloud with corresponding indexes
        mesh_face_indexes, ray_indexes, points = ray_cast_mesh(
            mesh,
            pose.camera_to_world(self.rays_origins),
            pose.camera_to_world(self.rays_directions, translate=0)
        )

        if len(points) == 0:  # we hit nothing; discard this attempt
            raise DataGenerationException('Object out of frame; discarding patch')

        # extract normals
        normals = mesh.face_normals[mesh_face_indexes]

        # compute an image, defined in camera frame
        image = self.points_to_image(pose.world_to_camera(points), ray_indexes)
        if self.validate_image and np.any(image < 0.):
            raise DataGenerationException('Negative values found in depthmap; discarding image')

        if return_hit_face_indexes:
            return image, points, normals, mesh_face_indexes
        else:
            return image, points, normals

    def points_to_image(self, points, ray_indexes, assign_channels=None):
        xy_to_ij = self.rays_screen_coords[ray_indexes]
        # note that `points` can have many data dimensions
        if None is assign_channels:
            assign_channels = [2]
        data_channels = len(assign_channels)
        image_height, image_width = self.resolution_image
        image = np.zeros((image_height, image_width, data_channels))
        # rays origins (h, w, 3), z is the same for all points of matrix
        # distance is absolute value
        image[xy_to_ij[:, 0], xy_to_ij[:, 1]] = points[[1, 0], assign_channels]
        return image.squeeze()

    def image_to_points(self, image):
        i = np.where(image.ravel() != 0)[0]
        points = np.zeros((len(i), 3))
        points[:, 1] = self.rays_origins[i, 0]
        points[:, 0] = self.rays_origins[i, 1]
        points[:, 2] = image.ravel()[i]
        return points


IMAGING_BY_TYPE = {
    'raycasting': RaycastingImaging,
}

