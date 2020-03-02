from abc import ABC, abstractmethod

import numpy as np

from sharpf.data import DataGenerationException
from sharpf.utils.mesh_utils.indexing import reindex_zerobased
from sharpf.utils.raycasting import generate_rays, ray_cast_mesh
from sharpf.utils.sampling import fibonacci_sphere_sampling
from sharpf.utils.view import from_pose, euclid_to_sphere


class ImagingFunc(ABC):
    """Implements obtaining depthmaps from meshes."""
    @abstractmethod
    def get_image(self, mesh):
        """Extracts a point cloud.

        :param mesh: an input mesh
        :type mesh: MeshType (must be present attributes `vertices`, `faces`, and `edges`)

        :returns: depthmap: the depth image
        :rtype: np.ndarray
        """
        pass


class RaycastingImaging(ImagingFunc):
    def __init__(self, resolution_image, resolution_3d, projection):
        self.resolution_image = resolution_image
        self.resolution_3d = resolution_3d
        self.projection = projection
        self.rays_screen_coords, self.rays_origins, self.rays_directions = None, None, None

    @classmethod
    def from_config(cls, config):
        return cls(config['resolution_image'], config['resolution_3d'],
                   config['projection'])

    def prepare(self, scanning_radius):
        # scanning radius is determined from the mesh extent
        self.rays_screen_coords, self.rays_origins, self.rays_directions = generate_rays(
            self.resolution_image, self.resolution_3d, radius=scanning_radius)

    def get_image(self, mesh):
        if any(value is None for value in [self.rays_screen_coords, self.rays_origins, self.rays_directions]):
            raise DataGenerationException('Raycasting was not prepared')

        # get a point cloud with corresponding indexes
        print(self.rays_origins, self.rays_directions)
        mesh_face_indexes, ray_indexes, points = ray_cast_mesh(
            mesh, self.rays_origins, self.rays_directions)

        # extract normals
        normals = mesh.face_normals[mesh_face_indexes]

        # compute indexes of faces and vertices in the original mesh
        mesh_face_indexes = np.unique(mesh_face_indexes)
        if mesh.faces[mesh_face_indexes].shape[0] != 0:
            mesh_vertex_indexes = np.unique(
                mesh.faces[mesh_face_indexes].reshape(-1, 1), axis=1)
        else:
            mesh_vertex_indexes = mesh_face_indexes
           
        # assemble mesh fragment into a submesh
        nbhood = reindex_zerobased(mesh, mesh_vertex_indexes, mesh_face_indexes)
 
        return ray_indexes, points, normals, nbhood, mesh_vertex_indexes, mesh_face_indexes

    def points_to_image(self, points, ray_indexes, assign_channels=None):
        xy_to_ij = self.rays_screen_coords[ray_indexes]
        # note that `points` can have many data dimensions
        if None is assign_channels:
            assign_channels = [2]
        data_channels = len(assign_channels)
        image = np.zeros((self.resolution_image, self.resolution_image, data_channels))
        image[xy_to_ij[:, 0], xy_to_ij[:, 1]] = points[:, assign_channels]
        return image.squeeze()

    # TODO implement `image_to_points`
    # def image_to_points(self, image):
    #     points = np.zeros((self.resolution_image * self.resolution_image, 3))
    #     points[:, 0] = self.rays_origins[:, 0]
    #     points[:, 1] = self.rays_origins[:, 1]
    #
    #     xy_to_ij = self.rays_screen_coords[ray_indexes]
    #     points[:, 2] = image[xy_to_ij[:, 0], xy_to_ij[:, 1]]
    #     return points


IMAGING_BY_TYPE = {
    'raycasting': RaycastingImaging,
}

class ScanningSequence(ABC):
    """Implements obtaining camera poses."""
    def __init__(self, n_images):
        self.n_images = n_images

    @classmethod
    def from_config(cls, config):
        return cls(config['n_images'])

    @abstractmethod
    def next_camera_pose(self, mesh):
        """Rotates the mesh to match the next camera pose.

        :param mesh: an input mesh
        :type mesh: MeshType (must be present attributes `vertices`, `faces`, and `edges`)

        :returns: tuple(mesh, pose): mesh in the camera pose, and the camera pose itself
        :rtype: Tuple[trimesh.base.Trimesh, tuple]

        """
        pass


class FibonacciSamplingScanningSequence(ScanningSequence):
    def __init__(self, n_images):
        super().__init__(n_images)
        self.camera_poses = None
        self.current_pose_index = 0

    def prepare(self, scanning_radius):
        # scanning radius is determined from the mesh extent
        self.camera_poses = fibonacci_sphere_sampling(
            self.n_images, radius=scanning_radius)

    def next_camera_pose(self, mesh):
        if None is self.camera_poses:
            raise DataGenerationException('Raycasting was not prepared')

        pose = self.camera_poses[self.current_pose_index]
        angles = euclid_to_sphere(pose)  # [0, theta, phi]
        transform = from_pose(angles, [0, 0, 0])  # rotation + translation

        # z_shift = -3
        mesh_transformed = mesh.copy()
        mesh_transformed.apply_transform(transform)
        # mesh_transformed.apply_translation([0, 0, z_shift])

        return mesh_transformed, pose


SCANNING_SEQ_BY_TYPE = {
    'fibonacci_sampling': FibonacciSamplingScanningSequence
}
