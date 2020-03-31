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
    def get_image(self, mesh, features):
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

    def get_image(self, mesh, features):
        if any(value is None for value in [self.rays_screen_coords, self.rays_origins, self.rays_directions]):
            raise DataGenerationException('Raycasting was not prepared')

        mesh.apply_translation([0, 0, z_shift])

        # get a point cloud with corresponding indexes
        mesh_face_indexes, ray_indexes, points = ray_cast_mesh(
            mesh, self.rays_origins, self.rays_directions)

        # extract normals
        normals = mesh.face_normals[mesh_face_indexes]

        # compute indexes of faces and vertices in the original mesh
        hit_surfaces_face_indexes = []
        for idx, surface in enumerate(features['surfaces']):
            surface_face_indexes = np.array(surface['face_indices'])
            if np.any(np.isin(surface_face_indexes, mesh_face_indexes, assume_unique=True)):
                hit_surfaces_face_indexes.extend(surface_face_indexes)
        mesh_face_indexes = np.unique(hit_surfaces_face_indexes)
        mesh_vertex_indexes = np.unique(mesh.faces[mesh_face_indexes])

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
    def image_to_points(self, image):
        points = np.zeros((self.resolution_image * self.resolution_image, 3))
        points[:, 0] = self.rays_origins[:, 0]
        points[:, 1] = self.rays_origins[:, 1]

        xy_to_ij = self.rays_screen_coords[ray_indexes]
        points[:, 2] = image[xy_to_ij[:, 0], xy_to_ij[:, 1]]
        return points


IMAGING_BY_TYPE = {
    'raycasting': RaycastingImaging,
}

class ScanningSequence(ABC):
    """Implements obtaining camera poses."""
    def __init__(self, n_perspectives, n_images_per_perspective):
        self.n_images = n_perspectives
        self.n_images_per_perspective = n_images_per_perspective

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

    def next_image(self, mesh):
        """Translate mesh to capture next object part.

            :param mesh: mesh in the camera pose
            :type

            :returns: (mesh, point): translated mesh, mesh point moved to origin
            :rtype:

        """


class FibonacciSamplingScanningSequence(ScanningSequence):
    def __init__(self, n_perspectives, n_images_per_perspective):
        super().__init__(n_perspectives, n_images_per_perspective)
        self.camera_poses, self.transforms = None, None
        self.current_pose_index = 0

    def prepare(self, scanning_radius):
        # scanning radius is determined from the mesh extent
        self.camera_poses = fibonacci_sphere_sampling(
            self.n_perspectives, radius=scanning_radius)
        #creating transforms matrices
        self.transforms = [] # replace with numpy array
        for pose in self.camera_poses:
            angles = euclid_to_sphere(pose)  # [0, theta, phi]
            self.transforms = from_pose(angles, [0, 0, 0])  # rotation + translation

        self.current_pose_index = 0

    def next_camera_pose(self, mesh, images_sampling):
        if None is self.camera_poses:
            raise DataGenerationException('Raycasting was not prepared')

        transform = self.transforms[self.current_pose_index]

        # z_shift = -3
        mesh_transformed = mesh.copy()
        mesh_transformed.apply_transform(transform)
        # mesh_transformed.apply_translation([0, 0, z_shift])

        top_left, bottom_right = mesh.bounding_box.bounds
        # sample images origins on the mesh surface

        if self.current_image_index == 0.0:
            self.images = images_sampling.prepare(top_left, bottom_right)
            self.current_image_index = len(images)
            self.current_pose_index += 1

        # translate mesh to image
        x, y  = self.images[self.current_image_index]
        mesh_transformed.apply_transform([-x, -y, 0])

        return mesh_transformed, pose

    def iterate_camera_poses(self, mesh):
        mesh_transformed, camera_pose = self.next_camera_pose(mesh)



SCANNING_SEQ_BY_TYPE = {
    'fibonacci_sampling': FibonacciSamplingScanningSequence
}
IMAGES_SAMPLING_BY_TYPE = {
    'poisson_disc_sampling': PoissonDiscSamplingImages
}
