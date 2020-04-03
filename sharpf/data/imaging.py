from abc import ABC, abstractmethod

import numpy as np

from sharpf.data import DataGenerationException
from sharpf.utils.mesh_utils.indexing import reindex_zerobased
from sharpf.utils.raycasting import generate_rays, ray_cast_mesh
from sharpf.utils.sampling import fibonacci_sphere_sampling, poisson_disc_sampling
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

        #mesh.apply_translation([0, 0, z_shift])

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

class PoissonDiscSamplingImages():
    def __init__(self, radius, K, random_seed=42):
        self.K = K
        self.radius = radius
        self.points = None
        self.seed = random_seed

    @classmethod
    def from_config(cls, config):
        if 'random_seed' in config.keys():
            return cls(config['radius'], config['K'], config['random_seed'])
        else:
            return cls(config['radius'], config['K'])

    def prepare(self, width, height):
        ''''''
        self.points = poisson_disc_sampling(width, height, r=self.radius, k=self.K, seed=self.seed)

    def choose_n_points(self, n):
        self.points = np.random.choice(self.points, size=n, replace=False)

    # def _get_3d_points(self):
    #
    #     return points_3d

    def __len__(self):
        if self.points is not None:
            return len(self.points)
        else:
            return 0

    def __getitem__(self, item):
        return self.points[item]


class ScanningSequence(ABC):
    """Implements obtaining camera poses."""
    def __init__(self, n_perspectives, n_images_per_perspective):
        self.n_perspectives = n_perspectives
        self.n_images_per_perspective = n_images_per_perspective

    @classmethod
    def from_config(cls, config):
        return cls(config['n_images'], config['n_images_per_perspective'])

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
    def __init__(self, n_perspectives, n_images_per_perspective, random_seed=None):
        super().__init__(n_perspectives, n_images_per_perspective)
        self.camera_poses, self.transforms = None, None
        self.current_pose_index = 0
        self.seed = random_seed

    @classmethod
    def from_config(cls, config):
        if 'seed' in config.keys():
            return cls(config['n_images'], config['n_images_per_perspective'], config['seed'])
        else:
            return cls(config['n_images'], config['n_images_per_perspective'])


    def prepare(self, scanning_radius, image_sampling_method):#, image_sampling_params):
        # scanning radius is determined from the mesh extent
        self.camera_poses = fibonacci_sphere_sampling(
            self.n_perspectives, radius=scanning_radius, seed=self.seed)

        # prepare image-points sampling method
        self.image_sampling = IMAGES_SAMPLING_BY_TYPE[image_sampling_method]#.from_config(image_sampling_params)

        #creating transforms matrices
        self.angles = np.zeros((len(self.camera_poses), 3))
        self.transforms = np.zeros((self.n_perspectives*self.n_images_per_perspective, 4, 4))
        for i, pose in enumerate(self.camera_poses):
            self.angles[i] = np.array(euclid_to_sphere(pose))  # [0, theta, phi]

        self.current_pose_index = 0
        self.current_image_index = 0
        self.iteration_index = 0

    def next_camera_pose(self, mesh):
        if None is self.camera_poses:
            raise DataGenerationException('Raycasting was not prepared')

        angle = self.angles[self.current_pose_index+int(self.current_image_index == 0.0)]
        if self.current_image_index > 0:
            x, y = self.image_sampling_function[self.current_image_index]
            translation = [-x, -y, 0]
        else:
            translation = [0, 0, 0]

        self.transforms[self.iteration_index] = from_pose(angle, translation)  # rotation
        transform = self.transforms[self.iteration_index]

        mesh_transformed = mesh.copy()
        mesh_transformed.apply_transform(transform)
        mesh_transformed.apply_translation(translation)

        # sample images origins on the mesh surface
        if self.current_image_index == 0.0:
            bottom_right, top_left = mesh.bounding_box.bounds
            width, height = top_left[0] - bottom_right[0], top_left[1] - bottom_right[1]
            image_sampling_params = {'radius': 2, 'K': 30}
            self.image_sampling_function = self.image_sampling.from_config(image_sampling_params)
            self.image_sampling_function.prepare(width//2, height//2)
            self.image_sampling.choose_n_points(self.n_images_per_perspective)
            self.current_image_index = len(self.image_sampling_function)
            self.current_pose_index += 1

        # translate mesh to image
        self.current_image_index -= 1
        self.iteration_index += 1

        return mesh_transformed, transform

    def iterate_camera_poses(self, mesh):
        i = 0
        while i < self.n_perspectives * self.n_images_per_perspective:
            yield self.next_camera_pose(mesh)
            i += 1


SCANNING_SEQ_BY_TYPE = {
    'fibonacci_sampling': FibonacciSamplingScanningSequence
}
IMAGES_SAMPLING_BY_TYPE = {
    'poisson_disc_sampling': PoissonDiscSamplingImages
}
