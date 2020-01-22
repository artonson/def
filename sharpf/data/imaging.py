from abc import ABC, abstractmethod


class ImagingFunc(ABC):
    """Implements obtaining depthmaps from meshes."""
    def __init__(self, n_images, resolution):
        self.n_images = n_images
        self.resolution = resolution

    @abstractmethod
    def get_image(self, mesh):
        """Extracts a point cloud.

        :param mesh: an input mesh
        :type mesh: MeshType (must be present attributes `vertices`, `faces`, and `edges`)

        :returns: depthmap: the depth image
        :rtype: np.ndarray
        """
        pass

    @classmethod
    def from_config(cls, config):
        return cls(config['n_images'], config['resolution'])


class RaycastingImaging(ImagingFunc):
    def get_image(self, mesh):



IMAGING_BY_TYPE = {
    'raycasting': RaycastingImaging,
}


