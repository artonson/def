from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np


class CameraProjection(ABC):
    @abstractmethod
    def project(self, points: np.ndarray) -> np.ndarray:
        """Project 3D points [n, 3] onto the image plane.
        This does not compute a rasterized pixel image."""
        pass

    @abstractmethod
    def unproject(self, image: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Un-project image plane points 3D points [n, 3] onto the 3D space.
        This assumes we start with , not a rasterized pixel image."""
        pass


class ParallelProjection(CameraProjection):
    def __init__(self, ):
        pass


class PerspectiveProjection(CameraProjection):
    def __init__(self, intrinsics: np.ndarray):
        self.intrinsics = intrinsics
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)

    @classmethod
    def from_params(cls, focal_length: Union[float, Tuple[float, float]]):
        intrinsics = get_camera_intrinsic_f(focal_length)
        return cls(intrinsics)

    def project(self, points: np.ndarray) -> np.ndarray:
        # assume [n, 3] array
        assert len(points.shape) == 2 and points.shape[-1] == 3, 'cannot pixelize image'
        image = np.dot(
            self.intrinsics,
            points.T).T
        return image

    def unproject(self, canvas: np.ndarray, depth: np.ndarray) -> np.ndarray:
        # assume [n, 3] array
        assert len(canvas.shape) == 2
        assert len(points.shape) == 2 and points.shape[-1] == 3, 'cannot pixelize image'

        image =
        points = np.dot(
            self.intrinsics_inv,
            image.T).T

        return points

