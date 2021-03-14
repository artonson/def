from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np

from sharpf.utils.camera_utils import matrix
from sharpf.utils.camera_utils.common import check_is_image, check_is_points


class CameraProjectionBase(ABC):
    @abstractmethod
    def project(
            self,
            points: np.ndarray,
            signal: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points [n, 3] onto the image plane.
        This does not compute a rasterized pixel image."""
        pass

    @abstractmethod
    def unproject(
            self,
            image: np.ndarray,
            signal: np.ndarray = None
    ) -> np.ndarray:
        """Un-project image plane points 3D points [n, 3] onto the 3D space.
        This assumes we start with , not a rasterized pixel image."""
        pass

#
# class ParallelProjection(CameraProjectionBase):
#     def __init__(self, ):
#         pass


class PerspectiveProjection(CameraProjectionBase):
    def __init__(self, intrinsics: np.ndarray):
        self.intrinsics = intrinsics
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)

    @classmethod
    def from_params(cls, focal_length: Union[float, Tuple[float, float]]):
        intrinsics = matrix.get_camera_intrinsic_f(focal_length)
        return cls(intrinsics)

    def project(
            self,
            points: np.ndarray,
            signal: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        check_is_points(points, signal)

        if None is not signal:
            signal = np.atleast_2d(signal)

        z = points[:, [2]].copy()
        image = np.dot(
            self.intrinsics,
            points.T).T
        image[:, [0, 1]] /= z

        return image, signal

    def unproject(
            self,
            image: np.ndarray,
            signal: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        check_is_image(image, signal)

        if None is not signal:
            signal = np.atleast_2d(signal)

        image_to_unproject = image.copy()
        z = image_to_unproject[:, [2]].copy()
        image_to_unproject[:, 2] = 1
        points = np.dot(
            self.intrinsics_inv,
            image.T).T
        points[:, [0, 1]] *= z

        return points, signal
