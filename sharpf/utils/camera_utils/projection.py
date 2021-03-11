from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np


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
            canvas: np.ndarray,
            signal: np.ndarray = None
    ) -> np.ndarray:
        """Un-project image plane points 3D points [n, 3] onto the 3D space.
        This assumes we start with , not a rasterized pixel image."""
        pass


class ParallelProjectionBase(CameraProjectionBase):
    def __init__(self, ):
        pass


class PerspectiveProjectionBase(CameraProjectionBase):
    def __init__(self, intrinsics: np.ndarray):
        self.intrinsics = intrinsics
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)

    @classmethod
    def from_params(cls, focal_length: Union[float, Tuple[float, float]]):
        intrinsics = get_camera_intrinsic_f(focal_length)
        return cls(intrinsics)

    def project(
            self,
            points: np.ndarray,
            signal: np.ndarray=None
    ) -> Tuple[np.ndarray, np.ndarray]:

        # assume [n, 3] array for points, [n, d] array for signal
        assert len(points.shape) == 2 and points.shape[-1] == 3, \
            'cannot project points to image: expected shape [n, 3], got: {}'.format(points.shape)
        if None is not signal:
            assert len(signal.shape) in [1, 2], \
                'cannot project signal: expected shape [n, d] or [n, ], got: {}'.format(signal.shape)
            assert points.shape[0] == signal.shape[0], \
                'cannot project points/signal: points and signal have different shapes'
            signal = np.atleast_2d(signal)

        image = np.dot(
            self.intrinsics,
            points.T).T
        return image, signal

    def unproject(
            self,
            image: np.ndarray,
            signal: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        # assume [n, 3] array for canvas, [n, d] array for signal
        assert len(image.shape) == 2 and image.shape[-1] == 3, \
            'cannot unproject image to points: expected shape [n, 3], got: {}'.format(image.shape)
        if None is not signal:
            assert len(signal.shape) in [1, 2], \
                'cannot project signal: expected shape [n, d] or [n, ], got: {}'.format(signal.shape)
            assert image.shape[0] == signal.shape[0], \
                'cannot project points/signal: points and signal have different shapes'
            signal = np.atleast_2d(signal)

        points = np.dot(
            self.intrinsics_inv,
            image.T).T

        return points, signal
