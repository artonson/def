from abc import ABC, abstractmethod

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


class OrthogonalProjection(CameraProjection):
    def __init__(self, ):
        pass


class PerspectiveProjection(CameraProjection):
    pass

