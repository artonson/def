from abc import ABC, abstractmethod

import numpy as np

# This should generally be called rasterization, but I was afraid
# to get something wrong, so I called it pixelization instead,
# which kind of has the same aim, but a toy background.


class ImagePixelizer(ABC):
    @abstractmethod
    def pixelize(self, image: np.ndarray) -> np.ndarray:
        """Given 3D points [n, 3] defined in the image plane,
        compute a rasterized [h, w] pixel image."""
        pass

    @abstractmethod
    def unpixelize(self, image: np.ndarray) -> np.ndarray:
        """Un-project image plane points 3D points [n, 3] onto the 3D space.
        This assumes we start with , not a rasterized pixel image."""
        pass

