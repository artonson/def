from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

# This should generally be called rasterization, but I was afraid
# to get something wrong, so I called it pixelization instead,
# which kind of has the same aim, but a toy background.

# In fact, image pixelization is a not-so-trivial task
# where one must:
#  - convert between pixel and image coordinate frames
#  - carefully use z-buffer to avoid projecting
#    'invisible' (from a particular viewing direction) surfaces
#  - evaluate the resulting function on a regular grid
from sharpf.utils.camera_utils.common import check_is_image, check_is_pixels


class ImagePixelizerBase(ABC):
    """Conversions between images defined in the image (canvas)
    coordinate frames, and pixel images defined in the pixel coordinate frame.o

    Image (canvas) coordinate frame:
        u, v defined in millimeters
        origin (o_x, o_y): image center exactly
        axes: X looks in direction of increasing X (right, same as pixels space)
              Y looks in direction of increasing Y (up, inverse wrt pixels space)
    """
    @abstractmethod
    def pixelize(
            self,
            image: np.ndarray,
            signal: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Given 3D points [n, 3] defined in the image plane,
        compute a rasterized [h, w] pixel image."""
        pass

    @abstractmethod
    def unpixelize(
            self,
            pixels: np.ndarray,
            signal: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Un-project a rasterized [h, w] pixel image
        into 3D points [n, 3] onto the 3D space.
        This assumes we start with , not a rasterized pixel image."""
        pass


class ImagePixelizer(ImagePixelizerBase):
    def __init__(self, intrinsics: np.ndarray):
        self.intrinsics = intrinsics

    @classmethod
    def from_params(cls, pixel_size=None, camera_center=None):
        pass

    def __transform_coords(self, image, signal=None):
        pixels_realvalued = np.dot(
            self.intrinsics,
            image[:, :2].T
        ).T
        return pixels_realvalued, signal

    def __compute_visibility(self, image, signal=None):
        pass

    def __assign_pixels(self, image, signal=None):
        pass

    def pixelize(
            self,
            image: np.ndarray,
            signal: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        check_is_image(image, signal)

        pixels_realvalued, signal_realvalued = self.__transform_coords(image)

        pixels_visible, signal_visible = self.__compute_visibility(image)

        pixels, signal_visible = self.__compute_visibility(image)

        # assume [n, 3] array
        assert len(canvas.shape) == 2 and canvas.shape[-1] == 3, 'cannot pixelize image'

        signal = None
        if self.view.signal is not None:
            signal = self.view.signal.ravel()[np.flatnonzero(view.image)]
        return pixels

    def unpixelize(
            self,
            image: np.ndarray,
            signal: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        check_is_pixels(image, signal)

