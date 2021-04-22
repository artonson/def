from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from sharpf.utils.camera_utils import matrix
from sharpf.utils.camera_utils.common import check_is_image, check_is_pixels

# This should generally be called rasterization, but I was afraid
# to get something wrong, so I called it pixelization instead,
# which kind of has the same aim, but a toy background.

# In fact, image pixelization is a not-so-trivial task
# where one must:
#  - convert between pixel and image coordinate frames
#  - carefully use z-buffer to avoid projecting
#    'invisible' (from a particular viewing direction) surfaces
#  - evaluate the resulting function on a regular grid


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
            signal: np.ndarray = None,
            faces: np.ndarray = None
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


def simple_z_buffered_rendering(
        pixels: np.ndarray,
        depth: np.ndarray,
        signal: np.ndarray,
        image_size: Tuple[int, int],
        depth_func_type: str = 'max',
):
    """A (over-)simplified z-buffer implementation.

    Given a [n, 3] array of rounded pixel coordinates (indexes)
    and corresponding [n,] array of depth values,
    return a 2d image with pixel values assigned to smallest (closest to camera) depth values.

    Normally we do:
    >>> image = np.zeros((image_size[1], image_size[0]))
    >>> depth_out[
    >>>     pixels[:, 1] - image_offset[1],
    >>>     pixels[:, 0] - image_offset[0]] = depth

    However, if multiple values in x_pixel index into the same (x, y) pixel in image,
    i.e. for i != j we have x_pixel[i] == x_pixel[j],
    we end up assigning arbitrary depth value in image.

    Instead we do (conceptually):
    >>> depth_out[pixels] = func(depth[i] for i where pixels[i] == c)
    """
    depth_out = np.zeros(image_size[::-1])
    signal_out = None
    if None is not signal:
        if len(signal.shape) == 1:
            signal_out_shape = depth_out.shape
        else:
            signal_out_shape = (image_size[1], image_size[0], signal.shape[-1])
        signal_out = np.zeros(signal_out_shape)

    if len(pixels) == 0:
        return pixels, depth_out, signal_out

    # sort XY array by row/col index
    sort_idx_1 = pixels[:, 1].argsort()
    pixels = pixels[sort_idx_1]
    sort_idx_0 = pixels[:, 0].argsort(kind='mergesort')
    pixels = pixels[sort_idx_0]

    idx_sort = sort_idx_1[sort_idx_0]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    unique_pixels_xy, unique_pixels_indexes, _ = np.unique(
        pixels,
        axis=0,
        return_counts=True,
        return_index=True)

    # splits the indices into separate arrays
    same_point_indexes = np.split(idx_sort, unique_pixels_indexes[1:])

    z_getter = {'min': np.argmin, 'max': np.argmax}[depth_func_type]
    for pixel_xy, point_indexes in zip(unique_pixels_xy, same_point_indexes):
        target_ji = (pixel_xy[1], pixel_xy[0])
        z_index = z_getter(depth[point_indexes])
        depth_out[target_ji] = depth[point_indexes][z_index]
        if None is not signal:
            signal_out[target_ji] = signal[point_indexes][z_index]

    return pixels, depth_out, signal_out


class ImagePixelizer(ImagePixelizerBase):
    def __init__(self, intrinsics: np.ndarray, image_size_in_pixels: Tuple[int, int] = None):
        self.intrinsics = intrinsics
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)
        self.image_size_in_pixels = image_size_in_pixels
        if None is self.image_size_in_pixels:
            self.image_size_in_pixels = (
                int(2 * self.intrinsics[0, 2]), int(2 * self.intrinsics[1, 2]))

    @classmethod
    def from_params(cls, pixel_size, image_size_in_pixels):
        intrinsics = matrix.get_camera_intrinsic_s(
            pixel_size[0],
            pixel_size[1],
            image_size_in_pixels[0],
            image_size_in_pixels[1])
        return cls(intrinsics, image_size_in_pixels)

    def split_input(self, image, signal=None):
        image_in = image.copy()
        depth_in = image_in[:, 2].copy()
        image_in[:, 2] = 1.
        return image_in, depth_in, signal

    def transform_image_to_pixel_coords(self, image, depth, signal=None):
        pixels_realvalued = np.dot(
            self.intrinsics,
            image.T
        ).T
        return pixels_realvalued, depth, signal

    def transform_pixel_to_image_coords(self, image, depth, signal=None):
        image_realvalued = np.dot(
            self.intrinsics_inv,
            image.T
        ).T
        return image_realvalued, depth, signal

    def compute_visibility(self, pixels, depth, signal=None):
        return pixels, depth, signal

    def assign_pixels(self, pixels, depth, signal=None):

        # Here we could interpolate depth and signal values
        # into the regular grid, if we wanted to.
        pixels = np.round(pixels).astype(np.int32)
        width, height = self.image_size_in_pixels
        frustum_indexes = np.where(
            (pixels[:, 0] > 0) & (pixels[:, 0] < width) &
            (pixels[:, 1] > 0) & (pixels[:, 1] < height))[0]

        pixels_out, depth_out, signal_out = simple_z_buffered_rendering(
            pixels[frustum_indexes],
            depth[frustum_indexes],
            signal[frustum_indexes] if None is not signal else None,
            self.image_size_in_pixels,
            depth_func_type='min')
        return pixels_out, depth_out, signal_out

    def unassign_pixels(self, pixels, signal=None):
        height, width = pixels.shape
        i, j = np.meshgrid(np.arange(width), np.arange(height))

        image_integers = np.stack((
            i.ravel(),
            j.ravel(),
            np.ones_like(i).ravel()
        )).T  # [n, 3]
        depth_integers = pixels.ravel()
        image_integers = image_integers[depth_integers != 0]
        depth_integers = depth_integers[depth_integers != 0]

        signal_integers = signal
        if None is not signal_integers:
            signal_integers = signal_integers[pixels != 0]

        return image_integers, depth_integers, signal_integers

    def pixelize(
            self,
            image: np.ndarray,
            signal: np.ndarray = None,
            faces: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        check_is_image(image, signal)

        # separate XY (image coordinates) and Z
        image_in, depth_in, signal_in = self.split_input(image, signal)

        # obtain UV in floating-point pixel coordinates (no rounding, nothing)
        pixels_realvalued, depth_realvalued, signal_realvalued = self.transform_image_to_pixel_coords(
            image_in, depth_in, signal_in)

        # computes a mask indicating which pixels are visible
        # and returns only visible pixels, associated depth, and signals
        pixels_visible, depth_visible, signal_visible = self.compute_visibility(
            pixels_realvalued, depth_realvalued, signal_realvalued)

        # compute actual pixel depth image and signal
        pixels_out, depth_out, signal_out = self.assign_pixels(
            pixels_visible, depth_visible, signal_visible)

        return depth_out, signal_out

    def unpixelize(
            self,
            pixels: np.ndarray,
            signal: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        check_is_pixels(pixels, signal)
        # print(pixels.shape, signal.shape)

        # obtain IJ integer coordinates in pixel space
        image_integers, depth_integers, signal_integers = self.unassign_pixels(
            pixels, signal)
        # print(image_integers.shape, depth_integers.shape, signal_integers.shape)

        # obtain UV in floating-point pixel coordinates (no rounding, nothing)
        image_realvalued, depth_realvalued, signal_realvalued = self.transform_pixel_to_image_coords(
            image_integers, depth_integers, signal_integers)
        # print(image_realvalued.shape, depth_realvalued.shape, signal_realvalued.shape)

        # merge UV (image coordinates) and Z (depth)
        image_out, signal_out = self.unsplit_input(
            image_realvalued, depth_realvalued, signal_realvalued)
        # print(image_out.shape, signal_out.shape)

        return image_out, signal_out

    def unsplit_input(self, image, depth, signal=None):
        image_out = image.copy()
        image_out[:, 2] = depth
        return image_out, signal
