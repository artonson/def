from typing import Tuple

import numpy as np
from skimage.morphology import square, binary_erosion

from sharpf.fusion.images import interpolate
from sharpf.utils.camera_utils.view import CameraView


def remove_boundary_by_erosion(
        image,
        boundary_width=1,
):
    """For a grayscale image, run binary_erosion and return an image
    with pixels remaining after erosion set to what they were before."""
    s = square(boundary_width * 2 + 1)
    binary_image = (image != 0).astype(np.int)
    eroded = binary_erosion(binary_image, s)
    output_image = np.zeros_like(image)
    output_image[eroded] = image[eroded]
    return output_image


def interpolate_views_as_images(
        source_view: CameraView,
        target_view: CameraView,
        distance_interp_thr: float = 1.0,
        nn_set_size: int = 8,
        z_distance_threshold: int = 2.0,
        interp_ratio_thr: float = 1.0,
        boundary_width_to_remove = 0,
        verbose: bool = False,
) -> Tuple[CameraView, np.ndarray]:

    """Given two views represented as depth/prediction images +
    camera parameters, run view-view interpolation in image space.

    :param source_view: the view to interpolate from
    :param target_view: the view to interpolate to (only depth here is used)
    :param verbose: be verbose (more prints)
    :return: a view
    """
    source_view = source_view.copy()
    source_view.depth = remove_boundary_by_erosion(source_view.depth, boundary_width_to_remove)
    source_view.signal = remove_boundary_by_erosion(source_view.signal, boundary_width_to_remove)

    # bring target view to the image coordinate space
    # shared with source view
    source_view = source_view.to_image()
    target_view = target_view.reproject_to(source_view)

    target_signal, valid_mask = interpolate.pointwise_interpolate_image(
        source_view.depth,
        source_view.signal,
        target_view.depth,
        distance_interp_thr=distance_interp_thr,
        nn_set_size=nn_set_size,
        z_distance_threshold=z_distance_threshold,
        interp_ratio_thr=interp_ratio_thr,
        verbose=verbose)
    target_view.signal = target_signal

    return target_view, valid_mask


def interpolate_pixels(

):
    pass


def interpolate_3d_points(

):
    pass
