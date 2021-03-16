from sharpf.fusion.images import interpolate
from sharpf.utils.camera_utils.view import CameraView


def interpolate_views_as_images(
        source_view: CameraView,
        target_view: CameraView,
        distance_interp_thr: float = 1.0,
        nn_set_size: int = 8,
        z_distance_threshold: int = 2.0,
        verbose: bool = False,
) -> CameraView:

    """Given two views represented as depth/prediction images +
    camera parameters, run view-view interpolation in image space.

    :param source_view: the view to interpolate from
    :param target_view: the view to interpolate to (only depth here is used)
    :param verbose: be verbose (more prints)
    :return: a view
    """

    # bring target view to the image coordinate space
    # shared with source view
    source_view = source_view.to_image()
    target_view = target_view.reproject_to(source_view)

    target_signal, can_interpolate = interpolate.pointwise_interpolate_image(
        source_view.depth,
        source_view.signal,
        target_view.depth,
        distance_interp_thr=distance_interp_thr,
        nn_set_size=nn_set_size,
        z_distance_threshold=z_distance_threshold,
        verbose=verbose)
    target_view.depth = target_view.depth[can_interpolate]
    target_view.signal = target_signal[can_interpolate]

    return target_view


def interpolate_pixels(

):
    pass


def interpolate_3d_points(

):
    pass
