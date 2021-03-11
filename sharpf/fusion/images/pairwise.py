from abc import abstractmethod
from functools import partial
import itertools
from typing import Callable, List, Mapping, Tuple

import numpy as np

from sharpf.utils.camera_utils.view import CameraView, view_to_points
from sharpf.utils.py_utils.parallel import multiproc_parallel


def get_orthogonal_view(
        view_index,
        images,
        predictions,
        camera_parameters,
):
    # use exterior variables
    pose_i = CameraPose(gt_cameras[view_index])
    image_i = images[view_index]
    points_i = pose_i.camera_to_world(imaging.image_to_points(image_i))
    predictions_i = np.zeros_like(image_i)
    predictions_i[image_i != 0.] = predictions[view_index][image_i != 0.]
    return pose_i, image_i, points_i, predictions_i


def get_perspective_view(
        view_index
):
    # use exterior variables
    pose_i = CameraPose(gt_extrinsics_[view_index])
    Kf_i, Ks_i = gt_intrinsics_f_[i], gt_intrinsics_s_[i]
    offset_i = gt_offsets_[i].astype(int)
    image_i = gt_images_[i]

    image_crop_i = data_crop(image_i)
    _, _, _, points_i = reproject_image_to_points(
        image_crop_i, pose_i, Kf_i, Ks_i, offset_i)

    transform_i = transforms_[i]
    assert None is not transform_i
    points_i = tt.transform_points(points_i, transform_i, translate=True)

    predictions_i = np.zeros_like(image_i)
    predictions_i[image_i != 0.] = pred_images_distances_masked_[i][image_i != 0.]
    return (transform_i, pose_i, Kf_i, Ks_i, offset_i), image_i, points_i, predictions_i


def interpolate_wrapper(f):
    # source_view_idx,
    # target_view_idx,
    # if verbose:
    #     eprint_t('Propagating views {} -> {}'.format(i, j))
    pass


def interpolate_views_as_images(
        source_view: CameraView,
        target_view: CameraView,
        verbose: bool = False,
) -> CameraView:
    """Given two views represented as depth/prediction images +
    camera parameters, run view-view interpolation in image space.

    :param source_view:
    :param target_view:
    :param verbose:
    :return: a view

    """


    if source_view_idx == target_view_idx:
        source_view = source_view.to_points()
        source_indexes = np.arange(len(source_view.depth))
        return source_view.signal, source_indexes, source_view.depth

    source_view = source_view.to_image()
    target_view = target_view.reproject_to(source_view)

    image_pixelizer = load_func_from_config(source_view)
    source_image = image_pixelizer.unpixelize(source_view.image)

    image_pixelizer = load_func_from_config(target_view)
    target_image = image_pixelizer.unpixelize(target_view.image)

    target_view = target_view.reproject_to(source_view)
    target_view = target_view.to_image(to_other=source_view)

    target_signal, can_interpolate = pointwise_interpolate_image(
        source_view.depth,
        source_view.signal,
        target_view.depth,
    )

    camera_pose = CameraPose(self.view.extrinsics)
    camera_projection = load_func_from_config()
    image_pixelizer = load_func_from_config()

    image = image_pixelizer.unpixelize(view.image)
    points_in_camera_frame = camera_projection.unproject(image)
    points_in_world_frame = camera_pose.camera_to_world(points_in_camera_frame)

    n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
    image_space_tree = cKDTree(imaging.rays_origins[:, :2], leafsize=100)

    list_predictions, list_indexes_in_whole, list_points = [], [], []
    n_points_per_image = np.cumsum([len(np.nonzero(image.ravel())[0]) for image in images])

    # Propagating predictions from view i into all other views
    pose_i, image_i, points_i, predictions_i = view_i_info

    start_idx, end_idx = (0, n_points_per_image[j]) if 0 == j \
        else (n_points_per_image[j - 1], n_points_per_image[j])
    indexes_in_whole = np.arange(start_idx, end_idx)


    if i == j:
        list_points.append(points_i)
        list_predictions.append(predictions_i[image_i != 0.].ravel())
        list_indexes_in_whole.append(indexes_in_whole)

    else:
        pose_j, image_j, points_j, predictions_j = view_j_info

        # reproject points from one view j to view i, to be able to interpolate in view i
        reprojected = pose_i.world_to_camera(points_j)

        # extract pixel indexes in view i (for each reprojected points),
        # these are source pixels to interpolate from
        _, nn_indexes = image_space_tree.query(
            reprojected[:, :2],
            k=4,
            n_jobs=n_omp_threads)

        can_interpolate = np.zeros(len(reprojected)).astype(bool)
        interpolated_distances_j = np.zeros_like(can_interpolate).astype(float)

        for idx, reprojected_point in enumerate(reprojected):
            # build neighbourhood of each reprojected point by taking
            # xy values from pixel grid and z value from depth image
            nbhood_of_reprojected = np.hstack((
                imaging.rays_origins[:, :2][nn_indexes[idx]],
                np.atleast_2d(image_i.ravel()[nn_indexes[idx]]).T
            ))
            distances_to_nearest = np.linalg.norm(reprojected_point - nbhood_of_reprojected, axis=1)
            can_interpolate[idx] = np.all(distances_to_nearest < distance_interpolation_threshold)

            if can_interpolate[idx]:
                try:
                    interpolator = interpolate.interp2d(
                        nbhood_of_reprojected[:, 0],
                        nbhood_of_reprojected[:, 1],
                        predictions_i.ravel()[nn_indexes[idx]],
                        kind='linear')
                    interpolated_distances_j[idx] = interpolator(
                        reprojected_point[0],
                        reprojected_point[1])[0]
                except ValueError as e:
                    eprint_t('Error while interpolating point {idx}: {what}, skipping this point'.format(
                        idx=idx, what=str(e)))
                    can_interpolate[idx] = False


        list_points.append(points_j[can_interpolate])
        list_predictions.append(interpolated_distances_j[can_interpolate])
        list_indexes_in_whole.append(indexes_in_whole[can_interpolate])

    return list_predictions, list_indexes_in_whole, list_points


def interpolate_pixels(

):
    pass


def interpolate_3d_points(

):
    pass
