from functools import partial
import itertools
from typing import Callable, List, Mapping, Tuple

import numpy as np

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


def image_pair_interpolate_image(
        i,
        j,
        view_i_info: Tuple,
        view_j_info: Tuple,
        verbose: bool = False,
):

    n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
    image_space_tree = cKDTree(imaging.rays_origins[:, :2], leafsize=100)

    list_predictions, list_indexes_in_whole, list_points = [], [], []
    n_points_per_image = np.cumsum([len(np.nonzero(image.ravel())[0]) for image in images])

    # Propagating predictions from view i into all other views
    pose_i, image_i, points_i, predictions_i = view_i_info

    start_idx, end_idx = (0, n_points_per_image[j]) if 0 == j \
        else (n_points_per_image[j - 1], n_points_per_image[j])
    indexes_in_whole = np.arange(start_idx, end_idx)

    if verbose:
        eprint_t('Propagating views {} -> {}'.format(i, j))

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


def multi_view_interpolate_image(
        images: List[np.array],
        predictions: List[np.array],
        camera_parameters: List,
        camera_type: str = 'orthogonal',
        n_jobs: int = 1,
        verbose: bool = False,
        distance_interpolation_threshold: float = HIGH_RES * 6.
):
    """Given a set of image-prediction pairs,
    interpolates predictions between views in the image space
    (i.e. 3d points are not discretized into pixels).

    :param images: list of depth images
    :param predictions: list of prediction images
    :param camera_type: string identifying camera specification to use
                        (currently: either 'orthogonal' or 'rangevision')
                        TODO: refactor camera classes to use a more generalised code
    :param camera_parameters: list of camera parameters, interpreted differently depending
                              on the `camera_type` argument:
                              <ADD HERE>
    :param n_jobs: number of CPU threads to use
    :param verbose: if true, prints debugging information during fusion steps
    :param distance_interpolation_threshold:
    :return:
    """

    if camera_type == 'orthogonal':
        get_view = get_orthogonal_view
    elif camera_type == 'rangevision':
        get_view = get_perspective_view
    else:
        raise ValueError('Unknown camera type: {}'.format(camera_type))

    get_view = partial(
        get_view,
        images=images,
        predictions=predictions,
        camera_parameters=camera_parameters)


    list_predictions, list_indexes_in_whole, list_points = [], [], []

    n_images = len(images)
    n_cum_points_per_image = np.cumsum([
        len(np.nonzero(image.ravel())[0])
        for image, t in zip(gt_images_, transforms_) if None is not t])

    data_iterable = (
        (i, j, get_view(i), get_view(j), n_cum_points_per_image, verbose)
        for i, j in itertools.product(range(n_images), range(n_images)))

    for predictions, indexes_in_whole, points in multiproc_parallel(
            image_pair_interpolate_image,
            data_iterable):
        list_points.append(points)
        list_predictions.append(predictions)
        list_indexes_in_whole.append(indexes_in_whole)

    return list_predictions, list_indexes_in_whole, list_points




