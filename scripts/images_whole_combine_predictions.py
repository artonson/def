#!/usr/bin/env python3

import argparse
import itertools
from functools import partial
import glob
import os
import re
import sys
from typing import List, Mapping

import h5py
import numpy as np
from scipy.interpolate import fitpack
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy import interpolate
from sklearn.linear_model import HuberRegressor

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)
sys.path[1:1] = [__dir__]

from sharpf.utils.py_utils.parallel import multiproc_parallel
import sharpf.data.datasets.sharpf_io as sharpf_io
from sharpf.data.imaging import RaycastingImaging
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
import sharpf.utils.abc_utils.hdf5.io_struct as io_struct
from sharpf.utils.camera_utils.camera_pose import CameraPose
import sharpf.consolidation.combiners as cc
import sharpf.consolidation.smoothers as cs


SingleViewPredictionsIO = io_struct.HDF5IO(
    {'image': io_struct.Float64('image'),
     'distances': io_struct.Float64('distances')},
    len_label='distances',
    compression='lzf')


def save_predictions(patches, filename):
    collate_fn = partial(io_struct.collate_mapping_with_io, io=SingleViewPredictionsIO)
    patches = collate_fn(patches)
    with h5py.File(filename, 'w') as f:
        SingleViewPredictionsIO.write(f, 'image', patches['image'])
        SingleViewPredictionsIO.write(f, 'distances', patches['distances'])


def convert_npylist_to_hdf5(input_dir, gt_images, output_filename):
    def get_num(basename):
        match = re.match('^test_(\d+)\.npy$', basename)
        return int(match.groups()[0])

    datafiles = glob.glob(os.path.join(input_dir, '*.npy'))
    datafiles.sort(key=lambda name: get_num(os.path.basename(name)))
    patches = [
        {'image': image, 'distances': np.load(f)}
        for image, f in zip(gt_images, datafiles)]
    save_predictions(patches, output_filename)


def save_full_model_predictions(points, predictions, filename):
    PointPatchPredictionsIO = io_struct.HDF5IO(
        {'points': io_struct.Float64('points'),
         'distances': io_struct.Float64('distances')},
        len_label='distances',
        compression='lzf')
    with h5py.File(filename, 'w') as f:
        PointPatchPredictionsIO.write(f, 'points', [points])
        PointPatchPredictionsIO.write(f, 'distances', [predictions])


def bisplrep_interpolate(xs, ys, vs, xt, yt):
    x_min, x_max = np.amin(xs), np.amax(xs)
    y_min, y_max = np.amin(ys), np.amax(ys)

    out_of_bounds_x = (xt < x_min) | (xt > x_max)
    out_of_bounds_y = (yt < y_min) | (yt > y_max)

    any_out_of_bounds_x = np.any(out_of_bounds_x)
    any_out_of_bounds_y = np.any(out_of_bounds_y)

    if any_out_of_bounds_x or any_out_of_bounds_y:
        raise ValueError("Values out of range; x must be in %r, y in %r"
                         % ((x_min, x_max),
                            (y_min, y_max)))

    tck = fitpack.bisplrep(
        xs.squeeze(),
        ys.squeeze(),
        vs.squeeze(),
        kx=1,
        ky=1,
        s=len(xs.squeeze()))
    vt = fitpack.bisplev(xt, yt, tck)
    return vt


def interp2d_interpolate(xs, ys, vs, xt, yt):
    interpolator = interpolate.interp2d(
        xs.squeeze(),
        ys.squeeze(),
        vs.squeeze(),
        kind='linear',
        bounds_error=True)

    vt = interpolator(xt, yt)[0]
    return vt


def do_interpolate(
        i,
        j,
        view_i,
        view_j,
        point_indexes,
        interp_params,
):
    distance_interpolation_threshold = interp_params['distance_interpolation_threshold']
    nn_set_size = interp_params['nn_set_size']
    verbose = interp_params['verbose']
    interpolator_function = interp_params['interpolator_function']

    image_i, distances_i, points_i, pose_i, imaging_i = view_i
    _, distances_j, points_j, _, _ = view_j

    start_idx, end_idx = (0, point_indexes[j]) if 0 == j \
        else (point_indexes[j - 1], point_indexes[j])
    indexes_j = np.arange(start_idx, end_idx)

    interp_fn = {
        'bisplrep': bisplrep_interpolate,
        'interp2d': interp2d_interpolate,
    }[interpolator_function]

    if i == j:
        predictions_interp, indexes_interp, points_interp = \
            distances_i[image_i != 0.].ravel(), indexes_j, points_i

    else:
        # reproject points from one view j to view i, to be able to interpolate in view i
        reprojected_j = pose_i.world_to_camera(points_j)

        # extract pixel indexes in view i (for each reprojected points),
        # these are source pixels to interpolate from
        uv_i = imaging_i.rays_origins[:, :2]
        _, nn_indexes_in_i = cKDTree(uv_i).query(reprojected_j[:, :2], k=nn_set_size)

        # Create interpolation mask: True for points which
        # can be stably interpolated (i.e. they have K neighbours present
        # within a predefined radius).
        interp_mask = np.zeros(len(reprojected_j)).astype(bool)
        # Distances to be produces as output.
        distances_j_interp = np.zeros(len(points_j), dtype=float)

        for idx, point_from_j in tqdm(enumerate(reprojected_j)):
            point_nn_indexes = nn_indexes_in_i[idx]
            # Build an [n, 3] array of XYZ coordinates for each reprojected point by taking
            # UV values from pixel grid and Z value from depth image.
            point_from_j_nns = np.hstack((
                uv_i[point_nn_indexes],
                np.atleast_2d(image_i.ravel()[point_nn_indexes]).T
            ))

            distances_to_nearest = np.linalg.norm(point_from_j - point_from_j_nns, axis=1)
            interp_mask[idx] = np.all(distances_to_nearest < distance_interpolation_threshold)

            if interp_mask[idx]:
                # Actually perform interpolation
                try:
                    distances_j_interp[idx] = interp_fn(
                        point_from_j_nns[:, 0],
                        point_from_j_nns[:, 1],
                        distances_i.ravel()[point_nn_indexes],
                        point_from_j[0],
                        point_from_j[1]
                    )

                except ValueError as e:
                    if verbose:
                        print('Error while interpolating point {idx}:'
                              '{what}, skipping this point'.format(idx=idx, what=str(e)))
                    interp_mask[idx] = False

                except RuntimeWarning:
                    break

        points_interp = points_j[interp_mask]
        indexes_interp = indexes_j[interp_mask]
        predictions_interp = distances_j_interp[interp_mask]

    return predictions_interp, indexes_interp, points_interp


def get_view(
        images: List[np.array],
        distances: List[np.array],
        extrinsics: List[np.array],
        intrinsics_dict: List[Mapping],
        i):
    """A helper function to conveniently prepare view information."""
    image_i = images[i]  # [h, w]
    distances_image_i = distances[i]  # [h, w]
    # Kill background for nicer visuals
    distances_i = np.zeros_like(distances_image_i)
    distances_i[np.nonzero(image_i)] = distances_image_i[np.nonzero(image_i)]

    pose_i = CameraPose(extrinsics[i])
    imaging_i = RaycastingImaging(**intrinsics_dict[i])
    points_i = pose_i.camera_to_world(imaging_i.image_to_points(image_i))

    return image_i, distances_i, points_i, pose_i, imaging_i


def multi_view_interpolate_predictions(
        images: List[np.array],
        distances: List[np.array],
        extrinsics: List[np.array],
        intrinsics_dict: List[Mapping],
        **config,
):
    get_view_local = partial(get_view, images, distances, extrinsics, intrinsics_dict)

    point_indexes = np.cumsum([
        np.count_nonzero(image) for image in images])

    n_jobs = config.get('n_jobs')
    n_images = len(images)
    data_iterable = (
        (i, j, get_view_local(i), get_view_local(j), point_indexes, config)
        for i, j in itertools.product(range(n_images), range(n_images)))

    os.environ['OMP_NUM_THREADS'] = str(n_jobs)
    list_predictions, list_indexes_in_whole, list_points = [], [], []
    for predictions_interp, indexes_interp, points_interp in multiproc_parallel(
            do_interpolate,
            data_iterable):
        list_predictions.append(predictions_interp)
        list_indexes_in_whole.append(indexes_interp)
        list_points.append(points_interp)

    return list_predictions, list_indexes_in_whole, list_points


def interpolate_ground_truth(
        images: List[np.array],
        distances: List[np.array],
        extrinsics: List[np.array],
        intrinsics_dict: List[Mapping],
):
    # Partially specify view extraction parameters.
    get_view_local = partial(get_view, images, distances, extrinsics, intrinsics_dict)

    fused_points_gt = []
    fused_predictions_gt = []
    for view_index in range(len(images)):
        image_i, distances_i, points_i, pose_i, imaging_i = get_view_local(view_index)
        fused_points_gt.append(points_i)
        fused_predictions_gt.append(distances_i.ravel()[np.flatnonzero(image_i)])

    fused_points_gt = np.concatenate(fused_points_gt)
    fused_predictions_gt = np.concatenate(fused_predictions_gt)

    return fused_points_gt, fused_predictions_gt


def main(options):
    name = os.path.splitext(os.path.basename(options.true_filename))[0]

    print('Loading ground truth data...')
    ground_truth_dataset = Hdf5File(
        options.true_filename,
        io=sharpf_io.WholeDepthMapIO,
        preload=PreloadTypes.LAZY,
        labels='*')
    gt_dataset = [view for view in ground_truth_dataset]
    # depth images captured from a variety of views around the 3D shape
    gt_images = [view['image'] for view in gt_dataset]
    # ground-truth distances (multi-view consistent for the global 3D shape)
    gt_distances = [view.get('distances', np.ones_like(view['image'])) for view in gt_dataset]
    # extrinsic camera matrixes describing the 3D camera poses used to capture depth images
    gt_extrinsics = [view['camera_pose'] for view in gt_dataset]
    # intrinsic camera parameters describing how to compute image from points and vice versa
    gt_intrinsics = [dict(resolution_image=gt_images[0].shape, resolution_3d=options.resolution_3d) for view in
                     gt_dataset]

    print('Fusing ground truth data...')
    fused_points_gt, fused_distances_gt = interpolate_ground_truth(
        gt_images,
        gt_distances,
        gt_extrinsics,
        gt_intrinsics)
    n_points = len(fused_points_gt)

    gt_output_filename = os.path.join(
        options.output_dir,
        '{}__{}.hdf5'.format(name, 'ground_truth'))
    print('Saving ground truth to {}'.format(gt_output_filename))
    save_full_model_predictions(
        fused_points_gt,
        fused_distances_gt,
        gt_output_filename)

    print('Loading predictions...')
    predictions_filename = os.path.join(
        options.output_dir,
        '{}__{}.hdf5'.format(name, 'predictions'))
    convert_npylist_to_hdf5(options.pred_dir, gt_images, predictions_filename)
    predictions_dataset = Hdf5File(
        predictions_filename,
        io=sharpf_io.WholeDepthMapIO,
        preload=PreloadTypes.LAZY,
        labels='*')
    pred_distances = [view['distances'] for view in predictions_dataset]

    diff_images = [{
        'image': image,
        'distances': np.abs(gt - predictions)}
        for image, predictions, gt in zip(gt_images, pred_distances, gt_distances)]
    diff_filename = os.path.join(
        options.output_dir,
        '{}__{}.hdf5'.format(name, 'absdiff'))
    save_predictions(diff_images, diff_filename)

    threshold = options.resolution_3d * options.distance_interp_factor
    config = {
        'verbose': options.verbose,
        'n_jobs': options.n_jobs,
        'nn_set_size': options.nn_set_size,
        'distance_interpolation_threshold': threshold,
        'interpolator_function': options.interpolator_function,
    }
    list_predictions, list_indexes_in_whole, list_points = multi_view_interpolate_predictions(
        gt_images,
        pred_distances,
        gt_extrinsics,
        gt_intrinsics,
        **config)

    # run various algorithms for consolidating predictions
    combiners = [
#       MedianPredictionsCombiner(),
        cc.MinPredictionsCombiner(),
#       AvgPredictionsCombiner(),
        cc.TruncatedAvgPredictionsCombiner(func=np.min),
#       MinsAvgPredictionsCombiner(signal_thr=0.9),
#       SmoothingCombiner(
#           combiner=MinPredictionsCombiner(),
#           smoother=L2Smoother(regularizer_alpha=0.01)
#       ),
#       SmoothingCombiner(
#           combiner=MinPredictionsCombiner(),
#           smoother=TotalVariationSmoother(regularizer_alpha=0.001)
#       ),
      cc.SmoothingCombiner(
          cc.TruncatedAvgPredictionsCombiner(func=np.min),
          smoother=cs.RobustLocalLinearFit(
              HuberRegressor(epsilon=4., alpha=1.),
              n_jobs=32)),
    ]

    for combiner in combiners:
        print('Fusing predictions...')
        fused_points_pred, fused_distances_pred, prediction_variants = \
            combiner(n_points, list_predictions, list_indexes_in_whole, list_points)

        pred_output_filename = os.path.join(
            options.output_dir,
            '{}__{}.hdf5'.format(name, combiner.tag))
        print('Saving preds to {}'.format(pred_output_filename))
        save_full_model_predictions(
            fused_points_pred,
            fused_distances_pred,
            pred_output_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')

    parser.add_argument('-t', '--true-filename', dest='true_filename', required=True,
                        help='Path to GT file with whole model point patches.')
    parser.add_argument('-p', '--pred-dir', dest='pred_dir', required=True,
                        help='Path to prediction directory with npy files.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True,
                        help='Path to output (suffixes indicating various methods will be added).')

    parser.add_argument('-j', '--jobs', dest='n_jobs', default=4,
                        required=False, help='number of jobs to use for fusion.')

    parser.add_argument('-k', '--nn_set_size', dest='nn_set_size', required=False, default=4, type=int,
                        help='Number of neighbors used for interpolation.')
    parser.add_argument('-r', '--resolution_3d', dest='resolution_3d', required=False, default=0.02, type=float,
                        help='3D resolution of scans.')
    parser.add_argument('-f', '--distance_interp_factor', dest='distance_interp_factor', required=False, type=float, default=6.,
                        help='distance_interp_factor * resolution_3d is the distance_interpolation_threshold')
    parser.add_argument('-l', '--interpolator_function', dest='interpolator_function',
                        required=False, choices=['bisplrep', 'interp2d'], default='bisplrep',
                        help='interpolator function to use.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
