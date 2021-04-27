#!/usr/bin/env python3

# Given an input dir with predictions (list of npy files)
# and an input file with ground truth (multiple point patches),
# this script:
#  1) builds a single file with ground truth predictions
#  2) runs a series of various prediction consolidation algorithms,
#     saving consolidated predictions into an output folder
#
# Convenient file structure for using this script:
#

import argparse
import itertools
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from functools import partial
import glob
import os
import re
import sys
from typing import Callable, List, Mapping, Tuple

import h5py
import numpy as np
from scipy.interpolate import fitpack
from tqdm import tqdm, trange
import torch
from torch import optim
from scipy.spatial import cKDTree
from scipy.stats.mstats import mquantiles
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.preprocessing import PolynomialFeatures
from joblib import Parallel, delayed

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


class AveragingNoDataException(ValueError):
    pass


class TruncatedMean(object):
    def __init__(self, probability, func=np.mean):
        self._probability = probability
        self._func = func

    def __call__(self, values):
        quantiles = mquantiles(values,
                               [(1.0 - self._probability) / 2.0, (1.0 + self._probability) / 2.0],
                               alphap=0.5, betap=0.5)
        values = list(filter(lambda value: quantiles[0] <= value <= quantiles[1], values))
        if not values:
            raise AveragingNoDataException('No values after trimming')
        result = self._func(values)
        return result


class PredictionsCombiner:
    def __call__(
            self,
            n_points: int,
            list_predictions: List[np.array],
            list_indexes_in_whole: List[np.array],
            list_points: List[np.array],
    ) -> Tuple[np.array, Mapping]:

        raise NotImplemented()


class PointwisePredictionsCombiner(PredictionsCombiner):
    """Combines predictions where each point is predicted multiple times
    by taking a simple function of the point's predictions.

    Returns:
        whole_model_distances_pred: consolidated predictions
        predictions_variants: enumerates predictions per each point
        """

    def __init__(self, tag: str, func: Callable = np.median):
        self._func = func
        self.tag = tag

    def __call__(
            self,
            n_points: int,
            list_predictions: List[np.array],
            list_indexes_in_whole: List[np.array],
            list_points: List[np.array],
    ) -> Tuple[np.array, Mapping]:

        whole_model_distances_pred = np.ones(n_points) * np.inf

        # step 1: gather predictions
        predictions_variants = defaultdict(list)
        iterable = zip(list_predictions, list_indexes_in_whole, list_points)
        for distances, indexes_gt, points_gt in tqdm(iterable):
            for i, idx in enumerate(indexes_gt):
                predictions_variants[idx].append(distances[i])

        # step 2: consolidate predictions
        for idx, values in predictions_variants.items():
            whole_model_distances_pred[idx] = self._func(values)

        return whole_model_distances_pred, predictions_variants


class MedianPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self): super().__init__(func=np.median, tag='median')


class AvgPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self): super().__init__(func=np.mean, tag='mean')


class MinPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self): super().__init__(func=np.min, tag='min')


class TruncatedAvgPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self, func=np.mean):
        super().__init__(func=TruncatedMean(0.6, func=func), tag='adv60__min')


class MinsAvgPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self, signal_thr=0.9):
        def mean_of_mins(values):
            values = np.array(values)
            if len(values[values < signal_thr]) > 0:
                return np.mean(values[values < signal_thr])
            else:
                return np.mean(values)

        super().__init__(func=mean_of_mins, tag='mom')


class CenterCropPredictionsCombiner(PointwisePredictionsCombiner):
    """Makes a central crop, then performs the same as above."""

    def __init__(self, func: Callable = np.median, brd_thr=90):
        super().__init__(func=func, tag='crop__adv60')
        self._brd_thr = brd_thr

    def __call__(
            self,
            n_points: int,
            list_predictions: List[np.array],
            list_indexes_in_whole: List[np.array],
            list_points: List[np.array],
    ) -> Tuple[np.array, Mapping]:

        whole_model_distances_pred = np.ones(n_points) * np.inf

        # step 1: gather predictions
        predictions_variants = defaultdict(list)
        iterable = zip(list_predictions, list_indexes_in_whole, list_points)
        for distances, indexes_gt, points_gt in tqdm(iterable):
            # here comes difference from the previous variant
            points_radii = np.linalg.norm(points_gt - points_gt.mean(axis=0), axis=1)
            center_indexes = np.where(points_radii < np.percentile(points_radii, self._brd_thr))[0]
            for i, idx in enumerate(center_indexes):
                predictions_variants[indexes_gt[idx]].append(distances[center_indexes[i]])

        # step 2: consolidate predictions
        for idx, values in predictions_variants.items():
            whole_model_distances_pred[idx] = self._func(values)

        return whole_model_distances_pred, predictions_variants


class PredictionsSmoother(ABC):
    def __init__(self, tag='', n_neighbours=51):
        self.tag = tag
        self._n_neighbours = n_neighbours

    def __call__(
            self,
            predictions: np.array,
            points: np.array,
            predictions_variants: Mapping
    ) -> Tuple[np.array, Mapping]:

        n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
        nn_distances, nn_indexes = cKDTree(points, leafsize=100) \
            .query(points, k=self._n_neighbours, n_jobs=n_omp_threads)

        smoothed_predictions = self.perform_smoothing(
            predictions, points, predictions_variants, nn_distances, nn_indexes)
        return smoothed_predictions

    def perform_smoothing(
            self,
            predictions: np.array,
            points: np.array,
            predictions_variants: Mapping,
            nn_distances: np.array,
            nn_indexes: np.array
    ) -> np.array:

        raise NotImplemented()


class OptimizationBasedSmoother(PredictionsSmoother):
    def __init__(self, regularizer_alpha=0.01, tag='', n_neighbours=51):
        super().__init__(tag, n_neighbours)
        self._regularizer_alpha = regularizer_alpha

    def smoothing_loss(self, *args, **kwargs):
        raise NotImplemented()

    def perform_smoothing(self, predictions, points, predictions_variants, nn_distances, nn_indexes):
        init_predictions_th = torch.Tensor(predictions)
        predictions_th = torch.ones(predictions.shape)
        predictions_th.requires_grad_()

        optimizer = optim.SGD([predictions_th], lr=0.001, momentum=0.9)
        t = trange(300, desc='Optimization', leave=True)
        for i in t:
            optimizer.zero_grad()
            loss = self.smoothing_loss(predictions_th, init_predictions_th, nn_indexes, self._regularizer_alpha)
            loss.backward()
            optimizer.step()
            s = 'Optimization: step #{0:}, loss: {1:3.1f}'.format(i, loss.item())
            t.set_description(s)
            t.refresh()

        return predictions_th.detach().numpy()


class L2Smoother(OptimizationBasedSmoother):
    def __init__(self, regularizer_alpha=0.01):
        super().__init__(regularizer_alpha, tag='l2')

    def smoothing_loss(self, predictions, init_predictions, nn_indexes, alpha):
        data_fidelity_term = (predictions - init_predictions) ** 2
        regularization_term = torch.sum(
            (predictions[nn_indexes[:, 1:]] -
             predictions.reshape((len(predictions), 1))) ** 2,
            dim=1)
        return torch.sum(data_fidelity_term) + alpha * torch.sum(regularization_term)


class TotalVariationSmoother(OptimizationBasedSmoother):
    def __init__(self, regularizer_alpha=0.01):
        super().__init__(regularizer_alpha, tag='tv')

    def smoothing_loss(self, predictions, init_predictions, nn_indexes, alpha):
        data_fidelity_term = (predictions - init_predictions) ** 2
        regularization_term = torch.sum(
            torch.abs(
                predictions[nn_indexes[:, 1:]] -
                predictions.reshape((len(predictions), 1))),
            dim=1)
        return torch.sum(data_fidelity_term) + alpha * torch.sum(regularization_term)


class SmoothingCombiner(PredictionsCombiner):
    """Predict on a pointwise basis, then smoothen."""

    def __init__(self, combiner: PointwisePredictionsCombiner, smoother: PredictionsSmoother):
        self._combiner = combiner
        self._smoother = smoother
        self.tag = combiner.tag + '__' + smoother.tag

    def __call__(
            self,
            n_points: int,
            list_predictions: List[np.array],
            list_indexes_in_whole: List[np.array],
            list_points: List[np.array],
    ) -> Tuple[np.array, Mapping]:

        combined_predictions, predictions_variants = self._combiner(
            n_points, list_predictions, list_indexes_in_whole, list_points)

        points = np.zeros((n_points, 3))
        iterable = zip(list_indexes_in_whole, list_points)
        for indexes_gt, points_gt in tqdm(iterable):
            points[indexes_gt] = points_gt.reshape((-1, 3))
        combined_predictions = self._smoother(combined_predictions, points, predictions_variants)

        return combined_predictions, predictions_variants


class RobustLocalLinearFit(PredictionsSmoother):
    def __init__(self, estimator, n_jobs=1, n_neighbours=51):
        super().__init__(tag='linreg', n_neighbours=n_neighbours)
        self._n_jobs = n_jobs
        self._estimator = estimator

    def perform_smoothing(self, predictions, points, predictions_variants, nn_distances, nn_indexes):

        def make_xy(point_index, points, nn_indexes, predictions_variants):
            X, y = [], []
            for neighbour_index in nn_indexes[point_index]:
                for y_value in predictions_variants[neighbour_index]:
                    X.append(points[neighbour_index])
                    y.append(y_value)

            return np.array(X), np.array(y), np.unique(X, axis=0, return_index=True)[1]

        def data_maker(points, nn_indexes, predictions_variants):
            for point_index in range(len(points)):
                X, y, uniq_indexes = make_xy(point_index, points, nn_indexes, predictions_variants)
                yield point_index, X, y, uniq_indexes

        def local_linear_fit(X, y, estimator):
            X_trans = PCA(n_components=2).fit_transform(X)
            X_trans = PolynomialFeatures(2).fit_transform(X_trans)
            try:
                y_pred = estimator.fit(X_trans, y).predict(X_trans)
            except ValueError:
                y_pred = None
            return y_pred

        parallel = Parallel(n_jobs=self._n_jobs, backend='loky', verbose=100)
        delayed_iterable = (delayed(local_linear_fit)(X, y, deepcopy(self._estimator))
                            for point_index, X, y, uniq_indexes in data_maker(points, nn_indexes, predictions_variants))
        refined_predictions = parallel(delayed_iterable)

        refined_predictions_variants = defaultdict(list)
        for refined_prediction, (point_index, X, y, uniq_indexes) in tqdm(
                zip(refined_predictions, data_maker(points, nn_indexes, predictions_variants))):
            if None is refined_prediction:
                continue
            for i, nn_index in enumerate(nn_indexes[point_index]):
                refined_predictions_variants[nn_index].append(refined_prediction[uniq_indexes[i]])

        refined_combined_predictions = np.zeros_like(predictions)
        for idx, values in refined_predictions_variants.items():
            refined_combined_predictions[idx] = np.mean(values)

        return refined_combined_predictions



def convert_npylist_to_hdf5(input_dir, output_filename):
    PointPatchPredictionsIO = io_struct.HDF5IO(
        {'distances': io_struct.Float64('distances')},
        len_label='distances',
        compression='lzf')

    def save_predictions(patches, filename):
        collate_fn = partial(io_struct.collate_mapping_with_io, io=PointPatchPredictionsIO)
        patches = collate_fn(patches)
        with h5py.File(filename, 'w') as f:
            PointPatchPredictionsIO.write(f, 'distances', patches['distances'])

    def get_num(basename):
        match = re.match('^test_(\d+)\.npy$', basename)
        return int(match.groups()[0])

    datafiles = glob.glob(os.path.join(input_dir, '*.npy'))
    datafiles.sort(key=lambda name: get_num(os.path.basename(name)))
    patches = [{'distances': np.load(f)} for f in datafiles]
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
    imaging_i = RaycastingImaging(**intrinsics_dict[i], validate_image=False, projection='fuckit' )
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
    convert_npylist_to_hdf5(options.pred_dir, predictions_filename)
    predictions_dataset = Hdf5File(
        predictions_filename,
        io=sharpf_io.WholeDepthMapIO,
        preload=PreloadTypes.LAZY,
        labels='*')
    pred_distances = [view['distances'] for view in predictions_dataset]

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
#       MinPredictionsCombiner(),
#       AvgPredictionsCombiner(),
        TruncatedAvgPredictionsCombiner(func=np.min),
#       MinsAvgPredictionsCombiner(signal_thr=0.9),
#       SmoothingCombiner(
#           combiner=MinPredictionsCombiner(),
#           smoother=L2Smoother(regularizer_alpha=0.01)
#       ),
#       SmoothingCombiner(
#           combiner=MinPredictionsCombiner(),
#           smoother=TotalVariationSmoother(regularizer_alpha=0.001)
#       ),
#       SmoothingCombiner(
#           combiner=MinPredictionsCombiner(),
#           smoother=RobustLocalLinearFit(
#               HuberRegressor(epsilon=4., alpha=1.),
#               n_jobs=32
#           )
#       ),
    ]

    for combiner in combiners:
        print('Fusing predictions...')
        combined_predictions, \
        prediction_variants = \
            combiner(
                n_points,
                list_predictions,
                list_indexes_in_whole,
                list_points)

        pred_output_filename = os.path.join(
            options.output_dir,
            '{}__{}.hdf5'.format(name, combiner.tag))
        print('Saving preds to {}'.format(pred_output_filename))
        save_full_model_predictions(
            fused_points_gt,
            combined_predictions,
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


def main_errors(options):
    try: 
        main(options)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    options = parse_args()
    main(options)