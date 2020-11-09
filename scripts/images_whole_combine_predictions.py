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
import json
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

import sharpf.data.datasets.sharpf_io as sharpf_io
from sharpf.data.imaging import IMAGING_BY_TYPE, RaycastingImaging
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
import sharpf.utils.abc_utils.hdf5.io_struct as io_struct
from sharpf.utils.camera_utils.camera_pose import CameraPose
from sharpf.utils.py_utils.config import load_func_from_config
from sharpf.utils.py_utils.console import eprint_t


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
    def __init__(self): super().__init__(func=TruncatedMean(0.6), tag='adv60')


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
        super().__init__(func=func, tag='crop')
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


HIGH_RES = 0.02


def multi_view_interpolate_predictions(
        imaging: RaycastingImaging,
        gt_cameras: List[np.array],
        gt_images: List[np.array],
        predictions: List[np.array],
        verbose: bool = False,
        distance_interpolation_threshold: float = HIGH_RES * 6.
):
    """Interpolates predictions between views.

    :return:
    """
    def get_view(i):
        # use exterior variables
        pose_i = CameraPose(gt_cameras[i])
        image_i = gt_images[i]
        points_i = pose_i.camera_to_world(imaging.image_to_points(image_i))
        predictions_i = np.zeros_like(image_i)
        predictions_i[image_i != 0.] = predictions[i][image_i != 0.]
        return pose_i, image_i, points_i, predictions_i

    n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
    image_space_tree = cKDTree(imaging.rays_origins[:, :2], leafsize=100)

    list_predictions, list_indexes_in_whole, list_points = [], [], []
    n_points_per_image = np.array([len(np.nonzero(image.ravel())[0]) for image in gt_images])

    n_images = len(gt_images)
    for i in range(n_images):
        # Propagating predictions from view i into all other views
        pose_i, image_i, points_i, predictions_i = get_view(i)

        for j in range(n_images):
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
                pose_j, image_j, points_j, predictions_j = get_view(j)

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
                        interpolator = interpolate.interp2d(
                            nbhood_of_reprojected[:, 0],
                            nbhood_of_reprojected[:, 1],
                            predictions_i.ravel()[nn_indexes[idx]],
                            kind='linear')
                        interpolated_distances_j[idx] = interpolator(
                            reprojected_point[0],
                            reprojected_point[1])[0]

                list_points.append(points_j[can_interpolate])
                list_predictions.append(interpolated_distances_j[can_interpolate])
                list_indexes_in_whole.append(indexes_in_whole[can_interpolate])

    return list_predictions, list_indexes_in_whole, list_points


def main(options):
    name = os.path.splitext(os.path.basename(options.true_filename))[0]

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)
    imaging = load_func_from_config(IMAGING_BY_TYPE, config['imaging'])

    # load ground truth and save to a single patch
    ground_truth_dataset = Hdf5File(
        options.true_filename,
        io=sharpf_io.WholeDepthMapIO,
        preload=PreloadTypes.LAZY,
        labels='*')
    ground_truth = [view for view in ground_truth_dataset]
    gt_images = [view['image'] for view in ground_truth]
    gt_distances = [view['distances'] for view in ground_truth]
    gt_cameras = [view['camera_pose'] for view in ground_truth]

    n_points = int(np.sum([len(np.nonzero(image.ravel())[0]) for image in gt_images]))
    whole_model_points_gt = []
    whole_model_distances_gt = []
    for camera_to_world_4x4, image, distances in zip(gt_cameras, gt_images, gt_distances):
        points_in_world_frame = CameraPose(camera_to_world_4x4).camera_to_world(imaging.image_to_points(image))
        whole_model_points_gt.append(points_in_world_frame)
        whole_model_distances_gt.append(distances.ravel()[np.nonzero(image.ravel())[0]])
    whole_model_points_gt = np.concatenate(whole_model_points_gt)
    whole_model_distances_gt = np.concatenate(whole_model_distances_gt)

    ground_truth_filename = os.path.join(options.output_dir, '{}__{}.hdf5'.format(name, 'ground_truth'))
    save_full_model_predictions(whole_model_points_gt, whole_model_distances_gt, ground_truth_filename)

    # convert and load predictions
    predictions_filename = os.path.join(options.output_dir, '{}__{}.hdf5'.format(name, 'predictions'))
    convert_npylist_to_hdf5(options.pred_dir, predictions_filename)
    predictions_dataset = Hdf5File(
        predictions_filename,
        io=sharpf_io.WholeDepthMapIO,
        preload=PreloadTypes.LAZY,
        labels='*')
    list_predictions = [view['distances'] for view in predictions_dataset]

    list_predictions, list_indexes_in_whole, list_points = multi_view_interpolate_predictions(
        imaging, gt_cameras, gt_images, list_predictions, verbose=options.verbose)

    # run various algorithms for consolidating predictions
    combiners = [
        MedianPredictionsCombiner(),
        MinPredictionsCombiner(),
        AvgPredictionsCombiner(),
        TruncatedAvgPredictionsCombiner(),
        MinsAvgPredictionsCombiner(signal_thr=0.9),
        SmoothingCombiner(
            combiner=MinPredictionsCombiner(),
            smoother=L2Smoother(regularizer_alpha=0.01)
        ),
        SmoothingCombiner(
            combiner=MinPredictionsCombiner(),
            smoother=TotalVariationSmoother(regularizer_alpha=0.001)
        ),
        SmoothingCombiner(
            combiner=MinPredictionsCombiner(),
            smoother=RobustLocalLinearFit(
                HuberRegressor(epsilon=4., alpha=1.),
                n_jobs=32
            )
        ),
    ]

    for combiner in combiners:
        if options.verbose:
            print('Running {}'.format(combiner.tag))

        combined_predictions, prediction_variants = \
            combiner(n_points, list_predictions, list_indexes_in_whole, list_points)

        output_filename = os.path.join(options.output_dir, '{}__{}.hdf5'.format(name, combiner.tag))
        save_full_model_predictions(whole_model_points_gt, combined_predictions, output_filename)


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
    parser.add_argument('-g', '--dataset-config', dest='dataset_config',
                        required=True, help='dataset configuration file.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
