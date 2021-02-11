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


HIGH_RES = 0.02


def main(options):
    name = os.path.splitext(os.path.basename(options.true_filename))[0]

    # load ground truth and save to a single patch
    ground_truth_dataset = Hdf5File(
        options.true_filename,
        io=sharpf_io.WholeDepthMapIO,
        preload=PreloadTypes.LAZY,
        labels='*')
    ground_truth = [view for view in ground_truth_dataset]
    gt_images = [view['image'] for view in ground_truth]
    gt_distances = [view.get('distances', np.ones_like(view['image'])) for view in ground_truth]
    gt_cameras = [view['camera_pose'] for view in ground_truth]

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)
        config['imaging']['resolution_image'] = gt_images[0].shape[0]
    imaging = load_func_from_config(IMAGING_BY_TYPE, config['imaging'])
    print(imaging.resolution_image, gt_images[0].shape)


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
        imaging, gt_cameras, gt_images, list_predictions, verbose=options.verbose,
        distance_interpolation_threshold=imaging.resolution_3d * 6.)

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
