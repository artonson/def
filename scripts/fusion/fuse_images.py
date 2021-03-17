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
import os
import sys

import numpy as np
import sklearn.linear_model as lm
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../..')
)
sys.path[1:1] = [__dir__]

import sharpf.data.datasets.sharpf_io as sharpf_io
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.py_utils.config import load_func_from_config
from sharpf.fusion.images import interpolators
import sharpf.fusion.combiners as combiners
import sharpf.fusion.smoothers as smoothers
import sharpf.fusion.io as fusion_io
import sharpf.utils.convertor_utils.convertors_io as convertors_io
from sharpf.fusion.images.interpolators import MVS_INTERPOLATORS
from sharpf.utils.camera_utils.view import CameraView
import sharpf.utils.convertor_utils.rangevision_utils as rv_utils


def load_ground_truth(true_filename, unlabeled=False):
    """Loads a specified representation of """
    data_io = convertors_io.AnnotatedViewIO

    # load ground truth and save to a single patch
    ground_truth_dataset = Hdf5File(
        true_filename,
        io=data_io,
        preload=PreloadTypes.LAZY,
        labels='*')

    views = [
        CameraView(
            depth=scan['points'],
            signal=scan['distances'],
            faces=scan['faces'].reshape((-1, 3)),
            extrinsics=np.dot(scan['points_alignment'], scan['extrinsics']),
            intrinsics=scan['intrinsics'],
            state='pixels')
        for scan in ground_truth_dataset]
    view_alignments = [scan['points_alignment'] for scan in ground_truth_dataset]
    return views, view_alignments


def load_predictions(pred_data, name, pred_key='distances'):
    if os.path.isdir(pred_data):
        predictions_filename = os.path.join(
            options.output_dir,
            '{}__{}.hdf5'.format(name, 'predictions'))
        fusion_io.convert_npylist_to_hdf5(
            pred_data,
            predictions_filename,
            fusion_io.ImagePredictionsIO)

    else:
        assert os.path.isfile(pred_data)
        predictions_filename = pred_data

    if pred_key is not None:
        data_io = fusion_io.ComparisonsIO
    else:
        data_io = sharpf_io.WholeDepthMapIO

    predictions_dataset = Hdf5File(
        predictions_filename,
        io=data_io,
        preload=PreloadTypes.LAZY,
        labels=[pred_key])
    list_predictions = [patch[pred_key] for patch in predictions_dataset]

    return list_predictions


def main(options):
    name = os.path.splitext(os.path.basename(options.true_filename))[0]

    # Load the input ground-truth dataset (this holds
    # per-view ground truth depth images, distances, and
    # view parameters). If running inference, distances
    # can be absent (run with --unlabeled flag).
    views, view_alignments = load_ground_truth(
        options.true_filename,
        unlabeled=options.unlabeled)

    # Compute the fused point cloud trivially:
    # perform un-projection and transform all points into
    # world coordinate frame. If running inference, distances
    # can be absent (run with --unlabeled flag).
    list_predictions, list_indexes_in_whole, list_points = \
        interpolators.GroundTruthInterpolator()(views)
    n_points = np.sum([len(points) for points in list_points])
    fused_points_gt, fused_distances_gt, prediction_variants_gt = combiners.GroundTruthCombiner()(
        n_points,
        list_predictions,
        list_indexes_in_whole,
        list_points,
        max_distance=options.max_distance_to_feature)

    ground_truth_filename = os.path.join(
        options.output_dir,
        '{}__{}.hdf5'.format(name, 'ground_truth'))
    fusion_io.save_full_model_predictions(
        fused_points_gt,
        fused_distances_gt,
        ground_truth_filename)

    # This assumes we have predictions for each of the images
    # in the ground-truth file.
    list_predictions = load_predictions(
        options.pred_dir,
        name,
        pred_key=options.pred_key or 'distances')
    for view, predictions in zip(views, list_predictions):
        full_h, full_w = rv_utils.RV_SPECTRUM_CAM_RESOLUTION[::-1]
        h, w = predictions.shape
        if (full_h, full_w) != (h, w):
            predictions_uncropped = np.zeros((full_h, full_w))
            predictions_uncropped[
                slice(full_h // 2 - h // 2, full_h // 2 + h // 2),
                slice(full_w // 2 - w // 2, full_w // 2 + w // 2)] = predictions
        else:
            predictions = predictions_uncropped
        view.signal = predictions_uncropped

    # Run fusion of predictions using multiple view
    # interpolator algorithm.
    with open(options.fusion_config, 'r') as yml_file:
        config = yaml.load(yml_file.read(), Loader=yaml.Loader)
    config.update({
        'verbose': options.verbose,
        'n_jobs': options.n_jobs
    })
    mvs_interpolator = load_func_from_config(MVS_INTERPOLATORS, config)
    list_predictions, list_indexes_in_whole, list_points = mvs_interpolator(views)

    # run various algorithms for consolidating predictions
    # this selection is up to you, user
    combiners_list = [
#       combiners.MedianPredictionsCombiner(),
#       combiners.MinPredictionsCombiner(),
#       combiners.AvgPredictionsCombiner(),
        combiners.TruncatedAvgPredictionsCombiner(
            func=combiners.TruncatedMean(0.6, func=np.min)),
#       combiners.MinsAvgPredictionsCombiner(signal_thr=0.9),
#       combiners.SmoothingCombiner(
#           combiner=combiners.MinPredictionsCombiner(),
#           smoother=smoothers.L2Smoother(regularizer_alpha=0.01)
#       ),
#       combiners.SmoothingCombiner(
#           combiner=combiners.MinPredictionsCombiner(),
#           smoother=smoothers.TotalVariationSmoother(regularizer_alpha=0.001)
#       ),
#       combiners.SmoothingCombiner(
#           combiner=combiners.MinPredictionsCombiner(),
#           smoother=smoothers.RobustLocalLinearFit(
#               lm.HuberRegressor(epsilon=4., alpha=1.),
#               n_jobs=32
#           )
#       ),
    ]

    for combiner in combiners_list:
        if options.verbose:
            print('Running {}'.format(combiner.tag))

        fused_points_pred, fused_distances_pred, _ = combiner(
            n_points,
            list_predictions,
            list_indexes_in_whole,
            list_points,
            max_distance=options.max_distance_to_feature)

        output_filename = os.path.join(
            options.output_dir,
            '{}__{}.hdf5'.format(name, combiner.tag))
        fusion_io.save_full_model_predictions(
            fused_points_pred,
            fused_distances_pred,
            output_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')

    parser.add_argument('-t', '--true-filename', dest='true_filename', required=True,
                        help='Path to GT file with whole model point patches.')
    parser.add_argument('-p', '--pred-dir', dest='pred_dir', required=True,
                        help='Path to prediction directory with npy files.')
    parser.add_argument('-k', '--pred-key', dest='pred_key', required=False,
                        help='if set, switch to compare-io and use this key.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True,
                        help='Path to output (suffixes indicating various methods will be added).')
    parser.add_argument('-f', '--fusion-config', dest='fusion_config',
                        required=True, help='fusion configuration YAML file.')
    parser.add_argument('-j', '--jobs', dest='n_jobs', default=4,
                        required=False, help='number of jobs to use for fusion.')
    parser.add_argument('-u', '--unlabeled', dest='unlabeled', action='store_true', default=False,
                        help='set if input data is unlabeled.')
    parser.add_argument('-s', '--max_distance_to_feature', dest='max_distance_to_feature',
                        default=1.0, type=float, required=False, help='max distance to sharp feature to compute.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
