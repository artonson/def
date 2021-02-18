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


def load_ground_truth(true_filename, unlabeled=False):
    """Loads a specified representation of """
    if unlabeled:
        data_io = fusion_io.UnlabeledImageIO
    else:
        data_io = sharpf_io.WholeDepthMapIO

    # load ground truth and save to a single patch
    ground_truth_dataset = Hdf5File(
        true_filename,
        io=data_io,
        preload=PreloadTypes.LAZY,
        labels='*')
    ground_truth = [view for view in ground_truth_dataset]

    list_images = [view['image'] for view in ground_truth]
    list_distances = [view.get('distances', np.ones_like(view['image'])) for view in ground_truth]
    list_extrinsics = [view['camera_pose'] for view in ground_truth]
    list_intrinsics = [view['???'] for view in ground_truth]
    #
    # with open(options.dataset_config) as config_file:
    #     config = json.load(config_file)
    #     config['imaging']['resolution_image'] = gt_images[0].shape[0]
    # imaging = load_func_from_config(IMAGING_BY_TYPE, config['imaging'])
    # print(imaging.resolution_image, gt_images[0].shape)

    n_points = np.sum([
        np.count_nonzero(image) for image in list_images],
        dtype=np.int)
    return n_points, list_distances, list_images, list_extrinsics, list_intrinsics


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
    n_points, \
    list_images_preds, \
    list_images, \
    list_extrinsics, \
    list_intrinsics = load_ground_truth(
        options.true_filename,
        unlabeled=options.unlabeled)

    # Compute the fused point cloud trivially:
    # perform un-projection and transform all points into
    # world coordinate frame. If running inference, distances
    # can be absent (run with --unlabeled flag).
    list_predictions, list_indexes_in_whole, list_points = interpolators.GroundTruthInterpolator()(
        list_images_preds,
        list_images,
        list_extrinsics,
        list_intrinsics)
    fused_points_gt, fused_distances_gt, prediction_variants_gt = combiners.GroundTruthCombiner()(
        n_points,
        list_predictions,
        list_indexes_in_whole,
        list_points)

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
        options.pred_data,
        name,
        pred_key=options.pred_key or 'distances')

    # Run fusion of predictions using multiple view
    # interpolator algorithm.
    mvs_interpolator = load_func_from_config()
    # verbose = options.verbose,
    # distance_interpolation_threshold = imaging.resolution_3d * 6.
    list_predictions, list_indexes_in_whole, list_points = mvs_interpolator(
        n_points,
        list_predictions,
        list_images,
        list_extrinsics,
        list_intrinsics)

    # run various algorithms for consolidating predictions
    # this selection is up to you, user
    combiners_list = [
#       combiners.MedianPredictionsCombiner(),
#       combiners.MinPredictionsCombiner(),
#       combiners.AvgPredictionsCombiner(),
        combiners.TruncatedAvgPredictionsCombiner(),
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
            list_points)

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
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True,
                        help='Path to output (suffixes indicating various methods will be added).')
    parser.add_argument('-g', '--dataset-config', dest='dataset_config',
                        required=True, help='dataset configuration file.')
    parser.add_argument('-u', '--unlabeled', dest='unlabeled', action='store_true', default=False,
                        help='set if input data is unlabeled.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
