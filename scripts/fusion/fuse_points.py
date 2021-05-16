#!/usr/bin/env python3

# Given an input dir with predictions (list of npy files)
# and an input file with ground truth (multiple point patches),
# this script:
#  1) builds a single file with ground truth predictions
#     (or a single point-cloud file without ground truth,
#     indicated by a key --unlabeled);
#  2) runs a series of various prediction fusion algorithms,
#     saving consolidated predictions into an output folder

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

from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
import sharpf.data.datasets.sharpf_io as sharpf_io
import sharpf.fusion.combiners as combiners
import sharpf.fusion.smoothers as smoothers
import sharpf.fusion.io as fusion_io


def load_ground_truth(true_filename, unlabeled=False):
    """Loads a specified representation of """
    if unlabeled:
        data_io = fusion_io.UnlabeledPointCloudIO
    else:
        data_io = sharpf_io.WholePointCloudIO

    # load ground truth and save to a single patch
    ground_truth_dataset = Hdf5File(
        true_filename,
        io=data_io,
        preload=PreloadTypes.LAZY,
        labels='*')
    ground_truth = [patch for patch in ground_truth_dataset]

    n_points = np.concatenate([patch['indexes_in_whole'] for patch in ground_truth]).max() + 1
    if n_points >= 1000000:
        raise ValueError('Too large file ({} points); skipping'.format(n_points))

    list_distances = [patch['distances'] for patch in ground_truth]
    list_indexes_in_whole = [patch['indexes_in_whole'].astype(int) for patch in ground_truth]
    list_points = [patch['points'].reshape((-1, 3)) for patch in ground_truth]

    return n_points, list_distances, list_indexes_in_whole, list_points


def load_predictions(pred_data, name, pred_key='distances'):
    if os.path.isdir(pred_data):
        predictions_filename = os.path.join(
            options.output_dir,
            '{}__{}.hdf5'.format(name, 'predictions'))
        fusion_io.convert_npylist_to_hdf5(
            pred_data,
            predictions_filename,
            fusion_io.PointPatchPredictionsIO)

    else:
        assert os.path.isfile(pred_data)
        predictions_filename = pred_data

    if pred_key is not None:
        data_io = fusion_io.ComparisonsIO
    else:
        data_io = sharpf_io.WholePointCloudIO

    predictions_dataset = Hdf5File(
        predictions_filename,
        io=data_io,
        preload=PreloadTypes.LAZY,
        labels=[pred_key])
    list_predictions = [patch[pred_key] for patch in predictions_dataset]

    return list_predictions


def main(options):
    name = os.path.splitext(os.path.basename(options.true_filename))[0]

    n_points, list_distances, list_indexes_in_whole, list_points = load_ground_truth(
        options.true_filename,
        unlabeled=options.unlabeled)

    fused_points_gt, fused_distances_gt, _ = combiners.GroundTruthCombiner()(
        n_points,
        list_distances,
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

    list_predictions = load_predictions(
        options.pred_data,
        name,
        pred_key=options.pred_key or 'distances')
    list_predictions = [distances * options.pred_distance_scale_ratio
                        for distances in list_predictions]

    # run various algorithms for consolidating predictions
    # this selection is up to you, user
    combiners_list = [
        # combiners for probabilities
        # combiners.AvgProbaPredictionsCombiner(thr=0.25),
        # combiners.AvgProbaPredictionsCombiner(thr=0.5),
        # combiners.AvgProbaPredictionsCombiner(thr=0.75),

        # combiners for distances
        # MedianPredictionsCombiner(),
        combiners.MinPredictionsCombiner(),
        # AvgPredictionsCombiner(),
        # TruncatedAvgPredictionsCombiner(),
        # CenterCropPredictionsCombiner(brd_thr=80, func=np.min, tag='crop__min'),
        combiners.CenterCropPredictionsCombiner(
           brd_thr=80,
           func=combiners.TruncatedMean(0.6, func=np.min),
           tag='adv60'),
        # MinsAvgPredictionsCombiner(signal_thr=0.9),

        # combiners + smoothers for distances
        # SmoothingCombiner(
        #     combiner=CenterCropPredictionsCombiner(brd_thr=80, func=np.min),
        #     smoother=L2Smoother(regularizer_alpha=0.01)
        # ),
        # SmoothingCombiner(
        #     combiner=CenterCropPredictionsCombiner(brd_thr=80, func=np.min),
        #     smoother=TotalVariationSmoother(regularizer_alpha=0.001)
        # ),
        combiners.SmoothingCombiner(
            combiner=combiners.CenterCropPredictionsCombiner(
                brd_thr=80,
                func=combiners.TruncatedMean(0.6, func=np.min),
                tag='adv60'),
            smoother=smoothers.RobustLocalLinearFit(
                lm.HuberRegressor(epsilon=4., alpha=1.),
                n_jobs=options.n_jobs)),
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
    parser.add_argument('-w', '--overwrite', action='store_true', default=False, dest='overwrite',
                        help='overwrite each output [default: False].')

    parser.add_argument('-t', '--true-filename', dest='true_filename', required=True,
                        help='Path to GT file with whole model point patches.')
    parser.add_argument('-p', '--pred-data', dest='pred_data', required=True,
                        help='Path to prediction directory with npy files or a single HDF5 file.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True,
                        help='Path to output (suffixes indicating various methods will be added).')
    parser.add_argument('-u', '--unlabeled', dest='unlabeled', action='store_true', default=False,
                        help='set if input data is unlabeled.')
    parser.add_argument('-j', '--jobs', dest='n_jobs', default=4, type=int,
                        required=False, help='number of jobs to use for fusion.')
    parser.add_argument('-k', '--key', dest='pred_key',
                        help='if set, switch to compare-io and use this key.')
    parser.add_argument('-s', '--max_distance_to_feature', dest='max_distance_to_feature',
                        default=1.0, type=float, required=False, help='max distance to sharp feature to compute.')
    parser.add_argument('-r', '--pred_distance_scale_ratio', dest='pred_distance_scale_ratio',
                        default=1.0, type=float, required=False, help='factor by which to multiply the predicted distances.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
