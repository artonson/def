#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import PreloadTypes, Hdf5File
import sharpf.metrics.numpy_metrics as nm
import sharpf.fusion.io as fusion_io


def main(options):
    true_dataset = Hdf5File(
        options.true_filename,
        io=fusion_io.FusedPredictionsIO,
        preload=PreloadTypes.LAZY,
        labels=['distances'])
    true_distances = {'distances': true_dataset[0]['distances']}

    pred_dataset = Hdf5File(
        options.pred_filename,
        io=fusion_io.FusedPredictionsIO,
        preload=PreloadTypes.LAZY,
        labels=['distances'])
    pred_distances = {'distances': pred_dataset[0]['distances']}

    rmse = nm.RMSE()
    q95rmse = nm.RMSEQuantile(0.95)
    r1 = options.resolution_3d
    bad_points_1r = nm.BadPoints(r1)
    r4 = options.resolution_3d * 4
    bad_points_4r = nm.BadPoints(r4)
    iou = nm.IOU(r1)

    all_mask = nm.DistanceLessThan(np.max(true_distances) + 1e-6, name='ALL')
    RMSE_ALL = nm.MaskedMetric(all_mask, rmse)
    q95RMSE_ALL = nm.MaskedMetric(all_mask, q95rmse)

    closesharp_mask = nm.DistanceLessThan(options.max_distance_to_feature, name='Close-Sharp')
    mBadPoints_1r_CloseSharp = nm.MaskedMetric(closesharp_mask, bad_points_1r)
    mBadPoints_4r_CloseSharp = nm.MaskedMetric(closesharp_mask, bad_points_4r)

    # for our whole models, we keep all points
    sharp_mask = nm.DistanceLessThan(np.max(true_distances) + 1e-6, name='Sharp')
    IOU_Sharp = nm.MaskedMetric(sharp_mask, iou)

    metrics = [
        RMSE_ALL,
        q95RMSE_ALL,
        mBadPoints_1r_CloseSharp,
        mBadPoints_4r_CloseSharp,
        IOU_Sharp,
    ]
    values = [metric(true_distances, pred_distances) for metric in metrics]
    print(
        '{metrics_names}\n{metrics_values}'.format(
            metrics_names=','.join([str(metric) for metric in metrics]),
            metrics_values=','.join([str(value) for value in values])),
        file=options.out_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')

    parser.add_argument(
        '-t', '--true-filename',
        dest='true_filename',
        type=argparse.FileType('r'),
        required=True,
        help='path to GT file with fused points and distances.')
    parser.add_argument(
        '-p', '--pred-filename',
        dest='pred_filename',
        type=argparse.FileType('r'),
        required=True,
        help='path to PRED file with fused points and distances.')
    parser.add_argument(
        '-o', '--out-filename',
        dest='out_filename',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help='path to OUTPUT file with metrics (if None, print metrics to stdout).')

    parser.add_argument(
        '-r', '--resolution_3d',
        dest='resolution_3d',
        default=1.0,
        type=float,
        help='presupposed 3d resolution.')
    parser.add_argument(
        '-s', '--max_distance_to_feature',
        dest='max_distance_to_feature',
        type=float,
        default=1.0,
        help='max distance to sharp feature to evaluate.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
