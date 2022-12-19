#!/usr/bin/env python3

import argparse
import os
import sys
import io

import numpy as np

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..'))
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import PreloadTypes, Hdf5File
import sharpf.metrics.numpy_metrics as nm
import sharpf.fusion.io as fusion_io
import sharpf.data.datasets.sharpf_io as sharpf_io
from sharpf.utils.convertor_utils import convertors_io


def main(options):
    if options.single_view:
        true_data_io = convertors_io.AnnotatedViewIO
        pred_data_io = fusion_io.ImagePredictionsIO
    elif options.single_patch:
        true_data_io = sharpf_io.WholePointCloudIO
        pred_data_io = sharpf_io.WholePointCloudIO
    else:
        true_data_io = fusion_io.FusedPredictionsIO
        pred_data_io = fusion_io.FusedPredictionsIO

    true_dataset = Hdf5File(
        options.true_filename,
        io=true_data_io,
        preload=PreloadTypes.LAZY,
        labels=['distances'])
    sharpness_masks = [item['distances'] != options.sharpness_bg_value for item in true_dataset]
    true_distances = [{'distances': item['distances'][mask]} for item, mask in zip(true_dataset, sharpness_masks)]

    pred_dataset = Hdf5File(
        options.pred_filename,
        io=pred_data_io,
        preload=PreloadTypes.LAZY,
        labels=['distances'])
    pred_distances = [{'distances': item['distances'][mask]} for item, mask in zip(pred_dataset, sharpness_masks)]

    rmse = nm.RMSE()
    q95rmse = nm.RMSEQuantile(0.95)
    r1 = options.resolution_3d
    bad_points_1r = nm.BadPoints(r1, normalize=True)
    r4 = options.resolution_3d * 4
    bad_points_4r = nm.BadPoints(r4, normalize=True)
    iou_1r = nm.IntersectionOverUnion(r1, r1)
    iou_4r = nm.IntersectionOverUnion(r1, r4)
    ap = nm.AveragePrecision(r1)
    fpr_1r = nm.FalsePositivesRate(r1, r1)
    fpr_4r = nm.FalsePositivesRate(r1, r4)

    tol = 1e-6
    max_distances_all = np.max([np.max(item['distances']) for item in true_distances]) + tol
    all_mask = nm.DistanceLessThan(max_distances_all, name='ALL')
    RMSE_ALL = nm.MaskedMetric(all_mask, rmse)
    q95RMSE_ALL = nm.MaskedMetric(all_mask, q95rmse)

    sharp_mask = nm.DistanceLessThan(options.max_distance_to_feature - tol, name='Sharp')
    closesharp_mask = nm.DistanceLessThan(options.max_distance_to_feature - tol, name='Sharp')
#    close_mask = nm.DistanceLessThan(options.max_distance_to_feature - tol, name='Close')
#    closesharp_mask = nm.MaskedMetric(sharp_mask, close_mask)
    mBadPoints_1r_CloseSharp = nm.MaskedMetric(closesharp_mask, bad_points_1r)
    mBadPoints_4r_CloseSharp = nm.MaskedMetric(closesharp_mask, bad_points_4r)

    # for our whole models, we keep all points
    # sharp_mask = nm.DistanceLessThan(max_distances_all, name='Sharp')
    IOU_1r_Sharp = nm.MaskedMetric(sharp_mask, iou_1r)
    IOU_4r_Sharp = nm.MaskedMetric(sharp_mask, iou_4r)
    AP_Sharp = nm.MaskedMetric(sharp_mask, ap)

    far_all_mask = nm.DistanceGreaterThan(options.max_distance_to_feature - tol, name='Far-ALL')
    FPR_1r_Sharp = nm.MaskedMetric(far_all_mask, fpr_1r)
    FPR_4r_Sharp = nm.MaskedMetric(far_all_mask, fpr_4r)

    metrics = [
        RMSE_ALL,
        q95RMSE_ALL,
        mBadPoints_1r_CloseSharp,
        mBadPoints_4r_CloseSharp,
        IOU_1r_Sharp,
        IOU_4r_Sharp,
        AP_Sharp,
        FPR_1r_Sharp,
        FPR_4r_Sharp,
    ]
    values = []
    for idx, (true_item, pred_item) in enumerate(zip(true_distances, pred_distances)):
        print(idx)
        item_values = [metric(true_item, pred_item) for metric in metrics]
        values.append(item_values)
    fp = options.out_filename
    fp = open(fp, 'w') if not isinstance(fp, io.TextIOBase) else fp
    print(
        '{metrics_names}\n{items_values}'.format(
            metrics_names=','.join([str(metric) for metric in metrics]),
            items_values='\n'.join([
                ','.join([str(value) for value in item_values])
                for item_values in values])
        ),
        file=fp)
    fp.close()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')

    parser.add_argument(
        '-t', '--true-filename',
        dest='true_filename',
        required=True,
        help='path to GT file with fused points and distances.')
    parser.add_argument(
        '-p', '--pred-filename',
        dest='pred_filename',
        required=True,
        help='path to PRED file with fused points and distances.')
    parser.add_argument(
        '-o', '--out-filename',
        dest='out_filename',
        default=sys.stdout,
        help='path to OUTPUT file with metrics (if None, print metrics to stdout).')

    parser.add_argument(
        '-sv', '--single_view',
        dest='single_view',
        default=False,
        action='store_true',
        help='if set, this .')
    parser.add_argument(
        '-sp', '--single_patch',
        dest='single_patch',
        default=False,
        action='store_true',
        help='if set, this .')


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
    parser.add_argument(
        '-sbg', '--sharpness_bg_value',
        dest='sharpness_bg_value',
        default=0.0,
        type=float,
        help='if set, specifies sharpness value to be treated as background.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
