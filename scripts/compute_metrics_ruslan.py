#!/usr/bin/env python3

import argparse
import os
import sys
import io

import numpy as np
import torch

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import PreloadTypes, Hdf5File
import sharpf.metrics.numpy_metrics as nm
import sharpf.fusion.io as fusion_io
import sharpf.data.datasets.sharpf_io as sharpf_io
from sharpf.utils.convertor_utils import convertors_io

from metrics import MRMSE, Q95RMSE, MRecall, MFPR

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

    assert len(options.true_filenames) == len(options.pred_filenames), '-t and -p must be added the same number of times'

    true_distances, pred_distances = [], []
    for true_filename, pred_filename in zip(options.true_filenames, options.pred_filenames):
        true_dataset = Hdf5File(
            true_filename,
            io=true_data_io,
            preload=PreloadTypes.LAZY,
            labels=['distances'])
        pred_dataset = Hdf5File(
            pred_filename,
            io=pred_data_io,
            preload=PreloadTypes.LAZY,
            labels=['distances'])
        assert len(true_dataset) == len(pred_dataset), 'lengths of files: {} and {} are not equal'.format(true_filename, pred_filename)

        sharpness_masks = [item['distances'] != options.sharpness_bg_value for item in true_dataset]
        true_distances.extend([item['distances'][mask] for item, mask in zip(true_dataset, sharpness_masks)])
        pred_distances.extend([item['distances'][mask] for item, mask in zip(pred_dataset, sharpness_masks)])

    mfpr = MFPR()
    mrec = MRecall()
    mrmse = MRMSE()
    q95rmse = Q95RMSE()

    if options.is_binary:
        metrics = [
            mfpr,
            mrec,
        ]
    else:
        metrics = [
            mfpr,
            mrec,
            mrmse,
            q95rmse,
        ]
    thresh_4r = options.resolution_3d * 4
    for idx, (true_item, pred_item) in enumerate(zip(true_distances, pred_distances)):
        print(idx)
        for metric in metrics:
            if isinstance(metric, MFPR) or isinstance(metric, MRecall):
                pred_tensor = torch.tensor(pred_item, dtype=torch.float32).reshape(1, -1)
                if not options.is_binary:
                    pred_tensor = pred_tensor < thresh_4r
                metric.update(
                    pred_tensor,
                    torch.tensor(true_item, dtype=torch.float32).reshape(1, -1) < thresh_4r
                )
            else:
                metric.update(
                    torch.tensor(pred_item, dtype=torch.float32).reshape(1, -1),
                    torch.tensor(true_item, dtype=torch.float32).reshape(1, -1)
                )

    values = [metric.compute() for metric in metrics]
    fp = options.out_filename
    fp = open(fp, 'w') if not isinstance(fp, io.TextIOBase) else fp
    print(
        '{metrics_names}\n{items_values}'.format(
            metrics_names=','.join([str(metric) for metric in metrics]),
            items_values=','.join([str(value) for value in values])),
        file=fp)
    fp.close()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')

    parser.add_argument(
        '-t', '--true-filename',
        dest='true_filenames',
        required=True,
        action='append',
        help='path to GT file with fused points and distances.')
    parser.add_argument(
        '-p', '--pred-filename',
        dest='pred_filenames',
        required=True,
        action='append',
        help='path to PRED file with fused points and distances.')
    parser.add_argument(
        '-o', '--out-filename',
        dest='out_filename',
        default=sys.stdout,
        help='path to OUTPUT file with metrics (if None, print metrics to stdout).')
    
    parser.add_argument(
        '-bin', '--is_binary',
        dest='is_binary',
        default=False,
        action='store_true',
        help='whether predict is a binary vector, where 1 means sharp, 0 - not sharp')

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
