#!/usr/bin/env python3

import argparse
import os
import sys
import io

import numpy as np
import torch
from scipy.special import softmax
import scipy.io as sio

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import PreloadTypes, Hdf5File
import sharpf.metrics.numpy_metrics as nm
import sharpf.fusion.io as fusion_io
from sharpf.utils.convertor_utils import convertors_io

from metrics import MRMSE, Q95RMSE, MRecall, MFPR

def main(options):
    labels_gt = sio.loadmat(str(options.true_filename))['distances']

    pred_data = sio.loadmat(str(options.pred_filename))
    # "prob_edge" <= 0.7 that means point has distance of 1., otherwise it has 0. distance
    # pred_labels = (softmax(pred_data['pred_labels_key_p_val'], axis=2)[:, :, 1] <= 0.7).astype(float) * 2.
    pred_labels = (softmax(pred_data['pred_labels_key_p_val'], axis=2)[:, :, 1] >= 0.7).astype(float)
    
    points_gt = pred_data['input_point_cloud']
    
    brd_thr = 90
    points_radii = np.linalg.norm(points_gt - points_gt.mean(axis=1).reshape(-1,1,3), axis=2)
    true_distances = []
    pred_distances = []

    for idx, thresh_radii in enumerate(np.percentile(points_radii, brd_thr, axis=1)):
        center_mask = points_radii[idx] < thresh_radii
        mask = center_mask
        true_distances.append(labels_gt[idx][mask])
        pred_distances.append(pred_labels[idx][mask])

    mfpr = MFPR()
    mrec = MRecall()
    mrmse = MRMSE()
    q95rmse = Q95RMSE()

    metrics = [
        mfpr,
        mrec,
        # mrmse,
        # q95rmse,
    ]
    values = []
    thresh_4r = options.resolution_3d * 4
    for idx, (true_item, pred_item) in enumerate(zip(true_distances, pred_distances)):
        item_values = []
        for metric in metrics:
            if isinstance(metric, MFPR) or isinstance(metric, MRecall):
                metric.update(
                    torch.tensor(pred_item).reshape(1, -1),
                    torch.tensor(true_item).reshape(1, -1) < thresh_4r
                )
            else:
                metric.update(
                    torch.tensor(pred_item).reshape(1, -1),
                    torch.tensor(true_item).reshape(1, -1)
                )
    values.append([metric.compute() for metric in metrics])
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
        required=False,
        help='path to GT file with fused points and distances.')
    parser.add_argument(
        '-p', '--pred-filename',
        dest='pred_filename',
        required=False,
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
    from tqdm import tqdm
    options = parse_args()
    ARGS_PATH = "/home/appuser/ssd/ebogomolov/patches/args_list.txt"
    with open(ARGS_PATH, "r") as f:
        for line in tqdm(f.readlines()):
            line = line.strip().split(" ")
            options.true_filename = line[0]
            options.pred_filename = line[1]
            options.resolution_3d = float(line[2])
            options.out_filename = line[3]
            main(options)