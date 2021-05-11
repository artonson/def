#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../..'))
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
import sharpf.fusion.io as fusion_io


def main(options):
    true_dataset = Hdf5File(
        options.true_filename,
        io=fusion_io.FusedPredictionsIO,
        preload=PreloadTypes.LAZY,
        labels=['points', 'distances'])
    true_distances = true_dataset[0]['distances']
    true_points = true_dataset[0]['points']

    pred_dataset = Hdf5File(
        options.pred_filename,
        io=fusion_io.FusedPredictionsIO,
        preload=PreloadTypes.LAZY,
        labels=['distances'])
    pred_distances = pred_dataset[0]['distances']

    fused_distances_diff = np.abs(true_distances - pred_distances)
    fusion_io.save_full_model_predictions(
        true_points,
        fused_distances_diff,
        options.out_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')

    parser.add_argument('-t', '--true-filename', dest='true_filename', required=True,
                        help='Path to GT file with whole model point patches.')
    parser.add_argument('-p', '--pred-filename', dest='pred_filename', required=True,
                        help='Path to PRED file with whole model point patches.')
    parser.add_argument('-o', '--out-filename', dest='out_filename', required=True,
                        help='Path to output filename.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
