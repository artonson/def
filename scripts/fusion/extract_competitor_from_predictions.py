#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../..')
)
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
import sharpf.fusion.io as fusion_io


def main(options):
    hdf5_file = Hdf5File(
        filename=options.input,
        io=fusion_io.ComparisonsIO,
        labels=['points', options.key],
        preload=PreloadTypes.NEVER)

    points = np.array(hdf5_file[0]['points']).squeeze()
    predictions = np.array(hdf5_file[0][options.key], dtype=np.float).squeeze()

    fusion_io.save_full_model_predictions(
        points,
        predictions,
        options.output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='input',
                        required=True, help='input file with predictions.')
    parser.add_argument('-o', '--output', dest='output',
                        required=True, help='output file with partial predictions.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='be verbose')
    parser.add_argument('-k', '--key', dest='key', required=True,
                        help='what key to use for getting preds.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
