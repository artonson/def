#!/usr/bin/env python3

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


def main(options):
    pass




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        dest='verbose',
        help='verbose output [default: False].')

    parser.add_argument(
        '-p', '--pred-filename',
        dest='pred_filename',
        required=True,
        help='Path to predictions hdf5 file.')

    parser.add_argument(
        '-t', '--true-filename',
        dest='true_filename',
        required=True,
        help='Path to GT file with whole model point patches.')
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
