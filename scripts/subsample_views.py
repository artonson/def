#!/usr/bin/env python3

# Given N views (supposed to be arranged in a spherical spiral from 1 to N),
# output non-randomly n equidistant views.

import argparse
from functools import partial
import glob
import os
import re
import sys

import h5py
import numpy as np

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)
sys.path[1:1] = [__dir__]

import sharpf.data.datasets.sharpf_io as sharpf_io
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
import sharpf.utils.abc_utils.hdf5.io_struct as io_struct


SingleViewPredictionsIO = io_struct.HDF5IO(
    {'image': io_struct.Float64('image'),
     'distances': io_struct.Float64('distances')},
    len_label='distances',
    compression='lzf')


def save_predictions(patches, filename):
    collate_fn = partial(io_struct.collate_mapping_with_io, io=SingleViewPredictionsIO)
    patches = collate_fn(patches)
    with h5py.File(filename, 'w') as f:
        SingleViewPredictionsIO.write(f, 'image', patches['image'])
        SingleViewPredictionsIO.write(f, 'distances', patches['distances'])


def convert_npylist_to_hdf5(input_dir, gt_images, output_filename):
    def get_num(basename):
        match = re.match('^test_(\d+)\.npy$', basename)
        return int(match.groups()[0])

    datafiles = glob.glob(os.path.join(input_dir, '*.npy'))
    datafiles.sort(key=lambda name: get_num(os.path.basename(name)))
    patches = [
        {'image': image, 'distances': np.load(f)}
        for image, f in zip(gt_images, datafiles)]
    save_predictions(patches, output_filename)


def main(options):
    name = os.path.splitext(os.path.basename(options.true_filename))[0]

    print('Loading ground truth data...')
    ground_truth_dataset = Hdf5File(
        options.true_filename,
        io=sharpf_io.WholeDepthMapIO,
        preload=PreloadTypes.LAZY,
        labels='*')
    gt_dataset = [view for view in ground_truth_dataset]
    gt_images = [view['image'] for view in gt_dataset]

    n_views = len(gt_dataset)
    n_desired_views = options.n_views
    assert n_views >= n_desired_views, 'n_views < n_desired_views'
    views_indexes = np.linspace(0, n_views - 1, n_desired_views).astype(int)
    print('Selecting {} indexes: {}'.format(n_desired_views, str(views_indexes)))

    print('Saving ground truth data...')
    gt_dataset = [gt_dataset[i] for i in views_indexes]
    sharpf_io.save_depth_maps(
        gt_dataset,
        os.path.join(options.output_dir, os.path.basename(options.true_filename)))

    print('Loading predictions...')
    predictions_filename_samedir = os.path.join(
        os.path.dirname(options.pred_dir),
        '{}__{}.hdf5'.format(name, 'predictions'))
    if os.path.isdir(options.pred_dir):
        convert_npylist_to_hdf5(options.pred_dir, gt_images, predictions_filename_samedir)
    else:
        if not os.path.exists(predictions_filename_samedir):
            os.symlink(options.pred_dir, predictions_filename_samedir)
    predictions_dataset = Hdf5File(
        predictions_filename_samedir,
        io=sharpf_io.WholeDepthMapIO,
        preload=PreloadTypes.LAZY,
        labels='*')
    pred_dataset = [predictions_dataset[i] for i in views_indexes]

    print('Saving predictions...')
    save_predictions(
        pred_dataset,
        os.path.join(options.output_dir, '{}__{}.hdf5'.format(name, 'predictions')))


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

    parser.add_argument('-n', '--n_views', dest='n_views', default=4, type=int,
                        required=False, help='number of views to use for fusion.')
    parser.add_argument('-r', '--resolution_3d', dest='resolution_3d', required=False, default=0.02, type=float,
                        help='3D resolution of scans.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
