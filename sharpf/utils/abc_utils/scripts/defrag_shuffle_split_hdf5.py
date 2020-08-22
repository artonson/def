#!/usr/bin/env python3

import argparse
import collections
import os
import sys
from functools import partial

import torch
from torch.utils.data import DataLoader

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), *['..'] * 4)
)
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import LotsOfHdf5Files, PreloadTypes
# from sharpf.data.datasets.sharpf_io import (
#     save_point_patches as save_fn,
#     PointCloudIO as IO
# )
from sharpf.data.datasets.sharpf_io import (
    save_depth_maps as save_fn,
    DepthMapIO as IO
)
from sharpf.utils.abc_utils.hdf5.io_struct import collate_mapping_with_io


class BufferedHDF5Writer(object):
    def __init__(self, output_dir=None, prefix='', n_items_per_file=float('+inf'),
                 verbose=False, save_fn=None):
        self.output_dir = output_dir
        self.prefix = prefix
        self.file_id = 0
        self.n_items_per_file = n_items_per_file
        self.data = []
        self.verbose = verbose
        self.save_fn = save_fn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.data:
            self._flush()

    def append(self, data):
        assert isinstance(data, collections.abc.Mapping)
        self.data.append(data)
        self.check_flush()

    def extend(self, data):
        self.data.extend([
            dict(zip(data, item_values))
            for item_values in zip(*data.values())
        ])
        self.check_flush()

    def check_flush(self):
        if -1 != self.n_items_per_file and len(self.data) >= self.n_items_per_file:
            self._flush()
            self.file_id += 1
            self.data = self.data[self.n_items_per_file:]

    def _flush(self):
        filename = '{prefix}{id}.hdf5'.format(prefix=self.prefix, id=self.file_id)
        filename = os.path.join(self.output_dir, filename)
        self.save_fn(self.data[:self.n_items_per_file], filename)
        if self.verbose:
            print('Saved {} with {} items'.format(filename, len(self.data)))


def select_items_by_predicates(batch, true_keys=None, false_keys=None):
    """Selects sub-batch where item[key] == True for each key in true_keys
    and item[key] == False for each key in false_keys"""
    any_key = next(iter(batch.keys()))
    batch_size = len(batch[any_key])

    if None is not true_keys:
        true_mask = torch.stack([batch[key] for key in true_keys]).all(axis=0)
    else:
        true_mask = torch.ones(batch_size).bool()

    if None is not false_keys:
        false_mask = torch.stack([~batch[key] for key in false_keys]).all(axis=0)
    else:
        false_mask = torch.ones(batch_size).bool()

    selected_idx = torch.where(true_mask * false_mask)[0]
    filtered_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            filtered_batch[key] = value[selected_idx]
        elif isinstance(value, list):
            filtered_batch[key] = [value[i] for i in selected_idx]

    return filtered_batch


def main(options):
    batch_size = min(128, options.num_items_per_file)
    loader = DataLoader(
        LotsOfHdf5Files(
            data_dir=options.hdf5_input_dir,
            io=IO,
            labels=options.keys or '*',
            max_loaded_files=options.max_loaded_files,
            preload=PreloadTypes.NEVER),
        num_workers=options.n_jobs,
        batch_size=batch_size,
        shuffle=options.random_shuffle,
        collate_fn=partial(collate_mapping_with_io, io=IO),
        # worker_init_fn=worker_init_fn,
    )

    writer_params = {
        'output_dir': options.hdf5_output_dir,
        'n_items_per_file': options.num_items_per_file,
        'save_fn': save_fn,
        'verbose': options.verbose,
    }
    train_writer_params = {'prefix': options.train_prefix, **writer_params}
    val_writer_params = {'prefix': options.val_prefix, **writer_params}

    with BufferedHDF5Writer(**train_writer_params) as train_writer, \
            BufferedHDF5Writer(**val_writer_params) as val_writer:

        for batch_idx, batch in enumerate(loader):
            seen_fraction = batch_idx * batch_size / len(loader.dataset)
            writer = train_writer if seen_fraction <= options.train_fraction else val_writer
            filtered_batch = select_items_by_predicates(
                batch, true_keys=options.true_keys, false_keys=options.false_keys)
            writer.extend(filtered_batch)


def parse_options():
    parser = argparse.ArgumentParser(
        description='De-fragment (make desired size batches), shuffle, and create train/test split from a dir of HDF5s')

    parser.add_argument('-i', '--input-dir', dest='hdf5_input_dir', help='directory of HDF5 input files.')
    parser.add_argument('-o', '--output-dir', dest='hdf5_output_dir', help='directory with HDF5 output files.')

    parser.add_argument('-n', '--num-items-per-file', dest='num_items_per_file', type=int, default=1000,
                        help='how many items to put into each output HDF5 file.')
    parser.add_argument('-t', '--train-fraction', dest='train_fraction', type=float, default=0.8,
                        help='fraction of items to keep as training set.')
    parser.add_argument('--train-prefix', dest='train_prefix', default='train_',
                        help='string prefix of output filename to use for train split.')
    parser.add_argument('--val-prefix', dest='val_prefix', default='val_',
                        help='string prefix of output filename to use for validation split.')
    parser.add_argument('-s', '--random-shuffle', dest='random_shuffle', action='store_true', default=False,
                        help='perform random shuffle of the output.')
    parser.add_argument('-r', '--random-seed', dest='random_seed',
                        help='specify a fixed random seed for reproducibility.')
    parser.add_argument('-k', '--key', dest='keys', action='append',
                        help='specify keys to put into resulting HDF5 files '
                             '(can me multiple; empty means ALL encountered keys).')
    parser.add_argument('-tk', '--true-key', dest='true_keys', action='append',
                        help='specify keys that must be TRUE to put into resulting HDF5 files (can me multiple).')
    parser.add_argument('-fk', '--false-key', dest='false_keys', action='append',
                        help='specify keys that must be FALSE to put into resulting HDF5 files (can me multiple).')
    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=4, help='CPU jobs to use in parallel [default: 4].')
    parser.add_argument('-x', '--max-loaded-files', dest='max_loaded_files',
                        type=int, default=10, help='max loaded sourcec HDF5 files.')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        default=False, help='overwrite existing files.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    options = parse_options()
    main(options)