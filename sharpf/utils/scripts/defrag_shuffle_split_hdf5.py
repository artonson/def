#!/usr/bin/env python3

import argparse
import os
import sys

from torch.utils.data import DataLoader

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')
)
sys.path[1:1] = [__dir__]

from sharpf.data.datasets.hdf5_datasets import LotsOfHdf5Files


class BufferedHDF5Writer(object):
    def __init__(self, output_dir=None, output_files=None, n_meshes_file=float('+inf'),
                 verbose=False, save_fn=None):
        self.output_dir = output_dir
        self.output_files = output_files
        self.file_id = 0
        self.n_meshes_file = n_meshes_file
        self.data = []
        self.verbose = verbose
        self.save_fn = save_fn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.data:
            self._flush()

    def append(self, data):
        self.data.append(data)
        if -1 != self.n_meshes_file and len(self.data) >= self.n_meshes_file:
            self._flush()
            self.file_id += 1
            self.data = []

    def _flush(self):
        """Create the next file (as indicated by self.file_id"""
        if self.output_files:
            filename = self.output_files[self.file_id]
        else:
            filename = os.path.join(self.output_dir, '{}.hdf5'.format(self.file_id))
        self.save_fn(self.data, filename)
        if self.verbose:
            print('Saved {} with {} items'.format(filename, len(self.data)))


def main(options):
    # TODO:
    #  1) understand the logic of dataloader in pytorch, and whether raw dataloader may be used (without BufferedHDF5Writer);
    #  2) understand how one should implement collate to ensure building a patch is a) correct and b) generic;
    #  3) understand if any modifications must be made to mesh denoising/vector fields
    #  4) rewrite LotsOfHdf5Files with custom data/target labels

    loader = DataLoader(
        LotsOfHdf5Files(
            data_dir=options.hdf5_input_dir,
            data_label=options.data_label,
            target_label=options.target_label,
            labels=options.keys,
            max_loaded_files=10),
        num_workers=options.n_jobs,
        batch_size=options.num_items_per_file,
        shuffle=options.random_shuffle,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_to_list,
    )

    for batch in loader:
        save_point_patches(batch, )

    hdf5_writer_params = {
        'output_dir': options.hdf5_output_dir,
        'output_files': output_hdf5_files,
        'n_meshes_file': options.num_items_per_file,
        'verbose': options.verbose,
        'save_fn': save_point_patches
    }
    with BufferedHDF5Writer(**hdf5_writer_params) as hdf_writer:
        for batch in loader:
            hdf_writer.append(item)


def parse_options():
    parser = argparse.ArgumentParser(
        description='Defragment (make N batches), sluffle, and create train/test split from a dir of HDF5s')

    parser.add_argument('-i', '--input-dir', dest='hdf5_input_dir', help='directory of HDF5 input files.')
    parser.add_argument('-o', '--output-dir', dest='hdf5_output_dir', help='directory with HDF5 output files.')

    parser.add_argument('-n', '--num-items-per-file', dest='num_items_per_file', type=int, default=1000,
                        help='how many items to put into each output HDF5 file.')
    parser.add_argument('-t', '--train-fraction', dest='train_fraction', type=float, default=0.8,
                        help='fraction of items to keep as training set.')
    parser.add_argument('-s', '--random-shuffle', dest='random_shuffle', action='store_true', default=False,
                        help='perform random shuffle of the output.')
    parser.add_argument('-r', '--random-seed', dest='random_seed',
                        help='specify a fixed random seed for reproducibility.')
    parser.add_argument('-k', '--key', dest='keys', action='append',
                        help='specify keys to put into resulting HDF5 files (can me multiple; empty means ALL encountered keys).')
    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=4, help='CPU jobs to use in parallel [default: 4].')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        default=False, help='overwrite existing files.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    options = parse_options()
    main(options)
