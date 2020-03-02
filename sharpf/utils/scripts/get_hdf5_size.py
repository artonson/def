#!/usr/bin/env python3

import argparse
import os
import sys

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)
sys.path[1:1] = [__dir__]

from sharpf.data.datasets.hdf5_datasets import Hdf5File, LotsOfHdf5Files


def main(options):

    if None is not options.h5_input:
        h5_file = Hdf5File(options.h5_input, None, None, preload=False)
        print('{} items in {}'.format(len(h5_file), h5_file.filename))
    else:
        assert None is not options.h5_input_dir
        lots = LotsOfHdf5Files(options.h5_input_dir, None, None)
        if not options.total_only:
            for h5_file in lots.files:
                print('{} items in {}'.format(len(h5_file), h5_file.filename))
        print('Total {} items'.format(len(lots)))


def parse_options():
    parser = argparse.ArgumentParser(
        description='Get length of files in HDF5 format.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input', dest='h5_input', help='HDF5 input file.')
    group.add_argument('--input-dir', dest='h5_input_dir', help='directory of HDF5 input files.')

    parser.add_argument('-t', '--total-only', dest='total_only', action='store_true', default=False,
                        help='print total only for directory.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    options = parse_options()
    main(options)
