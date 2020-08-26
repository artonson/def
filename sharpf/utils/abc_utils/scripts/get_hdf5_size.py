#!/usr/bin/env python3

import argparse
from functools import partial
import os
import sys

from torch.utils.data import DataLoader

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), *['..'] * 4)
)
sys.path[1:1] = [__dir__]

from sharpf.data.datasets.sharpf_io import PointCloudIO as IO, DepthMapIO
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, LotsOfHdf5Files, PreloadTypes
from sharpf.utils.abc_utils.hdf5.io_struct import collate_mapping_with_io, select_items_by_predicates


def main(options):
    labels = ['has_sharp'] + options.true_keys + options.false_keys

    if None is not options.h5_input:
        dataset = Hdf5File(options.h5_input, IO, labels=labels, preload=PreloadTypes.NEVER)
        print('{} items in {}'.format(len(dataset), dataset.filename))
    else:
        assert None is not options.h5_input_dir
        dataset = LotsOfHdf5Files(options.h5_input_dir, IO, labels=labels, preload=PreloadTypes.NEVER)
        if not options.total_only:
            for sub_dataset in dataset.files:
                print('{} items in {}'.format(len(sub_dataset), sub_dataset.filename))
        print('Total {} items'.format(len(dataset)))

    if len(options.true_keys) > 0 or len(options.false_keys) > 0:
        loader = DataLoader(
            dataset,
            num_workers=1,
            batch_size=128,
            shuffle=False,
            collate_fn=partial(collate_mapping_with_io, io=IO),
        )

        filtered_num_items = 0
        for batch in loader:
            filtered_batch = select_items_by_predicates(
                batch, true_keys=options.true_keys, false_keys=options.false_keys)
            filtered_num_items += filtered_batch['has_sharp']

        print('Filtered by TRUE [{}] and FALSE [{}]: {} items'.format(
            ', '.join(options.true_keys),
            ', '.join(options.false_keys),
            filtered_num_items))


def parse_options():
    parser = argparse.ArgumentParser(
        description='Get length of files in HDF5 format.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input', dest='h5_input', help='HDF5 input file.')
    group.add_argument('-d', '--input-dir', dest='h5_input_dir', help='directory of HDF5 input files.')

    parser.add_argument('-t', '--total-only', dest='total_only', action='store_true', default=False,
                        help='print total only for directory.')

    parser.add_argument('-tk', '--true-key', dest='true_keys', action='append', default=[],
                        help='specify keys that must be TRUE to put into resulting HDF5 files (can me multiple).')
    parser.add_argument('-fk', '--false-key', dest='false_keys', action='append', default=[],
                        help='specify keys that must be FALSE to put into resulting HDF5 files (can me multiple).')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    options = parse_options()
    main(options)
