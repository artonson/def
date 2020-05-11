#!/usr/bin/env python3

import argparse
from functools import partial
import os
import sys

import numpy as np
from torch.utils.data import DataLoader

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), *['..'] * 2)
)
sys.path[1:1] = [__dir__]

from sharpf.utils.py_utils.os import require_empty
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File
from sharpf.data.datasets.sharpf_io import (
    PointCloudIO as IO
)
# from sharpf.data.datasets.sharpf_io import (
#     DepthMapIO as IO
# )


def main(options):
    require_empty(options.output_dir, recreate=options.overwrite)

    loader = DataLoader(
        dataset=Hdf5File(
            filename=options.hdf5_input_file,
            io=IO,
            data_label=options.data_label,
            target_label=options.target_label,
            preload=True,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    filename_template = '{prefix}{index:06d}{suffix}.{fmt}'

    items = []  # used only for 'npy' format
    from tqdm import tqdm
    for item_idx, item in tqdm(enumerate(loader)):
        data = item[options.data_label].squeeze().numpy()
        target = item[options.target_label].squeeze().numpy()
        if None is not options.target_thr:
            target = np.array(target < options.target_thr, dtype=np.float)

        if options.output_format == 'txt':
            # save separate txt files
            output_filename = partial(
                filename_template.format,
                prefix=options.output_prefix,
                index=item_idx,
                fmt='txt'
            )
            if options.split_xy:
                np.savetxt(output_filename(suffix='_data'), data, fmt='%.10e')
                np.savetxt(output_filename(suffix='_target'), target, fmt='%.10e')
            else:
                X = np.column_stack((data, target))
                np.savetxt(output_filename(suffix=''), X, fmt='%.10e')

        else:
            assert options.output_format == 'npy', 'weird output format {}'.format(options.output_format)
            items.append((data, target))

    if options.output_format == 'npy':
        output_filename = partial(
            filename_template.format,
            prefix=options.output_prefix,
            index=0,
            fmt='npy'
        )
        if options.split_xy:
            data_items = np.stack([data for data, _ in items])
            np.save(output_filename(suffix='_data'), data_items)

            target_items = np.stack([target for _, target in items])
            np.save(output_filename(suffix='_target'), target_items)
        else:
            X = np.stack([
                np.column_stack((data, target))
                for data, target in items])
            np.save(output_filename(suffix=''), X)


def parse_options():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 file to a set of TXT/NPY files.')

    parser.add_argument('-i', '--input-file', dest='hdf5_input_file', help='HDF5 input files.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='directory for output files.')

    parser.add_argument('-d', '--data-label', dest='data_label', required=True, help='data label.')
    parser.add_argument('-t', '--target-label', dest='target_label', required=True, help='target label.')
    parser.add_argument('-T', '--target-thr', dest='target_thr', type=float,
                        required=False, help='if specified, targets with values smaller than T will be set to '
                                             'binary 1 (sharp), and those with values greater than T to binary 0.')

    parser.add_argument('-s', '--split-xy', dest='split_xy', action='store_true',
                        default=False, help='if set, data and target will be extracted as separate '
                                            'files (with "_data" suffix for data and "_target" suffix for target '
                                            '(useful for creating test/validation files).')
    parser.add_argument('-f', '--output-format', dest='output_format', default='txt',
                        choices=['txt', 'npy'], help='output format to use.')
    parser.add_argument('-p', '--output-prefix', dest='output_prefix', default='distance_',
                        help='string prefix of output filename to use.')

    parser.add_argument('-w', '--overwrite', dest='overwrite', action='store_true',
                        default=False, help='overwrite existing files.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        default=False, help='be verbose.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    options = parse_options()
    main(options)
