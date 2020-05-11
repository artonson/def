#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
from torch.utils.data import DataLoader

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), *['..'] * 4)
)
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File
from sharpf.data.datasets.sharpf_io import (
    PointCloudIO as IO
)
# from sharpf.data.datasets.sharpf_io import (
#     DepthMapIO as IO
# )


def main(options):
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

    for item_idx, item in enumerate(loader):
        data = item[options.data_label].numpy()
        target = item[options.target_label].numpy()
        X = np.stack((data, target), axis=1)

        output_filename = os.path.join(
            options.output_dir,
            '{prefix}{index: <5}.txt'.format(
                prefix=options.output_prefix,
                index=item_idx
            )
        )
        np.savetxt(output_filename, X, fmt='%.10e')


def parse_options():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 file to a set of TXT files.')

    parser.add_argument('-i', '--input-file', dest='hdf5_input_file', help='HDF5 input files.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='directory for output files.')

    parser.add_argument('-d', '--data-label', dest='data_label', required=True, help='data label.')
    parser.add_argument('-t', '--target-label', dest='target_label', required=True, help='target label.')

    parser.add_argument('--output-prefix', dest='output_prefix', default='distance_',
                        help='string prefix of output filename to use.')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        default=False, help='overwrite existing files.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    options = parse_options()
    main(options)
