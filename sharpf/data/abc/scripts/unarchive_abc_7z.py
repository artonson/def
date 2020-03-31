#!/usr/bin/env python3

import argparse
import os
import sys

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')
)
sys.path[1:1] = [__dir__]

from sharpf.data.abc.abc_data import ABC7ZFile
from sharpf.data.abc.abc_data_provider import ABC7zDataProvider, ABCUnarchivedDataProvider

# Unarchive 7z file and create the index files necessary to use
# the unarchived version of the ABC7ZFile.


def main(options):

    archive_provider = ABC7zDataProvider(options.input_filename)
    unarchived_provider = ABCUnarchivedDataProvider(options.output_dir)

    with ABC7ZFile(archive_provider) as abc_archive, ABC7ZFile(unarchived_provider) as abc_unarchived:
        abc_archive.names
        for item in abc_archive:
            abc_unarchived


def parse_options():
    parser = argparse.ArgumentParser(
        description='Read 7z archive of ABC, write separate files in their respective formats.')
    parser.add_argument('-i', '--input', dest='input_filename', help='input file.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='output directory with raw files.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    options = parse_options()
    main(options)
