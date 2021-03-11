#!/usr/bin/env python3

import argparse
from glob import glob
import os
import sys

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../../../..')
)

from io import BytesIO

import yaml

from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.abc_utils.mesh.io import trimesh_load

sys.path[1:1] = [__dir__]


def main(options):
    stl_filename = glob(os.path.join(options.input_dir, '*.stl'))[0]
    obj_filename = glob(os.path.join(options.input_dir, '*.obj'))[0]
    yml_filename = glob(os.path.join(options.input_dir, '*.yml'))[0]
    hdf5_filename = glob(os.path.join(options.input_dir, '*.hdf5'))[0]

    with open(obj_filename) as obj_file:
        obj_mesh, _, _ = trimesh_load(BytesIO(obj_file.read()))

    with open(yml_filename) as yml_file:
        yml_features = yaml.load(BytesIO(yml_file.read()), Loader=yaml.Loader)

    dataset = Hdf5File(
        hdf5_filename,
        io=data_io,
        preload=PreloadTypes.LAZY,
        labels='*')

    for view in dataset:

        points = view['points']

        distances, directions, has_sharp = annotator.annotate(
            obj_mesh,
            yml_features,
            points)






def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-dir', dest='input_dir',
                        required=True, help='input directory with scans.')
    parser.add_argument('-o', '--output', dest='output_filename',
                        required=True, help='output .hdf5 filename.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
