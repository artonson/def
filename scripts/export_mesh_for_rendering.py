#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import trimesh
import tqdm

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5 import io_struct
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def main(options):
    FusedPredictionsIO = io_struct.HDF5IO(
        {'points': io_struct.Float64('points'),
         'distances': io_struct.Float64('distances')},
        len_label='distances',
        compression='lzf')

    if options.verbose:
        print('Loading data...')
    dataset = Hdf5File(
        options.input_filename,
        io=FusedPredictionsIO,
        preload=PreloadTypes.LAZY,
        labels=['points', 'distances'])
    points = dataset[0]['points']
    distances = dataset[0]['distances']

    is_hard_label = False
    low, high = 0.0, options.max_distance_to_feature
    if None is not options.sharpness_hard_thr:
        is_hard_label = True
        if None is not options.sharpness_hard_values:
            assert isinstance(options.sharpness_hard_values, (tuple, list)) and len(options.sharpness_hard_values) == 2, \
                '"sharpness_hard_values" must be a tuple of size 2'
            low, high = options.sharpness_hard_values

    tol = 1e-3
    helper = MplColorHelper(
        options.input_cmap,
        -tol,
        options.max_distance_to_feature + tol)

    iterable = zip(points, distances)
    if options.verbose:
        iterable = tqdm(iterable)

    full_mesh = []
    for point, distance in iterable:
        if is_hard_label:
            distances[distances <= options.sharpness_hard_thr] = low
            distances[distances > options.sharpness_hard_thr] = high

        rgba = helper.get_rgb(distance)
        mesh = trimesh.creation.icosphere(
            subdivisions=1,
            radius=options.point_size,
            color=rgba[:3])
        mesh.vertices += point
        full_mesh.append(mesh)

    if options.verbose:
        print('Exporting...')
    mesh = trimesh.util.concatenate(full_mesh)
    _ = trimesh.exchange.export.export_mesh(
        mesh,
        options.output_filename,
        'obj')
    if options.verbose:
        print(options.output_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='be verbose')

    parser.add_argument('-i', '--input_filename', dest='input_filename',
                        required=True, help='input file with fused filename.')
    parser.add_argument('-o', '--output_filename', dest='output',
                        required=True, help='output .html filename.')
    parser.add_argument('-s', '--max_distance_to_feature', dest='max_distance_to_feature',
                        default=1.0, type=float, required=False, help='max distance to sharp feature to display.')
    parser.add_argument('-ps', '--point_size', dest='point_size',
                        default=0.02, type=float, required=False,
                        help='point size for plotting (should equal resolution_3d).')

    parser.add_argument('-t', '--sharpness_hard_thr', dest='sharpness_hard_thr',
                        default=None, type=float, help='if set, forces to compute and paint hard labels.')
    parser.add_argument('-v', '--sharpness_hard_values', dest='sharpness_hard_values', nargs=2, default=None, type=float, 
                        help='if set, specifies min and max sharpness values for hard labels.')

    parser.add_argument('-icm', '--input_cmap', dest='input_cmap', default='plasma_r',
                        help='if specified, this should be a list of string colormaps (e.g. "plasma_r", '
                             'one per input file).')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
