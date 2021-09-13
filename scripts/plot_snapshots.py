#!/usr/bin/env python3

import argparse
import os
import sys
import time

import k3d
import numpy as np
from tqdm import tqdm

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5 import io_struct
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.py_utils.os import change_ext


def main(options):
#   FusedPredictionsIO = io_struct.HDF5IO(
#       {'points': io_struct.Float64('points'),
#        'distances': io_struct.Float64('distances')},
#       len_label='distances',
#       compression='lzf')


    ComparisonsIO = io_struct.HDF5IO({
        'points': io_struct.VarFloat64('points'),
        'indexes_in_whole': io_struct.VarInt32('indexes_in_whole'),
        'distances': io_struct.VarFloat64('distances'),
        'item_id': io_struct.AsciiString('item_id'),
        'voronoi': io_struct.VarFloat64('voronoi'),
        'ecnet': io_struct.VarFloat64('ecnet'),
        'sharpness': io_struct.VarFloat64('sharpness'),
        'sharpness_seg': io_struct.VarFloat64('sharpness_seg'),
        'distances_for_sh': io_struct.VarFloat64('distances_for_sh'),
    },
        len_label='points',
        compression='lzf')


    plot_height = 768
    plot = k3d.plot(
        grid_visible=True,
        height=plot_height,
        background_color=options.bgcolor,
        camera_auto_fit=True)

    is_hard_label = False
    if None is not options.sharpness_hard_thr:
        is_hard_label = True
        if None is not options.sharpness_hard_values:
            assert isinstance(options.sharpness_hard_values, (tuple, list)) and len(options.sharpness_hard_values) == 2, \
                '"sharpness_hard_values" must be a tuple of size 2'
            low, high = options.sharpness_hard_values
        else:
            low, high = 0.0, options.max_distance_to_feature

    input_cmaps = options.input_cmaps or ['plasma_r'] * len(options.inputs)
    for input_filename, input_cmap, key in tqdm(zip(options.inputs, input_cmaps, options.keys), desc='Loading/plotting'):
        dataset = Hdf5File(
            input_filename,
            io=ComparisonsIO,
            preload=PreloadTypes.LAZY,
            labels=['points', key])
#        name_for_plot = change_ext(
#            os.path.basename(input_filename), '').split('__', maxsplit=1)[-1]
        name_for_plot = key

        points = dataset[0]['points']
        distances = np.array(dataset[0][key], dtype=np.float)

        if is_hard_label:
            distances[distances <= options.sharpness_hard_thr] = low
            distances[distances > options.sharpness_hard_thr] = high

        cmap = getattr(k3d.colormaps.matplotlib_color_maps, input_cmap)
        tol = 1e-3
        colors = k3d.helpers.map_colors(
            distances,
            cmap,
            [-tol, options.max_distance_to_feature + tol]
        ).astype(np.uint32)

        plot += k3d.points(
            points,
            point_size=options.point_size,
            colors=colors,
            shader=options.point_shader,
            name=name_for_plot)

    print('Making snapshot...')
    plot.fetch_snapshot()
    time.sleep(10)
    with open(options.output, 'w') as f:
        f.write(plot.get_snapshot())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='inputs', action='append',
                        required=True, help='input files with prediction.')
    parser.add_argument('-o', '--output', dest='output',
                        required=True, help='output .html filename.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='be verbose')
    parser.add_argument('-s', '--max_distance_to_feature', dest='max_distance_to_feature',
                        default=1.0, type=float, required=False, help='max distance to sharp feature to display.')
    parser.add_argument('-ps', '--point_size', dest='point_size',
                        default=0.02, type=float, required=False,
                        help='point size for plotting (should equal resolution_3d).')
    parser.add_argument('-ph', '--point_shader', dest='point_shader',
                        default='flat', choices=['flat', '3d', 'mesh'], required=False,
                        help='point shader for plotting.')

    parser.add_argument('-bg', '--bgcolor', dest='bgcolor',
                        default=0xffffff, help='set background color for print.')
    parser.add_argument('-t', '--sharpness_hard_thr', dest='sharpness_hard_thr',
                        default=None, type=float, help='if set, forces to compute and paint hard labels.')
    parser.add_argument('-v', '--sharpness_hard_values', dest='sharpness_hard_values', nargs=2, default=None, type=float, 
                        help='if set, specifies min and max sharpness values for hard labels.')

    parser.add_argument('-icm', '--input_cmaps', dest='input_cmaps', action='append',
                        help='if specified, this should be a list of string colormaps (e.g. "plasma_r", '
                             'one per input file).')

    parser.add_argument('-k', '--key', dest='keys', action='append',
                        default=[], help='what key to use for getting preds.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
