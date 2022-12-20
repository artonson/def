#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
from sharpf.utils.abc_utils.abc import feature_utils
from sharpf.utils.abc_utils.mesh.io import trimesh_load


def main(options):
    obj_filename = os.path.join(
        options.input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.OBJ.value,
            version='00'))
    feat_filename = os.path.join(
        options.input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.FEAT.value,
            version='00'))
    with ABCChunk([obj_filename, feat_filename]) as abc_data:
        for abc_id in options.abc_ids:
            # 00500041_5aa40dcd43fa0b14df9bdcf8_010__20.5mm.stl
            item_id, size_mm = abc_id.split('__')
            size_mm = float(size_mm[:-6])  # L

            abc_item = abc_data.get(item_id)
            mesh, _, _ = trimesh_load(abc_item.obj)
            size_mesh = np.max(mesh.bounding_box.extents)
            mesh = mesh.apply_scale(size_mm / size_mesh)
            features = yaml.load(abc_item.feat, Loader=yaml.Loader)

            sharp_curves_lengths = feature_utils.get_curves_extents(mesh, features)
            short_curve_quantile = 0.25
            size_q_mm = np.quantile(sharp_curves_lengths, short_curve_quantile)  # l
            n_points_per_short_curve = 8  # n
            grid_step_mm = size_q_mm / n_points_per_short_curve  # r
            grid_size_voxels = np.round(size_mm / grid_step_mm).astype(np.int_)  # N
            grid_size_depth = np.round(np.log2(grid_size_voxels)).astype(np.int_)  # d

            print(f'{size_mm:3.2f}\t{size_q_mm:3.2f}\t'
                  f'{grid_step_mm:3.2f}\t{grid_size_voxels}\t'
                  f'{grid_size_depth}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input-dir',
        dest='input_dir',
        required=True,
        help='input dir with the source dataset. '
             'The source dataset can be either a collection'
             'of .obj and .yml files (with the same name),'
             'or a .7z archived ABC dataset.')
    parser.add_argument(
        '-c', '--chunk',
        help='ABC chunk ID. ')

    parser.add_argument(
        '-a', '--abc-id',
        dest='abc_ids',
        nargs='*',
        help='ABC item id to process; '
             '(this should include the size the object was manufactured with, '
             'e.g., 00515801_589bdec38006660f7f91d8e2_005__50.5mm.stl).')

    parser.add_argument(
        '-v', '--verbose',
        dest='verbose',
        action='store_true',
        default=False,
        help='be verbose.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
