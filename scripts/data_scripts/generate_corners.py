#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..'))
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
import sharpf.utils.abc_utils.abc.feature_utils as feat
from sharpf.utils.abc_utils.mesh.io import trimesh_load


def scale_mesh(
        mesh,
        features,
        default_mesh_extent_mm,
        resolution_mm_per_point,
        short_curve_quantile=0.05,
        n_points_per_curve=4
):
    """Scale the mesh to achieve a desired sampling on feature curves.

    :param mesh: input mesh
    :param features: feature description for input mesh
    :param default_mesh_extent_mm: some default size of the mesh
    :param resolution_mm_per_point: inter-measurement distance (scanning resolution)
    :param short_curve_quantile: percentage of feature curves, below which we allow
                                 the curves to be undersampled
    :param n_points_per_curve: number of measurement samples that should be landing
                               on a typical feature curve along its linear spatial extent
    :return: scaled mesh
    """

    # First, make our mesh have some "default" spatial extent (in mm).
    scale_to_default = default_mesh_extent_mm / np.max(mesh.bounding_box.extents)
    mesh = mesh.apply_scale(scale_to_default)

    # Our goal is to sample each feature with a specified
    # number of point samples. For this, we first compute distribution of
    # feature curve extents (measured by bounding boxes) in "default" scale.
    curves_extents_mm = feat.get_curves_extents(mesh, features)

    # We compute feature curve extents that we want sampled exactly
    # with the specified number of points.
    # Longer curves shall receive larger number of points,
    # while shorter curves will be under-sampled.
    default_curve_extent_mm = np.quantile(curves_extents_mm, short_curve_quantile)

    # We compute spatial extents that the specified number of points
    # must take when sampling the curve with the specified resolution.
    resolution_mm_per_curve = resolution_mm_per_point * n_points_per_curve

    scale_to_resolution = resolution_mm_per_curve / default_curve_extent_mm
    mesh = mesh.apply_scale(scale_to_resolution)

    return mesh, scale_to_default * scale_to_resolution



def get_corners(abc_item, config):
    mesh, _, _ = trimesh_load(abc_item.obj)
    features = yaml.load(abc_item.feat, Loader=yaml.Loader)

    shape_fabrication_extent = config.get('shape_fabrication_extent', 10.0)
    base_n_points_per_short_curve = config.get('base_n_points_per_short_curve', 8)
    base_resolution_3d = config.get('base_resolution_3d', 0.125)
    short_curve_quantile = config.get('short_curve_quantile', 0.05)
    mesh, _ = scale_mesh(
        mesh,
        features,
        shape_fabrication_extent,
        base_resolution_3d,
        short_curve_quantile=short_curve_quantile,
        n_points_per_curve=base_n_points_per_short_curve)

    corner_vertex_indexes = feat.get_corner_vertex_indexes(features)
    corners_xyz = mesh.vertices[corner_vertex_indexes]
    return corners_xyz


def make_patches(options):
    obj_filename = os.path.join(
        options.data_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.OBJ.value,
            version='00'))
    feat_filename = os.path.join(
        options.data_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.FEAT.value,
            version='00'))

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)

    with ABCChunk([obj_filename, feat_filename]) as abc_data:
        abc_item = abc_data.get_one(options.input_shape_id)
        corner_xyz = get_corners(abc_item, config)
        np.savetxt(options.output_filename, corner_xyz)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--data-dir', dest='data_dir',
                        required=True, help='input dir with ABC dataset.')
    # parser.add_argument('-i', '--input-filename', dest='input_filename',
    #                     required=True, help='input filename with fused model.')
    parser.add_argument('-id', '--input-shape-id', dest='input_shape_id',
                        required=True, help='input_shape_id')
    parser.add_argument('-c', '--chunk', required=True, help='ABC chunk id to process.')
    parser.add_argument('-o', '--output-filename', dest='output_filename',
                        required=True, help='output filename.')
    # parser.add_argument('-thr', '--annotation-threshold', dest='output_filename',
    #                     required=True, help='output filename.')
    parser.add_argument('-g', '--dataset-config', dest='dataset_config',
                        required=True, help='dataset configuration file.')

    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    make_patches(options)
