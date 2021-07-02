#!/usr/bin/env python3

import argparse
from glob import glob
import os
import sys

import numpy as np
from tqdm import tqdm

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../..')
)
sys.path[1:1] = [__dir__]

from sharpf.data.patch_cropping import patches_from_point_cloud
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.data.datasets.sharpf_io import save_whole_patches, WholePointCloudIO
import sharpf.data.data_smells as smells

DEFAULT_PATCH_SIZE = 4096
PATCH_SIZE_MINIMUM = 30

def process_fused_pc(
        dataset,
        item_id,
        sharpness_thresh,
        max_distance_to_feature=1.0
):

    point_cloud = dataset[0]['points']
    all_distances = dataset[0]['distances']
    n_patches = int(len(point_cloud) * 10 / DEFAULT_PATCH_SIZE)
    print('n_patches = ', n_patches)

    smell_sharpness_discontinuities = smells.SmellSharpnessDiscontinuities.from_config({})

    point_patches = []
    iterable = patches_from_point_cloud(point_cloud, n_patches)
    for patch_point_indexes in tqdm(iterable, desc='Cropping annotated patches'):
        if len(patch_point_indexes) > PATCH_SIZE_MINIMUM:
            points = point_cloud[patch_point_indexes]
            distances = all_distances[patch_point_indexes]
            has_smell_sharpness_discontinuities = smell_sharpness_discontinuities.run(points, distances)
            point_patches.append({
                'points': np.ravel(points),
                'normals': np.ravel(np.zeros_like(points)),
                'distances': np.ravel(distances),
                'directions': np.ravel(np.zeros_like(distances)),
                'has_sharp': any(distances < sharpness_thresh),
                'orig_vert_indices': np.ravel(np.zeros_like(patch_point_indexes)),
                'orig_face_indexes': np.ravel(np.zeros_like(patch_point_indexes)),
                'indexes_in_whole': patch_point_indexes,
                'item_id': item_id,
                'num_sharp_curves': -1,
                'num_surfaces': -1,
                'has_smell_coarse_surfaces_by_num_faces': False,
                'has_smell_coarse_surfaces_by_angles': False,
                'has_smell_deviating_resolution': False,
                'has_smell_sharpness_discontinuities': has_smell_sharpness_discontinuities,
                'has_smell_bad_face_sampling': False,
                'has_smell_mismatching_surface_annotation': False,
            })

    print('Total {} patches'.format(len(point_patches)))

    return point_patches


def main(options):
    hdf5_filename = glob(os.path.join(options.input_dir, '*__ground_truth.hdf5'))[0]
    item_id = os.path.basename(hdf5_filename).split('__')[0]

    print('Reading input data...', end='')

    print('hdf5...')
    dataset = Hdf5File(
        hdf5_filename,
        io=WholePointCloudIO,
        preload=PreloadTypes.LAZY,
        labels='*')

    print('Converting fused pointcloud...')
    output_patches = process_fused_pc(
        dataset,
        item_id,
        options.sharpness_thresh,
        options.max_distance_to_feature,
    )

    print('Writing output file...')
    if output_patches:
        save_whole_patches(output_patches, options.output_filename)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-dir', dest='input_dir',
                        required=True, help='input directory with scans.')
    parser.add_argument('-o', '--output', dest='output_filename',
                        required=True, help='output .hdf5 filename.')
    parser.add_argument('-sht', '--sharpness_thresh', dest='sharpness_thresh',
                        required=True, default=0.02, type=float, help='sharpness threshold')                   
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='be verbose')
    parser.add_argument('-s', '--max_distance_to_feature', dest='max_distance_to_feature',
                        default=1.0, type=float, required=False, help='max distance to sharp feature to compute.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
