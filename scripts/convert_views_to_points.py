#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path[1:1] = [__dir__]

from sharpf.data.datasets import sharpf_io
from sharpf.utils.camera_utils.view import CameraView
from sharpf.utils.convertor_utils import convertors_io
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes


def main(options):
    ground_truth_dataset = Hdf5File(
        options.input_filename,
        io=convertors_io.AnnotatedViewIO,
        preload=PreloadTypes.LAZY,
        labels='*')

    views = [
        CameraView(
            depth=scan['points'],
            signal=scan['distances'],
            faces=scan['faces'].reshape((-1, 3)),
            extrinsics=scan['extrinsics'],
            intrinsics=scan['intrinsics'],
            state='pixels')
        for scan in ground_truth_dataset]

    point_patches = []
    for view in tqdm(views):
        view = view.to_points()

        point_patches.append({
            'points': np.ravel(view.depth),
            'distances': np.array(view.signal),
            'normals': [],
            'directions': [],
            'indexes_in_whole': [],
            'item_id': os.path.basename(options.input_filename),
            'orig_vert_indices': [],
            'orig_face_indexes': [],
            'has_sharp': True,
            'num_sharp_curves': 0,
            'num_surfaces': 0,
            'has_smell_coarse_surfaces_by_num_faces': False,
            'has_smell_coarse_surfaces_by_angles': False,
            'has_smell_deviating_resolution': False,
            'has_smell_sharpness_discontinuities': False,
            'has_smell_bad_face_sampling': False,
            'has_smell_mismatching_surface_annotation': False,
        })


    sharpf_io.save_whole_patches(point_patches, options.output_filename)
    print(options.output_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='be verbose')

    parser.add_argument('-i', '--input_filename', dest='input_filename',
                        required=True, help='input file with AnnotatedViewIO type.')
    parser.add_argument('-o', '--output_filename', dest='output_filename',
                        required=True, help='output filename with .')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
