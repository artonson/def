#!/usr/bin/env python3

import argparse
from glob import glob
from io import BytesIO
import os
import sys

import numpy as np
import trimesh.transformations as tt
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../../../..')
)
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.abc_utils.mesh.io import trimesh_load
from sharpf.utils.convertor_utils.meshlab_project_parsers import load_meshlab_project
import sharpf.utils.convertor_utils.rangevision_utils as rv_utils


def main(options):
    stl_filename = glob(os.path.join(options.input_dir, '*.stl'))[0]
    obj_filename = glob(os.path.join(options.input_dir, '*.obj'))[0]
    yml_filename = glob(os.path.join(options.input_dir, '*.yml'))[0]
    hdf5_filename = glob(os.path.join(options.input_dir, '*.hdf5'))[0]
    meshlab_filename = glob(os.path.join(options.input_dir, '*.mlp'))[0]

    print('Reading input data...')
    with open(obj_filename, 'rb') as obj_file:
        obj_mesh, _, _ = trimesh_load(BytesIO(obj_file.read()))

    with open(yml_filename, 'rb') as yml_file:
        yml_features = yaml.load(BytesIO(yml_file.read()), Loader=yaml.Loader)

    _, manual_alignment_transforms, _ = load_meshlab_project(meshlab_filename)

    dataset = Hdf5File(
        hdf5_filename,
        io=RangeVisionIO,
        preload=PreloadTypes.LAZY,
        labels='*')

    rvn = rv_utils.RangeVisionNames

    for scan_index, scan in enumerate(dataset):

        # The algorithm:
        # * check if scan is usable?
        #  1) load the raw scanned 3D points from HDF5 (directly exported from scanner)
        points = np.array(scan[rvn.points]).reshape((-1, 3))
        faces = np.array(scan[rvn.faces]).reshape((-1, 3))

        #  2a) load the automatic scanner alignment transformation
        rv_alignment_transform = np.array(scan[rvn.alignment]).reshape((4, 4))

        #  2b) apply scanner alignments to approximately align scans
        points_aligned = tt.transform_points(points, rv_alignment_transform)

        #  3a) load the transformation that we thought was the extrinsic of scanner (they are not)
        rxyz_euler_angles = np.array(scan[rvn.rxyz_euler_angles])
        translation = np.array(scan[rvn.translation])
        wrong_extrinsics = rv_utils.get_camera_extrinsic(
            rxyz_euler_angles, translation).camera_to_world_4x4

        #  3b) transform points to wrong global frame
        points_wrong_aligned = tt.transform_points(points_aligned, wrong_extrinsics)

        #  4a) load manual alignments from meshlab project
        manual_alignment_transform = manual_alignment_transforms[scan_index]

        #  4b) apply manual alignments, mirror if necessary
        points_wrong_aligned = tt.transform_points(points_wrong_aligned, manual_alignment_transform)

        # >>>> You are here: we have precisely aligned scans <<<<

        # * get obj aligned / mirrored
        # * annotate
        #
        #  2) load the scanner-to-calibration board transformation
        # * transform back to scanner frame
        # * transform to right camera frame
        # * project / pixelize







        scan_to_board_transform = np.array(scan[rvn.vertex_matrix]).reshape((4, 4))


        intrinsics_f = rv_utils.get_camera_intrinsic_f(
            calibration_parser.focal_length.value)

        intrinsics_s = rv_utils.get_camera_intrinsic_s(
            calibration_parser.pixel_size_xy.value[0] * 1e3,
            calibration_parser.pixel_size_xy.value[1] * 1e3,
            calibration_parser.center_xy.value[0],
            calibration_parser.center_xy.value[1])
        intrinsics = np.dstack((intrinsics_f, intrinsics_s))




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
