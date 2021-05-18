#!/usr/bin/env python3

# This script is based on the DirectX format presented in
# http://paulbourke.net/dataformats/directx/
# (but may not 100% fully implement everything).

import argparse
import os
import sys

import numpy as np
import pyparsing as pp
from tqdm import tqdm

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../../../..')
)
sys.path[1:1] = [__dir__]

from sharpf.utils.convertor_utils.convertors_io import write_raw_rv_scans_to_hdf5
from sharpf.utils.convertor_utils.rangevision_utils import \
    RangeVisionNames as rvn, get_camera_extrinsic
import sharpf.utils.convertor_utils.directx_parsers as dp
from sharpf.utils.py_utils.os import change_ext


def create_parser():
    # parse header
    magic = pp.Literal('xof')
    version_major = pp.Combine(pp.Char(pp.nums)[2])
    version_minor = pp.Combine(pp.Char(pp.nums)[2])
    format_type = pp.Literal('txt')
    float_size = pp.Word(pp.nums)

    # example: 'xof 0302txt 0064'
    header = magic + version_major + version_minor + format_type + float_size

    # parse data objects
    literal_containers = [
        'Frame', 'Processing', 'RV_Calibration', 'FrameTransformMatrix',
        'Mesh']
    one_of_literals = pp.Or([pp.Literal(name) for name in literal_containers])
    alnum_ = pp.Word(pp.alphas + pp.nums + '_')
    opener = one_of_literals + alnum_[0, 2] + pp.Literal('{')
    closer = pp.Literal('}')
    content = pp.Word(pp.nums + ',.;-')
    data_objects = pp.nestedExpr(
        opener=opener,
        closer=closer,
        content=content)

    directx_parser = header + data_objects

    return directx_parser


def extract_scans(result: pp.ParseResults):
    header, frame_side = result[:-1], result[-1]
    frame_group = frame_side[1]

    scans = []
    for frame in tqdm(frame_group[1:]):
        calibration, transform, mesh = frame

        calibration_parser = dp.ParsableRVCalibration()
        calibration_parser.parse(calibration[:])

        mesh_parser = dp.ParsableMesh()
        mesh_parser.parse(mesh[:])

        alignment_parser = dp.ParsableMatrix4x4()
        alignment_parser.parse(transform[:])
        alignment_transform = np.array(alignment_parser.value).reshape((4, 4)).T

        points, faces = mesh_parser.value

        scans.append({
            rvn.points: np.ravel(points),
            rvn.faces: np.ravel(faces),
            rvn.vertex_matrix: np.array(calibration_parser.vertex_matrix.value),
            rvn.rxyz_euler_angles: np.array(calibration_parser.angles.value),
            rvn.translation: np.array(calibration_parser.translation.value),
            rvn.focal_length: np.array(calibration_parser.focal_length.value),
            rvn.pixel_size_xy: np.array(calibration_parser.pixel_size_xy.value),
            rvn.center_xy: np.array(calibration_parser.center_xy.value),
            rvn.alignment: alignment_transform,
        })

    return scans


def write_scans_to_ply(output_prefix, scans):
    import open3d as o3d
    import trimesh.transformations as tt

    def open3d_pointcloud_from_numpy(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def open3d_mesh_from_numpy(points, faces):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        return mesh

    for scan_index, scan in enumerate(scans):

        points = np.array(scan[rvn.points]).reshape((-1, 3))
        faces = np.array(scan[rvn.faces]).reshape((-1, 3))

        extrinsics = get_camera_extrinsic(
            scan[rvn.rxyz_euler_angles],
            scan[rvn.translation]).camera_to_world_4x4
        alignment = np.array(scan[rvn.alignment])

        transform = np.dot(extrinsics, alignment)
        points_global = tt.transform_points(points, transform)

        mesh = open3d_mesh_from_numpy(points_global, faces)
        output_filename_scan = "{}_{}.ply".format(output_prefix, scan_index)
        o3d.io.write_triangle_mesh(
            output_filename_scan,
            mesh,
            write_ascii=True)
        print(output_filename_scan)


def main(options):
    scan_name = change_ext(
        os.path.basename(options.input_filename), '')

    print('Reading input data...')
    with open(options.input_filename) as input_file:
        input_data = input_file.read()

    print('Creating parsers...')
    parser = create_parser()

    print('Parsing input file...')
    result = parser.parseString(input_data)

    print('Converting to standard representation...')
    scans = extract_scans(result)
    for scan_index, scan in enumerate(scans):
        scan.update({'scan_id': '{}_{}'.format(scan_name, scan_index)})

    if options.is_output_hdf5:
        print('Writing scans to output HDF5 container...')
        output_hdf5_filename = os.path.join(options.output_dir, scan_name + '.hdf5')
        write_raw_rv_scans_to_hdf5(output_hdf5_filename, scans)

    if options.is_output_ply:
        print('Writing scans to output PLY containers...')
        output_plyfile_prefix = os.path.join(options.output_dir, scan_name)
        write_scans_to_ply(output_plyfile_prefix, scans)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='input_filename',
                        required=True, help='input .x filename.')
    parser.add_argument('-o', '--output', dest='output_dir',
                        required=True, help='output directory.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    parser.add_argument('--hdf5', dest='is_output_hdf5', action='store_true', default=False,
                        required=False, help='produce HDF5 output (default: no).')
    parser.add_argument('--ply', dest='is_output_ply', action='store_true', default=False,
                        required=False, help='produce PLY output (default: no).')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)