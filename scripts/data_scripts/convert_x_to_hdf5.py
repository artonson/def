#!/usr/bin/env python3

import argparse

import numpy as np
import pyparsing as pp

# This script is based on the DirectX format presented in
# http://paulbourke.net/dataformats/directx/
# (but may not 100% fully implement everything).
import sharpf.utils.camera_utils.rangevision_utils as rv_utils


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
    opener = (
            pp.Literal('Frame') ^
            pp.Literal('Processing') ^
            pp.Literal('RV_Calibration') ^
            pp.Literal('FrameTransformMatrix') ^
            pp.Literal('Mesh')) + \
        pp.Word(pp.alphas + pp.nums + '_')[0,2] + \
        pp.Literal('{')
    closer = pp.Literal('}')
    content = pp.Word(pp.nums + ',.;-')
    data_objects = pp.nestedExpr(
        opener=opener,
        closer=closer,
        content=content)

    directx_parser = header + data_objects

    return directx_parser


def get_floats(s):
    s = s.strip(';,')
    return [float(value) for value in s.split(',')]


def transform_parsed_result(result):
    header, frame_side = result[:-1], result[-1]
    frame_group = frame_side[1]

    scans = []
    for frame in frame_group[1:]:
        calibration, transform, mesh = frame

        rotation_matrix, \
            focal_length, \
            angles, \
            translation, \
            pixel_xy, \
            center_xy, \
            correction = [get_floats(part) for part in calibration]

        extrinsics = rv_utils.get_camera_extrinsic(angles, translation)
        intrinsics_f = rv_utils.get_camera_intrinsic_f(focal_length[0])
        intrinsics_s = rv_utils.get_camera_intrinsic_s(
            pixel_xy[0] * 1e3,
            pixel_xy[1] * 1e3,
            center_xy[0],
            center_xy[1])
        intrinsics = np.dstack((intrinsics_f, intrinsics_s))

        alignment_transform = get_floats(transform[0])
        alignment_transform = np.array(alignment_transform).reshape((4, 4)).T

        scans.append({
            'image': image,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        })


    return scans


def main(options):

    print('Reading input data...')
    with open(options.input_filename) as input_file:
        input_data = input_file.read()

    print('Creating parsers...')
    parser = create_parser()

    print('Parsing input file...')
    result = parser.parseString(input_data)

    print('Converting to standard representation...')
    scans = transform_parsed_result(result)

    print('Writing...')
    options.output_filename


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=4, help='CPU jobs to use in parallel [default: 4].')
    parser.add_argument('-i', '--input', dest='input_filename',
                        required=True, help='input .x filename.')
    parser.add_argument('-o', '--output', dest='output_filename',
                        required=True, help='output .hdf5 filename.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
