#!/usr/bin/env python3

import argparse
from functools import partial

import h5py
import numpy as np
import pyparsing as pp

# This script is based on the DirectX format presented in
# http://paulbourke.net/dataformats/directx/
# (but may not 100% fully implement everything).
from tqdm import tqdm

import sharpf.utils.abc_utils.hdf5.io_struct as io_struct
import sharpf.utils.camera_utils.rangevision_utils as rv_utils
import sharpf.utils.convertor_utils.directx_parsers as dp


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


def transform_parse_results(result: pp.ParseResults):
    header, frame_side = result[:-1], result[-1]
    frame_group = frame_side[1]

    scans = []
    for frame in tqdm(frame_group[1:]):
        calibration, transform, mesh = frame

        calibration_parser = dp.ParsableRVCalibration()
        calibration_parser.parse(calibration[:])

        mesh_parser = dp.ParsableMesh()
        mesh_parser.parse(mesh[:])

        extrinsics = rv_utils.get_camera_extrinsic(
            calibration_parser.angles.value,
            calibration_parser.translation.value).camera_to_world_4x4

        intrinsics_f = rv_utils.get_camera_intrinsic_f(
            calibration_parser.focal_length.value)

        intrinsics_s = rv_utils.get_camera_intrinsic_s(
            calibration_parser.pixel_size_xy.value[0] * 1e3,
            calibration_parser.pixel_size_xy.value[1] * 1e3,
            calibration_parser.center_xy.value[0],
            calibration_parser.center_xy.value[1])
        intrinsics = np.dstack((intrinsics_f, intrinsics_s))

        alignment_parser = dp.ParsableMatrix4x4()
        alignment_parser.parse(transform[:])
        alignment_transform = np.array(alignment_parser.value).reshape((4, 4)).T

        points, _ = mesh_parser.value

        scans.append({
            'points': np.ravel(points),
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'alignment': alignment_transform,
            'item_id': '123',
        })

    return scans


def write_scans(output_filename, scans):
    RangeVisionIO = io_struct.HDF5IO({
        'points': io_struct.VarFloat64('points'),
        'extrinsics': io_struct.Float64('extrinsics'),
        'intrinsics': io_struct.Float64('intrinsics'),
        'alignment': io_struct.Float64('alignment'),
        'item_id': io_struct.AsciiString('item_id'),
    },
        len_label='has_sharp',
        compression='lzf')
    
    collate_fn = partial(io_struct.collate_mapping_with_io, io=RangeVisionIO)
    scans = collate_fn(scans)

    with h5py.File(output_filename, 'w') as f:
        for key in ['extrinsics', 'intrinsics', 'alignment']:
            RangeVisionIO.write(f, key, scans[key].numpy())
        RangeVisionIO.write(f, 'item_id', scans['item_id'])
        RangeVisionIO.write(f, 'points', scans['points'])


def main(options):
    print('Reading input data...')
    with open(options.input_filename) as input_file:
        input_data = input_file.read()

    print('Creating parsers...')
    parser = create_parser()

    print('Parsing input file...')
    result = parser.parseString(input_data)

    print('Converting to standard representation...')
    scans = transform_parse_results(result)

    print('Writing scans to output...')
    write_scans(options.output_filename, scans)


def parse_args():
    parser = argparse.ArgumentParser()

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
