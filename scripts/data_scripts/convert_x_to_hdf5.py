#!/usr/bin/env python3

import argparse

import pyparsing as pp

# This script is based on the DirectX format presented in
# http://paulbourke.net/dataformats/directx/
# (but may not 100% fully implement everything).

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


def transform_parsed_result(result):
    frame_obj = result[-1]
    processing, frame_group = frame_obj

    scans = []
    for frame in frame_group[1:]:
        calibration, transform, mesh = frame

        scans.append({
            'image': image,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics_f
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
