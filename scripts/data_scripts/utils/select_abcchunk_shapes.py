#!/usr/bin/env python3

import argparse
import os
import sys

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)

from typing import Mapping

sys.path[1:1] = [__dir__]


def parse_item_info(line: str):
    """
    Expected line format:
    <index> <item_id> [key=value]+
    e.g.:
    5000 00996964_0c378724bacc399c9c303d73_008 num_verts=13185 num_faces=26366 ...
    """
    index, item_id, keys_values = line.strip().split(maxsplit=2)
    info = {'index': index, 'item_id': item_id, 'params': {}}
    for key_value in keys_values.split():
        key, value = key_value.split('=')
        info['params'][key] = value
    return info


PATCH_TYPES = ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus', 'Revolution', 'Extrusion', 'BSpline', 'Other']
CURVE_TYPES = ['Line', 'Circle', 'Ellipse', 'BSpline', 'Other']


def main(options):
    input_file = options.input_file or sys.stdin
    output_file = options.output_file or sys.stdout

    def boolean_condition(info: Mapping) -> bool:
        num_verts = int(info['params']['num_verts'])
        num_sharp_curve_ellipse = int(info['params']['num_sharp_curve_ellipse'])
        num_sharp_curve_other = int(info['params']['num_sharp_curve_other'])

        # num_sharp_curves_by_type = []
        # for curve_type in CURVE_TYPES:
        #     key = f'num_sharp_curve_{curve_type.lower()}'
        #     value = int(info['params'][key])
        #     num_sharp_curves_by_type.append(value)
        # num_sharp_curves = sum(num_sharp_curves_by_type)
        #
        # OK_CURVE_TYPES = ['Line', 'Circle', 'BSpline']
        # num_ok_sharp_curves_by_type = []
        # for curve_type in OK_CURVE_TYPES:
        #     key = f'num_sharp_curve_{curve_type.lower()}'
        #     value = int(info['params'][key])
        #     num_ok_sharp_curves_by_type.append(value)
        # num_ok_sharp_curves = sum(num_ok_sharp_curves_by_type)
        # num_ok_sharp_curves == num_sharp_curves

        is_ok = \
            num_verts < 30000 and \
            num_sharp_curve_ellipse == 0 and \
            num_sharp_curve_other == 0
        return is_ok

    for line in input_file:
        info = parse_item_info(line)
        if boolean_condition(info):
            output_file.write('{index} {item_id}\n'.format(**info))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Given per-shape ABC item info in text format, '
                    'filter out item IDs with certain requirements.')

    parser.add_argument(
        '-i', '--input-file',
        dest='input_file',
        type=argparse.FileType('r'),
        help='if specified, this is the filename of the input file; '
             'otherwise, standard input will be used.')
    parser.add_argument(
        '-o', '--output-file',
        dest='output_file',
        type=argparse.FileType('w'),
        help='if specified, this is the filename of the output file; '
             'otherwise, standard output will be used.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
