#!/usr/bin/env python3

import argparse
import glob
import os
import sys

import joblib
import trimesh


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def change_ext(filename, new_ext):
    name, old_ext = os.path.splitext(filename)
    return name + new_ext


def convert_svg_to_txt(svg_filename, txt_filename):
    path = trimesh.load(svg_filename)

    output_vertices, output_edges = [], []
    num_vertices = 0

    p = path.polygons_full
    num_edges = 0
    for i in range(0, len(p)):
        x, y = p[i].exterior.coords.xy
        start = num_vertices
        for j in range(start, start + len(x)):
            num_vertices += 1
            # exterior vertices
            output_vertices.append((x[j - start], y[j - start]))

        for j in range(start, num_vertices - 1):
            num_edges += 1
            output_edges.append((j, j + 1))

        for interior in p[i].interiors:
            t, s = interior.coords.xy
            start = num_vertices
            for k in range(start, start + len(t)):
                num_vertices += 1
                output_vertices.append((t[k - start], s[k - start]))

            for k in range(start, num_vertices - 1):
                num_edges += 1
                output_edges.append((k, k + 1))

    with open(txt_filename, 'w') as output_file:
        output_file.write('{} {}\n'.format(num_vertices, num_edges))
        output_file.writelines('\n'.join(
            ['{:.3f} {:.3f}'.format(x, y) for x, y in output_vertices]
        ))
        output_file.write('\n')
        output_file.writelines('\n'.join(
            ['{} {}'.format(i, j) for i, j in output_edges]
        ))


def main(options):
#    if os.path.exists(options.txt_output) and not options.overwrite:
#        eprint('\'{}\' exists and no --overwrite specified; exiting'.format(options.txt_output))
#    elif :
#
#
    if None is not options.svg_input:
        input_svg_files = [options.svg_input]
        if None is not options.txt_output:
            output_txt_files = [options.txt_output]
        else:
            assert None is not options.txt_output_dir
            svg_filename = os.path.basename(options.svg_input)
            output_txt_files = [os.path.join(options.txt_output_dir, change_ext(svg_filename, '.txt'))]
    else:
        assert None is not options.txt_output_dir and None is options.txt_output, \
            'Could not output a whole directory into a single file \'{}\''.format(options.txt_output)
        input_svg_files = glob.glob(os.path.join(options.svg_input_dir, '*.svg'))
        output_txt_files = []
        for svg_filename in input_svg_files:
            svg_filename = os.path.basename(svg_filename)
            txt_filename = os.path.join(options.txt_output_dir, change_ext(svg_filename, '.txt'))
            output_txt_files.append(txt_filename)

    assert None is not input_svg_files and \
        None is not output_txt_files and \
        len(input_svg_files) == len(output_txt_files)

    for svg_filename, txt_filename in zip(input_svg_files, output_txt_files):
        eprint('Processing {}'.format(svg_filename))
        try:
            convert_svg_to_txt(svg_filename, txt_filename)
        except Exception as e:
            eprint('Could not process {}: {}'.format(svg_filename, e))


def parse_options():
   parser = argparse.ArgumentParser(description='Sample point cloud from mesh.')

   group = parser.add_mutually_exclusive_group(required=True)
   group.add_argument('-w', '--overwrite-all', dest='overwrite_all', action='store_true',
       default=False, help='overwrite existing files.')
   group.add_argument('-e', '--overwrite-missing', dest='overwrite_all', action='store_true',
                      default=False, help='try to  existing files.')

   group = parser.add_mutually_exclusive_group(required=True)
   group.add_argument('-i', '--input', dest='svg_input', help='SVG input file.')
   group.add_argument('--input-dir', dest='svg_input_dir', help='directory of SVG input files.')

   group = parser.add_mutually_exclusive_group(required=True)
   group.add_argument('-o', '--output', dest='txt_output', help='TXT output file.')
   group.add_argument('--output-dir', dest='txt_output_dir', help='directory with TXT output files.')

   args = parser.parse_args()

   if None is not args.svg_input_dir and args.txt_output:
       raise ValueError('Could not output a whole directory into a single file \'{}\''.format(args.txt_output))
   if None is not args.txt_output and os.path.isdir(args.txt_output):
       eprint('\'{}\' is a dir specified by -o switch; selecting it as output-dir'.format(args.txt_output))
       args.txt_output_dir = args.txt_output
       args.txt_output = None
   return args


if __name__ == '__main__':
    options = parse_options()
    main(options)
