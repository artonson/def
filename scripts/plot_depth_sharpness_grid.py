#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)
sys.path[1:1] = [__dir__]

from sharpf.data.datasets.sharpf_io import WholeDepthMapIO
from sharpf.utils.convertor_utils.convertors_io import AnnotatedViewIO
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.camera_utils.view import CameraView
from sharpf.utils.plotting import display_depth_sharpness


def main(options):
    labels = []
    if options.depth_images:
        labels.append('image')

    if options.sharpness_images:
        labels.append('distances')

    if not options.depth_images and not options.sharpness_images:
        print('At least one of -di or -si must be set')

    if options.real_world:
        dataset = Hdf5File(
            options.input_filename,
            io=AnnotatedViewIO,
            preload=PreloadTypes.LAZY,
            labels='*')
        pixel_views = [
            CameraView(
                depth=scan['points'],
                signal=scan['distances'],
                faces=scan['faces'].reshape((-1, 3)),
                extrinsics=scan['extrinsics'],
                intrinsics=scan['intrinsics'],
                state='pixels')
            for scan in dataset]
        dataset = [
            {'image': view.depth, 'distances': view.signal}
            for view in pixel_views]

    else:
        dataset = Hdf5File(
            options.input_filename,
            io=WholeDepthMapIO,
            preload=PreloadTypes.LAZY,
            labels=labels)

    rx, ry = map(int, options.resolution)
    if None is not options.crop_size:
        sy, sx = options.crop_size, options.crop_size
    elif options.depth_images:
        list_images = [patch['image'] for patch in dataset]
        sy, sx = list_images[0].shape
    elif options.sharpness_images:
        list_predictions = [patch['distances'] for patch in dataset]
        sy, sx = list_predictions[0].shape

    slices = slice(ry // 2 - sy // 2, ry // 2 + sy // 2),\
             slice(rx // 2 - sx // 2, rx // 2 + sx // 2)

    depth_images_for_display = None
    if options.depth_images:
        list_images = [patch['image'] for patch in dataset]
        depth_images_for_display = [image[slices] for image in list_images]

    sharpness_images_for_display = None
    if options.sharpness_images:
        list_predictions = [patch['distances'] for patch in dataset]
        sharpness_images_for_display = [distances[slices] for distances in list_predictions]

    f_x, f_y = map(float, options.figsize)
    display_depth_sharpness(
        depth_images=depth_images_for_display,
        sharpness_images=sharpness_images_for_display,
        ncols=options.ncols,
        axes_size=(f_x, f_y),
        max_sharpness=options.max_distance_to_feature)

    plt.savefig(options.output_filename)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='input_filename',
                        required=True, help='input files with prediction.')
    parser.add_argument('-o', '--output', dest='output_filename',
                        required=True, help='output .png filename.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='be verbose')
    parser.add_argument('-s', '--max_distance_to_feature', dest='max_distance_to_feature',
                        default=1.0, type=float, required=False, help='max distance to sharp feature to compute.')
    parser.add_argument('-c', '--crop_size', dest='crop_size',
                        default=None, type=int, required=False, help='make a crop of pixels of this size.')
    parser.add_argument('-di', '--depth_images', dest='depth_images',
                        default=False, action='store_true', required=False,
                        help='display depth images.')
    parser.add_argument('-si', '--sharpness_images', dest='sharpness_images',
                        default=False, action='store_true', required=False,
                        help='display sharpness images.')
    parser.add_argument('-r', '--resolution', dest='resolution', nargs=2,
                        required=True, help='resolution of the input image in pixels [width, height].')
    parser.add_argument('--ncols', dest='ncols',
                        default=1, type=int, required=False, help='number of cols.')
    parser.add_argument('-f', '--figsize', dest='figsize', nargs=2, default=(16, 16),
                        required=False, help='figure size in inches.')
    parser.add_argument('-w', '--real_world', dest='real_world',
                        default=False, action='store_true', required=False,
                        help='if set, this will read the input file as Views.')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
