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
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.plotting import display_depth_sharpness


def main(options):
    labels = []
    if options.depth_images:
        labels.append('image')

    if options.sharpness_images:
        labels.append('distances')

    if not options.depth_images and not options.sharpness_images:
        print('At least one of -di or -si must be set')

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

    depth_images_for_display = None
    if options.depth_images:
        list_images = [patch['image'] for patch in dataset]
        depth_images_for_display = [
            image[
                slice(ry // 2 - sy, ry // 2 + sy),
                slice(rx // 2 - sx, rx // 2 + sx)]
            for image in list_images]

    sharpness_images_for_display = None
    if options.sharpness_images:
        list_predictions = [patch['distances'] for patch in dataset]
        sharpness_images_for_display = [
            distances[
                slice(ry // 2 - sy, ry // 2 + sy),
                slice(rx // 2 - sx, rx // 2 + sx)]
            for distances in list_predictions]

    display_depth_sharpness(
        depth_images=depth_images_for_display,
        sharpness_images=sharpness_images_for_display,
        ncols=4,
        axes_size=(16, 16),
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
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
