#!/usr/bin/env python3

import argparse
from glob import glob
from io import BytesIO
import os
import sys

import numpy as np
import trimesh
import trimesh.transformations as tt
from tqdm import tqdm
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../..')
)
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.abc_utils.mesh.io import trimesh_load
from sharpf.utils.convertor_utils.meshlab_project_parsers import load_meshlab_project
from sharpf.utils.convertor_utils.convertors_io import RangeVisionIO, write_realworld_views_to_hdf5
import sharpf.utils.convertor_utils.rangevision_utils as rv_utils
from sharpf.utils.camera_utils import matrix
from sharpf.utils.abc_utils.abc.feature_utils import get_curves_extents


def scale_mesh(
        mesh,
        features,
        default_mesh_extent_mm,
        resolution_mm_per_point,
        short_curve_quantile=0.05,
        n_points_per_curve=4
):
    """Scale the mesh to achieve a desired sampling on feature curves.

    :param mesh: input mesh
    :param features: feature description for input mesh
    :param default_mesh_extent_mm: some default size of the mesh
    :param resolution_mm_per_point: inter-measurement distance (scanning resolution)
    :param short_curve_quantile: percentage of feature curves, below which we allow
                                 the curves to be undersampled
    :param n_points_per_curve: number of measurement samples that should be landing
                               on a typical feature curve along its linear spatial extent
    :return: scaled mesh
    """

    # First, make our mesh have some "default" spatial extent (in mm).
    scale_to_default = default_mesh_extent_mm / np.max(mesh.bounding_box.extents)
    mesh = mesh.apply_scale(scale_to_default)

    # Our goal is to sample each feature with a specified
    # number of point samples. For this, we first compute distribution of
    # feature curve extents (measured by bounding boxes) in "default" scale.
    curves_extents_mm = get_curves_extents(mesh, features)

    # We compute feature curve extents that we want sampled exactly
    # with the specified number of points.
    # Longer curves shall receive larger number of points,
    # while shorter curves will be under-sampled.
    default_curve_extent_mm = np.quantile(curves_extents_mm, short_curve_quantile)

    # We compute spatial extents that the specified number of points
    # must take when sampling the curve with the specified resolution.
    resolution_mm_per_curve = resolution_mm_per_point * n_points_per_curve

    scale_to_resolution = resolution_mm_per_curve / default_curve_extent_mm
    mesh = mesh.apply_scale(scale_to_resolution)

    return mesh, scale_to_default * scale_to_resolution


# parameters used to generate the fabricated mesh
DEFAULT_MESH_EXTENT_MM = 10.0  # mm
BASE_N_POINTS_PER_SHORT_CURVE = 8
BASE_RESOLUTION_3D = 0.15  # mm
SHORT_CURVE_QUANTILE = 0.25  # 25%
TARGET_RESOLUTION_3D = 1.0  # mm


def process_scans(
        dataset,
        obj_mesh,
        yml_features,
        manual_alignment_transforms,
        stl_transform,
        item_id,
        is_mirrored=False,
):
    rvn = rv_utils.RangeVisionNames

    #  1) Get obj scale
    obj_mesh, obj_scale = scale_mesh(
        obj_mesh.copy(),
        yml_features,
        DEFAULT_MESH_EXTENT_MM,
        TARGET_RESOLUTION_3D,
        short_curve_quantile=SHORT_CURVE_QUANTILE,
        n_points_per_curve=BASE_N_POINTS_PER_SHORT_CURVE)

    output_scans = []
    for scan_index, scan in tqdm(enumerate(dataset)):

        # The algorithm:
        # * check if scan is usable?
        #  1) load the raw scanned 3D points from HDF5 (directly exported from scanner)
        points = np.array(scan[rvn.points]).reshape((-1, 3))
        faces = np.array(scan[rvn.faces]).reshape((-1, 3))

        #  1a) if we know that the scan must be mirrored, do this
        if is_mirrored:
            scan_mesh = trimesh.base.Trimesh(points, faces, process=False, validate=False)
            scan_mesh.vertices[:, 2] = -scan_mesh.vertices[:, 2]
            scan_mesh.invert()
            points, faces = scan_mesh.vertices, scan_mesh.faces

        #  2a) load the automatic scanner alignment transformation
        rv_alignment_transform = np.array(scan[rvn.alignment]).reshape((4, 4))

        #  2b) load the transformation that we thought was the extrinsic of scanner (they are not)
        rxyz_euler_angles = np.array(scan[rvn.rxyz_euler_angles])
        translation = np.array(scan[rvn.translation])
        wrong_extrinsics = rv_utils.get_camera_extrinsic(
            rxyz_euler_angles, translation).camera_to_world_4x4

        #  2c) load manual alignments from meshlab project
        manual_alignment_transform = manual_alignment_transforms[scan_index]

        #  2d) load vertex_matrix, this is transform to board coordinate system
        scan_to_board_transform = np.linalg.inv(
            np.array(scan[rvn.vertex_matrix]).reshape((4, 4)))

        #  2e) compute transform that would align scans in source frame
        fine_alignment_transform = np.dot(
            np.linalg.inv(wrong_extrinsics),
            np.dot(
                manual_alignment_transform,
                wrong_extrinsics))
        alignment_transform = np.dot(fine_alignment_transform, rv_alignment_transform)

        #  2d) compute transform that would align scans in board frame
        board_alignment_transform = np.dot(
            scan_to_board_transform,
            np.dot(
                alignment_transform,
                np.linalg.inv(scan_to_board_transform))
        )
        points_wrt_calib_board = tt.transform_points(points, scan_to_board_transform)

        #  3) compute transform that would align mesh to points in board frame
        obj_fine_alignment_transform = np.dot(
            np.linalg.inv(wrong_extrinsics),
            stl_transform)

        obj_alignment_transform = np.dot(
            scan_to_board_transform,
            obj_fine_alignment_transform)

        #  4) get extrinsics of right camera
        right_extrinsics_4x4 = rv_utils.get_right_camera_extrinsics(
            rxyz_euler_angles, translation).camera_to_world_4x4

        #  5) get projection matrix
        focal_length = scan[rvn.focal_length]
        intrinsics_f = matrix.get_camera_intrinsic_f(focal_length)

        #  6) get intrinsics matrix
        pixel_size_xy = scan[rvn.pixel_size_xy]
        center_xy = scan[rvn.center_xy]
        intrinsics_s = matrix.get_camera_intrinsic_s(
            pixel_size_xy[0],
            pixel_size_xy[1],
            rv_utils.RV_SPECTRUM_CAM_RESOLUTION[0],
            rv_utils.RV_SPECTRUM_CAM_RESOLUTION[1])
        intrinsics = np.stack((intrinsics_f, intrinsics_s))

        output_scan = {
            'points': np.ravel(points_wrt_calib_board),
            'faces': np.ravel(faces),
            'points_alignment': board_alignment_transform,
            'extrinsics': right_extrinsics_4x4,
            'intrinsics': intrinsics,
            'obj_alignment': obj_alignment_transform,
            'obj_scale': obj_scale,
            'item_id': item_id,
        }
        output_scans.append(output_scan)

    return output_scans


def debug_plot(output_scans, obj_mesh, output_filename):
    import k3d
    import time
    import randomcolor
    import matplotlib.pyplot as plt

    from sharpf.utils.camera_utils.view import CameraView
    from sharpf.utils.plotting import display_depth_sharpness
    from sharpf.utils.py_utils.os import change_ext


    all_points = [
        tt.transform_points(
            scan['points'].reshape((-1, 3)),
            scan['points_alignment'])
        for scan in output_scans
    ]

    views = [
        CameraView(
            depth=scan['points'].reshape((-1, 3)),
            signal=None,
            faces=scan['faces'].reshape((-1, 3)),
            extrinsics=scan['extrinsics'],
            intrinsics=scan['intrinsics'],
            state='points')
        for scan in output_scans]
    processed_views = [view.to_pixels().to_points() for view in views]

    all_points_processed = [
        tt.transform_points(view.depth, scan['points_alignment'])
        for view, scan in zip(processed_views, output_scans)
    ]

    plot_height = 768
    plot = k3d.plot(grid_visible=True, height=plot_height)

    # rand_color = randomcolor.RandomColor()
    # color = rand_color.generate(hue='red')[0]
    # color = int('0x' + color[1:], 16)
    #
    plot += k3d.points(
        np.concatenate(all_points),
        point_size=0.25,
        color=0xff0000,
        shader='flat')

    plot += k3d.points(
        np.concatenate(all_points_processed),
        point_size=0.25,
        color=0x00ff00,
        shader='flat')

    obj_scale = output_scans[0]['obj_scale']
    obj_alignment = output_scans[0]['obj_alignment']
    mesh = obj_mesh.copy().apply_scale(obj_scale).apply_transform(obj_alignment)
    plot += k3d.mesh(
        mesh.vertices,
        mesh.faces,
        color=0xaaaaaa)

    plot.fetch_snapshot()

    time.sleep(3)

    output_html = change_ext(output_filename, '') + '_alignment.html'
    with open(output_html, 'w') as f:
        f.write(plot.get_snapshot())

    pixels_views = [v.to_pixels() for v in views]
    s = 256
    depth_images_for_display = [
        view.depth[
            slice(1536 // 2 - s, 1536 // 2 + s),
            slice(2048 // 2 - s, 2048 // 2 + s)]
        for view in pixels_views]
    display_depth_sharpness(
        depth_images=depth_images_for_display,
        ncols=4,
        axes_size=(16, 16))
    output_png = change_ext(output_filename, '') + '_depthmaps.png'
    plt.savefig(output_png)


def main(options):
    stl_filename = glob(os.path.join(options.input_dir, '*.stl'))[0]
    obj_filename = glob(os.path.join(options.input_dir, '*.obj'))[0]
    yml_filename = glob(os.path.join(options.input_dir, '*.yml'))[0]
    hdf5_filename = glob(os.path.join(options.input_dir, '*.hdf5'))[0]
    meshlab_filename = glob(os.path.join(options.input_dir, '*.mlp'))[0]
    item_id = os.path.basename(stl_filename).split('__')[0]

    print('Reading input data...', end='')
    with open(obj_filename, 'rb') as obj_file:
        print('obj...', end='')
        obj_mesh, _, _ = trimesh_load(BytesIO(obj_file.read()))

    with open(yml_filename, 'rb') as yml_file:
        print('yml...', end='')
        yml_features = yaml.load(BytesIO(yml_file.read()), Loader=yaml.Loader)

    print('meshlab...', end='')
    _, manual_alignment_transforms, _, stl_transform = load_meshlab_project(meshlab_filename)

    print('hdf5...')
    dataset = Hdf5File(
        hdf5_filename,
        io=RangeVisionIO,
        preload=PreloadTypes.LAZY,
        labels='*')

    print('Converting scans...')
    output_scans = process_scans(
        dataset,
        obj_mesh,
        yml_features,
        manual_alignment_transforms,
        stl_transform,
        item_id,
        is_mirrored=options.is_mirrored)

    print('Writing output file...')
    write_realworld_views_to_hdf5(options.output_filename, output_scans)

    if options.debug:
        print('Plotting debug figures...')
        debug_plot(output_scans, obj_mesh, options.output_filename)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-dir', dest='input_dir',
                        required=True, help='input directory with scans.')
    parser.add_argument('-o', '--output', dest='output_filename',
                        required=True, help='output .hdf5 filename.')
    parser.add_argument('--mirrored', dest='is_mirrored', action='store_true', default=False,
                        required=False, help='treat data as mirrored')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='be verbose')
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                         help='produce debug output')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
