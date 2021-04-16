#!/usr/bin/env python3

import argparse
from glob import glob
from io import BytesIO
import os
import sys

import igl
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
from sharpf.utils.convertor_utils.convertors_io import RangeVisionIO, write_realworld_views_to_hdf5, ViewIO
import sharpf.utils.convertor_utils.rangevision_utils as rv_utils
from sharpf.utils.camera_utils import matrix
from sharpf.utils.abc_utils.abc.feature_utils import get_curves_extents
from sharpf.utils.numpy_utils.transformations import transform_to_frame
from sharpf.utils.plotting import plot_views


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

    flip_z = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]])

    output_scans = []
    for scan_index, scan in enumerate(tqdm(dataset)):
        # The algorithm:
        # * check if scan is usable?
        #  1) load the raw scanned 3D points from HDF5 (directly exported from scanner)
        points = np.array(scan[rvn.points]).reshape((-1, 3))
        faces = np.array(scan[rvn.faces]).reshape((-1, 3))

        #  1a) we know that the scan must be mirrored, do this
        scan_mesh = trimesh.base.Trimesh(points, faces, process=False, validate=False)
        scan_mesh.vertices = tt.transform_points(scan_mesh.vertices, flip_z)
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

        #  2e) compute transform that would align scans in source frame
        manual_alignment_transform = transform_to_frame(
            manual_alignment_transform, np.linalg.inv(wrong_extrinsics))
        alignment_transform = manual_alignment_transform @ rv_alignment_transform
        alignment_transform = transform_to_frame(alignment_transform, flip_z)

        #  2d) load vertex_matrix, this is transform to board coordinate system
        scan_to_board_transform = np.linalg.inv(
            np.array(scan[rvn.vertex_matrix]).reshape((4, 4)))

        #  2d) compute transform that would align scans in board frame
        board_alignment_transform = transform_to_frame(alignment_transform, scan_to_board_transform)
        points_wrt_calib_board = tt.transform_points(points, scan_to_board_transform)

        #  3) compute transform that would align mesh to points in board frame
        obj_alignment_transform = flip_z @ np.linalg.inv(wrong_extrinsics) @ stl_transform
        obj_alignment_transform = scan_to_board_transform @ obj_alignment_transform

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


def debug_plot(views_iterable, obj_mesh, output_filename):
    def plot_alignment_quality(views, mesh):
        distances = np.concatenate(
            [np.sqrt(igl.point_mesh_squared_distance(
                view.depth, mesh.vertices, mesh.faces)[0])
             for view in views])

        plt.figure(figsize=(12, 6))
        _ = plt.hist(distances, bins=100, range=[0, 10])
        plt.gca().set_yscale('log')
        plt.gca().set_ylim([0, 1e5])
        plt.gca().set_xlim([0, 10])
        plt.gca().set_xlabel('Distance to ground truth CAD model, mm', fontsize=14)
        plt.gca().set_ylabel('Number of point samples', fontsize=14)
        plt.gca().tick_params(axis='both', which='major', labelsize=14)
        plt.gca().tick_params(axis='both', which='minor', labelsize=14)

        for q in [0.5, 0.95, 0.99]:
            q_val = np.quantile(distances, q)
            label = '{0:3.0f}%: {1:0.2f}'.format(q * 100, q_val)
            plt.axvline(q_val, 0, 1e5,
                        color=plt.get_cmap('rainbow')(q), linewidth=3,
                        label=label)
            print(label)
        plt.legend(fontsize=16, loc='upper right')

    import time
    import matplotlib.pyplot as plt

    from sharpf.utils.camera_utils.view import CameraView
    from sharpf.utils.plotting import display_depth_sharpness
    from sharpf.utils.py_utils.os import change_ext

    views = [
        CameraView(
            depth=tt.transform_points(
                scan['points'].reshape((-1, 3)),
                scan['points_alignment']),
            signal=None,
            faces=scan['faces'].reshape((-1, 3)),
            extrinsics=np.dot(scan['points_alignment'], scan['extrinsics']),
            intrinsics=scan['intrinsics'],
            state='points')
        for scan in views_iterable]

    obj_scale = views_iterable[0]['obj_scale']
    obj_alignment = views_iterable[0]['obj_alignment']
    mesh = obj_mesh.copy().apply_scale(obj_scale).apply_transform(obj_alignment)

    plot = plot_views(
        views, mesh,
        camera_l=2000,
        camera_w=1)
    plot.fetch_snapshot()
    time.sleep(10)
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

    plot_alignment_quality(views, mesh)
    output_png = change_ext(output_filename, '') + '_alignment_hist.png'
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
        item_id)

    print('Writing output file...')
    write_realworld_views_to_hdf5(options.output_filename, output_scans)

    if options.debug:
        print('Plotting debug figures...')
        saved_dataset = Hdf5File(
            options.output_filename,
            ViewIO,
            preload=PreloadTypes.LAZY,
            labels='*')
        debug_plot(saved_dataset, obj_mesh, options.output_filename)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-dir', dest='input_dir',
                        required=True, help='input directory with scans.')
    parser.add_argument('-o', '--output', dest='output_filename',
                        required=True, help='output .hdf5 filename.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='be verbose')
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                         help='produce debug output')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
