#!/usr/bin/env python3

import argparse
from glob import glob
from io import BytesIO
import os
import sys
from typing import Mapping
import warnings

import igl
import numpy as np
import trimesh
import trimesh.transformations as tt
from tqdm import tqdm
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../..'))
sys.path[1:1] = [__dir__]

from sharpf.data.annotation import ANNOTATOR_BY_TYPE, AnnotatorFunc
from sharpf.utils.camera_utils.view import CameraView
from sharpf.utils.py_utils.config import load_func_from_config
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.abc_utils.mesh.io import trimesh_load
from sharpf.utils.convertor_utils.convertors_io import (
    ViewIO,
    write_annotated_views_to_hdf5)
from sharpf.utils.abc_utils.abc.feature_utils import (
    compute_features_nbhood,
    remove_boundary_features,
    submesh_from_hit_surfaces)
import sharpf.data.data_smells as smells
from sharpf.utils.numpy_utils.sliding_window import crop_around_nonzero, sliding_window_with_offset


def annotate_view(
        view: CameraView,
        view_alignment: np.ndarray,
        aligned_mesh: trimesh.base.Trimesh,
        features: Mapping,
        annotator: AnnotatorFunc,
        smell_sharpness_discontinuities: smells.DataSmell,
        max_point_mesh_distance=1.0,
        full_mesh=False,
):
    points = view.depth
    aligned_points = tt.transform_points(points, view_alignment)

    # select only well-aligned points
    point_mesh_distance, mesh_face_indexes, _ = igl.point_mesh_squared_distance(
        aligned_points,
        aligned_mesh.vertices,
        aligned_mesh.faces)
    well_aligned_mask = np.sqrt(point_mesh_distance) < max_point_mesh_distance
    aligned_points = aligned_points[well_aligned_mask]
    mesh_face_indexes = mesh_face_indexes[well_aligned_mask]
    view.depth = points[well_aligned_mask]

    if full_mesh:
        nbhood = aligned_mesh
        nbhood_features = features
        mesh_vertex_indexes = np.arange(len(aligned_mesh.vertices))
        mesh_face_indexes = np.arange(len(aligned_mesh.faces))

    else:
        nbhood, mesh_vertex_indexes, mesh_face_indexes = submesh_from_hit_surfaces(
            aligned_mesh,
            features,
            mesh_face_indexes)

        nbhood_features = compute_features_nbhood(
            aligned_mesh,
            features,
            mesh_face_indexes,
            mesh_vertex_indexes=mesh_vertex_indexes)

        nbhood_features = remove_boundary_features(
            nbhood,
            nbhood_features,
            how='edges')

    distances, directions, has_sharp = annotator.annotate(
        nbhood,
        nbhood_features,
        aligned_points)
    has_smell_sharpness_discontinuities = smell_sharpness_discontinuities.run(
        aligned_points, distances)

    view.signal = np.hstack((np.atleast_2d(distances).T, directions))
    pixel_view = view.to_pixels()

    num_sharp_curves = len([curve for curve in nbhood_features['curves'] if curve['sharp']])
    num_surfaces = len(nbhood_features['surfaces'])
    annotation = {
        'points': pixel_view.depth,
        'faces': np.ravel(pixel_view.faces),
        'distances': pixel_view.signal[:, :, 0],
        'directions': pixel_view.signal[:, :, 1:],
        'has_sharp': has_sharp,
        'orig_vert_indices': mesh_vertex_indexes,
        'orig_face_indexes': mesh_face_indexes,
        'num_sharp_curves': num_sharp_curves,
        'num_surfaces': num_surfaces,
        'has_smell_sharpness_discontinuities': has_smell_sharpness_discontinuities,
    }
    return annotation


def crop_patch_views(view, patch_size, step_size):
    pixel_view = view.to_pixels()
    crop_depth, (top, bottom, left, right) = crop_around_nonzero(
        pixel_view.depth, return_bbox=True)

    patch_views = []
    for patch_image, (y_off, x_off) in sliding_window_with_offset(
            crop_depth, patch_size, step_size):
        patch_view = pixel_view.copy()
        patch_view.depth = np.zeros_like(patch_view.depth)
        s_y, s_x = patch_image.shape
        patch_view.depth[
            top + y_off:top + y_off + s_y,
            left + x_off:left + x_off + s_x] = patch_image
        patch_view.to_points(inplace=True)
        patch_views.append(patch_view)

    return patch_views


def process_scans(
        dataset,
        obj_mesh,
        yml_features,
        item_id,
        max_point_mesh_distance=1.0,
        max_distance_to_feature=1.0,
        full_mesh=False,
        patch_size=None,
        step_size=None,
):
    if None is not patch_size and None is step_size:
        step_size = patch_size / 2

    if None is not step_size and None is patch_size:
        warnings.warn('step_size set but patch_size unknown, skipping')

    obj_alignment_transform = dataset[0]['obj_alignment']
    obj_scale = dataset[0]['obj_scale']
    mesh = obj_mesh.copy() \
        .apply_scale(obj_scale) \
        .apply_transform(obj_alignment_transform)

    views = [
        CameraView(
            depth=scan['points'].reshape((-1, 3)),
            signal=None,
            faces=scan['faces'].reshape((-1, 3)),
            extrinsics=scan['extrinsics'],
            intrinsics=scan['intrinsics'],
            state='points')
        for scan in tqdm(dataset, desc='Loading scans')]
    view_alignments = [scan['points_alignment'] for scan in dataset]

    # Prepare annotation stuff
    annotation_config = {
        "type": "surface_based_aabb",
        "distance_upper_bound": max_distance_to_feature,
        "always_check_adjacent_surfaces": True,
        "distance_computation_method": 'geom',
    }
    annotator = load_func_from_config(ANNOTATOR_BY_TYPE, annotation_config)
    smell_sharpness_discontinuities = smells.SmellSharpnessDiscontinuities.from_config({})

    depth_images = []
    for view, view_alignment in tqdm(zip(views, view_alignments), desc='Annotating depth views'):

        if None is not patch_size:
            views_for_annotation = crop_patch_views(
                view,
                (patch_size, patch_size),
                (step_size, step_size))
        else:
            views_for_annotation = [view]

        for view_for_annotation in views_for_annotation:
            annotation = annotate_view(
                view_for_annotation,
                view_alignment,
                mesh,
                yml_features,
                annotator,
                smell_sharpness_discontinuities,
                max_point_mesh_distance=max_point_mesh_distance,
                full_mesh=full_mesh)
            annotation.update({
                'points_alignment': view_alignment,
                'extrinsics': view_for_annotation.extrinsics,
                'intrinsics': view_for_annotation.intrinsics,
                'obj_alignment': obj_alignment_transform,
                'obj_scale': obj_scale,
                'item_id': item_id,
            })
            depth_images.append(annotation)

    print('Total {} patches'.format(len(depth_images)))

    return depth_images


def debug_plot(output_scans, obj_mesh, output_filename, max_distance_to_feature):
    def fuse_points(n_points, list_predictions, list_indexes_in_whole, list_points, max_distance_to_feature):
        fused_points = np.zeros((n_points, 3))
        fused_distances = np.ones(n_points) * np.inf
        # fused_directions = np.ones((n_points, 3)) * np.inf

        iterable = zip(list_predictions, list_indexes_in_whole, list_points)
        for distances, indexes, points in tqdm(iterable):
            fused_points[indexes] = points
            assign_mask = fused_distances[indexes] > distances
            fused_distances[indexes[assign_mask]] = np.minimum(distances[assign_mask], max_distance_to_feature)
            # fused_directions[indexes[assign_mask]] = directions[assign_mask]

        return fused_points, fused_distances, {}

    import k3d
    import time
    import matplotlib.pyplot as plt
    from sharpf.utils.py_utils.os import change_ext
    from sharpf.utils.plotting import display_depth_sharpness

    pixel_views = [
        CameraView(
            depth=scan['points'],
            signal=scan['distances'],
            faces=scan['faces'].reshape((-1, 3)),
            extrinsics=scan['extrinsics'],
            intrinsics=scan['intrinsics'],
            state='pixels')
        for scan in output_scans]
    views = [view.to_points() for view in pixel_views]

    list_distances, list_indexes_in_whole, list_points = [], [], []
    n_points = 0
    for view, scan in zip(views, output_scans):
        points_alignment = scan['points_alignment']
        list_indexes_in_whole.append(
            np.arange(n_points, n_points + len(view.depth)))
        list_points.append(tt.transform_points(view.depth, points_alignment))
        list_distances.append(view.signal)
        n_points += len(view.depth)

    fused_points_gt, fused_distances_gt, _ = fuse_points(
        n_points, list_distances, list_indexes_in_whole, list_points,
        max_distance_to_feature)


    plot_height = 768
    plot = k3d.plot(grid_visible=True, height=plot_height)

    tol = 1e-3
    colors = k3d.helpers.map_colors(
        fused_distances_gt,
        k3d.colormaps.matplotlib_color_maps.plasma_r,
        [-tol, options.max_distance_to_feature + tol]
    ).astype(np.uint32)

    plot += k3d.points(
        fused_points_gt,
        point_size=0.5,
        colors=colors,
        shader='3d')

    plot.fetch_snapshot()

    time.sleep(10)

    output_html = change_ext(output_filename, '') + '_fused.html'
    with open(output_html, 'w') as f:
        f.write(plot.get_snapshot())


    s = 256
    depth_images_for_display = [
        view.depth[
            slice(1536 // 2 - s, 1536 // 2 + s),
            slice(2048 // 2 - s, 2048 // 2 + s)]
        for view in pixel_views]
    sharpness_images_for_display = [
        view.signal[
            slice(1536 // 2 - s, 1536 // 2 + s),
            slice(2048 // 2 - s, 2048 // 2 + s)]
        for view in pixel_views]
    display_depth_sharpness(
        depth_images=depth_images_for_display,
        sharpness_images=sharpness_images_for_display,
        ncols=4,
        axes_size=(16, 16),
        max_sharpness=max_distance_to_feature,
        bgcolor='black')
    output_png = change_ext(output_filename, '') + '_depthmaps.png'
    plt.savefig(output_png)


def main(options):
    stl_filename = glob(os.path.join(options.input_dir, '*.stl'))[0]
    obj_filename = glob(os.path.join(options.input_dir, '*.obj'))[0]
    yml_filename = glob(os.path.join(options.input_dir, '*.yml'))[0]
    hdf5_filename = glob(os.path.join(options.input_dir, '*_preprocessed.hdf5'))[0]
    item_id = os.path.basename(stl_filename).split('__')[0]

    print('Reading input data...', end='')
    with open(obj_filename, 'rb') as obj_file:
        print('obj...', end='')
        obj_mesh, _, _ = trimesh_load(BytesIO(obj_file.read()))

    with open(yml_filename, 'rb') as yml_file:
        print('yml...', end='')
        yml_features = yaml.load(BytesIO(yml_file.read()), Loader=yaml.Loader)

    print('hdf5...')
    dataset = Hdf5File(
        hdf5_filename,
        io=ViewIO,
        preload=PreloadTypes.LAZY,
        labels='*')

    print('Converting scans...')
    output_patches = process_scans(
        dataset,
        obj_mesh,
        yml_features,
        item_id,
        options.max_point_mesh_distance,
        options.max_distance_to_feature,
        options.full_mesh,
        options.patch_size,
        options.step_size,
    )

    print('Writing output file...')
    write_annotated_views_to_hdf5(options.output_filename, output_patches)

    if options.debug:
        print('Plotting debug figures...')
        debug_plot(
            output_patches,
            obj_mesh,
            options.output_filename,
            max_distance_to_feature=options.max_distance_to_feature)


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
    parser.add_argument('-d', '--max_point_mesh_distance', dest='max_point_mesh_distance',
                        default=1.0, type=float, required=False, help='max distance from point to mesh.')
    parser.add_argument('-s', '--max_distance_to_feature', dest='max_distance_to_feature',
                        default=1.0, type=float, required=False, help='max distance to sharp feature to compute.')
    parser.add_argument('-f', '--full_mesh', action='store_true', default=False,
                        required=False, help='use the full mesh annotation (no removal of curves due to visibility).')
    parser.add_argument('-p', '--patch_size', dest='patch_size', default=None, type=int,
                        help='if set, specifies a size of the (square) patch to be used.')
    parser.add_argument('-t', '--step_size', dest='step_size', default=None, type=int,
                        help='if set, specifies a size of the stride to be used (otherwise set to patch_size / 2).')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
