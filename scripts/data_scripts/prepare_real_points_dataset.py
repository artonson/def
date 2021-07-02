#!/usr/bin/env python3

import argparse
from glob import glob
from io import BytesIO
import os
import sys

import igl
import numpy as np
import trimesh.transformations as tt
from tqdm import tqdm
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../..')
)
sys.path[1:1] = [__dir__]

from sharpf.data.annotation import ANNOTATOR_BY_TYPE
from sharpf.data.patch_cropping import patches_from_point_cloud
from sharpf.utils.py_utils.config import load_func_from_config
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.abc_utils.mesh.io import trimesh_load
from sharpf.utils.convertor_utils.convertors_io import ViewIO
from sharpf.utils.abc_utils.abc.feature_utils import (
    compute_features_nbhood,
    remove_boundary_features,
    submesh_from_hit_surfaces)
from sharpf.data.datasets.sharpf_io import save_whole_patches, WholePointCloudIO
import sharpf.data.data_smells as smells

DEFAULT_PATCH_SIZE = 4096


def process_scans(
        dataset,
        obj_mesh,
        yml_features,
        item_id,
        max_point_mesh_distance_mm=1.0,
        max_distance_to_feature=1.0
):

    obj_alignment_transform = dataset[0]['obj_alignment']
    obj_scale = dataset[0]['obj_scale']
    mesh = obj_mesh.copy() \
        .apply_scale(obj_scale) \
        .apply_transform(obj_alignment_transform)

    # Get the whole point cloud from scans
    point_cloud = []
    for scan in tqdm(dataset, desc='Loading scans'):
        points = scan['points'].reshape((-1, 3))
        points_alignment = scan['points_alignment']

        aligned_points = tt.transform_points(points, points_alignment)
        point_cloud.append(aligned_points)

    point_cloud = np.concatenate(point_cloud)
    distance_sq, _, _ = igl.point_mesh_squared_distance(
        point_cloud,
        mesh.vertices,
        mesh.faces)
    close_enough_indexes = np.where(np.sqrt(distance_sq) < max_point_mesh_distance_mm)[0]
    print('{} out of {} points are closer to GT mesh than {} mm '
          'and have been selected for annotation'.format(
        len(close_enough_indexes), len(point_cloud), max_point_mesh_distance_mm))
    point_cloud = point_cloud[close_enough_indexes]

    n_patches = int(len(point_cloud) * 10 / DEFAULT_PATCH_SIZE)
    print('n_patches = ', n_patches)

    # Prepare annotation stuff
    annotation_config = {
        "type": "surface_based_aabb",
        "distance_upper_bound": max_distance_to_feature,
        "always_check_adjacent_surfaces": True,
        "distance_computation_method": 'geom',
    }
    annotator = load_func_from_config(ANNOTATOR_BY_TYPE, annotation_config)
    smell_sharpness_discontinuities = smells.SmellSharpnessDiscontinuities.from_config({})

    point_patches = []
    iterable = patches_from_point_cloud(point_cloud, n_patches)
    for patch_point_indexes in tqdm(iterable, desc='Cropping annotated patches'):
        if len(patch_point_indexes) == DEFAULT_PATCH_SIZE:
            points = point_cloud[patch_point_indexes]

            distance_sq, mesh_face_indexes, _ = igl.point_mesh_squared_distance(
                points,
                mesh.vertices,
                mesh.faces)
            indexes = np.where(np.sqrt(distance_sq) < max_point_mesh_distance_mm)[0]
            if len(indexes) < DEFAULT_PATCH_SIZE:
                continue

            nbhood, mesh_vertex_indexes, mesh_face_indexes = \
                submesh_from_hit_surfaces(mesh, yml_features, mesh_face_indexes)

#           mesh_vertex_indexes = np.unique(mesh.faces[mesh_face_indexes])
#           nbhood = reindex_zerobased(
#               mesh,
#               mesh_vertex_indexes,
#               mesh_face_indexes)

            # create annotations: condition the features onto the nbhood
            nbhood_features = compute_features_nbhood(
                mesh,
                yml_features,
                mesh_face_indexes,
                mesh_vertex_indexes=mesh_vertex_indexes)

            # remove vertices lying on the boundary (sharp edges found in 1 face only)
            nbhood_features = remove_boundary_features(
                nbhood,
                nbhood_features,
                how='edges')

            distances, directions, has_sharp = annotator.annotate(
                nbhood,
                nbhood_features,
                points)

            num_sharp_curves = len([curve for curve in nbhood_features['curves'] if curve['sharp']])
            num_surfaces = len(nbhood_features['surfaces'])
            has_smell_sharpness_discontinuities = smell_sharpness_discontinuities.run(points, distances)
            point_patches.append({
                'points': np.ravel(points),
                'normals': np.ravel(np.zeros_like(points)),
                'distances': np.ravel(distances),
                'directions': np.ravel(directions),
                'has_sharp': has_sharp,
                'orig_vert_indices': mesh_vertex_indexes,
                'orig_face_indexes': mesh_face_indexes,
                'indexes_in_whole': patch_point_indexes,
                'item_id': item_id,
                'num_sharp_curves': num_sharp_curves,
                'num_surfaces': num_surfaces,
                'has_smell_coarse_surfaces_by_num_faces': False,
                'has_smell_coarse_surfaces_by_angles': False,
                'has_smell_deviating_resolution': False,
                'has_smell_sharpness_discontinuities': has_smell_sharpness_discontinuities,
                'has_smell_bad_face_sampling': False,
                'has_smell_mismatching_surface_annotation': False,
            })

    print('Total {} patches'.format(len(point_patches)))

    return point_patches


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
    from sharpf.utils.py_utils.os import change_ext

    ground_truth_dataset = Hdf5File(
        output_filename,
        io=WholePointCloudIO,
        preload=PreloadTypes.LAZY,
        labels='*')

    ground_truth = [patch for patch in ground_truth_dataset]
    list_distances = [patch['distances'] for patch in ground_truth]
    list_indexes_in_whole = [patch['indexes_in_whole'].astype(int) for patch in ground_truth]
    list_points = [patch['points'].reshape((-1, 3)) for patch in ground_truth]
    n_points = np.concatenate([patch['indexes_in_whole'] for patch in ground_truth]).max() + 1

    fused_points_gt, fused_distances_gt, _ = fuse_points(
        n_points, list_distances, list_indexes_in_whole, list_points,
        max_distance_to_feature)


    plot_height = 768
    plot = k3d.plot(grid_visible=True, height=plot_height)

    colors = k3d.helpers.map_colors(
        fused_distances_gt,
        k3d.colormaps.matplotlib_color_maps.coolwarm_r,
        [0, max_distance_to_feature]
    ).astype(np.uint32)

    plot += k3d.points(
        fused_points_gt,
        point_size=0.25,
        colors=colors,
        shader='flat')

    plot.fetch_snapshot()

    time.sleep(3)

    output_html = change_ext(output_filename, '') + '_fused.html'
    with open(output_html, 'w') as f:
        f.write(plot.get_snapshot())


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
    )

    print('Writing output file...')
    save_whole_patches(output_patches, options.output_filename)

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
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
