#!/usr/bin/env python3

import argparse
import json
from copy import deepcopy
import os
import sys

import h5py
from joblib import Parallel, delayed
import numpy as np
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)

sys.path[1:1] = [__dir__]

from sharpf.data import DataGenerationException
from sharpf.data.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
from sharpf.data.annotation import ANNOTATOR_BY_TYPE
from sharpf.data.mesh_nbhoods import NBHOOD_BY_TYPE
from sharpf.data.noisers import NOISE_BY_TYPE
from sharpf.data.point_samplers import SAMPLER_BY_TYPE
from sharpf.utils.common import eprint
from sharpf.utils.mesh_utils.io import trimesh_load


def load_func_from_config(func_dict, config):
    return func_dict[config['type']].from_config(config)


def compute_curves_nbhood(features, vert_indices, face_indexes):
    """Extracts curves for the neighbourhood."""
    nbhood_sharp_curves = []
    for curve in features['curves']:
        nbhood_vert_indices = np.array([
            vert_index for vert_index in curve['vert_indices']
            if vert_index in vert_indices
        ])
        if len(nbhood_vert_indices) == 0:
            continue
        for index, reindex in zip(vert_indices, np.arange(len(vert_indices))):
            nbhood_vert_indices[np.where(nbhood_vert_indices == index)] = reindex
        nbhood_curve = deepcopy(curve)
        nbhood_curve['vert_indices'] = nbhood_vert_indices
        nbhood_sharp_curves.append(nbhood_curve)

    nbhood_features = {'curves': nbhood_sharp_curves}
    return nbhood_features


def remove_boundary_features(mesh, features, how='none'):
    """Removes features indexed into vertex edges adjacent to 1 face only.
    :param how: 'all_verts': remove entire feature curve if all vertices are boundary
                'edges': remove vertices that belong to boundary edges only (not to other edges)
                'verts': remove vertices that are boundary
                'none': do nothing
    """
    if how == 'none':
        return features

    mesh_edge_indexes, mesh_edge_counts = np.unique(
        mesh.faces_unique_edges.flatten(), return_counts=True)

    boundary_edges = mesh.edges_unique[mesh_edge_indexes[np.where(mesh_edge_counts == 1)[0]]]
    boundary_vertex_indexes = np.unique(boundary_edges.flatten())

    non_boundary_curves = []
    for curve in features['curves']:
        non_boundary_curve = deepcopy(curve)

        if how == 'all_verts':
            if np.all([vert_index in boundary_vertex_indexes
                       for vert_index in curve['vert_indices']]):
                continue

        elif how == 'verts':
            non_boundary_vert_indices = np.array([
                vert_index for vert_index in curve['vert_indices']
                if vert_index not in boundary_vertex_indexes
            ])
            if len(non_boundary_vert_indices) == 0:
                continue
            non_boundary_curve['vert_indices'] = non_boundary_vert_indices

        elif how == 'edges':
            curve_edges = mesh.edges_unique[
                np.where(
                    np.all(np.isin(mesh.edges_unique, curve['vert_indices']), axis=1)
                )[0]
            ]
            non_boundary = (curve_edges[:, None] != boundary_edges).any(2).all(1)
            non_boundary_vert_indices = np.unique(curve_edges[non_boundary])
            non_boundary_curve['vert_indices'] = non_boundary_vert_indices

        non_boundary_curves.append(non_boundary_curve)

    non_boundary_features = {'curves': non_boundary_curves}
    return non_boundary_features


def scale_mesh(mesh, features, shape_fabrication_extent, resolution_3d,
               short_curve_quantile=0.05, n_points_per_short_curve=4):
    # compute standard size spatial extent
    mesh_extent = np.max(mesh.bounding_box.extents)
    mesh = mesh.apply_scale(shape_fabrication_extent / mesh_extent)

    # compute lengths of curves
    sharp_curves_lengths = []
    for curve in features['curves']:
        if curve['sharp']:
            curve_vertices = mesh.vertices[curve['vert_indices']]
            sharp_curves_lengths.append(
                np.sum(np.linalg.norm(curve_vertices[:-1] - curve_vertices[1:]))
            )

    least_len = np.quantile(sharp_curves_lengths, short_curve_quantile)
    least_len_mm = resolution_3d * n_points_per_short_curve

    mesh = mesh.apply_scale(least_len_mm / least_len)

    return mesh


def generate_patches(meshes_filename, feats_filename, data_slice, config, output_file):
    n_patches_per_mesh = config['n_patches_per_mesh']
    shape_fabrication_extent = config.get('shape_fabrication_extent', 10.0)
    n_points_per_short_curve = config.get('n_points_per_short_curve', 4)
    short_curve_quantile = config.get('short_curve_quantile', 0.05)

    nbhood_extractor = load_func_from_config(NBHOOD_BY_TYPE, config['neighbourhood'])
    sampler = load_func_from_config(SAMPLER_BY_TYPE, config['sampling'])
    noiser = load_func_from_config(NOISE_BY_TYPE, config['noise'])
    annotator = load_func_from_config(ANNOTATOR_BY_TYPE, config['annotation'])

    # Specific to this script only: override radius of neighbourhood extractor
    # to reflect actual point cloud resolution:
    # we extract spheres of radius r, such that area of a (plane) disk with radius r
    # is equal to the total area of 3d points (as if we scanned a plane wall)
    nbhood_extractor.radius_base = np.sqrt((sampler.n_points * np.pi * sampler.resolution_3d ** 2)) / np.pi

    slice_start, slice_end = data_slice
    with ABCChunk([meshes_filename, feats_filename]) as data_holder:
        point_patches = []
        for item in data_holder[slice_start:slice_end]:
            eprint("Processing chunk file {chunk}, item {item}".format(
                chunk=meshes_filename, item=item.item_id))
            try:
                # load the mesh and the feature curves annotations
                mesh = trimesh_load(item.obj)
                features = yaml.load(item.feat, Loader=yaml.Loader)

                # fix mesh fabrication size in physical mm
                mesh = scale_mesh(mesh, features, shape_fabrication_extent, sampler.resolution_3d,
                                  short_curve_quantile=short_curve_quantile,
                                  n_points_per_short_curve=n_points_per_short_curve)
                # index the mesh using a neighbourhood functions class
                # (this internally may call indexing, so for repeated invocation one passes the mesh)
                nbhood_extractor.index(mesh)

                for patch_idx in range(n_patches_per_mesh):
                    # extract neighbourhood
                    try:
                        nbhood, mesh_vertex_indexes, mesh_face_indexes, scaler = nbhood_extractor.get_nbhood()
                    except DataGenerationException as e:
                        eprint(str(e))
                        continue

                    # create annotations: condition the features onto the nbhood
                    nbhood_features = compute_curves_nbhood(features, mesh_vertex_indexes, mesh_face_indexes)

                    # remove vertices lying on the boundary (sharp edges found in 1 face only)
                    nbhood_features = remove_boundary_features(nbhood, nbhood_features, how='edges')

                    # sample the neighbourhood to form a point patch
                    try:
                        points, normals = sampler.sample(nbhood, centroid=nbhood_extractor.centroid)
                    except DataGenerationException as e:
                        eprint(str(e))
                        continue

                    # create a noisy sample
                    noisy_points = noiser.make_noise(points, normals)

                    # compute the TSharpDF
                    distances, directions = annotator.annotate(nbhood, nbhood_features, noisy_points, scaler)

                    has_sharp = any(curve['sharp'] for curve in nbhood_features['curves'])
                    if not has_sharp:
                        distances = np.ones(distances.shape) * config['annotation']['distance_upper_bound']
                    patch_info = {
                        'points': noisy_points,
                        'normals': normals,
                        'distances': distances,
                        'directions': directions,
                        'item_id': item.item_id,
                        'orig_vert_indices': mesh_vertex_indexes,
                        'orig_face_indexes': mesh_face_indexes,
                        'has_sharp': has_sharp
                    }
                    point_patches.append(patch_info)

            except Exception as e:
                eprint('Error processing item {item_id} from chunk {chunk}: {what}'.format(
                    item_id=item.item_id, chunk='[{},{}]'.format(meshes_filename, feats_filename), what=e
                ))

    with h5py.File(output_file, 'w') as hdf5file:
        points = np.stack([patch['points'] for patch in point_patches])
        hdf5file.create_dataset('points', data=points, dtype=np.float64)

        normals = np.stack([patch['normals'] for patch in point_patches])
        hdf5file.create_dataset('normals', data=normals, dtype=np.float64)

        distances = np.stack([patch['distances'] for patch in point_patches])
        hdf5file.create_dataset('distances', data=distances, dtype=np.float64)

        directions = np.stack([patch['directions'] for patch in point_patches])
        hdf5file.create_dataset('directions', data=directions, dtype=np.float64)

        item_ids = [patch['item_id'] for patch in point_patches]
        hdf5file.create_dataset('item_id', data=np.string_(item_ids), dtype=h5py.string_dtype(encoding='ascii'))

        mesh_vertex_indexes = [patch['orig_vert_indices'].astype('int32') for patch in point_patches]
        vert_dataset = hdf5file.create_dataset('orig_vert_indices',
                                               shape=(len(mesh_vertex_indexes),),
                                               dtype=h5py.special_dtype(vlen=np.int32))
        for i, vert_indices in enumerate(mesh_vertex_indexes):
            vert_dataset[i] = vert_indices

        mesh_face_indexes = [patch['orig_face_indexes'].astype('int32') for patch in point_patches]
        face_dataset = hdf5file.create_dataset('orig_face_indexes',
                                               shape=(len(mesh_face_indexes),),
                                               dtype=h5py.special_dtype(vlen=np.int32))
        for i, face_indices in enumerate(mesh_face_indexes):
            face_dataset[i] = face_indices.flatten()

        has_sharp = np.stack([patch['has_sharp'] for patch in point_patches]).astype(bool)
        hdf5file.create_dataset('has_sharp', data=has_sharp, dtype=np.bool)


def make_patches(options):
    obj_filename = os.path.join(
        options.input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.OBJ.value,
            version='00'
        )
    )
    feat_filename = os.path.join(
        options.input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.FEAT.value,
            version='00'
        )
    )

    with ABCChunk([obj_filename, feat_filename]) as abc_data:
        num_data_items = len(abc_data)

    processes_to_spawn = 10 * options.n_jobs
    chunk_size = num_data_items // processes_to_spawn
    abc_data_slices = [(start, start + chunk_size)
                       for start in range(0, num_data_items, chunk_size)]
    output_files = [
        os.path.join(
            options.output_dir,
            'abc_{chunk}_{slice_start}_{slice_end}.hdf5'.format(
                chunk=options.chunk.zfill(4), slice_start=slice_start, slice_end=slice_end)
        )
        for slice_start, slice_end in abc_data_slices]

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)

    parallel = Parallel(n_jobs=options.n_jobs, backend='multiprocessing')
    delayed_iterable = (delayed(generate_patches)(obj_filename, feat_filename, data_slice, config, out_filename)
                        for data_slice, out_filename in zip(abc_data_slices, output_files))
    parallel(delayed_iterable)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=4, help='CPU jobs to use in parallel [default: 4].')
    parser.add_argument('-i', '--input-dir', dest='input_dir',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-c', '--chunk', required=True, help='ABC chunk id to process.')
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        required=True, help='output dir.')
    parser.add_argument('-g', '--dataset-config', dest='dataset_config',
                        required=True, help='dataset configuration file.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    make_patches(options)
