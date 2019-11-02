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

from sharpf.data.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
from sharpf.data.annotation import ANNOTATOR_BY_TYPE
from sharpf.data.mesh_nbhoods import NBHOOD_BY_TYPE
from sharpf.data.noisers import NOISE_BY_TYPE
from sharpf.data.point_samplers import SAMPLER_BY_TYPE
from sharpf.utils.common import eprint
from sharpf.utils.mesh_utils import trimesh_load


def load_func_from_config(func_dict, config):
    return func_dict[config['type']].from_config(config)


def compute_curves_nbhood(features, vert_indices, face_indexes):
    """Extracts curves for the neighbourhood."""
    nbhood_sharp_curves = []
    for curve in features['curves']:
        nbhood_vert_indices = np.array([
            vert_index for vert_index in curve['vert_indices']
            if vert_index + 1 in vert_indices
        ])
        if len(nbhood_vert_indices) == 0:
            continue
        for index, reindex in zip(vert_indices, np.arange(len(vert_indices))):
            nbhood_vert_indices[np.where(nbhood_vert_indices == index - 1)] = reindex
        nbhood_curve = deepcopy(curve)
        nbhood_curve['vert_indices'] = nbhood_vert_indices
        nbhood_sharp_curves.append(nbhood_curve)

    nbhood_features = {'curves': nbhood_sharp_curves}
    return nbhood_features


def generate_patches(meshes_filename, feats_filename, data_slice, config, output_file):
    n_patches_per_mesh = config['n_patches_per_mesh']
    nbhood_extractor = load_func_from_config(NBHOOD_BY_TYPE, config['neighbourhood'])
    sampler = load_func_from_config(SAMPLER_BY_TYPE, config['sampling'])
    noiser = load_func_from_config(NOISE_BY_TYPE, config['noise'])
    annotator = load_func_from_config(ANNOTATOR_BY_TYPE, config['annotation'])

    slice_start, slice_end = data_slice
    with ABCChunk([meshes_filename, feats_filename]) as data_holder:
        point_patches = []
        for item in data_holder[slice_start:slice_end]:
            # load the mesh and the feature curves annotations
            try:
                mesh = trimesh_load(item.obj)
                features = yaml.load(item.feat, Loader=yaml.Loader)

                # index the mesh using a neighbourhood functions class
                # (this internally may call indexing, so for repeated invocation one passes the mesh)
                nbhood_extractor.index(mesh)

                for patch_idx in range(n_patches_per_mesh):
                    # extract neighbourhood
                    nbhood, orig_vert_indices, orig_face_indexes = nbhood_extractor.get_nbhood()

                    # sample the neighbourhood to form a point patch
                    points, normals = sampler.sample(nbhood)

                    # create a noisy sample
                    noisy_points = noiser.make_noise(points, normals)

                    # create annotations: condition the features onto the nbhood, then compute the TSharpDF
                    nbhood_features = compute_curves_nbhood(features, orig_vert_indices, orig_face_indexes)
                    distances, directions = annotator.annotate(nbhood, nbhood_features, noisy_points)

                    has_sharp = any(curve['sharp'] for curve in nbhood_features['curves'])
                    patch_info = {
                        'points': noisy_points,
                        'normals': normals,
                        'distances': distances,
                        'directions': directions,
                        'item_id': item.item_id,
                        'orig_vert_indices': orig_vert_indices,
                        'orig_face_indexes': orig_face_indexes,
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

        orig_vert_indices = [patch['orig_vert_indices'].astype('int32') for patch in point_patches]
        vert_dataset = hdf5file.create_dataset('orig_vert_indices',
                                               shape=(len(orig_vert_indices),),
                                               dtype=h5py.special_dtype(vlen=np.int32))
        for i, vert_indices in enumerate(orig_vert_indices):
            vert_dataset[i] = vert_indices

        orig_face_indexes = [patch['orig_face_indexes'].astype('int32') for patch in point_patches]
        face_dataset = hdf5file.create_dataset('orig_face_indexes',
                                               shape=(len(orig_face_indexes),),
                                               dtype=h5py.special_dtype(vlen=np.int32))
        for i, face_indices in enumerate(orig_face_indexes):
            face_dataset[i] = face_indices.flatten()

        has_sharp = np.stack([patch['has_sharp'] for patch in point_patches]).astype(bool)
        hdf5file.create_dataset('has_sharp', data=has_sharp, dtype=np.bool)


def make_patches(options):
    """Filter the shapes using a number of filters, saving intermediate results."""

    # create the data source iterator
    # abc_data = ABCData(options.data_dir,
    #                    modalities=[ABCModality.FEAT.value, ABCModality.OBJ.value],
    #                    chunks=[options.chunk],
    #                    shape_representation='trimesh')

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
    abc_data = ABCChunk([obj_filename, feat_filename])

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)

    processes_to_spawn = 10 * options.n_jobs
    chunk_size = len(abc_data) // processes_to_spawn
    abc_data_slices = [(start, start + chunk_size)
                       for start in range(0, len(abc_data), chunk_size)]
    output_files = [
        os.path.join(
            options.output_dir,
            'abc_{chunk}_{slice_start}_{slice_end}.hdf5'.format(
                chunk=options.chunk.zfill(4), slice_start=slice_start, slice_end=slice_end)
        )
        for slice_start, slice_end in abc_data_slices]

    # run the filtering job in parallel
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
