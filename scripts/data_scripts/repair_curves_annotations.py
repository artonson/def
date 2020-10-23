#!/usr/bin/env python3

import argparse
from collections.abc import Mapping
from functools import partial
import json
from multiprocessing import Pool
import os
import sys
from typing import List

import h5py
import numpy as np
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)

sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
from sharpf.data.annotation import ANNOTATOR_BY_TYPE
from sharpf.utils.abc_utils.abc.feature_utils import compute_features_nbhood, remove_boundary_features, get_curves_extents
from sharpf.utils.py_utils.config import load_func_from_config
from sharpf.utils.abc_utils.mesh.io import trimesh_load
from sharpf.utils.abc_utils.mesh.indexing import reindex_zerobased
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, LotsOfHdf5Files, PreloadTypes
import sharpf.utils.abc_utils.hdf5.io_struct as io
import sharpf.data.data_smells as smells
from sharpf.data.datasets.sharpf_io import (
    save_point_patches as save_fn,
    PointCloudIO as IO
)


class BufferedHDF5Writer(object):
    def __init__(self, output_dir=None, prefix='', n_items_per_file=float('+inf'),
                 verbose=False, save_fn=None):
        self.output_dir = output_dir
        self.prefix = prefix
        self.file_id = 0
        self.n_items_per_file = n_items_per_file
        self.data = []
        self.index = []
        self.verbose = verbose
        self.save_fn = save_fn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.data:
            self._flush()

    # def append(self, data):
    #     assert isinstance(data, Mapping)
    #     self.data.append(data)
    #     self.check_flush()
    #
    # def extend(self, data):
    #     self.data.extend([
    #         dict(zip(data, item_values))
    #         for item_values in zip(*data.values())
    #     ])
    #     self.check_flush()

    def check_flush(self):
        if -1 != self.n_items_per_file and len(self.data) >= self.n_items_per_file:
            required_index = list(range(self.file_id * int(self.n_items_per_file),
                                        (self.file_id + 1) * int(self.n_items_per_file)))
            if self.index[:self.n_items_per_file] == required_index:
                self._flush()
                self.file_id += 1
                self.data = self.data[self.n_items_per_file:]
                self.index = self.index[self.n_items_per_file:]

    def _flush(self):
        filename = '{prefix}{id}.hdf5'.format(prefix=self.prefix, id=self.file_id)
        filename = os.path.join(self.output_dir, filename)
        self.save_fn(self.data[:self.n_items_per_file], filename)
        if self.verbose:
            print('Saved {} with {} items'.format(filename, len(self.data[:self.n_items_per_file])))

    def insert(self, data_idx, data):
        assert isinstance(data, Mapping)
        insert_index = np.searchsorted(self.index, data_idx)
        self.index.insert(insert_index, data_idx)
        self.data.insert(insert_index, data)
        self.check_flush()


# mm/pixel
HIGH_RES = 0.02
MED_RES = 0.05
LOW_RES = 0.125
XLOW_RES = 0.25


DATA_HOLDERS_BY_CHUNK = dict()


def get_data_holder(abc_dir, chunk_id):
    global DATA_HOLDERS_BY_CHUNK

    if None is DATA_HOLDERS_BY_CHUNK.get(chunk_id):
        obj_filename = os.path.join(
            abc_dir,
            ABC_7Z_FILEMASK.format(
                chunk=str(chunk_id).zfill(4),
                modality=ABCModality.OBJ.value,
                version='00'
            )
        )
        feat_filename = os.path.join(
            abc_dir,
            ABC_7Z_FILEMASK.format(
                chunk=str(chunk_id).zfill(4),
                modality=ABCModality.FEAT.value,
                version='00'
            )
        )
        DATA_HOLDERS_BY_CHUNK[chunk_id] = ABCChunk([obj_filename, feat_filename])

    data_holder = DATA_HOLDERS_BY_CHUNK[chunk_id]
    return data_holder


def scale_mesh(mesh, features, shape_fabrication_extent, resolution_3d,
               short_curve_quantile=0.05, n_points_per_short_curve=4):
    # compute standard size spatial extent
    mesh_extent = np.max(mesh.bounding_box.extents)
    mesh = mesh.apply_scale(shape_fabrication_extent / mesh_extent)

    # compute lengths of curves
    sharp_curves_lengths = get_curves_extents(mesh, features)

    least_len = np.quantile(sharp_curves_lengths, short_curve_quantile)
    least_len_mm = resolution_3d * n_points_per_short_curve

    mesh = mesh.apply_scale(least_len_mm / least_len)

    return mesh


def fix_patch_unpack(args):
    return fix_patch(*args)


def fix_patch(idx, patch, chunk_ids, config, abc_dir):
    print('Fixing patch {}, item_id= {}'.format(idx, patch['item_id']))
#   item = None
#   for chunk_id in chunk_ids:
#       try:
#           data_holder = get_data_holder(abc_dir, chunk_id)
#           item = data_holder.get(patch['item_id'].decode())
#       except KeyError:
#           continue
#
#   if None is item:
#       raise ValueError()
#
#   mesh, _, _ = trimesh_load(item.obj)
#   features = yaml.load(item.feat, Loader=yaml.Loader)

    dirname, hash, number = patch['item_id'].decode().split('_')
    obj_basename = '_'.join([dirname, hash, 'trimesh', number]) + '.obj'
    feat_basename = '_'.join([dirname, hash, 'features', number]) + '.yml'

    for chunk_id in chunk_ids:
        obj_filename = os.path.join(abc_dir, 'obj', str(chunk_id).zfill(4), dirname, obj_basename)
        feat_filename = os.path.join(abc_dir, 'feat', str(chunk_id).zfill(4), dirname, feat_basename)
        if os.path.exists(obj_filename):
            break

    print('Opening {} and {}'.format(obj_filename, feat_filename))

    with open(obj_filename) as obj_file:
        mesh, _, _ = trimesh_load(obj_file, need_decode=False)

    with open(feat_filename) as feat_file:
        features = yaml.load(feat_file, Loader=yaml.Loader)

    shape_fabrication_extent = config.get('shape_fabrication_extent', 10.0)
    base_n_points_per_short_curve = config.get('base_n_points_per_short_curve', 8)
    base_resolution_3d = config.get('base_resolution_3d', LOW_RES)
    short_curve_quantile = config.get('short_curve_quantile', 0.05)
    annotator = load_func_from_config(ANNOTATOR_BY_TYPE, config['annotation'])

    mesh = scale_mesh(mesh, features, shape_fabrication_extent, base_resolution_3d,
                      short_curve_quantile=short_curve_quantile,
                      n_points_per_short_curve=base_n_points_per_short_curve)

    nbhood = reindex_zerobased(
        mesh,
        patch['orig_vert_indices'],
        patch['orig_face_indexes'])

    nbhood_features = compute_features_nbhood(
        mesh,
        features,
        patch['orig_face_indexes'],
        patch['orig_vert_indices'],
        deduce_verts_from_faces=False)

    nbhood_features = remove_boundary_features(
        nbhood,
        nbhood_features,
        how='edges')
    num_sharp_curves = len([curve for curve in nbhood_features['curves'] if curve['sharp']])

    distances, directions, has_sharp = annotator.annotate(
        nbhood,
        nbhood_features,
        patch['points'])

    smell_sharpness_discontinuities = smells.SmellSharpnessDiscontinuities()
    has_smell_sharpness_discontinuities = smell_sharpness_discontinuities.run(patch['points'], distances)

    fixed_part = {
        'distances': distances,
        'directions': directions,
        'has_sharp': has_sharp,
        'num_sharp_curves': num_sharp_curves,
        'has_smell_sharpness_discontinuities': has_smell_sharpness_discontinuities
    }

    return idx, fixed_part


def uncollate(collated: Mapping) -> List[Mapping]:
    """
    Given a collated batch (i.e., a mapping of lists/arrays),
    produce a list of mappings.
    """
    any_value = next(iter(collated.values()))
    list_len = len(any_value)
    return [
        {key: value[idx] for key, value in collated.items()}
        for idx in range(list_len)
    ]


#IO = io.HDF5IO({
#   'points': io.Float64('points'),
#   'normals': io.Float64('normals'),
#   'distances': io.Float64('distances'),
#   'directions': io.Float64('directions'),
#   'item_id': io.AsciiString('item_id'),
#   'orig_vert_indices': io.VarInt32('orig_vert_indices'),
#   'orig_face_indexes': io.VarInt32('orig_face_indexes'),
#   'has_sharp': io.Bool('has_sharp'),
#   'num_sharp_curves': io.Int8('num_sharp_curves'),
#   'num_surfaces': io.Int8('num_surfaces'),
#   'has_smell_coarse_surfaces_by_num_faces': io.Bool('has_smell_coarse_surfaces_by_num_faces'),
#   'has_smell_coarse_surfaces_by_angles': io.Bool('has_smell_coarse_surfaces_by_angles'),
#   'has_smell_deviating_resolution': io.Bool('has_smell_deviating_resolution'),
#   'has_smell_sharpness_discontinuities': io.Bool('has_smell_sharpness_discontinuities'),
#   'has_smell_bad_face_sampling': io.Bool('has_smell_bad_face_sampling'),
#   'has_smell_mismatching_surface_annotation': io.Bool('has_smell_mismatching_surface_annotation'),
#   'voronoi': io.Float64('voronoi'),
#   'normals_estimation_10': io.Float64('normals_estimation_10'),
#   'normals_estimation_100': io.Float64('normals_estimation_100'),
#},
#len_label='has_sharp',
#compression='lzf')


#def save_point_patches(patches, filename):
#   # turn a list of dicts into a dict of torch tensors:
#   # default_collate([{'a': 'str1', 'x': np.random.normal()}, {'a': 'str2', 'x': np.random.normal()}])
#   # Out[26]: {'a': ['str1', 'str2'], 'x': tensor([0.4252, 0.1414], dtype=torch.float64)}
#   collate_fn = partial(io.collate_mapping_with_io, io=IO)
#   patches = collate_fn(patches)
#
#   with h5py.File(filename, 'w') as f:
#       for key in ['points', 'normals', 'distances', 'directions',
#                   'voronoi', 'normals_estimation_10', 'normals_estimation_100']:
#           IO.write(f, key, patches[key].numpy())
#       IO.write(f, 'item_id', patches['item_id'])
#       IO.write(f, 'orig_vert_indices', patches['orig_vert_indices'])
#       IO.write(f, 'orig_face_indexes', patches['orig_face_indexes'])
#       IO.write(f, 'has_sharp', patches['has_sharp'].numpy().astype(np.bool))
#       IO.write(f, 'num_sharp_curves', patches['num_sharp_curves'].numpy())
#       IO.write(f, 'num_surfaces', patches['num_surfaces'].numpy())
#       has_smell_keys = [key for key in IO.datasets.keys()
#                         if key.startswith('has_smell')]
#       for key in has_smell_keys:
#           IO.write(f, key, patches[key].numpy().astype(np.bool))


def main(options):
    chunk_id = sorted(options.chunk_id)

#   dataset = LotsOfHdf5Files(
#       data_dir=options.hdf5_input_dir,
#       io=IO,
#       labels='*',
#       max_loaded_files=1,
#       preload=PreloadTypes.LAZY)

    dataset = Hdf5File(
        filename=options.hdf5_input_filename,
        io=IO,
        labels='*',
        preload=PreloadTypes.LAZY)

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)

    writer_params = {
        'output_dir': options.hdf5_output_dir,
        'n_items_per_file': 10240,
        'save_fn': save_fn,
        'verbose': options.verbose,
        'prefix': 'fix_' + os.path.splitext(os.path.basename(options.hdf5_input_filename))[0] + '_',
    }
    with BufferedHDF5Writer(**writer_params) as writer, \
            Pool(processes=options.n_jobs) as pool:

        data_to_fix = ((idx, dataset[idx], chunk_id, config, options.abc_dir)
                       for idx in range(len(dataset)))
        for proc_idx, (patch_idx, fixed_part) in enumerate(pool.imap_unordered(fix_patch_unpack, data_to_fix)):
            patch = dataset[patch_idx]
            patch.update(fixed_part)
            writer.insert(patch_idx, patch)
            print('{} items processed'.format(proc_idx + 1))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=1, help='CPU jobs to use in parallel [default: 1].')
    parser.add_argument('-a', '--abc-dir', dest='abc_dir',
                        required=True, help='directory with ABC dataset source files.')
    parser.add_argument('-i', '--input-filename', dest='hdf5_input_filename',
                        required=True, help='HDF5 input filename.')
    parser.add_argument('-o', '--output-dir', dest='hdf5_output_dir',
                        required=True, help='directory with HDF5 output files.')
    parser.add_argument('-c', '--chunk', type=int, dest='chunk_id', action='append',
                        required=True, help='ABC chunk id to process.')
    parser.add_argument('-g', '--dataset-config', dest='dataset_config',
                        required=True, help='dataset configuration file.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
