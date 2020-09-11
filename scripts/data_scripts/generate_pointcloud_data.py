#!/usr/bin/env python3

import argparse
from collections import defaultdict
import json
import os
import sys
import traceback

from joblib import Parallel, delayed
import numpy as np
import trimesh
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)

sys.path[1:1] = [__dir__]

from sharpf.data import DataGenerationException
from sharpf.utils.abc_utils.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
from sharpf.data.annotation import ANNOTATOR_BY_TYPE
from sharpf.data.datasets.sharpf_io import save_point_patches
from sharpf.data.mesh_nbhoods import NBHOOD_BY_TYPE
from sharpf.data.noisers import NOISE_BY_TYPE
from sharpf.data.point_samplers import SAMPLER_BY_TYPE
from sharpf.utils.abc_utils.abc.feature_utils import compute_features_nbhood, remove_boundary_features, get_curves_extents
from sharpf.utils.py_utils.console import eprint_t
from sharpf.utils.py_utils.os import add_suffix
from sharpf.utils.py_utils.config import load_func_from_config, Configurable
from sharpf.utils.abc_utils.mesh.io import trimesh_load
import sharpf.data.data_smells as smells


LARGEST_PROCESSABLE_MESH_VERTICES = 20000



# mm/pixel
HIGH_RES = 0.02
MED_RES = 0.05
LOW_RES = 0.125
XLOW_RES = 0.25

from abc import ABC, abstractmethod
from typing import Mapping, MutableMapping, List


class AbstractPipeline(ABC, Configurable):
    def __init__(self):
        self._filters = []

    @abstractmethod
    def run(self, data_item: MutableMapping) -> MutableMapping:
        """Applies a pipeline by executing a sequence of stages/filters."""
        pass

    def add_filter(self, filter_obj):
        self._filters.append(filter_obj)
        return self


class SequentialPipeline(AbstractPipeline):
    @classmethod
    def from_config(cls, config: Mapping) -> AbstractPipeline:
        pass

    @abstractmethod
    def run(self, data_item: MutableMapping) -> MutableMapping:
        """Applies a pipeline by executing a sequence of stages/filters."""
        for _filter in self._filters:
            data_item = _filter(data_item)
        return data_item


class ParallelPipeline(AbstractPipeline):
    @classmethod
    def from_config(cls, config: Mapping) -> AbstractPipeline:
        pass

    @abstractmethod
    def run(self, data_item: MutableMapping) -> List[MutableMapping]:
        """Applies a pipeline by executing a sequence of stages/filters."""
        for _filter in self._filters:
            data_item = _filter(data_item)
        return data_item


PIPELINE_BY_NAME = {
    "sequential_pipeline": SequentialPipeline,
    "parallel_pipeline": ParallelPipeline,
}


class AbstractFilter(ABC, Configurable):
    @abstractmethod
    def __call__(self, data_item: Mapping) -> Mapping:
        """Applies a filter, in-place modification of the input data_item."""
        pass


class MeshFilter(AbstractFilter):
    def __call__(self, data_item: MutableMapping) -> MutableMapping:
        assert 'mesh' in data_item, 'cannot operate on data item'
        return data_item


def require_keys(*keys):
    def inner(func):
        '''
           do operations with func
        '''
        return func

    return inner  # this is the fun_obj mentioned in the above content


class MeshScaler(MeshFilter):
    def __init__(
            self,
            shape_fabrication_extent,
            short_curve_quantile,
            resolution_3d,
            n_points_per_short_curve
    ):
        self._shape_fabrication_extent = shape_fabrication_extent
        self._short_curve_quantile = short_curve_quantile
        self._resolution_3d = resolution_3d
        self._n_points_per_short_curve = n_points_per_short_curve

    def __call__(self, data_item: MutableMapping) -> MutableMapping:
        data_item = super().__call__(data_item)

        mesh = data_item['mesh']
        features = data_item['features']

        # compute standard size spatial extent
        mesh_extent = np.max(mesh.bounding_box.extents)
        mesh = mesh.apply_scale(self._shape_fabrication_extent / mesh_extent)

        # compute lengths of curves
        sharp_curves_lengths = get_curves_extents(mesh, features)

        least_len = np.quantile(sharp_curves_lengths, self._short_curve_quantile)
        least_len_mm = self._resolution_3d * self._n_points_per_short_curve

        data_item['mesh'] = mesh.apply_scale(least_len_mm / least_len)
        return data_item


def get_annotated_patches(abc_item, config):

    # load the pipeline
    pipeline = load_func_from_config(PIPELINE_BY_NAME, config)

    # load the mesh and the feature curves annotations
    mesh, _, _ = trimesh_load(abc_item.obj)
    features = yaml.load(abc_item.feat, Loader=yaml.Loader)

    # construct the initial data item
    data_item = {
        'mesh': mesh,
        'features': features
    }

    try:
        result = pipeline.run(data_item)
    except DataGenerationException as e:
        eprint_t(str(e))



    shape_fabrication_extent = config.get('shape_fabrication_extent', 10.0)
    base_n_points_per_short_curve = config.get('base_n_points_per_short_curve', 8)
    base_resolution_3d = config.get('base_resolution_3d', LOW_RES)

    short_curve_quantile = config.get('short_curve_quantile', 0.05)

    nbhood_extractor = load_func_from_config(NBHOOD_BY_TYPE, config['neighbourhood'])
    sampler = load_func_from_config(SAMPLER_BY_TYPE, config['sampling'])
    noiser = load_func_from_config(NOISE_BY_TYPE, config['noise'])
    annotator = load_func_from_config(ANNOTATOR_BY_TYPE, config['annotation'])

    smell_coarse_surfaces_by_num_edges = smells.SmellCoarseSurfacesByNumEdges.from_config(config['smell_coarse_surfaces_by_num_edges'])
    smell_coarse_surfaces_by_angles = smells.SmellCoarseSurfacesByAngles.from_config(config['smell_coarse_surfaces_by_angles'])
    smell_deviating_resolution = smells.SmellDeviatingResolution.from_config(config['smell_deviating_resolution'])
    smell_sharpness_discontinuities = smells.SmellSharpnessDiscontinuities.from_config(config['smell_sharpness_discontinuities'])
    smell_bad_face_sampling = smells.SmellBadFaceSampling.from_config(config['smell_bad_face_sampling'])

    # Specific to this script only: override radius of neighbourhood extractor
    # to reflect actual point cloud resolution:
    # we extract spheres of radius r, such that area of a (plane) disk with radius r
    # is equal to the total area of 3d points (as if we scanned a plane wall)
    nbhood_extractor.radius_base = np.sqrt(sampler.n_points) * 0.5 * sampler.resolution_3d

    processed_mesh = trimesh.base.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True, validate=True)
    if processed_mesh.vertices.shape != mesh.vertices.shape or \
            processed_mesh.faces.shape != mesh.faces.shape or not mesh.is_watertight:
        raise DataGenerationException('Will not process mesh {}: likely the mesh is broken'.format(item.item_id))

    has_smell_mismatching_surface_annotation = any([
        np.array(np.unique(mesh.faces[surface['face_indices']]) != np.sort(surface['vert_indices'])).all()
        for surface in features['surfaces']
    ])

    # fix mesh fabrication size in physical mm
    mesh = scale_mesh(mesh, features, shape_fabrication_extent, base_resolution_3d,
                      short_curve_quantile=short_curve_quantile,
                      n_points_per_short_curve=base_n_points_per_short_curve)
    # index the mesh using a neighbourhood functions class
    # (this internally may call indexing, so for repeated invocation one passes the mesh)
    nbhood_extractor.index(mesh)

    for patch_idx in range(nbhood_extractor.n_patches_per_mesh):
        # extract neighbourhood
        try:
            nbhood, mesh_vertex_indexes, mesh_face_indexes, scaler = nbhood_extractor.get_nbhood()
            if len(nbhood.vertices) > LARGEST_PROCESSABLE_MESH_VERTICES:
                raise DataGenerationException('Too large number of vertices in crop: {}'.format(len(nbhood.vertices)))
        except DataGenerationException as e:
            eprint_t(str(e))
            continue

        has_smell_coarse_surfaces_by_num_edges = smell_coarse_surfaces_by_num_edges.run(mesh, mesh_face_indexes, features)
        has_smell_coarse_surfaces_by_angles = smell_coarse_surfaces_by_angles.run(mesh, mesh_face_indexes, features)

        # create annotations: condition the features onto the nbhood
        nbhood_features = compute_features_nbhood(mesh, features, mesh_face_indexes, mesh_vertex_indexes)

        # remove vertices lying on the boundary (sharp edges found in 1 face only)
        nbhood_features = remove_boundary_features(nbhood, nbhood_features, how='edges')

        # sample the neighbourhood to form a point patch
        try:
            points, normals = sampler.sample(nbhood, centroid=nbhood_extractor.centroid)
        except DataGenerationException as e:
            eprint_t(str(e))
            continue

        has_smell_deviating_resolution = smell_deviating_resolution.run(points)
        has_smell_bad_face_sampling = smell_bad_face_sampling.run(nbhood, points)

        # create a noisy sample
        for configuration, noisy_points in noiser.make_noise(points, normals):
            # compute the TSharpDF
            try:
                distances, directions, has_sharp = annotator.annotate(nbhood, nbhood_features, noisy_points)
            except DataGenerationException as e:
                eprint_t(str(e))
                continue

            has_smell_sharpness_discontinuities = smell_sharpness_discontinuities.run(points, distances)

            num_sharp_curves = len([curve for curve in nbhood_features['curves'] if curve['sharp']])
            num_surfaces = len(nbhood_features['surfaces'])
            patch_info = {
                'points': np.array(noisy_points).astype(np.float64),
                'normals': np.array(normals).astype(np.float64),
                'distances': np.array(distances).astype(np.float64),
                'directions': np.array(directions).astype(np.float64),
                'item_id': item.item_id,
                'orig_vert_indices': np.array(mesh_vertex_indexes).astype(np.int32),
                'orig_face_indexes': np.array(mesh_face_indexes).astype(np.int32),
                'has_sharp': has_sharp,
                'num_sharp_curves': num_sharp_curves,
                'num_surfaces': num_surfaces,
                'has_smell_coarse_surfaces_by_num_faces': has_smell_coarse_surfaces_by_num_edges,
                'has_smell_coarse_surfaces_by_angles': has_smell_coarse_surfaces_by_angles,
                'has_smell_deviating_resolution': has_smell_deviating_resolution,
                'has_smell_sharpness_discontinuities': has_smell_sharpness_discontinuities,
                'has_smell_bad_face_sampling': has_smell_bad_face_sampling,
                'has_smell_mismatching_surface_annotation': has_smell_mismatching_surface_annotation
            }
            yield configuration, patch_info


def generate_patches(meshes_filename, feats_filename, data_slice, config, output_file):
    slice_start, slice_end = data_slice
    with ABCChunk([meshes_filename, feats_filename]) as data_holder:
        point_patches_by_config = defaultdict(list)
        for item in data_holder[slice_start:slice_end]:
            eprint_t("Processing chunk file {chunk}, item {item}".format(
                chunk=meshes_filename, item=item.item_id))
            try:
                for configuration, patch_info in get_annotated_patches(item, config):
                    config_name = configuration.get('name')
                    point_patches_by_config[config_name].append(patch_info)

            except Exception as e:
                eprint_t('Error processing item {item_id} from chunk {chunk}: {what}'.format(
                    item_id=item.item_id, chunk='[{},{}]'.format(meshes_filename, feats_filename), what=e))
                eprint_t(traceback.format_exc())

            else:
                eprint_t('Done processing item {item_id} from chunk {chunk}'.format(
                    item_id=item.item_id, chunk='[{},{}]'.format(meshes_filename, feats_filename)))

    for config_name, point_patches in point_patches_by_config.items():
        if len(point_patches) == 0:
            continue

        output_file_config = add_suffix(output_file, config_name) if config_name else output_file
        try:
            save_point_patches(point_patches, output_file_config)
        except Exception as e:
            eprint_t('Error writing patches to disk at {output_file}: {what}'.format(
                output_file=output_file_config, what=e))
            eprint_t(traceback.format_exc())
        else:
            eprint_t('Done writing {num_patches} patches to disk at {output_file}'.format(
                num_patches=len(point_patches), output_file=output_file_config))


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

    if all([opt is not None for opt in (options.slice_start, options.slice_end)]):
        slice_start, slice_end = options.slice_start, options.slice_end
    else:
        with ABCChunk([obj_filename, feat_filename]) as abc_data:
            slice_start, slice_end = 0, len(abc_data)
        if options.slice_start is not None:
            slice_start = options.slice_start
        if options.slice_end is not None:
            slice_end = options.slice_end

    processes_to_spawn = 10 * options.n_jobs
    chunk_size = max(1, (slice_end - slice_start) // processes_to_spawn)
    abc_data_slices = [(start, start + chunk_size)
                       for start in range(slice_start, slice_end, chunk_size)]

    output_files = [
        os.path.join(
            options.output_dir,
            'abc_{chunk}_{slice_start}_{slice_end}.hdf5'.format(
                chunk=options.chunk.zfill(4), slice_start=slice_start, slice_end=slice_end)
        )
        for slice_start, slice_end in abc_data_slices]

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)

    MAX_SEC_PER_PATCH = 100 * 6
    max_patches_per_mesh = config['neighbourhood'].get('max_patches_per_mesh', 32)
    parallel = Parallel(n_jobs=options.n_jobs, backend='multiprocessing',
                        timeout=chunk_size * max_patches_per_mesh * MAX_SEC_PER_PATCH)
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
    parser.add_argument('-n1', dest='slice_start', type=int,
                        required=False, help='min index of data to process')
    parser.add_argument('-n2', dest='slice_end', type=int,
                        required=False, help='max index of data to process')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    make_patches(options)
