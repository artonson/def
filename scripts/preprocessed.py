import os
import pickle

from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

import vectran.data.graphics_primitives as graphics_primitives
from vectran.data.syndata.preprocessed import PreprocessedSyntheticHandcrafted
from vectran.data.vectordata.preprocessed import PreprocessedSVG


patch_parameters = {
    'patch_width': 64,
    'patch_height': 64,
    'max_lines': 10,
    'max_curves': 0
}

svg_parameters = {
    'widths': ('*',),
    'orientations': ('*',),
    'rotations': ('*',),
    'translations': (('*', '*'),)
}

class SynParameters:
    from vectran.data.syndata.utils import renormalize as _normalize_probas

    strokes_probas = _normalize_probas({1: .45, 2: .45, 3: .1})

    directions_probas = {}
    for angle in np.linspace(0, np.pi, 13):
        if angle in [0, np.pi / 2]:                 # - horizontal, | vertical
            directions_probas[angle] = .30
        elif angle in [np.pi / 4, np.pi * 3 / 4]:   # /, \
            directions_probas[angle] = .15
        elif angle < np.pi:
            directions_probas[angle] = .1 / 8

    directions_probas = _normalize_probas(directions_probas)

    ## def get_offset_angles():
    ##     rotations = [0, 90, 180, 270]
    ##     max_rot_deviation = 10
    ##     rotations += [base_rot + dev_rot * dev_sign for dev_rot in range(1, max_rot_deviation + 1, 1) for base_rot in (0, 90, 180, 270) for dev_sign in (1, -1)]
    ##     return rotations
    def get_offset_angles():
        return range(0, 360, 4)
    offset_directions_probas = _normalize_probas({np.deg2rad(angle): 1 for angle in get_offset_angles()})

    padding_factor = 2

syn_parameters = {
    'samples_n': 1000000,
    'border': 8,
    'min_directions': 1, 'max_directions': 2,
    'min_primitives_gap': 2, 'max_primitives_gap': 10,
    'min_stroke_width': 1, 'max_stroke_width': 7,
    'min_stroke_length': max(patch_parameters['patch_width'], patch_parameters['patch_height']) * .90 * SynParameters.padding_factor,
    'max_stroke_length': np.sqrt(patch_parameters['patch_width']**2 + patch_parameters['patch_height']**2) * SynParameters.padding_factor,
    'primitives_endpoint_noise_sigma': 0.5,
    'primitives_direction_noise_sigma': np.pi / 270.,
    'directions_probas': SynParameters.directions_probas,
    'offset_directions_probas': SynParameters.offset_directions_probas,
    'strokes_probas': SynParameters.strokes_probas,
}


# TODO add loading from multiple files: thus we could generate separate datasets
#  with distinct single line topologies (e.g. one dataset with L-junctions only,
#  another with T-junctions only, etc.)
class _PreprocessedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        meta_file = _PreprocessedDataset._meta_path(data_dir)
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)

        images_file = _PreprocessedDataset._images_path(data_dir)
        images_shape = [meta['samples_n'], 3, meta['patch_height'], meta['patch_width']]
        self.images = torch.FloatTensor(torch.FloatStorage.from_file(images_file, size=np.prod(images_shape), shared=True)).reshape(*images_shape)

        targets_file = _PreprocessedDataset._targets_path(data_dir)
        targets_shape = [meta['samples_n'], meta['max_lines'], graphics_primitives.repr_len_by_type[graphics_primitives.PrimitiveType.PT_LINE] + 1]
        self.targets = torch.FloatTensor(torch.FloatStorage.from_file(targets_file, size=np.prod(targets_shape), shared=True)).reshape(*targets_shape)
            

    @classmethod
    def from_preprocessed(cls, datasets, out_root, shuffle_seed=None):
        assert False, 'too slow'
        # initialize concatenated dataset
        concat_dataset = torch.utils.data.ConcatDataset(datasets)
        # all subsets must have the same shape
        assert all(d.images.shape[1:] == concat_dataset.datasets[0].images.shape[1:] for d in concat_dataset.datasets[1:])
        assert all(d.targets.shape[1:] == concat_dataset.datasets[0].targets.shape[1:] for d in concat_dataset.datasets[1:])

        # save meta
        os.makedirs(out_root, exist_ok=True)
        patch_height, patch_width = concat_dataset.datasets[0].images.shape[-2:]
        max_lines = concat_dataset.datasets[0].targets.shape[1]
        meta = {'patch_width': patch_width, 'patch_height': patch_height, 'max_lines': max_lines}
        meta['sources'] = [d.data_dir for d in concat_dataset.datasets]
        meta['samples_n'] = len(concat_dataset)
        meta['shuffle_seed'] = shuffle_seed
        with open(_PreprocessedDataset._meta_path(out_root), 'wb') as f:
            pickle.dump(meta, f)

        # shuffle dataset if necessary
        if shuffle_seed is not None:
            concat_dataset = torch.utils.data.Subset(concat_dataset, np.random.RandomState(seed=shuffle_seed).permutation(len(concat_dataset)))

        # initialize dataset
        dataset = cls(out_root)
        dataset.dataset = concat_dataset
        dataset.convert_images = dataset.convert_targets = lambda x: x
        return dataset


    @classmethod
    def from_svgdataset(cls, data_root, patch_parameters, svg_parameters, out_root, shuffle_seed=None):
        # make primitive_types and max_primitives dicts based on the parameters
        primitive_types = []
        max_primitives = {}
        if patch_parameters['max_lines'] > 0:
            primitive_types.append(graphics_primitives.PrimitiveType.PT_LINE)
            max_primitives[graphics_primitives.PrimitiveType.PT_LINE] = patch_parameters['max_lines']
        if patch_parameters['max_curves'] > 0:
            primitive_types.append(graphics_primitives.PrimitiveType.PT_BEZIER)
            max_primitives[graphics_primitives.PrimitiveType.PT_BEZIER] = patch_parameters['max_curves']

        # initialize svgdataset
        svg_dataset = PreprocessedSVG(
            patch_size=(patch_parameters['patch_width'], patch_parameters['patch_height']), normalize_image=True,
            primitive_types=primitive_types, max_primitives=max_primitives, sample_primitives_randomly=False,
            data_root=data_root,
            **svg_parameters)

        # shuffle dataset if necessary
        if shuffle_seed is not None:
            svg_dataset = torch.utils.data.Subset(svg_dataset, np.random.RandomState(seed=shuffle_seed).permutation(len(svg_dataset)))

        # save meta
        os.makedirs(out_root, exist_ok=True)
        meta = {**patch_parameters, **svg_parameters}
        meta['data_root'] = data_root
        meta['samples_n'] = len(svg_dataset)
        meta['shuffle_seed'] = shuffle_seed
        with open(_PreprocessedDataset._meta_path(out_root), 'wb') as f:
            pickle.dump(meta, f)

        # initialize dataset
        dataset = cls(out_root)
        dataset.dataset = svg_dataset
        dataset.convert_images = dataset.convert_targets = torch.from_numpy
        return dataset


    @classmethod
    def from_synthetic_handcrafted(cls, patch_parameters, syn_parameters, topologies_with_probas, out_root, random_seed):
        # make primitive_types and max_primitives dicts based on the parameters
        primitive_types = []
        max_primitives = {}
        if patch_parameters['max_lines'] > 0:
            primitive_types.append(graphics_primitives.PrimitiveType.PT_LINE)
            max_primitives[graphics_primitives.PrimitiveType.PT_LINE] = patch_parameters['max_lines']
        if patch_parameters['max_curves'] > 0:
            primitive_types.append(graphics_primitives.PrimitiveType.PT_BEZIER)
            max_primitives[graphics_primitives.PrimitiveType.PT_BEZIER] = patch_parameters['max_curves']

        # initialize syndataset
        np.random.seed(random_seed)
        syn_dataset = PreprocessedSyntheticHandcrafted(
            patch_size=(patch_parameters['patch_width'], patch_parameters['patch_width']), normalize_image=True,
            primitive_types=primitive_types, max_primitives=max_primitives, size=syn_parameters['samples_n'],
            topologies_with_probas=topologies_with_probas, **syn_parameters)

        # save meta
        os.makedirs(out_root, exist_ok=True)
        meta = {**patch_parameters, **syn_parameters}
        meta['random_seed'] = random_seed
        with open(_PreprocessedDataset._meta_path(out_root), 'wb') as f:
            pickle.dump(meta, f)

        # initialize dataset
        dataset = cls(out_root)
        dataset.dataset = syn_dataset
        dataset.convert_images = dataset.convert_targets = torch.from_numpy
        return dataset


    def __getitem__(self, idx):
        images, target = self.dataset[idx]
        self.images[idx] = self.convert_images(images)
        self.targets[idx] = self.convert_targets(target)
        return []


    def __len__(self):
        return len(self.images)


    @staticmethod
    def _images_path(root):
        return os.path.join(root, 'images.bin')

    @staticmethod
    def _meta_path(root):
        return os.path.join(root, 'meta.pickle')

    @staticmethod
    def _targets_path(root):
        return os.path.join(root, 'targets.bin')


class PreprocessedDataset(_PreprocessedDataset):
    '''For optimal performance load data in the main thread and don't use shuffling.'''
    def __init__(self, data_dir=None, range=None):
        if data_dir is None: return

        super().__init__(data_dir)

        if range is not None:
            start, end = self._subset_range(range)
            self.images = self.images[start:end]
            self.targets = self.targets[start:end]


    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


    def slice(self, start=0, end=-1):
        subset = PreprocessedDataset()
        subset.images = self.images[start:end]
        subset.targets = self.targets[start:end]
        return subset


    def subset(self, range):
        subset = PreprocessedDataset()
        start, end = self._subset_range(range)
        subset.images = self.images[start:end]
        subset.targets = self.targets[start:end]
        return subset


    def _subset_range(self, range):
        start, end = range
        assert (0 <= start < 1) and (0 < end <= 1)
        start = int(round(len(self) * start))
        end = int(round(len(self) * end))
        return start, end
