import logging
import random
from typing import List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from defs.utils.abc_utils import LotsOfHdf5Files
from defs.utils.abc_utils.hdf5 import HDF5DistributedSampler
from defs.utils.comm import get_batch_size
from defs.utils.hydra import instantiate

log = logging.getLogger(__name__)

__all__ = ['build_datasets', 'build_datasets']


def build_datasets(cfg: DictConfig, partition: str) -> List[Tuple[DictConfig, Dataset]]:
    assert partition in ['train', 'val', 'test']
    if partition not in cfg.datasets:
        return []

    datasets_params = cfg.datasets[partition]
    assert isinstance(datasets_params, ListConfig)

    if partition == 'train':
        assert len(datasets_params) == 1, 'multiple train datasets is not supported yet'

    datasets = []
    _dataset_names = []

    for dataset_param in datasets_params:
        assert dataset_param.dataset_name not in _dataset_names
        dataset = instantiate(dataset_param.dataset_class)
        log.info(f"Dataset ({dataset_param.dataset_name}) contains {len(dataset)} elements")
        datasets.append((dataset_param.dataset_name, dataset))
        _dataset_names.append(dataset_param.dataset_name)

    return datasets


def build_loaders(cfg: DictConfig, datasets: List[Tuple[str, Dataset]], partition: str):
    if len(datasets) == 0:
        return None

    assert partition in ['train', 'val', 'test']
    if partition == 'train':
        assert len(datasets) == 1

    num_workers = cfg.dataloader[partition].num_workers
    pin_memory = cfg.dataloader[partition].pin_memory
    batch_size = get_batch_size(cfg.dataloader[partition].total_batch_size)

    data_loaders = []
    shuffle = cfg.dataloader.train.shuffle if partition == 'train' else False
    drop_last = partition == 'train'

    for dataset_name, dataset in datasets:
        sampler = cfg.dataloader[partition].sampler

        if sampler == 'hdf5':
            assert isinstance(dataset, LotsOfHdf5Files)
            assert dataset.real_len() <= len(dataset), "Not implemented yet"
            sampler = HDF5DistributedSampler(dataset.file_sizes, shuffle)
        else:
            assert sampler is None

        data_loaders.append(DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            worker_init_fn=worker_init_reset_seed if sampler is not None else None))

    return data_loaders


def worker_init_reset_seed(worker_id):
    seed = np.random.randint(2 ** 31) + worker_id
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
