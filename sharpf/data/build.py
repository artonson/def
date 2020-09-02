import logging
from typing import List, Tuple

from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sharpf.utils.comm import get_batch_size
from sharpf.utils.hydra import instantiate

log = logging.getLogger(__name__)


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
        datasets.append((dataset_param, dataset))
        _dataset_names.append(dataset_param.dataset_name)

    return datasets


def build_loaders(cfg: DictConfig, datasets: List[Tuple[DictConfig, Dataset]], partition: str):
    if len(datasets) == 0:
        return None

    assert partition in ['train', 'val', 'test']
    if partition == 'train':
        assert len(datasets) == 1

    num_workers = cfg.data_loader[partition].num_workers
    pin_memory = cfg.data_loader[partition].pin_memory
    batch_size = get_batch_size(cfg.data_loader[partition].total_batch_size)

    data_loaders = []
    shuffle = partition == 'train'
    drop_last = partition == 'train'

    for dataset_param, dataset in datasets:
        data_loaders.append(
            DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle,
                       drop_last=drop_last))

    return data_loaders
