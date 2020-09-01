import logging
from collections import OrderedDict

from omegaconf import ListConfig
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sharpf.utils.comm import get_batch_size
from sharpf.utils.hydra import instantiate

log = logging.getLogger(__name__)


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def build_datasets(cfg, partition):
    assert partition in ['train', 'val', 'test']
    if partition not in cfg.datasets:
        return None

    datasets_params = cfg.datasets[partition]
    assert isinstance(datasets_params, ListConfig)

    datasets = OrderedDict()

    for dataset_param in datasets_params:
        dataset_name = dataset_param.dataset_name
        assert dataset_name not in datasets, f"dataset_name {dataset_name} is duplicated"
        datasets[dataset_name] = instantiate(dataset_param.dataset_class)
        log.info(f"Dataset ({dataset_name}) contains {len(datasets[dataset_name])} elements")

    if partition == 'train' and len(datasets) > 1:
        key = 'concat_' + '_'.join(list(datasets.keys()))
        return {key: ConcatDataset(datasets)}

    return datasets


def build_loaders(cfg, partition):
    assert partition in ['train', 'val', 'test']

    num_workers = cfg.data_loader[partition].num_workers
    pin_memory = cfg.data_loader[partition].pin_memory
    batch_size = get_batch_size(cfg.data_loader[partition].total_batch_size)

    datasets = build_datasets(cfg, partition)
    if datasets is None:
        return None

    data_loaders = []
    shuffle = partition == 'train'
    drop_last = partition == 'train'

    for dataset_name, dataset in datasets.items():
        data_loaders.append(DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                       shuffle=shuffle, drop_last=drop_last))

    if partition == 'train':
        assert len(data_loaders) == 1

    return data_loaders
