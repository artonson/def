import logging
from typing import Optional, List

import torch
from torch.utils.data.sampler import Sampler

from defs.utils import comm

log = logging.getLogger(__name__)


class HDF5DistributedSampler(Sampler):

    def __init__(self, sizes: List[int], shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            sizes (List[int]): list of hdf5 files sizes
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        assert isinstance(sizes, list)
        for size in sizes:
            assert size > 0

        self._shuffle = shuffle
        if seed is None:
            log.info("SEED IS NOT INITIALIZED")
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        self._sizes = torch.tensor(sizes)
        self._num_files = len(sizes)
        self.num_samples = len(torch.arange(torch.sum(self._sizes))[self._rank::self._world_size])
        if not shuffle:
            log.info(f"{self._rank}_{self._world_size}_{self._num_files}_{self.num_samples}")

        if shuffle:
            self.g = []
            for i in range(self._num_files + 1):
                self.g.append(torch.Generator())
                self.g[-1].manual_seed(self._seed + i)

    def __iter__(self):
        if self._shuffle:
            inds = []
            start = torch.tensor(0)
            for i, size in enumerate(self._sizes[torch.randperm(self._num_files, generator=self.g[0])]):
                inds.append(torch.arange(start, start + size)[torch.randperm(size, generator=self.g[i + 1])])
                start += size
            inds = torch.cat(inds)
        else:
            inds = torch.arange(torch.sum(self._sizes))

        inds = inds[self._rank::self._world_size]
        assert len(inds) == self.num_samples
        return iter(inds)

    def __len__(self):
        return self.num_samples
