import logging

import torch

from .evaluator import DatasetEvaluator
from ..utils.comm import all_gather, synchronize

log = logging.getLogger(__name__)


class StatsEvaluator(DatasetEvaluator):
    """
    Evaluate distance statistics
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self._min_dist = None
        self._sum_dist = None
        self._max_dist = None
        self._num_points = 0

    def process(self, inputs, outputs):
        """
        Args:
            inputs (dict): the inputs to a model.
            outputs (dict): the outputs of a model.
        """
        x = inputs['points']  # (B, N, C)
        inner = torch.matmul(x, x.transpose(2, 1))  # (B, N, N)
        xx = torch.sum(x ** 2, dim=2, keepdim=True)  # (B, N, 1)
        pairwise_distance = xx - 2 * inner + xx.transpose(2, 1)  # (B, N, N)
        # pairwise_distance = torch.sqrt(pairwise_distance)
        b, n, _ = pairwise_distance.size()
        pairwise_distance = pairwise_distance.view(b * n, n)  # (B * N, N)

        # for i in range(n):
        #     print(i, pairwise_distance[i][i])

        pairwise_distance = torch.sort(pairwise_distance, dim=1)[0]  # (B * N, N)
        pairwise_distance[:, 1:] = torch.sqrt(pairwise_distance[:, 1:])

        # print()
        # print(pairwise_distance)
        # print()

        batch_min_dist = pairwise_distance.min(dim=0, keepdim=True).values.detach().cpu()  # (1, N)
        batch_max_dist = pairwise_distance.max(dim=0, keepdim=True).values.detach().cpu()  # (1, N)
        batch_sum_dist = pairwise_distance.sum(dim=0, keepdim=True).detach().cpu()  # (1, N)

        self._num_points += torch.tensor(b * n)

        if self._min_dist is None:
            self._min_dist = batch_min_dist
        else:
            self._min_dist = torch.cat([self._min_dist, batch_min_dist], dim=0).min(dim=0, keepdim=True).values

        if self._max_dist is None:
            self._max_dist = batch_max_dist
        else:
            self._max_dist = torch.cat([self._max_dist, batch_max_dist], dim=0).max(dim=0, keepdim=True).values

        if self._sum_dist is None:
            self._sum_dist = batch_sum_dist
        else:
            self._sum_dist += batch_sum_dist

    def evaluate(self):

        # gather results across gpus
        synchronize()
        min_dist = torch.cat(all_gather(self._min_dist), dim=0).min(dim=0).values  # (N,)
        max_dist = torch.cat(all_gather(self._max_dist), dim=0).max(dim=0).values  # (N,)
        sum_dist = torch.cat(all_gather(self._sum_dist), dim=0).sum(dim=0)  # (N,)
        num_points = torch.cat(all_gather(self._num_points.view(1))).sum()
        avg_dist = sum_dist / num_points

        print("i min_dist avg_dist max_dist")
        for i in range(min_dist.size(0)):
            print(f"{i}:\t{min_dist[i]}\t{avg_dist[i]}\t{max_dist[i]}")

        return {'scalars': {}, 'images': {}}
