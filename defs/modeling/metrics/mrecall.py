import logging

import torch
from pytorch_lightning.metrics import Metric
from typing import Union

log = logging.getLogger(__name__)


class MeanRecall(Metric):

    def __init__(self, pos_label):
        """
        pos_label: 0 or 1
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        assert pos_label == 1 or pos_label == 0
        self.pos_label = pos_label
        self.neg_label = 1 - pos_label
        self.add_state("recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds (Tensor): Predictions (binary, sharp=1) of size (batch, num_points)
            target (Tensor): Ground truth values (binary, sharp=1) of size (batch, num_points)
        """
        preds = preds.float()
        target = target.float()

        # itereative version (slower)
        # for i in range(preds.shape[0]):
        #     mask = (target[i] == self.pos_label)
        #     if torch.any(mask):
        #         self.recall_sum += ((preds[i][mask] == self.pos_label).float().sum() / mask.float().sum()).detach().cpu()
        #         self.total += 1

        # batch version
        pos_mask = (target == self.pos_label)
        valid_mask = torch.any(pos_mask, dim=1)
        denom = torch.where(valid_mask, pos_mask.float().sum(dim=1), preds.new_ones(preds.size(0)))
        self.recall_sum += torch.where(
            valid_mask,
            ((preds.where(pos_mask, self.neg_label * preds.new_ones(preds.shape)) == self.pos_label).float().sum(dim=1) / denom),
            preds.new_zeros(preds.size(0))
        ).sum().detach().cpu()
        self.total += valid_mask.float().sum().detach().cpu()

    def compute(self):
        return (self.recall_sum / self.total).item()
