import logging

import torch
from pytorch_lightning.metrics import Metric
from typing import Union

log = logging.getLogger(__name__)


class MFPR(Metric):

    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("fpr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
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
        pos_label = 0
        neg_label = 1

        # ----iterative version----
        # for i in range(preds.shape[0]):
        #     mask = (target[i] == pos_label)
        #     if torch.any(mask):
        #         recall = ((preds[i][mask] == pos_label).float().sum() / mask.float().sum()).detach().cpu()
        #         self.mfpr_sum += (1 - recall).detach().cpu()
        #         self.total += 1
        # -------------------------

        # ------batch version------
        pos_mask = (target == pos_label)
        valid_mask = torch.any(pos_mask, dim=1)
        denom = torch.where(valid_mask, pos_mask.float().sum(dim=1), preds.new_ones(preds.size(0)))
        recall = torch.where(
            valid_mask,
            ((preds.where(pos_mask, neg_label * preds.new_ones(preds.shape)) == pos_label).float().sum(dim=1) / denom),
            preds.new_zeros(preds.size(0))
        )
        self.fpr_sum += (1 - recall).sum().detach()
        self.total += valid_mask.float().sum().detach().cpu()
        # -------------------------

    def compute(self):
        return (self.fpr_sum / self.total).item()
