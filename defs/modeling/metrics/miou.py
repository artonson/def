import logging

import torch
from pytorch_lightning.metrics import Metric

log = logging.getLogger(__name__)


class MIOU(Metric):

    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("iou_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds (Tensor): Predictions from model of size (batch, num_points)
            target (Tensor): Ground truth values of size (batch, num_points)
        """
        assert preds.dtype == torch.bool and target.dtype == torch.bool
        assert torch.all(torch.any(target, dim=1))
        preds = preds.float()
        target = target.float()

        intersection = preds * target
        union = (target + preds - intersection).sum(dim=1)
        self.iou_sum += (intersection.sum(dim=1) / union).sum().detach().cpu()
        self.total += preds.size(0)

    def compute(self):
        return (self.iou_sum / self.total).item()
