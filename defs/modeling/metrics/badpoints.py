import logging

import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import Metric

log = logging.getLogger(__name__)


class MeanBadPoints(Metric):

    def __init__(self, threshold: float):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.threshold = threshold
        self.add_state("bp_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds (Tensor): Predictions from model of size (batch, num_points)
            target (Tensor): Ground truth values of size (batch, num_points)
        """
        squared_errors = F.mse_loss(preds, target, reduction='none')
        bad_points_mask = squared_errors > self.threshold ** 2
        self.bp_sum += bad_points_mask.float().mean(dim=1).sum().detach().cpu()
        self.total += preds.size(0)

    def compute(self):
        return (self.bp_sum / self.total).item()
