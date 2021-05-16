import logging
from typing import Optional, Callable, Any

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import Metric

log = logging.getLogger(__name__)


class MRMSE(Metric):

    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("rmse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds (Tensor): Predictions from model of size (batch, num_points)
            target (Tensor): Ground truth values of size (batch, num_points)
        """
        squared_errors = F.mse_loss(preds, target, reduction='none')
        self.rmse_sum += torch.sqrt(squared_errors.mean(dim=1)).sum().detach().cpu()
        self.total += preds.size(0)

    def compute(self):
        return (self.rmse_sum / self.total).item()


class Q95RMSE(Metric):

    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("rmses", default=torch.zeros(0), dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds (Tensor): Predictions from model of size (batch, num_points)
            target (Tensor): Ground truth values of size (batch, num_points)
        """
        squared_errors = F.mse_loss(preds, target, reduction='none')
        self.rmses = torch.cat([self.rmses.cpu(), torch.sqrt(squared_errors.mean(dim=1)).detach().cpu()], dim=0)

    def compute(self):
        return np.quantile(self.rmses.cpu().numpy(), 0.95)
