import logging
from typing import Optional, Callable, Any

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


class MRMSE():

    def __init__(self):
        self.set_defaults()

    def __str__(self):
        return "MRMSE"

    def set_defaults(self):
        self.rmse_sum = torch.tensor(0.0)
        self.total = torch.tensor(0)

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


class Q95RMSE():

    def __init__(self):
        self.set_defaults()

    def __str__(self):
        return "Q95RMSE"

    def set_defaults(self):
        self.rmses = torch.zeros(0)

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
