import logging
from typing import Optional, Callable, Any

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import Metric
import sklearn.metrics

log = logging.getLogger(__name__)


class MAP(Metric):

    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("ap_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds (Tensor): Predictions (probabilities) from model of size (batch, num_points)
            target (Tensor): Ground truth values of size (batch, num_points)
        """
        for i in range(preds.size(0)):
            self.ap_sum += sklearn.metrics.average_precision_score(
                target[i].detach().cpu().numpy(),
                preds[i].detach().cpu().numpy(),
            )
        self.total += preds.size(0)

    def compute(self):
        return (self.ap_sum / self.total).item()
