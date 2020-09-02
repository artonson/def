import logging
from typing import Optional

from pytorch_lightning import TrainResult, EvalResult

from . import BaseLightningModule
from ...utils.hydra import instantiate, call

log = logging.getLogger(__name__)


class SharpFeaturesRegressionTask(BaseLightningModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = instantiate(self.hparams.model.model_class)
        self.example_input_array = instantiate(self.hparams.model.example_input_array)

    def forward(self, x, clamp=True):
        out = self.model(x)
        if out.ndim == 3:  # (B, N, 1) -> (B, N)
            out = out.squeeze(2)
        if out.ndim == 4:  # (B, 1, H, W) -> (B, H, W)
            out = out.squeeze(1)
        if clamp:
            out = out.clamp(0.0, 1.0)
        # todo: I think model should output dict
        return out

    def _check_range(self, tensor, left=0.0, right=1.0):
        min_value, max_value = tensor.min().item(), tensor.max().item()
        if not (left <= min_value and max_value <= right):
            log.warning(f"The violation of assumed range in train partition: min={min_value}, max={max_value}")

    def training_step(self, batch, batch_idx: int):
        points, distances = batch['points'], batch['distances']
        self._check_range(distances)
        preds = self.forward(points, clamp=False)
        if preds.ndim == 4:
            preds = preds.permute(0, 2, 3, 1)  # (B, H, W, C)
            b, h, w, c = preds.size()
            preds = preds.view(b, h * w, c)
            distances = distances.view(b, h * w)
        loss = call(self.hparams.task.loss, preds, distances)
        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return result

    def _shared_eval_step(self, batch, batch_idx: int, dataloader_idx: Optional[int], partition: str):
        points, distances = batch['points'], batch['distances']
        self._check_range(distances)
        outputs = {'pred_distances': self.forward(points)}
        evaluator = self._get_evaluator(dataloader_idx, partition)
        evaluator.process(batch, outputs)
        return EvalResult()
