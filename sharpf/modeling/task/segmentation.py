import logging
from typing import Optional

from pytorch_lightning import TrainResult, EvalResult

from . import BaseLightningModule
from ...utils.hydra import call, instantiate

log = logging.getLogger(__name__)


class SharpFeaturesSegmentationTask(BaseLightningModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = instantiate(self.hparams.model.model_class)
        self.example_input_array = instantiate(self.hparams.model.example_input_array)

    def forward(self, x, as_mask=True):
        out = self.model(x)
        if as_mask:
            return (out.sigmoid() > 0.5).long()
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        points, target = batch['points'], batch['close_to_sharp_mask']
        preds = self.forward(points, as_mask=False)
        loss = call(self.hparams.task.loss, preds, target)
        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return result

    def _shared_eval_step(self, batch, batch_idx: int, dataloader_idx: Optional[int], partition: str):
        points, target = batch['points'], batch['close_to_sharp_mask']
        outputs = {'pred_mask': self.forward(points, as_mask=True)}
        evaluator = self._get_evaluator(dataloader_idx, partition)
        evaluator.process(batch, outputs)
        return EvalResult()
