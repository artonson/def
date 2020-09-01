import logging

from pytorch_lightning import TrainResult, EvalResult

from . import BaseLightningModule
from ...evaluation import build_evaluators
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

    def training_step(self, batch, batch_idx):
        points, target = batch['points'], batch['close_to_sharp_mask']
        preds = self.forward(points, as_mask=False)
        loss = call(self.hparams.task.loss, preds, target)
        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return result

    def _shared_eval_step(self, batch, batch_idx, dataloader_idx, partition):
        if self.evaluators is None:
            self.evaluators = build_evaluators(self.hparams, partition, self)
            if self.evaluators is None:
                return EvalResult()

        points, target = batch['points'], batch['close_to_sharp_mask']
        outputs = {'pred_mask': self.forward(points, as_mask=True)}
        evaluator_idx = dataloader_idx if not dataloader_idx is None else 0
        self.evaluators[evaluator_idx].process(batch, outputs)
        return EvalResult()
