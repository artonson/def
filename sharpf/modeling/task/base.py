import logging

import torch
from pytorch_lightning import EvalResult
from pytorch_lightning.core.lightning import LightningModule

from sharpf.utils.hydra import instantiate

log = logging.getLogger(__name__)


class BaseLightningModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.hparams = cfg
        self.evaluators = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def _shared_eval_step(self, batch, batch_idx, dataloader_idx, partition):
        raise NotImplementedError

    def _shared_eval_epoch_end(self, outputs, partition):
        results = {'scalars': {}, 'images': {}}
        for i in range(len(self.evaluators)):
            evaluator = self.evaluators[i]
            results_i = evaluator.evaluate()
            if results_i is None:
                continue
            assert isinstance(results_i, dict), \
                f"Evaluator must return a dict on the main process. Got {results_i} instead."
            for meta_key in results.keys():
                for k, v in results_i[meta_key].items():
                    assert (
                            k not in results[meta_key]
                    ), f"Different evaluators produce results ({meta_key}) with the same key {k}"
                    results[meta_key][k] = v

        if partition == 'val' and 'checkpoint_on' in self.hparams.task and self.hparams.task.checkpoint_on in results[
            'scalars']:
            checkpoint_on = torch.tensor(results['scalars'][self.hparams.task.checkpoint_on])
        else:
            checkpoint_on = None

        if partition == 'val' and 'early_stop_on' in self.hparams.task and self.hparams.task.early_stop_on in results[
            'scalars']:
            early_stop_on = torch.tensor(results['scalars'][self.hparams.task.early_stop_on])
        else:
            early_stop_on = None

        result = EvalResult(checkpoint_on=checkpoint_on, early_stop_on=early_stop_on)
        for name, image in results['images'].items():
            self.logger[0].experiment.add_image(name, image)
        result.log_dict(results['scalars'])
        return result

    def validation_step(self, batch, batch_idx, *args):
        dataloader_idx = args[0] if len(args) == 1 else None
        return self._shared_eval_step(batch, batch_idx, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx, *args):
        assert len(args) <= 1
        dataloader_idx = args[0] if len(args) == 1 else None
        return self._shared_eval_step(batch, batch_idx, dataloader_idx, 'test')

    def validation_epoch_end(self, outputs):
        results = self._shared_eval_epoch_end(outputs, 'val')
        self.evaluators = None
        return results

    def test_epoch_end(self, outputs):
        results = self._shared_eval_epoch_end(outputs, 'test')
        self.evaluators = None
        return results

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.opt, params=self.parameters())
        scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]
