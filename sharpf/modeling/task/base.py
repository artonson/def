import logging
from typing import Optional, Dict, List, Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import EvalResult
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import Dataset

from sharpf.data import build_loaders, build_datasets
from sharpf.utils.hydra import instantiate
from ...evaluation import build_evaluators, DatasetEvaluator

log = logging.getLogger(__name__)


class BaseLightningModule(LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.hparams = cfg
        self.datasets: Dict[str, Optional[List[Tuple[DictConfig, Dataset]]]] = {'train': None, 'val': None,
                                                                                'test': None}
        self.evaluators: Dict[str, Optional[List[DatasetEvaluator]]] = {'train': None, 'val': None, 'test': None}
        self.learning_rate = cfg.opt.lr

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int):
        raise NotImplementedError

    def _shared_eval_step(self, batch, batch_idx: int, dataloader_idx: Optional[int], partition: str):
        raise NotImplementedError

    def checkpoint_on(self, results: Dict[str, float]):
        checkpoint_on = None
        if 'checkpoint_on' in self.hparams.task and self.hparams.task.checkpoint_on is not None:
            assert self.hparams.task.checkpoint_on in results
            checkpoint_on = torch.tensor(results[self.hparams.task.checkpoint_on])
        return checkpoint_on

    def early_stop_on(self, results: Dict[str, float]):
        early_stop_on = None
        if 'early_stop' in self.hparams.task and self.hparams.task.early_stop.value is not None:
            assert self.hparams.task.early_stop.value in results
            early_stop_on = torch.tensor(results[self.hparams.task.early_stop.value])
        return early_stop_on

    def _shared_eval_epoch_end(self, outputs, partition: str):

        # all evaluators should return dict with two keys 'scalars' and 'images'
        results = {'scalars': {}, 'images': {}}

        # merge results from different evaluators
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

        # define EvalResult with possible checkpointing and early stopping
        if partition == 'val':
            result = EvalResult(checkpoint_on=self.checkpoint_on(results['scalars']),
                                early_stop_on=self.early_stop_on(results['scalars']))
        else:
            result = EvalResult()

        # log scalars
        result.log_dict(results['scalars'])

        # log images
        for name, image in results['images'].items():
            try:
                self.logger[0].experiment.add_image(name, image)
            except TypeError as e:
                print(f"Skipping TypeError: {e}")

        # reset evaluators
        self.evaluators[partition] = None
        # for evaluator in self.evaluators[partition]:
        #     evaluator.reset()

        return result

    def validation_step(self, batch, batch_idx: int, *args):
        dataloader_idx = args[0] if len(args) == 1 else None
        return self._shared_eval_step(batch, batch_idx, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx: int, *args):
        assert len(args) <= 1
        dataloader_idx = args[0] if len(args) == 1 else None
        return self._shared_eval_step(batch, batch_idx, dataloader_idx, 'test')

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.opt, params=self.parameters(), lr=self.learning_rate)
        scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
        # todo add possibility to turn off scheduler
        return [optimizer], [scheduler]

    def train_dataloader(self):
        self.datasets['train'] = build_datasets(self.hparams, 'train')
        loaders = build_loaders(self.hparams, self.datasets['train'], 'train')
        assert len(loaders) == 1, "There must be only one train dataloader"
        return loaders[0]

    def val_dataloader(self):
        self.datasets['val'] = build_datasets(self.hparams, 'train')
        return build_loaders(self.hparams, self.datasets['val'], 'val')

    def test_dataloader(self):
        self.datasets['test'] = build_datasets(self.hparams, 'test')
        return build_loaders(self.hparams, self.datasets['test'], 'test')

    def _get_evaluator(self, dataloader_idx: int, partition: str) -> Optional[DatasetEvaluator]:
        if self.evaluators[partition] is None:
            self.evaluators[partition] = build_evaluators(self.datasets[partition], self)

        evaluator_idx = dataloader_idx if dataloader_idx is not None else 0
        return self.evaluators[partition][evaluator_idx]
