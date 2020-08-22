import logging

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import TrainResult
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics import tensor_metric
from pytorch_lightning.metrics.sklearns import BalancedAccuracy
from torch.utils.data import DataLoader

from sharpf.utils.comm import get_batch_size
from pytorch_lightning.metrics.functional import stat_scores
from ..model.build import build_model
from ...data import DepthMapIO
from ...utils.abc_utils.hdf5 import DepthDataset
from ...utils.abc_utils.torch import CompositeTransform

log = logging.getLogger(__name__)


@tensor_metric()
def gather_sum(x: torch.Tensor) -> torch.Tensor:
    return x


class DepthSegmentator(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.hparams = cfg
        self.task = 'segmentation'
        self.model = build_model(self.hparams.model)
        self.example_input_array = torch.rand(1, 1, 64, 64)
        self.data_dir = hydra.utils.to_absolute_path(self.hparams.data.data_dir)

        dist_backend = self.hparams.trainer.distributed_backend
        if (dist_backend is not None and 'ddp' in dist_backend) or (
                dist_backend is None and self.hparams.trainer.gpus is not None and (
                self.hparams.trainer.gpus > 1 or self.hparams.trainer.num_nodes > 1)):
            log.info('Converting BatchNorm to SyncBatchNorm. Do not forget other batch-dimension dependent operations.')
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        points, target = batch['image'], batch['close_to_sharp_mask']
        points = points.unsqueeze(1) if points.dim() == 3 else points
        preds = self.forward(points)
        loss = hydra.utils.instantiate(self.hparams.meta_arch.loss, preds, target)
        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def _shared_eval_step(self, batch, batch_idx, prefix):
        stat_names = ['tpr', 'tnr']
        points, target = batch['image'], batch['close_to_sharp_mask']
        points = points.unsqueeze(1) if points.dim() == 3 else points
        preds = self.forward(points)

        statistics = torch.Tensor(list(map(lambda a: stat_scores(a[0], a[1], class_index=1),
                                           [preds, target]))).T.to(preds.device) # return: tp, fp, tn, fn, sup
                                                                # dim: (5, batch)

        tpr = statistics[0] / (statistics[0] + statistics[-2])  # 1(pred=1| true=1) / (1(pred=1| true=1) + 1(pred=0|true=1))
        tnr = statistics[2] / (statistics[2] + statistics[1])  # 1(pred=0| true=0) / (1(pred=0| true=0) + 1(pred=1|true=0))

        # loss = hydra.utils.instantiate(self.cfg.meta_arch.loss, preds, distances)
        # self.logger[0].experiment.add_scalars('losses', {f'{prefix}_loss': loss})
        # TODO Consider pl.EvalResult, once there are good examples how to use it
        return {f'{stat_names[0]}': tpr,
                f'{stat_names[1]}': tnr,
                'batch_size': torch.tensor(points.size(0), device=self.device)}

    def _shared_eval_epoch_end(self, outputs, prefix):
        stat_names = ['tpr', 'tnr']
        metric_sum = 0
        size = 0
        for output in outputs:
            metric_sum += sum((output[f'{stat_names[0]}'] + output[f'{stat_names[1]}']).double() / 2)
            size += output['batch_size']

        metric_name = 'balanced_accuracy'
        mean_metric = gather_sum(metric_sum) / gather_sum(size)

        logs = {f'{prefix}_mean_{metric_name}': mean_metric}
        return {f'{prefix}_mean_{metric_name}': mean_metric, 'log': logs}

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, prefix='val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, prefix='test')

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, prefix='val')

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, prefix='test')

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.opt, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def _get_dataset(self, partition):
        if hasattr(self, f'{partition}_set') and getattr(self, f'{partition}_set') is not None:
            return getattr(self, f'{partition}_set')
        transform = CompositeTransform([hydra.utils.instantiate(tf) for tf in self.hparams.transforms[partition]])
        if 'normalisation' in self.hparams.transforms.keys():
            normalisation = self.hparams.transforms['normalisation']
        else:
            normalisation = None

        return DepthDataset(
            data_dir=self.data_dir,
            io=DepthMapIO,
            data_label=self.hparams.data.data_label,
            target_label=self.hparams.data.target_label,
            task=self.task,
            partition=partition,
            transform=transform,
            max_loaded_files=self.hparams.data.max_loaded_files,
            normalisation=normalisation
        )

    def _get_dataloader(self, partition):
        dataset = self._get_dataset(partition)
        num_workers = self.hparams.data_loader[partition].num_workers
        batch_size = get_batch_size(self.hparams.data_loader[partition].total_batch_size)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    def setup(self, stage: str):
        self.train_set = self._get_dataset('train') if stage == 'fit' else None
        self.val_set = self._get_dataset('val') if stage == 'fit' else None
        self.test_set = self._get_dataset('test') if stage == 'test' else None

    def train_dataloader(self):
        return self._get_dataloader('train')

    def val_dataloader(self):
        return self._get_dataloader('val')

    def test_dataloader(self):
        return self._get_dataloader('val')  # FIXME
