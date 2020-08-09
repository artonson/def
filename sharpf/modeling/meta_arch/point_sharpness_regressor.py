import logging

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import TrainResult
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics import tensor_metric
from torch.utils.data import DataLoader

from sharpf.data import PointCloudIO
from sharpf.utils.comm import get_batch_size
from ..model.build import build_model
from ...utils.abc_utils import LotsOfHdf5Files
from ...utils.abc_utils.torch import CompositeTransform
from ...utils.config import flatten_omegaconf

log = logging.getLogger(__name__)


@tensor_metric()
def gather_sum(x: torch.Tensor) -> torch.Tensor:
    return x


class PointSharpnessRegressor(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.hparams = flatten_omegaconf(cfg)  # there should be better official way later
        self.cfg = cfg
        self.model = build_model(cfg.model.regression)
        self.example_input_array = torch.rand(1, 4096, 3)
        self.data_dir = hydra.utils.to_absolute_path(self.cfg.data.regression.data_dir)

        dist_backend = self.cfg.trainer.distributed_backend
        if (dist_backend is not None and 'ddp' in dist_backend) or (
                dist_backend is None and self.cfg.trainer.gpus is not None and (
                self.cfg.trainer.gpus > 1 or self.cfg.trainer.num_nodes > 1)):
            log.info('Converting BatchNorm to SyncBatchNorm. Do not forget other batch-dimension dependent operations.')
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        points, distances = batch['points'], batch['distances']
        preds = self.model(points)
        loss = hydra.utils.instantiate(self.cfg.meta_arch.loss, preds, distances)
        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def _shared_eval_step(self, batch, batch_idx, prefix):
        points, distances = batch['points'], batch['distances']
        preds = self.model(points)  # (batch, n_points)
        loss = hydra.utils.instantiate(self.cfg.meta_arch.loss, preds, distances)
        mean_squared_errors = F.mse_loss(preds, distances, reduction='none').mean(dim=1)  # (batch)
        root_mean_squared_errors = torch.sqrt(mean_squared_errors)
        self.logger[0].experiment.add_scalars('losses', {f'{prefix}_loss': loss})
        # TODO Consider pl.EvalResult, once there are good examples how to use it
        return {'rmse_sum': root_mean_squared_errors.sum(),
                'batch_size': torch.tensor(points.size(0), device=self.device)}

    def _shared_eval_epoch_end(self, outputs, prefix):
        rmse_sum = 0
        size = 0
        for output in outputs:
            rmse_sum += output['rmse_sum']
            size += output['batch_size']
        mean_rmse = gather_sum(rmse_sum) / gather_sum(size)
        logs = {f'{prefix}_mean_rmse': mean_rmse}
        return {f'{prefix}_mean_rmse': mean_rmse, 'log': logs}

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, prefix='val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, prefix='test')

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, prefix='val')

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, prefix='test')

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.opt, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def _get_dataset(self, partition):
        if hasattr(self, f'{partition}_set'):
            return getattr(self, f'{partition}_set')
        transform = CompositeTransform([hydra.utils.instantiate(tf) for tf in self.cfg.transforms[partition]])
        return LotsOfHdf5Files(
            data_dir=self.data_dir,
            io=PointCloudIO,
            data_label=self.cfg.data.regression.data_label,
            target_label=self.cfg.data.regression.target_label,
            partition=partition,
            transform=transform,
            max_loaded_files=self.cfg.data.regression.max_loaded_files
        )

    def _get_dataloader(self, partition):
        dataset = self._get_dataset(partition)
        num_workers = self.cfg.data_loader[partition].num_workers
        batch_size = get_batch_size(self.cfg.data_loader[partition].total_batch_size)
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
        return self._get_dataloader('test')
