import logging

import hydra
import torch
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics import tensor_metric
from torch.utils.data import DataLoader

from sharpf.data import PointCloudIO
from sharpf.utils.comm import get_world_size
from ..model.build import build_model
from ...utils.abc_utils import LotsOfHdf5Files
from ...utils.abc_utils.torch import CompositeTransform, TypeCast, Center, NormalizeL2, Random3DRotation
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
        self.example_input_array = torch.rand(2, 4096, 3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        points, distances = batch['points'], batch['distances']
        preds = self.model(points)
        loss = hydra.utils.instantiate(self.cfg.meta_arch.loss, preds, distances)
        self.logger[0].experiment.add_scalars("losses", {"train_loss": loss})
        return {'loss': loss}

    def _shared_eval_step(self, batch, batch_idx, prefix):
        points, distances = batch['points'], batch['distances']
        preds = self.model(points)  # (batch, n_points)
        loss = hydra.utils.instantiate(self.cfg.meta_arch.loss, preds, distances)
        mean_squared_errors = F.mse_loss(preds, distances, reduction='none').mean(dim=1)  # (batch)
        root_mean_squared_errors = torch.sqrt(mean_squared_errors)
        self.logger[0].experiment.add_scalars("losses", {f"{prefix}_loss": loss})
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
        return self._shared_eval_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, prefix="test")

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, prefix='val')

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, prefix='test')

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.opt, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def setup(self, stage: str):
        self.cfg.data.regression.data_dir = hydra.utils.to_absolute_path(self.cfg.data.regression.data_dir)
        if stage == 'fit':
            self.train_set = LotsOfHdf5Files(
                data_dir=self.cfg.data.regression.data_dir,
                io=PointCloudIO,
                data_label=self.cfg.data.regression.data_label,
                target_label=self.cfg.data.regression.target_label,
                partition='train',
                transform=CompositeTransform([hydra.utils.instantiate(tf) for tf in self.cfg.transforms.train]),
                max_loaded_files=self.cfg.data.regression.max_loaded_files
            )
            self.val_set = LotsOfHdf5Files(
                data_dir=self.cfg.data.regression.data_dir,
                io=PointCloudIO,
                data_label=self.cfg.data.regression.data_label,
                target_label=self.cfg.data.regression.target_label,
                partition='val',
                transform=CompositeTransform([hydra.utils.instantiate(tf) for tf in self.cfg.transforms.val]),
                max_loaded_files=self.cfg.data.regression.max_loaded_files
            )
        elif stage == 'test':
            self.val_set = LotsOfHdf5Files(
                data_dir=self.cfg.data.regression.data_dir,
                io=PointCloudIO,
                data_label=self.cfg.data.regression.data_label,
                target_label=self.cfg.data.regression.target_label,
                partition='test',
                transform=CompositeTransform([hydra.utils.instantiate(tf) for tf in self.cfg.transforms.test]),
                max_loaded_files=self.cfg.data.regression.max_loaded_files
            )
        else:
            raise ValueError(f"Unknown stage {stage}")

    def _get_batch_size(self, total_batch_size):
        world_size = get_world_size()
        assert (total_batch_size > 0 and total_batch_size % world_size == 0), \
            f"Total batch size ({total_batch_size}) must be divisible by the number of gpus ({world_size})."
        batch_size = total_batch_size // world_size
        return batch_size

    def train_dataloader(self):
        num_workers = self.cfg.data_loader.train.num_workers
        batch_size = self._get_batch_size(self.cfg.data_loader.train.total_batch_size)
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        num_workers = self.cfg.data_loader.val.num_workers
        batch_size = self._get_batch_size(self.cfg.data_loader.val.total_batch_size)
        return DataLoader(self.val_set, batch_size=batch_size,
                          num_workers=num_workers, pin_memory=True)

    def test_dataloader(self):
        num_workers = self.cfg.data_loader.test.num_workers
        batch_size = self._get_batch_size(self.cfg.data_loader.test.total_batch_size)
        return DataLoader(self.test_set, batch_size=batch_size,
                          num_workers=num_workers, pin_memory=True)
