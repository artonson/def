import logging

import hydra
import torch
import torch.nn as nn
from pytorch_lightning import TrainResult
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import stat_scores
from torch.utils.data import DataLoader

from sharpf.utils.comm import get_batch_size, all_gather, synchronize
from ..metrics import balanced_accuracy
from ..model.build import build_model
from ...data import DepthMapIO
from ...utils.abc_utils.hdf5 import DepthDataset
from ...utils.abc_utils.torch import CompositeTransform

log = logging.getLogger(__name__)


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

    def forward(self, x, as_mask=True):
        out = self.model(x)
        if as_mask:
            return (out.sigmoid() > 0.5).long()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        points, target = batch['image'], batch['close_to_sharp_mask']
        points = points.unsqueeze(1) if points.dim() == 3 else points
        preds = self.forward(points, as_mask=False)
        loss = hydra.utils.instantiate(self.hparams.meta_arch.loss, preds, target)
        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def _shared_eval_step(self, batch, batch_idx, prefix):
        points, target = batch['image'], batch['close_to_sharp_mask']
        points = points.unsqueeze(1) if points.dim() == 3 else points
        preds = self.forward(points, as_mask=True)
        stats = [list(stat_scores(preds[i], target[i], class_index=1)) for i in range(preds.size(0))]
        tp, fp, tn, fn, sup = torch.Tensor(stats).to(preds.device).T.unsqueeze(1)  # each of size (1, batch)
        return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'sup': sup}

    def _shared_eval_epoch_end(self, outputs, prefix):
        # gather across sub batches
        tp = torch.cat([output['tp'] for output in outputs])
        fp = torch.cat([output['fp'] for output in outputs])
        tn = torch.cat([output['tn'] for output in outputs])
        fn = torch.cat([output['fn'] for output in outputs])

        # gather results across gpus
        synchronize()
        tp = torch.cat(all_gather(tp))
        fp = torch.cat(all_gather(fp))
        tn = torch.cat(all_gather(tn))
        fn = torch.cat(all_gather(fn))

        # calculate metrics
        ba = balanced_accuracy(tp, fp, tn, fn)

        logs = {f'{prefix}_balanced_accuracy': ba}
        return {f'{prefix}_balanced_accuracy': ba, 'log': logs}

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
