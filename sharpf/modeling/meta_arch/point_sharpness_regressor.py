import logging

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from pytorch_lightning import TrainResult
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics import tensor_metric
from torch.utils.data import DataLoader

from sharpf.data import PointCloudIO
from sharpf.utils.comm import get_batch_size
from ..model.build import build_model
from ...utils.abc_utils import LotsOfHdf5Files
from ...utils.abc_utils.torch import CompositeTransform
from ...utils.image import plot_to_image

log = logging.getLogger(__name__)


@tensor_metric()
def gather_sum(x: torch.Tensor) -> torch.Tensor:
    return x


class PointSharpnessRegressor(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.hparams = cfg
        self.model = build_model(self.hparams.model)
        self.example_input_array = torch.rand(1, 4096, 3)
        self.dataset_names = {}
        self.num_dataloaders = {}

        dist_backend = self.hparams.trainer.distributed_backend
        if (dist_backend is not None and 'ddp' in dist_backend) or (
                dist_backend is None and self.hparams.trainer.gpus is not None and (
                self.hparams.trainer.gpus > 1 or self.hparams.trainer.num_nodes > 1)):
            log.info('Converting BatchNorm to SyncBatchNorm. Do not forget other batch-dimension dependent operations.')
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def forward(self, x, clamp=True):
        out = self.model(x)
        if clamp:
            out = out.clamp(0.0, 1.0)
        return out

    def training_step(self, batch, batch_idx):
        points, distances = batch['points'], batch['distances']
        if not (0.0 <= distances.min().item() and distances.max().item() <= 1.0):
            log.warning(
                f"The violation of assumed range in train partition: min={distances.min().item()}, max={distances.max().item()}")

        preds = self.forward(points, clamp=False)
        loss = hydra.utils.call(self.hparams.meta_arch.loss, preds, distances)
        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def _shared_eval_step(self, batch, batch_idx):
        points, distances = batch['points'], batch['distances']
        if not (0.0 <= distances.min().item() and distances.max().item() <= 1.0):
            log.warning(
                f"The violation of assumed range in eval partition: min={distances.min().item()}, max={distances.max().item()}")

        preds = self.forward(points)  # (batch, n_points)

        mean_squared_errors = F.mse_loss(preds, distances, reduction='none').mean(dim=1)  # (batch)
        root_mean_squared_errors = torch.sqrt(mean_squared_errors)
        rmse_hist = torch.histc(root_mean_squared_errors, bins=100, min=0.0, max=1.0)

        return {
            'rmse_sum': root_mean_squared_errors.sum(),
            'batch_size': torch.tensor(points.size(0), device=self.device),
            'rmse_hist': rmse_hist
        }

    def _plot_rmse_hist(self, rmse_hist, label):
        figure = plt.figure(figsize=(4.8, 3.6))
        prototype_hist = rmse_hist[0] if isinstance(rmse_hist, list) else rmse_hist
        bins = torch.linspace(0, 1, len(prototype_hist) + 1)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        if isinstance(rmse_hist, list):
            for label_i, hist in zip(label, rmse_hist):
                plt.bar(center, hist, label=label_i, align='center', width=width, alpha=0.5)
        else:
            plt.bar(center, rmse_hist, label=label, align='center', width=width)
        plt.xlabel('RMSE per patch value')
        plt.ylabel('# of patches')
        plt.xlim((0.0, 1.0))
        plt.yscale('log')
        plt.legend(loc='upper right')
        return figure

    def _shared_eval_epoch_end(self, outputs, partition, dataloader_idx=None):
        # TODO Consider pl.EvalResult, once there are good examples how to use it

        assert partition in ['train', 'val', 'test']
        num_dataloaders = self.num_dataloaders[partition]
        assert not (dataloader_idx is not None and num_dataloaders == 1), "not possible situation"

        if dataloader_idx is None and num_dataloaders > 1:
            # in this case outputs is a list, where list is enumerated by dataloader_idx
            results = {}
            rmse_hists = []
            hist_labels = []
            for i in range(num_dataloaders):
                result = self._shared_eval_epoch_end(outputs[i], partition, dataloader_idx=i)
                rmse_hists.append(result.pop('rmse_hist').cpu())
                hist_labels.append(self.dataset_names[partition][i])
                log = result.pop('log')
                results['log'] = {**results['log'], **log} if 'log' in results else log
                results = {**results, **result}
            self.logger[0].experiment.add_image(f'{partition}_rmse_hist',
                                                plot_to_image(self._plot_rmse_hist(rmse_hists, hist_labels)))
            return results
        else:
            # in this case outputs is a list of _shared_eval_step on sub batches

            # gather results across sub batches
            rmse_sum = 0
            size = 0
            rmse_hist = 0

            for output in outputs:
                rmse_sum += output['rmse_sum']
                size += output['batch_size']
                rmse_hist += output['rmse_hist']

            # gather results across gpus
            rmse_sum = gather_sum(rmse_sum)
            size = gather_sum(size)
            rmse_hist = gather_sum(rmse_hist)

            # calculate metrics
            mean_rmse = rmse_sum / size

            # set prefix
            prefix = f'{partition}_{self.dataset_names[partition][dataloader_idx]}' if dataloader_idx is not None else partition

            # result
            logs = {f'{prefix}_mean_rmse': mean_rmse}
            result = {f'{prefix}_mean_rmse': mean_rmse, 'log': logs}

            # plot histogram
            if num_dataloaders == 1:
                self.logger[0].experiment.add_image(f'{prefix}_rmse_hist',
                                                    plot_to_image(self._plot_rmse_hist(rmse_hist.cpu(), "")))
            else:
                result['rmse_hist'] = rmse_hist

            return result

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.opt, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def _get_dataset(self, partition):
        assert partition in ['train', 'val', 'test']

        if hasattr(self, f'{partition}_set') and getattr(self, f'{partition}_set') is not None:
            return getattr(self, f'{partition}_set')

        try:
            dataset_params = self.hparams.data[f'{partition}_dataset']
        except KeyError:
            return None

        transform = CompositeTransform([hydra.utils.instantiate(tf) for tf in self.hparams.transforms[partition]])

        if isinstance(dataset_params, ListConfig):
            datasets = []
            self.dataset_names[partition] = []
            self.num_dataloaders[partition] = len(dataset_params)
            for dataset_part_params in dataset_params:
                datasets.append(
                    LotsOfHdf5Files(
                        data_dir=dataset_part_params.data_dir,
                        filenames=dataset_part_params.filenames,
                        io=PointCloudIO,
                        data_label=dataset_part_params.data_label,
                        target_label=dataset_part_params.target_label,
                        partition=partition,
                        transform=transform,
                        max_loaded_files=dataset_part_params.max_loaded_files
                    )
                )
                self.dataset_names[partition].append(dataset_part_params.name)
            return datasets
        else:
            self.num_dataloaders[partition] = 1
            return LotsOfHdf5Files(
                data_dir=dataset_params.data_dir,
                filenames=dataset_params.filenames,
                io=PointCloudIO,
                data_label=dataset_params.data_label,
                target_label=dataset_params.target_label,
                partition=partition,
                transform=transform,
                max_loaded_files=dataset_params.max_loaded_files
            )

    def _get_dataloader(self, partition):
        num_workers = self.hparams.data_loader[partition].num_workers
        pin_memory = self.hparams.data_loader[partition].pin_memory
        batch_size = get_batch_size(self.hparams.data_loader[partition].total_batch_size)

        dataset = self._get_dataset(partition)
        if dataset is None:
            return None

        if isinstance(dataset, list):
            assert partition != 'train', 'PL does not support several train data loaders'
            assert len(dataset) > 1
            data_loaders = []
            for dataset_part in dataset:
                data_loaders.append(
                    DataLoader(dataset_part, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory))
            return data_loaders
        else:
            return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

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
