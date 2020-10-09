import logging
from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_lightning import TrainResult, EvalResult

from . import BaseLightningModule
from .. import logits_to_scalar, DGCNNHist, PointRegressorHist, PixelRegressorHist
from ...utils.hydra import instantiate, call

log = logging.getLogger(__name__)


class SharpFeaturesRegressionTask(BaseLightningModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = instantiate(self.hparams.model.model_class)
        self.example_input_array = instantiate(self.hparams.model.example_input_array)

    def forward(self, x, clamp=True):
        output = self.model(x)
        out = {}
        for output_element in self.hparams.model.output_elements:
            key = output_element['key']
            left, right = output_element['channel_range']
            if output.ndim == 3:
                out[key] = output[:, :, left:right]
            elif output.ndim == 4:
                out[key] = output[:, left:right, :, :]

        ##### post-process distances
        # convert logits to scalar if the output is a histogram
        if not self.training and (isinstance(self.model, DGCNNHist) or isinstance(self.model, PointRegressorHist)):
            out['distances'] = logits_to_scalar(out['distances'], self.model.a, self.model.b,
                                                self.model.discretization, self.model.margin)
        elif not self.training and isinstance(self.model, PixelRegressorHist):
            out['distances'] = out['distances'].permute(0, 2, 3, 1)  # (B, H, W, C)
            out['distances'] = logits_to_scalar(out['distances'], self.model.a, self.model.b,
                                                self.model.discretization, self.model.margin)  # (B, H, W, 1)
            out['distances'] = out['distances'].permute(0, 3, 1, 2)  # (B, 1, H, W)

        if clamp:
            out['distances'] = out['distances'].clamp(0.0, 1.0)
        if out['distances'].ndim == 3:
            out['distances'] = out['distances'].squeeze(2)  # (B, N, 1) -> (B, N)
        elif out['distances'].ndim == 4:
            out['distances'] = out['distances'].squeeze(1)  # (B, 1, H, W) -> (B, H, W)

        ##### post-process normals
        if 'normals' in out:
            if out['normals'].ndim == 3:  # (B, N, 3)
                out['normals'] = F.normalize(out['normals'], dim=2)
            elif out['normals'].ndim == 4:  # (B, 3, H, W)
                out['normals'] = F.normalize(out['normals'], dim=1)

        ##### post-process directions
        if 'directions' in out:
            if out['directions'].ndim == 3:  # (B, N, 3)
                out['directions'] = F.normalize(out['directions'], dim=2)
            elif out['directions'].ndim == 4:  # (B, 3, H, W)
                out['directions'] = F.normalize(out['directions'], dim=1)

        return out

    def _check_range(self, tensor, left=0.0, right=1.0):
        min_value, max_value = tensor.min().item(), tensor.max().item()
        if not (left <= min_value and max_value <= right):
            log.warning(f"The violation of assumed range: min={min_value}, max={max_value}")

    def training_step(self, batch, batch_idx: int):
        self._check_range(batch['distances'])
        outputs = self.forward(batch['points'], clamp=False)

        if outputs['distances'].ndim == 4:
            outputs['distances'] = outputs['distances'].permute(0, 2, 3, 1)  # (B, H, W, C)
            b, h, w, c = outputs['distances'].size()
            outputs['distances'] = outputs['distances'].view(b, h * w, c)
            batch['distances'] = batch['distances'].view(b, h * w)

        loss = 0
        loss_dict = {}
        for loss_param in self.hparams.task.losses:
            if loss_param.out_key == 'directions':
                assert batch['distances'].ndim == 2
                batch_size, num_points = batch['distances'].shape
                mask = (batch['distances'] < 1.0).unsqueeze(2).expand_as(batch['directions'])
                loss_value = call(loss_param.loss_func,
                                  torch.masked_select(outputs['directions'], mask),
                                  torch.masked_select(batch['directions'], mask), reduction='sum') / (
                                     batch_size * num_points)
            else:
                loss_value = call(loss_param.loss_func, outputs[loss_param.out_key], batch[loss_param.gt_key])

            loss_value = loss_value * loss_param['lambda']

            loss_dict[loss_param.name] = loss_value
            loss += loss_value

        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        result.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return result

    def _shared_eval_step(self, batch, batch_idx: int, dataloader_idx: Optional[int], partition: str):
        self._check_range(batch['distances'])
        evaluator = self._get_evaluator(dataloader_idx, partition)
        evaluator.process(batch, self.forward(batch['points']))
        return EvalResult()
