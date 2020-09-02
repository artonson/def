import logging

import torch
from pytorch_lightning.metrics.functional import stat_scores

from .evaluator import DatasetEvaluator
from ..modeling import balanced_accuracy
from ..utils.comm import all_gather, synchronize

log = logging.getLogger(__name__)


class SegmentationEvaluator(DatasetEvaluator):
    """
    Evaluate segmentation metrics.
    """

    def __init__(self, input_key, output_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.reset()

    def reset(self):
        self._tp = []
        self._fp = []
        self._tn = []
        self._fn = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs (dict): the inputs to a model.
            outputs (dict): the outputs of a model.
        """
        target = inputs[self.input_key]
        preds = outputs[self.output_key]

        stats = [list(stat_scores(preds[i], target[i], class_index=1)) for i in range(preds.size(0))]
        tp, fp, tn, fn, sup = torch.Tensor(stats).to(preds.device).T.unsqueeze(2)  # each of size (batch, 1)

        self._tp.append(tp)
        self._fp.append(fp)
        self._tn.append(tn)
        self._fn.append(fn)

    def evaluate(self):
        self._tp = torch.cat(self._tp, dim=0)
        self._fp = torch.cat(self._fp, dim=0)
        self._tn = torch.cat(self._tn, dim=0)
        self._fn = torch.cat(self._fn, dim=0)

        # gather results across gpus
        synchronize()
        tp = torch.cat(all_gather(self._tp), dim=0)
        fp = torch.cat(all_gather(self._fp), dim=0)
        tn = torch.cat(all_gather(self._tn), dim=0)
        fn = torch.cat(all_gather(self._fn), dim=0)

        # calculate metrics
        ba = balanced_accuracy(tp, fp, tn, fn)

        scalars = {f'balanced_accuracy/{self.dataset_name}': ba}

        return {'scalars': scalars, 'images': {}}
