import logging
from typing import List, Tuple

import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .evaluator import DatasetEvaluator, DatasetEvaluators
from ..utils.hydra import instantiate

log = logging.getLogger(__name__)


def build_evaluators(datasets: List[Tuple[DictConfig, Dataset]], model: nn.Module) -> List[DatasetEvaluator]:
    all_evaluators: List[DatasetEvaluator] = []
    for dataset_param, dataset in datasets:
        dataset_evaluators: List[DatasetEvaluator] = []
        if dataset_param.evaluators is not None:
            for evaluator_param in dataset_param.evaluators:
                dataset_evaluator = instantiate(
                    evaluator_param,
                    model=model,
                    dataset=dataset,
                    dataset_name=dataset_param.dataset_name)
                assert isinstance(dataset_evaluator, DatasetEvaluator)
                dataset_evaluators.append(dataset_evaluator)
        all_evaluators.append(DatasetEvaluators(dataset_evaluators))
    return all_evaluators
