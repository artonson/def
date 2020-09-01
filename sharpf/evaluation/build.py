import logging

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.core.lightning import LightningModule

from .evaluator import DatasetEvaluator, DatasetEvaluators
from ..utils.hydra import instantiate

log = logging.getLogger(__name__)


def build_evaluators(cfg: DictConfig, partition: str, pl_module: LightningModule):
    assert partition in ['val', 'test']
    if not partition in cfg.datasets:
        return None
    datasets_params = cfg.datasets[partition]
    assert isinstance(datasets_params, ListConfig)

    evaluators = []

    for dataset_param in datasets_params:
        dataset_evaluators = []
        if dataset_param.evaluators is not None:
            for evaluator_param in dataset_param.evaluators:
                dataset_evaluator = instantiate(evaluator_param, pl_module=pl_module,
                                                dataset_name=dataset_param.dataset_name)
                assert isinstance(dataset_evaluator, DatasetEvaluator)
                dataset_evaluators.append(dataset_evaluator)
        evaluators.append(DatasetEvaluators(dataset_evaluators))

    return evaluators
