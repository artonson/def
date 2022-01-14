import logging
import os
from typing import List, Iterable, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import LightningLoggerBase

from defs.utils.collect_env import collect_env_info
from defs.utils.hydra import instantiate
from defs.modeling.system import IdentityRegression, IdentitySegmentation

from configs import trainer_conf, optimizer, scheduler, resolvers

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # log.info(f"Environment info:\n{collect_env_info()}")
    # log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Current working directory: {os.getcwd()}")
    # log.info(f"Original working directory: {hydra.utils.get_original_cwd()}")

    loggers: Iterable[LightningLoggerBase] = instantiate(cfg.loggers)
    callbacks: Optional[List[pl.callbacks.Callback]] = instantiate(
        cfg.callbacks.fit) if not cfg.eval_only else instantiate(cfg.callbacks.test)
    model: pl.LightningModule = instantiate(cfg.system.system_class, cfg=cfg)
    trainer: pl.Trainer = pl.Trainer(**cfg.trainer, logger=loggers, callbacks=callbacks)

    # train / test
    if not cfg.eval_only:
        if cfg.preload_weights is not None:
            preload_weights_path = hydra.utils.to_absolute_path(cfg.preload_weights)
            assert os.path.exists(preload_weights_path), f"{preload_weights_path} does not exist"
            model.load_state_dict(torch.load(preload_weights_path)['state_dict'], strict=True)
            log.info("Preloaded weights successfully!")

        assert cfg.test_weights is None or cfg.test_weights == 'best'
        trainer.fit(model)
        log.info("Test model...")
        trainer.test(ckpt_path=cfg.test_weights)
        log.info(f"Test is ended for: {os.getcwd()}")
    else:
        if isinstance(model, IdentityRegression) or isinstance(model, IdentitySegmentation):
            trainer.test(model)
        else:
            test_weights_path = hydra.utils.to_absolute_path(cfg.test_weights)
            assert os.path.exists(test_weights_path), f"{test_weights_path} does not exist"
            model.load_state_dict(torch.load(test_weights_path)['state_dict'], strict=True)
            log.info("Loaded weights successfully!")
            trainer.test(model)
            log.info(f"Test is ended for: {os.getcwd()}\n\n\n")


if __name__ == "__main__":
    main()
