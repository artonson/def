import logging
import os

from sharpf.data import build_loaders

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler
import torch

from sharpf.utils.callbacks import FitDurationCallback
from sharpf.utils.collect_env import collect_env_info

from configs import trainer, optimizer, scheduler

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # log common information
    log.info(f"Environment info:\n{collect_env_info()}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Original working directory: {hydra.utils.get_original_cwd()}")

    # seed
    seed_everything(cfg.seed)

    # init loggers
    logger1 = TensorBoardLogger('tb_logs')

    # init useful callbacks
    callbacks = [LearningRateLogger(), FitDurationCallback()] if not cfg.eval_only else None

    # init checkpoint callback
    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=1, verbose=True)

    # init early stopping callback
    early_stop_callback = EarlyStopping(patience=1, verbose=True,
                                        mode=cfg.task.early_stop_mode) if cfg.task.early_stop_on is not None else None

    # init profiler
    profiler_output_filename = 'profile.txt'
    if cfg.trainer.gpus is not None and (
            cfg.trainer.distributed_backend is None or cfg.trainer.distributed_backend == 'ddp_spawn'):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1640#issuecomment-663981982
        profiler_output_filename = None
    profiler = SimpleProfiler(profiler_output_filename)

    # auto enable sync batchnorm
    dist_backend = cfg.trainer.distributed_backend
    if (dist_backend is not None and 'ddp' in dist_backend) or (
            dist_backend is None and cfg.trainer.gpus is not None and (
            cfg.trainer.gpus > 1 or cfg.trainer.num_nodes > 1)):
        log.info('Setting trainer.sync_batchnorm=true')
        cfg.trainer.sync_batchnorm = True

    # init model
    model = instantiate(cfg.task.pl_class, cfg=cfg)

    # init trainer
    trainer = Trainer(
        **cfg.trainer,
        logger=[logger1],
        profiler=profiler,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        callbacks=callbacks
    )

    # train / test
    if not cfg.eval_only:
        assert cfg.test_weights is None or cfg.test_weights == 'best'

        trainer.fit(model)
        trainer.test(ckpt_path=cfg.test_weights)
    else:
        test_weights_path = hydra.utils.to_absolute_path(cfg.test_weights)
        assert os.path.exists(test_weights_path), f"{test_weights_path} does not exist"
        model.load_state_dict(torch.load(test_weights_path)['state_dict'])

        trainer.test(model)


if __name__ == "__main__":
    main()
