import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import hydra
from omegaconf import OmegaConf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler

from sharpf.utils.callbacks import FitDurationCallback
from sharpf.utils.collect_env import collect_env_info

from configs import trainer, optimizer, scheduler

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training routine specific for this project
    :param cfg:
    """

    assert cfg.trainer.num_nodes == 1, 'trainer.test() does not work in multi-node regime'
    assert cfg.trainer.distributed_backend != 'dp', 'dp is the tricky and bad choice. It is currently not supported for now'

    log.info(f"Environment info:\n{collect_env_info()}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    seed_everything(cfg.seed)

    model = instantiate(cfg.meta_arch.pl_class, cfg=cfg)

    # init loggers
    logger1 = TensorBoardLogger('tb_logs')

    # init useful callbacks
    callbacks = [LearningRateLogger(), FitDurationCallback()] if not cfg.eval_only else None

    # init checkpoint callback
    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=1, verbose=True, monitor=cfg.meta_arch.monitor)

    # init profiler
    profiler_output_filename = 'profile.txt'
    if cfg.trainer.gpus is not None and (
            cfg.trainer.distributed_backend is None or cfg.trainer.distributed_backend == 'ddp_spawn'):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1640#issuecomment-663981982
        profiler_output_filename = None
    profiler = SimpleProfiler(profiler_output_filename)

    trainer = Trainer(
        **cfg.trainer,
        logger=[logger1],
        profiler=profiler,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks
    )
    if not cfg.eval_only:
        assert cfg.test_weights is None or cfg.test_weights == 'best'
        trainer.fit(model)
        trainer.test(ckpt_path=cfg.test_weights)
    else:
        model.load_state_dict(torch.load(cfg.test_weights)['state_dict'])
        trainer.test(model)


if __name__ == "__main__":
    main()
