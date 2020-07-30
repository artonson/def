import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.utilities import rank_zero_only

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
    assert cfg.trainer.distributed_backend != 'dp', 'dp is the tricky and bad choice. It is currently not supported for now'

    rank_zero_only(log.info(f"Environment info:\n{collect_env_info()}"))
    rank_zero_only(log.info(f"Config:\n{cfg.pretty()}"))
    seed_everything(cfg.seed)

    model = instantiate(cfg.meta_arch, cfg=cfg)

    # init loggers
    logger1 = TensorBoardLogger('tb_logs')
    # logger2 = TestTubeLogger('tt_logs')

    # init useful callbacks
    callbacks = [LearningRateLogger(), FitDurationCallback()]

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
        callbacks=callbacks,
    )
    trainer.fit(model)

    # does not work in ddp mode https://github.com/PyTorchLightning/pytorch-lightning/issues/2683
    # Very important: if model is provided, ckpth_path is not used
    # trainer.test()


if __name__ == "__main__":
    main()
