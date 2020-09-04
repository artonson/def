from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from hydra.types import TargetConf

cs = ConfigStore.instance()


@dataclass
class AdamConf(TargetConf):
    _target_: str = "torch.optim.Adam"
    betas: tuple = (0.9, 0.999)
    lr: float = 1e-3
    eps: float = 1e-8
    weight_decay: float = 0
    weight_decay_norm: float = 0
    amsgrad: bool = False


@dataclass
class AdamWConf(AdamConf):
    _target_: str = "torch.optim.AdamW"


cs.store(
    group="opt", name="adam", node=AdamConf(),
)
cs.store(
    group="opt", name="adamw", node=AdamWConf(),
)


@dataclass
class AdamaxConf(TargetConf):
    _target_: str = "torch.optim.Adamax"
    betas: tuple = (0.9, 0.999)
    lr: float = 1e-3
    eps: float = 1e-8
    weight_decay: float = 0
    weight_decay_norm: float = 0


cs.store(
    group="opt", name="adamax", node=AdamaxConf(),
)


@dataclass
class ASGDConf(TargetConf):
    _target_: str = "torch.optim.ASGD"
    alpha: float = 0.75
    lr: float = 1e-3
    lambd: float = 1e-4
    t0: float = 1e6
    weight_decay: float = 0
    weight_decay_norm: float = 0
    weight_decay_bias: float = 0


cs.store(
    group="opt", name="asgd", node=ASGDConf(),
)


@dataclass
class LBFGSConf(TargetConf):
    _target_: str = "torch.optim.LBFGS"
    lr: float = 1
    max_iter: int = 20
    max_eval: int = 25
    tolerance_grad: float = 1e-5
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: Optional[str] = None


cs.store(
    group="opt", name="lbfgs", node=LBFGSConf(),
)


@dataclass
class RMSpropConf(TargetConf):
    _target_: str = "torch.optim.RMSprop"
    lr: float = 1e-2
    momentum: float = 0
    alpha: float = 0.99
    eps: float = 1e-8
    centered: bool = True
    weight_decay: float = 0
    weight_decay_norm: float = 0


cs.store(
    group="opt", name="rmsprop", node=RMSpropConf(),
)


@dataclass
class RpropConf(TargetConf):
    _target_: str = "torch.optim.Rprop"
    lr: float = 1e-2
    etas: tuple = (0.5, 1.2)
    step_sizes: tuple = (1e-6, 50)


cs.store(
    group="opt", name="rprop", node=RpropConf(),
)


@dataclass
class SGDConf(TargetConf):
    _target_: str = "torch.optim.SGD"
    lr: float = 1e-2
    momentum: float = 0
    weight_decay: float = 0
    dampening: float = 0
    nesterov: bool = False
    weight_decay_norm: float = 0


cs.store(
    group="opt", name="sgd", node=SGDConf(),
)


@dataclass
class AdamPConf(TargetConf):
    _target_: str = "sharpf.optim.AdamP"
    betas: tuple = (0.9, 0.999)
    lr: float = 1e-3
    eps: float = 1e-8
    weight_decay: float = 0
    delta: float = 0.1
    wd_ratio: float = 0.1
    nesterov: bool = False
    weight_decay_norm: float = 0


cs.store(
    group="opt", name="adamp", node=AdamPConf(),
)
