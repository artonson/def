from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from hydra.types import TargetConf

cs = ConfigStore.instance()


@dataclass
class CosineConf(TargetConf):
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    T_max: int = 100
    eta_min: float = 0
    last_epoch: int = -1


cs.store(
    group="scheduler",
    name="cosine",
    node=CosineConf(),
)


@dataclass
class CosineWarmConf(TargetConf):
    _target_: str = "torch.optim.Adamax"
    T_0: int = 10
    T_mult: int = 1
    eta_min: float = 0
    last_epoch: int = -1


cs.store(
    group="scheduler",
    name="cosinewarm",
    node=CosineWarmConf(),
)


@dataclass
class CyclicConf(TargetConf):
    _target_: str = "torch.optim.Adamax"
    base_lr: Any = 1e-3
    max_lr: Any = 1e-2
    step_size_up: int = 2000
    step_size_down: int = 2000
    mode: str = "triangular"
    gamma: float = 1
    scale_fn: Optional[Any] = None
    scal_mode: str = "cycle"
    cycle_momentum: bool = True
    base_momentum: Any = 0.8
    max_momentum: Any = 0.9
    last_epoch: int = -1


cs.store(
    group="scheduler", name="cyclic", node=CyclicConf(),
)


@dataclass
class ExponentialConf(TargetConf):
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    gamma: float = 1
    last_epoch: int = -1


cs.store(
    group="scheduler",
    name="exponential",
    node=ExponentialConf(),
)


@dataclass
class Exponential09Conf(ExponentialConf):
    gamma: float = 0.9


cs.store(
    group="scheduler",
    name="exponential09",
    node=Exponential09Conf(),
)


@dataclass
class RedPlatConf(TargetConf):
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    verbose: bool = False
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: Any = 0
    eps: float = 1e-8


cs.store(
    group="scheduler",
    name="redplat",
    node=RedPlatConf(),
)


@dataclass
class MultiStepConf(TargetConf):
    _target_: str = "torch.optim.lr_scheduler.MultiStepLR"
    milestones: List = field(default_factory=lambda: [10, 20, 30, 40])
    gamma: float = 0.1
    last_epoch: int = -1


cs.store(
    group="scheduler",
    name="multistep",
    node=MultiStepConf(),
)


@dataclass
class OneCycleConf(TargetConf):
    _target_: str = "torch.optim.lr_scheduler.OneCycleLR"
    max_lr: Any = 1e-2
    total_steps: int = 2000
    epochs: int = 200
    steps_per_epoch: int = 100
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: Any = 0.8
    max_momentum: Any = 0.9
    div_factor: float = 25
    final_div_factor: float = 1e4
    last_epoch: int = -1


cs.store(
    group="scheduler",
    name="onecycle",
    node=OneCycleConf(),
)


@dataclass
class StepConf(TargetConf):
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = 20
    gamma: float = 0.1
    last_epoch: int = -1


cs.store(
    group="scheduler", name="step", node=StepConf(),
)
