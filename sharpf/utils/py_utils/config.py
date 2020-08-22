from abc import ABC, abstractmethod
from typing import Mapping


def load_func_from_config(func_dict, config):
    return func_dict[config['type']].from_config(config)


class Configurable(ABC):
    @abstractmethod
    def from_config(cls, config: Mapping) -> Mapping:
        pass
