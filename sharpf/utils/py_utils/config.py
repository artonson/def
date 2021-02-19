from abc import abstractmethod, ABC
from typing import Mapping


def load_func_from_config(func_dict, config):
    print(func_dict, config)
    assert 'type' in config, 'No "type" field in config: {}'.format(str(config))
    func_type = config['type']

    assert func_type in func_dict, 'Unknown requested type: {} (available: {})'.format(
        func_type, ', '.join(func_dict.keys()))

    func_class = func_dict[func_type]
    assert issubclass(func_class, Configurable) or callable(getattr(func_class, 'from_config', None)), \
        'Unable to construct an instance of type {}: from_config missing!'.format(func_class.__name__)
    return func_class.from_config(config)


class Configurable(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping):
        pass
