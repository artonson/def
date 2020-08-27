from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf


def call(config: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Implementation of recursive instantiation
    unless there is a better official way https://github.com/facebookresearch/hydra/issues/566#issuecomment-677713399

    :param config: An object describing what to call and what params to use. needs to have a _target_ field.
    :param args: optional positional parameters pass-through
    :param kwargs: optional named parameters pass-through
    :return: the return value from the specified class or method
    """
    for key, child_conf in config.items():
        if isinstance(child_conf, ListConfig) and isinstance(child_conf[0], DictConfig) and '_target_' in child_conf[0]:
            obj_list = []
            for conf_item in child_conf:
                obj_list.append(call(conf_item))
            kwargs[key] = obj_list
            primitive_conf = OmegaConf.to_container(config, resolve=True)
            del primitive_conf[key]
            config = DictConfig(primitive_conf)
            return call(config, *args, **kwargs)
        elif isinstance(child_conf, ListConfig) and not isinstance(child_conf[0], DictConfig) and not isinstance(
                child_conf[0], ListConfig):
            kwargs[key] = list(child_conf)
            primitive_conf = OmegaConf.to_container(config, resolve=True)
            del primitive_conf[key]
            config = DictConfig(primitive_conf)
            return call(config, *args, **kwargs)
        elif isinstance(child_conf, DictConfig) and '_target_' in child_conf and key not in kwargs:
            kwargs[key] = call(child_conf)
            primitive_conf = OmegaConf.to_container(config, resolve=True)
            del primitive_conf[key]
            config = DictConfig(primitive_conf)
            return call(config, *args, **kwargs)

    return hydra.utils.call(config, *args, **kwargs)


# Alias for call
instantiate = call
