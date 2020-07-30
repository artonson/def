from omegaconf import DictConfig

from sharpf.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for models that support building from DictConfig
"""


def build_model(cfg: DictConfig):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    model = MODEL_REGISTRY.get(cfg.model_name)(cfg.params)
    return model
