from omegaconf import OmegaConf

OmegaConf.register_resolver("logits", lambda discratization, margin: int(discratization) + 2 * int(margin))
