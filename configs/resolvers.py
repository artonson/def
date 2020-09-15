from omegaconf import OmegaConf

OmegaConf.register_resolver("logits", lambda discratization, margin: int(discratization) + 2 * int(margin))
OmegaConf.register_resolver("mul", lambda x, y: int(x) * int(y))
OmegaConf.register_resolver("add", lambda x, y: int(x) + int(y))
OmegaConf.register_resolver("fmul", lambda x, y: float(x) * float(y))
