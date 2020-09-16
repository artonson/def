from typing import List, Dict, Any, Set

import torch


def get_params_for_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, weight_decay_norm: float):
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            _weight_decay = weight_decay
            if isinstance(module, norm_module_types):
                _weight_decay = weight_decay_norm
            params += [{"params": [value], "lr": lr, "weight_decay": _weight_decay}]
    return params
