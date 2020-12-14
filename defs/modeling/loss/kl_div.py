from typing import List

import torch
import torch.nn.functional as F
from torch import erf

__all__ = ['kl_div_loss', 'logits_to_scalar']


def kl_div_loss(input: torch.Tensor, target: torch.Tensor,
                a: float = 0.0, b: float = 1.0, discretization: int = 240, margin: int = 2,
                ignore_mask=None,
                *args, **kwargs):
    """
    Implementation of the KL divergence (histogram) loss with target scalar->distribution preprocessing

    Args:
        input (Tensor): of shape (B, *, discretization + 2 * margin). Logits
        target (Tensor): of shape (B, *)
        a (float): left corner of the interval
        b (float): right corner of the interval
        discretization (int): the amount of bins inside the interval
        margin (int): how many bins to add to each corner (may improve performance)
    Returns:
        Tensor: loss value
    """
    assert margin >= 0 and input.size(-1) == discretization + 2 * margin

    with torch.no_grad():
        density = _calculate_density(target, a, b, discretization, margin)

    input = F.log_softmax(input, dim=-1)

    if ignore_mask is not None:
        size = input.view(-1).size(0)
        return F.kl_div(torch.masked_select(input, ~ignore_mask).view(-1, discretization + 2 * margin),
                        torch.masked_select(density, ~ignore_mask).view(-1, discretization + 2 * margin),
                        reduction='sum', *args, **kwargs) / size
    else:
        return F.kl_div(input, density, *args, **kwargs)


def _normal_cdf(x, mean, sigma):
    return 0.5 + 0.5 * erf((x - mean) / (torch.sqrt(torch.tensor(2.0)) * sigma))


def _calculate_density(mean: torch.Tensor, a: float, b: float, discretization: int, margin: int) -> torch.Tensor:
    """
    Args:
        mean (Tensor): of shape (B, *)
        a (float): left corner of the interval
        b (float): right corner of the interval
        discretization (int): the amount of bins inside the interval
        margin (int): how many bins to add to each corner (may improve performance)
    Returns:
        Tensor: of shape (B, *, discretization + margin * 2)
    """
    bin_limits = _calculate_bin_limits(mean.shape, a, b, discretization, margin,
                                       device=mean.device)  # (B, *, discretization + margin * 2)
    mean = mean.unsqueeze(-1)  # (B, *, 1)
    sigma = 1.0 / discretization
    norm_coefficient = _normal_cdf(b, mean, sigma) - _normal_cdf(a, mean, sigma)  # (B, *, 1)
    density = (_normal_cdf(bin_limits[..., 1:], mean, sigma) - _normal_cdf(bin_limits[..., :-1], mean, sigma))
    density = density / norm_coefficient
    return density


def _calculate_bin_limits(shape: List, a: float, b: float, discretization: int, margin: int, device: torch.device) \
        -> torch.Tensor:
    """
    Args:
        shape (list): desired shape
        a (float): left corner of the interval
        b (float): right corner of the interval
        discretization (int): the amount of bins inside the interval
        margin (int): how many bins to add to each corner (may improve performance)
    Returns:
        bin_limits (Tensor): of shape (*shape, discretization + margin * 2 + 1)
    """

    a = a - margin * (1 / discretization)
    b = b + margin * (1 / discretization)

    num_bins = discretization + margin * 2

    bin_limits = torch.linspace(a, b, num_bins + 1, device=device) \
        .view(*[1 for _ in range(len(shape))], num_bins + 1) \
        .expand(*shape, num_bins + 1)

    return bin_limits


def logits_to_scalar(input: torch.Tensor, a: float = 0.0, b: float = 1.0, discretization: int = 240, margin: int = 2):
    """
    Convert logits to scalar

    Args:
        input (Tensor): of shape (B, *, discretization + 2 * margin). Logits
        a (float): left corner of the interval
        b (float): right corner of the interval
        discretization (int): the amount of bins inside the interval
        margin (int): how many bins to add to each corner (may improve performance)
    Returns:
        Tensor: scalar values of size (B, *, 1)
    """
    assert margin >= 0 and input.size(-1) == discretization + 2 * margin

    with torch.no_grad():
        bin_limits = _calculate_bin_limits(input.shape[:-1], a, b, discretization, margin, input.device)
        bin_centers = bin_limits[..., :-1] + (1.0 / discretization / 2.0)  # (B, *, discretization + 2 * margin)

    return (input.softmax(dim=-1) * bin_centers).sum(dim=-1, keepdim=True)
