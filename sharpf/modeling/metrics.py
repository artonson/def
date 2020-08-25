import torch


def balanced_accuracy(tp: torch.Tensor, fp: torch.Tensor, tn: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """
    Calculate balanced accuracy for one class based on provided statistics

    Args:
        tp (Tensor): of shape (B, 1). True positive.
        fo (Tensor): of shape (B, 1). False positive stats of a shape.
        tn (Tensor): of shape (B, 1). True negative stats of a shape.
        fn (Tensor): of shape (B, 1). False negative stats of a shape.

    Returns:
        torch.Tensor: balanced accuracy value
    """
    tpr = tp / (tp + fn)  # (B, 1)
    tnr = tn / (tn + fp)  # (B, 1)
    tpr = torch.where(torch.isnan(tpr), tnr, tpr)  # (B, 1)
    tnr = torch.where(torch.isnan(tnr), tpr, tnr)  # (B, 1)
    return 0.5 * torch.mean(tpr + tnr)
