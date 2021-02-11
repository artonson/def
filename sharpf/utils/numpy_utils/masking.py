import numpy as np


def compress_mask(*in_masks):
    """Given a sequence of boolean masks,
    where each mask indexes into the previous mask,
    return a single mask indexing into the original array"""
    first_mask = in_masks[0]
    out_mask = np.zeros_like(first_mask, dtype=bool)
    true_indexes = np.where(first_mask)[0]
    for mask in in_masks[1:]:
        true_indexes = true_indexes[np.where(mask)[0]]
    out_mask[true_indexes] = True
    return out_mask
