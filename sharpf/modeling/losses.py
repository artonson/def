import torch
import torch.nn
import torch.nn.functional as F


def smooth_l1_reg_loss(pred, label):
    pred_distance, label_distance = pred[:, :, 0], label[:, :, 0]
    pred_direction, label_direction = pred[:, :, 1:], label[:, :, 1:]
    return F.smooth_l1_loss(pred_distance, label_distance) + torch.exp(-label_direction) * (
            1 - torch.dot(label_direction, pred_direction))


def dir_loss(pred, label, eps=1e-8):
    """ pred: BxNxC,
        label: BxNxC, """
    return 1 - torch.matmul(pred[:, :, None, :], label[:, :, :, None]).squeeze().mean(dim=1).mean(dim=0)


def dist_dir_loss(pred, label):
    """ pred: BxNxC,
        label: BxNxC, """
    return F.mse_loss(pred[:, :, 0], label[:, :, 0], reduction='mean') + dir_loss(pred[:, :, 1:], label[:, :, 1:])
