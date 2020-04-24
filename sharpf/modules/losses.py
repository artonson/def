#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""

import torch
import torch.nn
import torch.nn.functional as F


def bce_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    return loss(pred, label)


def smooth_l1_loss(pred, label):
    loss = torch.nn.SmoothL1Loss(reduction='mean')
    return loss(pred, label)


def smooth_l1_reg_loss(pred, label):
    pred_distance, label_distance = pred[:,:,0], label[:,:,0]
    pred_direction, label_direction = pred[:,:,1:], label[:,:,1:]
    return smooth_l1_loss(pred_distance, label_distance) + torch.exp(-label_direction) * (1 - torch.dot(label_direction, pred_direction))


def dist_loss(pred, label):
    """ pred: BxNxC,
        label: BxNxC, """
    loss = torch.nn.MSELoss(reduction='mean')
    return loss(pred, label)


def dir_loss(pred, label, eps=1e-8):
    """ pred: BxNxC,
        label: BxNxC, """
    return 1 - torch.matmul(pred[:,:,None,:],label[:,:,:,None]).squeeze().mean(dim=1).mean(dim=0)


def dist_dir_loss(pred, label):
    """ pred: BxNxC,
        label: BxNxC, """
    return dist_loss(pred[:,:,0], label[:,:,0]) + dir_loss(pred[:,:,1:], label[:,:,1:])

#def get_loss_tensor(pred, label):
#    """ pred: BxNxC,
#        label: BxN, """
#    classify_loss_tensor = tf.losses.sigmoid_cross_entropy(multi_class_labels=label, logits=pred, reduction = tf.losses.Reduction.NONE)
#    # classify_loss = tf.reduce_mean(loss, axis = 1)
#    # classify_loss = tf.reduce_mean(classify_loss, axis = 0)
#    #tf.summary.scalar('classify loss', classify_loss)
#    tf.add_to_collection('losse_tensors', classify_loss_tensor)
#    return classify_loss_tensor


#def cal_loss(pred, gold, smoothing=True):
#    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
#
#    gold = gold.contiguous().view(-1)
#
#    if smoothing:
#        eps = 0.2
#        n_class = pred.size(1)
#
#        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
#        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
#        log_prb = F.log_softmax(pred, dim=1)
#
#        loss = -(one_hot * log_prb).sum(dim=1).mean()
#    else:
#        loss = F.cross_entropy(pred, gold, reduction='mean')
#
#    return loss


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


LOSSES_BY_NAME = {
    'has_sharp': bce_loss,
    'segment_sharp': bce_loss,
    'regress_sharpdf': smooth_l1_loss,
    'regress_sharpdirf': smooth_l1_reg_loss
}


# an ugly hack to extract available losses from torch
TORCH_NN_LOSSES = list(filter(lambda f: f.endswith('Loss'), dir(torch.nn)))
LOSSES = list(LOSSES_BY_NAME.keys()) + TORCH_NN_LOSSES


def get_loss_function(metric_name, reduction='none'):
    """The metric should be either importable from torch.nn, or in LOSSES_BY_NAME."""

    if metric_name in LOSSES_BY_NAME:
        loss_class = LOSSES_BY_NAME[metric_name]

    elif metric_name in TORCH_NN_LOSSES:
        loss_class = getattr(torch.nn, metric_name)

    else:
        raise ValueError('Metric {} cannot be instantiated, skipping'.format(metric_name))

    loss_function = loss_class(reduction=reduction)
    return loss_function

