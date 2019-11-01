#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    classify_loss = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
    return classify_loss(pred, label)

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


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
