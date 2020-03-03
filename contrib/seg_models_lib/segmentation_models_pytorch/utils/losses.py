import torch
import torch.nn as nn
from . import functions as F


class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - F.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
#    __name__ = 'dice_loss'
#
#    def __init__(self, mean=None):
#        super(DiceLoss, self).__init__()
#        self.mean = mean
#
#    def __name__(self):
#        return 'dice_loss'
#
#    def forward(self, logits, target):
#        if self.mean is not None:
#            w_1 = 1/self.mean
#            w_0 = 1/(1-self.mean)
#        else:
#            w_1 = 1
#            w_0 = 1

        # skip the batch and class axis for calculating Dice score
#        epsilon = 1.
        #tmp = torch.ones(mask.shape).to('cuda')
        #tmp -= mask
        #mask = mask*w_1 + tmp
#        y_pred = torch.nn.Sigmoid()(logits)#*mask)#.permute(0, 2, 3, 1)
#        y_true = target#.permute(0, 2, 3, 1)
#        numerator = 2. * torch.sum(y_pred*y_true, dim=(2, 3))
#        denominator = torch.sum(y_pred + y_true, dim=(2,3))
#
#        return torch.mean(numerator / (denominator + epsilon)) # average over classes and batch
    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - F.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class BCEJaccardLoss(JaccardLoss):
    __name__ = 'bce_jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid', mean=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice = DiceLoss()

    def forward(self, y_pr, y_gt, mask=None):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce
