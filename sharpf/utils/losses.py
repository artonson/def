import torch
import numpy as np

def balanced_accuracy(y_pred, y_true):
    print('balanced accuracy print', y_pred.shape, y_true.shape)
    tpr = np.sum((y_pred[y_pred == 1] == y_true[y_true == 1]).float(), axis=1)
    tnr = np.sum((y_pred[y_pred == 0] == y_true[y_true == 1]).float(), axis=1)
    tpr /= (tpr + np.sum((y_pred[y_pred == 0] == y_true[y_true == 1]).float(), axis=1))
    tnr /= (tnr + np.sum((y_pred[y_pred == 1] == y_true[y_true == 0]).float().sum(), axis=1))
    acc = (tpr + tnr) / 2
    print(acc.shape)
    return acc