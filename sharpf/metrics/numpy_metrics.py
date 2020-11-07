from collections import Callable
from copy import deepcopy
from typing import Tuple, MutableMapping

import numpy as np


class Metric(Callable):
    def __call__(self, true_instance, pred_label):
        pass


class PointwiseSquaredError(Metric):
    """Points with error above threshold are considered bad."""

    def __call__(self, true_instance, pred_label):
        true_distances = true_instance['distances']
        pred_distances = pred_label['distances']
        diff_sq = np.square(true_distances - pred_distances)
        return diff_sq


class RMSE(Metric):
    def __init__(self, normalize=False):
        self._normalize = normalize

    def __call__(self, true_instance, pred_label):
        diff_sq = PointwiseSquaredError()(true_instance, pred_label)
        rmse = np.sqrt(
            np.mean(diff_sq))

        if self._normalize:
            rmse /= len(diff_sq)

        return rmse

    def __str__(self):
        return 'RMSE'


class BadPoints(Metric):
    """Points with error above threshold are considered bad."""

    def __init__(self, threshold, normalize=False):
        self._threshold_sq = threshold ** 2
        self._normalize = normalize

    def __str__(self):
        return 'Bad {0:3.3g}'.format(np.sqrt(self._threshold_sq))

    def __call__(self, true_instance, pred_label):
        diff_sq = PointwiseSquaredError()(true_instance, pred_label)
        bad_points = np.sum(diff_sq > self._threshold_sq)

        if self._normalize:
            bad_points /= len(diff_sq)

        return bad_points


class RMSEQuantile(Metric):
    """Determine what is error level for most points."""

    def __init__(self, proba, normalize=False):
        self._proba = proba
        self._normalize = normalize

    def __str__(self):
        return 'Q {0:3.3g}'.format(self._proba)

    def __call__(self, true_instance, pred_label):
        diff_sq = PointwiseSquaredError()(true_instance, pred_label)
        q = np.quantile(diff_sq, self._proba)
        return q


class PointwiseMaskSelector(Callable):
    def __call__(self, true_instance: MutableMapping, pred_label: MutableMapping) \
            -> Tuple[MutableMapping, MutableMapping]:
        pass


class DistanceLessThan(PointwiseMaskSelector):
    def __init__(self, threshold):
        self._threshold = threshold

    def __call__(self, true_instance, pred_label):
        masked_true_instance = deepcopy(true_instance)
        true_distances = masked_true_instance['distances']
        selection_mask = true_distances < self._threshold
        masked_true_instance['distances'] = true_distances[selection_mask]

        masked_pred_label = deepcopy(pred_label)
        pred_distances = masked_pred_label['distances']
        masked_pred_label['distances'] = pred_distances[selection_mask]
        return masked_true_instance, masked_pred_label


class MaskedMetric(Metric):
    def __init__(self, mask_threshold, metric):
        self._metric = metric
        self._masking = DistanceLessThan(mask_threshold)

    def __str__(self):
        return 'M{}'.format(str(self._metric))

    def __call__(self, true_instance, pred_label):
        true_instance, pred_label = self._masking(true_instance, pred_label)
        return self._metric(true_instance, pred_label)


class RescaledMetric(Metric):
    def __init__(self, scale_factor, metric):
        self._metric = metric
        self._scale_factor = scale_factor

    def __str__(self):
        return '{}'.format(str(self._metric))

    def __call__(self, true_instance, pred_label):
        value = self._metric(true_instance, pred_label)
        return value * self._scale_factor

