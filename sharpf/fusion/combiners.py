from abc import ABC
from collections import defaultdict
from typing import Callable, List, Mapping, Tuple

import numpy as np
from scipy.stats.mstats import mquantiles
from tqdm import tqdm

from sharpf.fusion.smoothers import PredictionsSmoother


class AveragingNoDataException(ValueError):
    pass


class TruncatedMean(object):
    def __init__(self, probability, func=np.mean):
        self._probability = probability
        self._func = func

    def __call__(self, values):
        quantiles = mquantiles(values,
                               [(1.0 - self._probability) / 2.0, (1.0 + self._probability) / 2.0],
                               alphap=0.5, betap=0.5)
        values = list(filter(lambda value: quantiles[0] <= value <= quantiles[1], values))
        if not values:
            raise AveragingNoDataException('No values after trimming')
        result = self._func(values)
        return result


class PredictionsCombiner(ABC):
    def __call__(
            self,
            n_points: int,
            list_predictions: List[np.array],
            list_indexes_in_whole: List[np.array],
            list_points: List[np.array],
            max_distance: float = 1.0
    ) -> Tuple[np.array, np.array, Mapping]:

        raise NotImplemented()


class PointwisePredictionsCombiner(PredictionsCombiner):
    """Combines predictions where each point is predicted multiple times
    by taking a simple function of the point's predictions.

    Returns:
        fused_points_pred: consolidated predictions
        fused_distances_pred: consolidated predictions
        predictions_variants: enumerates predictions per each point
        """

    def __init__(self, tag: str, func: Callable = np.median):
        self._func = func
        self.tag = tag

    def __call__(
            self,
            n_points: int,
            list_predictions: List[np.array],
            list_indexes_in_whole: List[np.array],
            list_points: List[np.array],
            max_distance: float = 1.0
    ) -> Tuple[np.array, np.array, Mapping]:

        fused_points_pred = np.zeros((n_points, 3))
        fused_distances_pred = np.ones(n_points) * max_distance

        # step 1: gather predictions
        predictions_variants = defaultdict(list)
        iterable = zip(list_predictions, list_indexes_in_whole, list_points)
        for distances, indexes_gt, points_gt in tqdm(iterable):
            fused_points_pred[indexes_gt] = points_gt
            for i, idx in enumerate(indexes_gt):
                predictions_variants[idx].append(distances[i])

        # step 2: consolidate predictions
        for idx, values in tqdm(predictions_variants.items()):
            fused_distances_pred[idx] = self._func(values)

        return fused_points_pred, fused_distances_pred, predictions_variants


class AvgProbaPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self, thr=0.5):
        def maj_voting(values):
            values = np.array(values)
            if np.mean(values) > thr:
                return 1.0
            else:
                return 0.0

        super().__init__(func=maj_voting, tag='vote_{0:3.2f}'.format(thr))


class MedianPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self): super().__init__(func=np.median, tag='median')


class AvgPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self): super().__init__(func=np.mean, tag='mean')


class MinPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self): super().__init__(func=np.min, tag='min')


class TruncatedAvgPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self, func=np.min):
        super().__init__(func=TruncatedMean(0.6, func=func), tag='adv60')


class MinsAvgPredictionsCombiner(PointwisePredictionsCombiner):
    def __init__(self, signal_thr=0.9):
        def mean_of_mins(values):
            values = np.array(values)
            if len(values[values < signal_thr]) > 0:
                return np.mean(values[values < signal_thr])
            else:
                return np.mean(values)

        super().__init__(func=mean_of_mins, tag='mom')


class CenterCropPredictionsCombiner(PointwisePredictionsCombiner):
    """Makes a central crop, then performs the same as above."""

    def __init__(self, func: Callable = np.median, brd_thr=90, tag='crop'):
        super().__init__(func=func, tag=tag)
        self._brd_thr = brd_thr

    def __call__(
            self,
            n_points: int,
            list_predictions: List[np.array],
            list_indexes_in_whole: List[np.array],
            list_points: List[np.array],
            max_distance: float = 1.0
    ) -> Tuple[np.array, np.array, Mapping]:

        fused_points_pred = np.zeros((n_points, 3))
        fused_distances_pred = np.ones(n_points) * max_distance

        # step 1: gather predictions
        predictions_variants = defaultdict(list)
        iterable = zip(list_predictions, list_indexes_in_whole, list_points)
        for distances, indexes_gt, points_gt in tqdm(iterable):
            fused_points_pred[indexes_gt] = points_gt
            # here comes difference from the previous variant
            points_radii = np.linalg.norm(points_gt - points_gt.mean(axis=0), axis=1)
            center_indexes = np.where(points_radii < np.percentile(points_radii, self._brd_thr))[0]
            for i, idx in enumerate(center_indexes):
                predictions_variants[indexes_gt[idx]].append(distances[center_indexes[i]])

        # step 2: consolidate predictions
        for idx, values in predictions_variants.items():
            fused_distances_pred[idx] = self._func(values)

        return fused_points_pred, fused_distances_pred, predictions_variants


class SmoothingCombiner(PredictionsCombiner):
    """Predict on a pointwise basis, then smoothen."""

    def __init__(self, combiner: PointwisePredictionsCombiner, smoother: PredictionsSmoother):
        self._combiner = combiner
        self._smoother = smoother
        self.tag = combiner.tag + '__' + smoother.tag

    def __call__(
            self,
            n_points: int,
            list_predictions: List[np.array],
            list_indexes_in_whole: List[np.array],
            list_points: List[np.array],
            max_distance: float = 1.0
    ) -> Tuple[np.array, np.array, Mapping]:

        fused_points, fused_predictions, predictions_variants = self._combiner(
            n_points,
            list_predictions,
            list_indexes_in_whole,
            list_points,
            max_distance=max_distance)

        fused_predictions = self._smoother(
            fused_predictions,
            fused_points,
            predictions_variants)

        return fused_points, fused_predictions, predictions_variants


class GroundTruthCombiner(PredictionsCombiner):
    """Combine ground-truth distances and directions:
    this is similar to MinPredictionsCombiner, but also computes directions."""
    def __call__(
            self,
            n_points: int,
            list_predictions: List[np.array],
            list_indexes_in_whole: List[np.array],
            list_points: List[np.array],
            max_distance: float = 1.0
    ) -> Tuple[np.array, np.array, Mapping]:

        fused_points = np.zeros((n_points, 3))
        fused_distances = np.ones(n_points) * max_distance
        # fused_directions = np.ones((n_points, 3)) * np.inf

        iterable = zip(list_predictions, list_indexes_in_whole, list_points)
        for distances, indexes, points in tqdm(iterable):
            fused_points[indexes] = points
            assign_mask = fused_distances[indexes] > distances
            fused_distances[indexes[assign_mask]] = np.minimum(distances[assign_mask], max_distance)
            # fused_directions[indexes[assign_mask]] = directions[assign_mask]

        return fused_points, fused_distances, {}
