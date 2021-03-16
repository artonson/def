from abc import ABC, abstractmethod
import itertools
import os
from typing import List, Tuple, Mapping

import numpy as np

import sharpf.fusion.images.pairwise as pairwise
from sharpf.utils.camera_utils.view import CameraView
from sharpf.utils.py_utils.config import Configurable
from sharpf.utils.py_utils.parallel import multiproc_parallel


class MultiViewInterpolatorBase(Configurable):
    @abstractmethod
    def __call__(
            self,
            views: List[CameraView],
    ) -> Tuple[np.array, np.array, np.array]:

        """Given a set of (image I, prediction D) pairs and corresponding
        camera extrinsic and intrinsic parameters, interpolates predictions
        between views.
        """
        raise NotImplemented()


class GroundTruthInterpolator(MultiViewInterpolatorBase):
    """Combine ground-truth distances and directions."""
    def __call__(
            self,
            views: List[CameraView],
    ) -> Tuple[np.array, np.array, np.array]:

        list_predictions, list_indexes_in_whole, list_points = [], [], []
        n_points = 0

        for pixels_view in views:
            points_view = pixels_view.to_points()
            points, distances = points_view.depth, points_view.signal
            list_indexes_in_whole.append(
                np.arange(n_points, n_points + len(points)))
            list_points.append(points)
            list_predictions.append(distances)
            n_points += len(points)

        return list_predictions, list_indexes_in_whole, list_points

    @classmethod
    def from_config(cls, config: Mapping):
        return cls()


class MultiViewPredictionsInterpolator(MultiViewInterpolatorBase):
    def __init__(
            self,
            n_jobs: int = 1,
            distance_interpolation_threshold: float = 1.0,
            nn_set_size: int = 8,
            z_distance_threshold: int = 2.0,
            verbose: bool = False,
    ):
        super().__init__()
        self.n_jobs = n_jobs
        self.distance_interp_thr = distance_interpolation_threshold
        self.nn_set_size = nn_set_size
        self.z_distance_threshold = z_distance_threshold
        self.verbose = verbose

    def __call__(
            self,
            views: List[CameraView],
    ) -> Tuple[np.array, np.array, np.array]:

        list_predictions, list_indexes_in_whole, list_points = [], [], []

        n_images = len(views)
        point_indexes = np.cumsum([
            np.count_nonzero(view.depth) for view in views])

        interp_params = {
            'distance_interp_thr': self.distance_interp_thr,
            'nn_set_size': self.nn_set_size,
            'z_distance_threshold': self.z_distance_threshold,
            'verbose': self.verbose,
        }
        data_iterable = (
            (i, j, views[i], views[j], point_indexes, interp_params)
            for i, j in itertools.product(range(n_images), range(n_images)))

        os.environ['OMP_NUM_THREADS'] = str(self.n_jobs)
        for predictions_interp, indexes_interp, points_interp in multiproc_parallel(
                self.do_interpolate,
                data_iterable):
            list_predictions.append(predictions_interp)
            list_indexes_in_whole.append(indexes_interp)
            list_points.append(points_interp)

        return list_predictions, list_indexes_in_whole, list_points

    @abstractmethod
    def do_interpolate(
            self,
            source_view_idx: int,
            target_view_idx: int,
            source_view: CameraView,
            target_view: CameraView,
            point_indexes: np.ndarray,
            interp_params: Mapping,
    ):
        pass

    @classmethod
    def from_config(cls, config: Mapping):
        return cls(
            n_jobs=config['n_jobs'],
            distance_interpolation_threshold=config['distance_interpolation_threshold'],
            z_distance_threshold=config['z_distance_threshold'],
            nn_set_size=config['nn_set_size'],
            verbose=config['verbose'],
        )


class MultiView3D(MultiViewPredictionsInterpolator):
    """Interpolates predictions between views in the 3D space directly."""
    pass


class MultiViewImage(MultiViewPredictionsInterpolator):
    """Interpolates predictions between views in the image space
    (i.e. 3d points are not discretized into pixels)."""

    def do_interpolate(
            self,
            source_view_idx: int,
            target_view_idx: int,
            source_view: CameraView,
            target_view: CameraView,
            point_indexes: np.ndarray,
            interp_params: Mapping,
    ):
        if 0 == target_view_idx:
            start_idx, end_idx = 0, point_indexes[target_view_idx]
        else:
            start_idx, end_idx = point_indexes[target_view_idx - 1], \
                                 point_indexes[target_view_idx]
        target_indexes = np.arange(start_idx, end_idx)

        if source_view_idx == target_view_idx:
            source_view = source_view.to_points()
            return source_view.signal, target_indexes, source_view.depth

        else:
            interpolated_view, valid_mask = pairwise.interpolate_views_as_images(
                source_view,
                target_view,
                **interp_params,
            )
            return interpolated_view.signal[valid_mask], target_indexes[valid_mask], interpolated_view.depth[valid_mask]


class MultiViewPixel(MultiViewPredictionsInterpolator):
    """Interpolates predictions between views in the 3D space directly."""
    pass


MVS_INTERPOLATORS = {
    'image': MultiViewImage,
}
