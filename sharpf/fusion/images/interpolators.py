from abc import ABC, abstractmethod
import itertools
from typing import List, Tuple, Mapping

import numpy as np

import sharpf.fusion.images.pairwise as pairwise
from sharpf.utils.camera_utils.view import CameraView, view_to_points
from sharpf.utils.py_utils.parallel import multiproc_parallel


class MultiViewInterpolatorBase(ABC, Configurable):
    @abstractmethod
    def __call__(
            self,
            views: List[CameraView],
    ) -> Tuple[np.array, np.array, np.array]:

        """Given a set of (image I, prediction D) pairs and corresponding
        camera extrinsic and intrinsic parameters, interpolates predictions
        between views. As a result, returns

        :param list_images: list of depth images
        :param list_images_preds: list of prediction images
                                  (must be the same length as list_images)
        :param list_extrinsics:
        :param list_intrinsics:
        :param camera_type: string identifying camera specification to use
                            (currently: either 'orthogonal' or 'rangevision')
                            TODO: refactor camera classes to use a more generalised code
        :param camera_parameters: list of camera parameters, interpreted differently depending
                                  on the `camera_type` argument:
                                  <ADD HERE>

        :return:
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
            points, signal = view_to_points(view)
            list_indexes_in_whole.append(
                np.arange(n_points, n_points + len(points)))
            list_points.append(points)
            list_predictions.append(signal)
            n_points += len(points)

        return list_predictions, list_indexes_in_whole, list_points


class MultiViewPredictionsInterpolator(MultiViewInterpolatorBase):
    def __init__(
            self,
            n_jobs: int = 1,
            verbose: bool = False,
            distance_interpolation_threshold: float = HIGH_RES * 6.):
        super().__init__(self)

    def __call__(
            self,
            views: List[CameraView],
    ) -> Tuple[np.array, np.array, np.array]:
        #
        #
        # if camera_type == 'orthogonal':
        #     get_view = get_orthogonal_view
        # elif camera_type == 'rangevision':
        #     get_view = get_perspective_view
        # else:
        #     raise ValueError('Unknown camera type: {}'.format(camera_type))
        #
        # get_view = partial(
        #     get_view,
        #     images=images,
        #     predictions=predictions,
        #     camera_parameters=camera_parameters)
        #
        list_predictions, list_indexes_in_whole, list_points = [], [], []

        n_images = len(views)
        n_cum_points_per_image = np.cumsum([
            np.count_nonzero(view.image) for view in views])

        data_iterable = (
            (self, i, j, views[i], views[j], n_cum_points_per_image)
            for i, j in itertools.product(range(n_images), range(n_images)))

        for predictions, indexes_in_whole, points in multiproc_parallel(
                self.do_interpolate,
                data_iterable):
            list_points.append(points)
            list_predictions.append(predictions)
            list_indexes_in_whole.append(indexes_in_whole)

        return list_predictions, list_indexes_in_whole, list_points

    @abstractmethod
    def do_interpolate(
            self,
            source_view_idx,
            target_view_idx,
            source_view,
            target_view,
    ):
        pass


class MultiView3D(MultiViewPredictionsInterpolator):
    """Interpolates predictions between views in the 3D space directly."""

    def __init__(self, n_jobs):
        super().__init__(self)

    def do_interpolate(self):
        return pairwise.image_pair_interpolate_image,


class MultiViewImage(MultiViewPredictionsInterpolator):
    """Interpolates predictions between views in the image space
    (i.e. 3d points are not discretized into pixels)."""

    def __init__(self, n_jobs):
        super().__init__()



class MultiViewPixel(MultiViewPredictionsInterpolator):
    """Interpolates predictions between views in the 3D space directly."""

    def __init__(self, n_jobs):
        super().__init__()

    def __call__(
            self,
            images: List[np.array],
            predictions: List[np.array],
            extrinsics: List[np.array],
            intrinsics: List[Mapping],
            camera_type: str = 'orthogonal',
            n_jobs: int = 1,
            verbose: bool = False,
            distance_interpolation_threshold: float = HIGH_RES * 6.
    ) -> Tuple[np.array, np.array, np.array]:
        pass

