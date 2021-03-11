from abc import ABC, abstractmethod
from typing import Mapping, Tuple

import numpy as np

import sharpf.utils.camera_utils.projection as projection
import sharpf.utils.camera_utils.pixelization as pixelization


class CameraView:
    """Data container representing a view of a scene/object:
    depth/prediction pixel images + camera parameters."""

    def __init__(
            self,
            depth: np.ndarray = None,
            signal: np.ndarray = None,
            extrinsics: np.ndarray = None,
            intrinsics: Mapping[str, np.ndarray] = None,
            params: Mapping = None,
            state: 'CameraViewStateBase' = None,
    ):
        self.depth = depth
        self.signal = signal
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.params = params
        self.state = state
        if None is self.state:
            self.state = PixelViewState(self)

    def to_points(self, inplace=False) -> 'CameraView':
        """Given a view represented as depth/prediction images +
        camera parameters, returns points."""
        view = self.state.to_points(inplace=inplace)
        return view

    def to_image(self, inplace=False) -> 'CameraView':
        view = self.state.to_image(inplace=inplace)
        return view

    def to_pixels(self, inplace=False) -> 'CameraView':
        view = self.state.to_pixels(inplace=inplace)
        return view

    def reproject_to(self, other: 'CameraView') -> 'CameraView':
        view = self.state.reproject_to(other)
        return view

    def copy(self) -> 'CameraView':
        from copy import deepcopy
        return CameraView(
            deepcopy(self.depth),
            deepcopy(self.signal),
            deepcopy(self.extrinsics),
            deepcopy(self.intrinsics),
            deepcopy(self.params))

    @property
    def as_dict(self):
        return { }


def _maybe_inplace(other_view: CameraView, inplace=False) -> CameraView:
    """Create a copy of other_view if inplace=False,
    otherwise return other_view."""
    view = other_view
    if not inplace:
        view = view.copy()
    return view


class CameraViewStateBase(ABC):
    def __init__(self, view: CameraView):
        self.view = view

    def to_points(self, inplace=False) -> CameraView:
        """Convert the representation of the container view
        to points (i.e., after calling this function, we guarantee
        that view.depth is [n, 3] array of point coordinates.

        Subclasses should override this method to provide the
        non-trivial implementation.

        :param inplace:
            if True, return the same view
            if False, create a deep copy of the view
        :return: modified view
        """
        return _maybe_inplace(self.view)

    def to_image(self, inplace=False) -> CameraView:
        """Convert the representation of the container view
        to image (i.e., after calling this function, we guarantee
        that view.depth has shape [n, 3], where
         - view.depth[:, [0, 1]] is the array of point XY coordinates
           in the canvas plane,
         - view.depth[:, 2] is the point depth (Z coordinate)
           in the camera reference frame.
        Moreover, view.signal has the shape [n, d] and is ordered
        in the same way as view.depth.

        Subclasses should override this method to provide the
        non-trivial implementation.

        :param inplace:
            if True, return the same view
            if False, create a deep copy of the view
        :return: modified view
        """
        return _maybe_inplace(self.view)

    def to_pixels(self, inplace=False) -> CameraView:
        """Convert the representation of the container view
        to pixels (i.e., after calling this function, we guarantee
        that view.depth is [h, w] dense pixel matrix.

        Subclasses should override this method to provide the
        non-trivial implementation.

        :param inplace:
            if True, return the same view
            if False, create a deep copy of the view
        :return: modified view
        """
        return _maybe_inplace(self.view)

    @abstractmethod
    def reproject_to(self, other: CameraView) -> CameraView:
        pass

    @abstractmethod
    def __str__(self): pass


def pixelizer_from_view(view: CameraView) -> pixelization.ImagePixelizerBase:
    pixelizer_type = view.params['projection']

    projection_by_type = {
        'parallel': pixelization.ParallelProjection,
        'perspective': projection.PerspectiveProjectionBase,
    }
    assert pixelizer_type in projection_by_type, 'unknown projection type: {}'.format(pixelizer_type)
    projection_cls = projection_by_type[pixelizer_type]
    return projection_cls()
    return pixelizer


class PixelViewState(CameraViewStateBase):
    def __str__(self): return 'pixels'

    def to_points(self, inplace=False) -> CameraView:
        view = self.to_image(inplace=inplace)
        view = view.to_points()  # calls ImageViewState.to_points()
        return view

    def to_image(self, inplace=False) -> CameraView:
        assert None is not self.view.intrinsics, 'view intrinsics not available'

        pixelizer = pixelizer_from_view(self.view)
        image, signal = pixelizer.unpixelize(self.view.depth, self.view.signal)

        view = _maybe_inplace(self.view, inplace=inplace)
        view.depth = image
        view.signal = signal
        view.state = ImageViewState(view)

        return view


def projection_from_view(view: CameraView) -> projection.CameraProjectionBase:
    projection_type = view.params['projection']

    projection_by_type = {
        'parallel': projection.ParallelProjectionBase,
        'perspective': projection.PerspectiveProjectionBase,
    }
    assert projection_type in projection_by_type, 'unknown projection type: {}'.format(projection_type)
    projection_cls = projection_by_type[projection_type]
    return projection_cls(view.intrinsics)


class ImageViewState(CameraViewStateBase):
    def __str__(self): return 'image'

    def to_points(self, inplace=False) -> CameraView:
        assert None is not self.view.intrinsics, 'view intrinsics not available'
        assert None is not self.view.params.get('projection'), 'view projection type unknown'

        projection = projection_from_view(self.view)
        image, signal = projection.unproject(self.view.depth, self.view.signal)

        view = _maybe_inplace(self.view, inplace=inplace)
        view.depth = image
        view.signal = signal
        view.state = PointsViewState(view)

        return view

    def to_pixels(self, inplace=False) -> CameraView:
        assert None is not self.view.intrinsics

        pixelizer = pixelizer_from_view(self.view)
        image, signal = pixelizer.pixelize(self.view.depth, self.view.signal)

        view = _maybe_inplace(self.view, inplace=inplace)
        view.depth = image
        view.signal = signal
        view.state = PixelViewState(view)

        return view


class PointsViewState(CameraViewStateBase):
    def __str__(self): return 'points'

    def to_image(self, inplace=False) -> CameraView:
        assert None is self.view.intrinsics, 'view intrinsics not available'
        assert None is self.view.params.get('projection'), 'view projection type unknown'

        projection = projection_from_view(self.view)
        image, signal = projection.project(self.view.depth, self.view.signal)

        view = _maybe_inplace(self.view, inplace=inplace)
        view.depth = image
        view.signal = signal
        view.state = ImageViewState(view)

        return view

    def to_pixels(self, inplace=False) -> CameraView:
        view = self.to_image(inplace=inplace)
        view = view.to_pixels()  # calls ImageViewState.to_pixels()
        return view

    def reproject_to(self, other: CameraView) -> CameraView:
        pass
