from abc import ABC, abstractmethod
from typing import Mapping, Tuple

import numpy as np

from .camera_pose import CameraPose


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
            state: 'CameraViewState' = None,
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

    def to_other(self, other: 'CameraView') -> 'CameraView':
        pass

    @classmethod
    def from_other(cls, other: 'CameraView') -> 'CameraView':
        from copy import deepcopy
        return cls(
            deepcopy(other.depth),
            deepcopy(other.signal),
            deepcopy(other.extrinsics),
            deepcopy(other.intrinsics),
            deepcopy(other.params))


class CameraViewState(ABC):
    def __init__(self, view: CameraView):
        self.view = view

    @abstractmethod
    def to_points(self, inplace=False) -> CameraView:
        pass

    @abstractmethod
    def to_image(self, inplace=False) -> CameraView:
        pass

    @abstractmethod
    def to_pixels(self, inplace=False) -> CameraView:
        pass

    @abstractmethod
    def reproject_to(self, other: CameraView) -> CameraView:
        pass

    @abstractmethod
    def __str__(self): pass


def maybe_inplace(other_view: CameraView, inplace=False) -> CameraView:
    """Create a copy of other_view if inplace=False,
    otherwise return other_view."""
    view = other_view
    if not inplace:
        view = CameraView.from_other(view)
    return view


def image_pixelizer_from_view(view: CameraView):
    view.params
    pass


class PixelViewState(CameraViewState):

    def __str__(self): return 'pixels'

    def to_pixels(self, inplace=False) -> CameraView:
        return maybe_inplace(self.view)

    def to_image(self, inplace=False) -> CameraView:
        assert None is not self.view.intrinsics

        image_pixelizer = image_pixelizer_from_view(self.view)
        image = image_pixelizer.unpixelize(self.view.depth)
        signal = None
        if self.view.signal is not None:
            signal = self.view.signal.ravel()[np.flatnonzero(view.image)]

        view = maybe_inplace(self.view)
        view.depth = image
        view.signal = signal
        view.state = ImageViewState(view)

        return view

    def to_points(self, inplace=False) -> CameraView:

        assert None is not self.view.intrinsics, 'View intrinsics not present'
        assert None is not self.view.extrinsics, 'View extrinsics not present'

        camera_pose = CameraPose(self.view.extrinsics)
        camera_projection = load_func_from_config()
        image_pixelizer = load_func_from_config()

        image = image_pixelizer.unpixelize(view.image)
        points_in_camera_frame = camera_projection.unproject(image)
        points_in_world_frame = camera_pose.camera_to_world(points_in_camera_frame)

        signal = None
        if view.signal is not None:
            signal = view.signal.ravel()[np.flatnonzero(view.image)]

        return points_in_world_frame, signal


class ImageViewState(CameraViewState):
    pass


class PointsViewState(CameraViewState):
    pass


def view_to_points(view: CameraView) -> Tuple[np.ndarray, np.ndarray]:


def points_to_view(image, signal, view: CameraView) -> Tuple[np.ndarray, np.ndarray]:
    pass


def view_to_image(view: CameraView) -> Tuple[np.ndarray, np.ndarray]:
    pass


def image_to_view(image, signal, view: CameraView) -> Tuple[np.ndarray, np.ndarray]:
    pass

