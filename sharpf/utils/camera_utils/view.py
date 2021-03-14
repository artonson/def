from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np

import sharpf.utils.camera_utils.projection as cam_proj
import sharpf.utils.camera_utils.pixelization as cam_pix
from sharpf.utils.camera_utils.camera_pose import CameraPose


def pixelizer_from_view(view: 'CameraView') -> cam_pix.ImagePixelizerBase:
    pixelizer_type = view.params['pixelization']

    pixelizer_by_type = {
        'default': cam_pix.ImagePixelizer,
    }
    assert pixelizer_type in pixelizer_by_type, 'unknown pixelizer type: {}'.format(pixelizer_type)

    pixelizer_cls = pixelizer_by_type[pixelizer_type]
    pixelizer_obj = pixelizer_cls(view.intrinsics[1])
    return pixelizer_obj


def projection_from_view(view: 'CameraView') -> cam_proj.CameraProjectionBase:
    projection_type = view.params['projection']

    projection_by_type = {
        # 'parallel': cam_proj.ParallelProjection,
        'perspective': cam_proj.PerspectiveProjection,
    }
    assert projection_type in projection_by_type, 'unknown projection type: {}'.format(projection_type)

    projection_cls = projection_by_type[projection_type]
    projection_obj = projection_cls(view.intrinsics[0])
    return projection_obj


class CameraView:
    """Data container representing a view of a scene/object:
    depth/prediction pixel images + camera parameters.
    See explanation of parameters in constructor."""
    default_params = {
        'pixelization': 'default',
        'projection': 'perspective',
    }

    def __init__(
            self,
            depth: np.ndarray = None,
            signal: np.ndarray = None,
            faces: np.ndarray = None,
            extrinsics: np.ndarray = np.eye(4),
            intrinsics: np.ndarray = None,
            params: Mapping = default_params,
            state: str = 'points',
            pixelizer: cam_pix.ImagePixelizerBase = None,
            projection: cam_proj.CameraProjectionBase = None,
    ):
        """A container for annotated range-images.

        :param depth: The actual geometry represented by the view.
                      This can be either [n, 3] array of 3D XYZ points (i.e. a point cloud) (PointsViewState),
                      [n, 3=2+1] array of hstacked UV image coordinates + associated depth (ImageViewState),
                      or [h, w] array of per-pixel depth (PixelViewState).

        :param signal: Per-point annotation for depth.
                       This can be either [n, d] array of per-point
                       d-dimensional features (PointsViewState, ImageViewState),
                       or [h, w, d] array of per-pixel features (PixelViewState).

        :param faces: If the input range-image has connectivity, e.g. coming from a scanner,
                      (and is thus a range surface), we can store its facets.

        :param extrinsics: Camera pose matrix (4x4, shape=[4, 4]) containing a rigid transformation
                           from the camera coordinate frame to the world coordinate frame.

        :param intrinsics: [2, 3, 3] intrinsic parameter matrices associated with the view.
                           The matrix intrinsics[0, ...] (K_f) represents the 3x3 projective transformation
                           from the 3D point coordinates to the UV image coordinates.
                           The matrix intrinsics[1, ...] (K_s) represents the 3x3 transformation
                           to be applied to homogeneous UV coordinates to transform from image coordinates
                           to pixel coordinates.

        :param params: Dict, a description of the extrinsic and intrinsic parameters
                       of the view. Description contains the following keys:
                       1) 'projection': one of 'perspective', 'parallel' -- defines
                       the type of projection to use.
                       2) 'pixelization': one of 'default' -- defines
                       the pixelization class to use.

        :param state: The state of the view (either 'points', 'image' or 'pixels')
                      which determines the view behaviour.
        """
        self.depth = depth
        self.signal = signal
        self.faces = faces
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.params = params

        self.state = {
            'points': PointsViewState,
            'image': ImageViewState,
            'pixels': PixelViewState,
        }[state](self)

        self.pose = CameraPose(self.extrinsics)
        self.pixelizer = pixelizer or pixelizer_from_view(self)
        self.projection = projection or projection_from_view(self)

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

    # def reproject_to(self, other: 'CameraView') -> 'CameraView':
    #     view = self.state.reproject_to(other)
    #     return view

    def copy(self) -> 'CameraView':
        from copy import deepcopy
        return CameraView(
            depth=deepcopy(self.depth),
            signal=deepcopy(self.signal),
            faces=deepcopy(self.faces),
            extrinsics=deepcopy(self.extrinsics),
            intrinsics=deepcopy(self.intrinsics),
            params=deepcopy(self.params),
            state=str(self.state),
            pixelizer=deepcopy(self.pixelizer),
            projection=deepcopy(self.projection),
        )

    @property
    def as_dict(self):
        return {}


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

    # @abstractmethod
    # def reproject_to(self, other: CameraView) -> CameraView:
    #     pass

    @abstractmethod
    def __str__(self): pass


class PixelViewState(CameraViewStateBase):
    def __str__(self): return 'pixels'

    def to_points(self, inplace=False) -> CameraView:
        view = self.to_image(inplace=inplace)
        view = view.to_points()  # calls ImageViewState.to_points()
        return view

    def to_image(self, inplace=False) -> CameraView:
        assert None is not self.view.intrinsics, 'view intrinsics not available'

        pixelizer = self.view.pixelizer
        image, signal = pixelizer.unpixelize(self.view.depth, self.view.signal)

        view = _maybe_inplace(self.view, inplace=inplace)
        view.depth = image
        view.signal = signal
        view.state = ImageViewState(view)

        return view


class ImageViewState(CameraViewStateBase):
    def __str__(self): return 'image'

    def to_points(self, inplace=False) -> CameraView:
        assert None is not self.view.intrinsics, 'view intrinsics not available'
        assert None is not self.view.params.get('projection'), 'view projection type unknown'

        projection = self.view.projection
        points, signal = projection.unproject(self.view.depth, self.view.signal)
        points = self.view.pose.camera_to_world(points)

        view = _maybe_inplace(self.view, inplace=inplace)
        view.depth = points
        view.signal = signal
        view.state = PointsViewState(view)

        return view

    def to_pixels(self, inplace=False) -> CameraView:
        assert None is not self.view.intrinsics

        pixelizer = self.view.pixelizer
        pixels, signal = pixelizer.pixelize(self.view.depth, self.view.signal)

        view = _maybe_inplace(self.view, inplace=inplace)
        view.depth = pixels
        view.signal = signal
        view.state = PixelViewState(view)

        return view


class PointsViewState(CameraViewStateBase):
    def __str__(self): return 'points'

    def to_image(self, inplace=False) -> CameraView:
        assert None is not self.view.intrinsics, 'view intrinsics not available'
        assert None is not self.view.params.get('projection'), 'view projection type unknown'

        points = self.view.pose.world_to_camera(self.view.depth)
        projection = self.view.projection
        image, signal = projection.project(points, self.view.signal)

        view = _maybe_inplace(self.view, inplace=inplace)
        view.depth = image
        view.signal = signal
        view.state = ImageViewState(view)

        return view

    def to_pixels(self, inplace=False) -> CameraView:
        view = self.to_image(inplace=inplace)
        view = view.to_pixels()  # calls ImageViewState.to_pixels()
        return view

    # def reproject_to(self, other: CameraView) -> CameraView:
    #     pass
