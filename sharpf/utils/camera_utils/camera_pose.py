import numpy as np
import trimesh.transformations as tt


def rotate_to_world_origin(camera_origin):
    # construct a 3x3 rotation matrix to a coordinate frame where:
    # Z axis points to world origin aka center of a mesh
    # Y axis points down
    # X axis is computed as Y cross Z

    camera_origin = np.asanyarray(camera_origin)

    e_z = -camera_origin / np.linalg.norm(camera_origin)  # Z axis points to world origin aka center of a mesh
    e_y = np.array([0, 0, -1])  # proxy to Y axis pointing directly down
    # note that real e_y must be
    # 1) orthogonal to e_z;
    # 2) lie in the plane spanned by e_y and e_z;
    # 3) point downwards so <e_y, [0, 0, -1]> >= <e_y, [0, 0, +1]>
    # 4) unit norm
    gamma = np.dot(e_y, e_z)
    e_y = -gamma / (1 + gamma ** 2) * e_z + 1. / (1 + gamma ** 2) * e_y
    if np.dot(e_y, [0, 0, -1]) < np.dot(e_y, [0, 0, 1]):
        e_y *= -1
    e_y /= np.linalg.norm(e_y)
    e_x = np.cross(e_y, e_z)  # X axis
    R = np.array([e_x, e_y, e_z])
    return R


def create_rotation_matrix_z(rz):
    return np.array([[np.cos(rz), -np.sin(rz), 0.0],
                     [np.sin(rz), np.cos(rz), 0.0],
                     [0.0, 0.0, 1.0]])


def camera_to_display(image):
    return image[::-1, ::-1].T


class CameraPose:
    def __init__(self, transform):
        self._camera_to_world_4x4 = transform
        # always store transform from world to camera frame
        self._world_to_camera_4x4 = np.linalg.inv(self._camera_to_world_4x4)

    @classmethod
    def from_camera_to_world(cls, rotation=None, translation=None):
        """Create camera pose from camera to world transform.

        :param rotation: 3x3 rotation matrix of camera frame axes in world frame
        :param translation: 3d location of camera frame origin in world frame
        """
        rotation = np.identity(3) if None is rotation else np.asanyarray(rotation)
        translation = np.zeros(3) if None is translation else np.asanyarray(translation)

        transform = np.identity(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation

        return cls(transform)

    @classmethod
    def from_camera_axes(cls, R=None, t=None):
        """Compute 4x4 camera pose from camera axes given in world frame.

        :param R: a list of 3D basis vectors (cx, cy, cz) defined in world frame
        :param t: 3D vector defining location of camera origin in world frame
        """
        if None is R:
            R = np.identity(3)

        return cls.from_camera_to_world(rotation=R.T, translation=t)

    def world_to_camera(self, points):
        """Transform points from world to camera coordinates.
        Useful for understanding where the objects are, as seen by the camera.

        :param points: either n * 3 array, or a single 3-vector
        """
        points = np.atleast_2d(points)
        return tt.transform_points(points, self._world_to_camera_4x4)

    def camera_to_world(self, points, translate=True):
        """Transform points from camera to world coordinates.
        Useful for understanding where objects bound to camera
        (e.g., image pixels) are in the world.

        :param points: either n * 3 array, or a single 3-vector
        :param translate: if True, also translate the points
        """
        points = np.atleast_2d(points)
        return tt.transform_points(points, self._camera_to_world_4x4, translate=translate)

    @property
    def world_to_camera_4x4(self):
        return self._world_to_camera_4x4

    @property
    def camera_to_world_4x4(self):
        return self._camera_to_world_4x4

    @property
    def frame_origin(self):
        """Return camera frame origin in world coordinates."""
        return self.camera_to_world_4x4[:3, 3]

    @property
    def frame_axes(self):
        """Return camera axes: a list of 3D basis
        vectors (cx, cy, cz) defined in world frame"""
        return self.camera_to_world_4x4[:3, :3].T

    def compose_world_to_camera(self, other_pose):
        """Compose camera poses C_1, C_2, ... (defined relative to each other),
        computing transforms from world frame to an innermost camera frame.

        Equivalent to:
        x_world = <some point>
        other_pose.world_to_camera(
            pose.world_to_camera(
                x_world
            )
        )
        """
        composed_world_to_camera_4x4 = np.dot(other_pose.world_to_camera_4x4, self._world_to_camera_4x4)
        composed_camera_to_world_4x4 = np.linalg.inv(composed_world_to_camera_4x4)
        return CameraPose(composed_camera_to_world_4x4)

    def compose_camera_to_world(self, other_pose):
        """Compose camera poses C_1, C_2, ... (defined relative to each other),
        computing transforms from innermost camera frame to the world frame.

        Equivalent to:
        x_local = <some point>
        pose.camera_to_world(
            pose_local.camera_to_world(
                x_local
            )
        )
        """
        composed_camera_to_world_4x4 = np.dot(self._camera_to_world_4x4, other_pose.camera_to_world_4x4, )
        return CameraPose(composed_camera_to_world_4x4)
