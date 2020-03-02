import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector


def generate_rays(image_resolution, resolution_3d, radius=1.0):
    """Creates an array of rays and ray directions used for mesh raycasting.

    :param image_resolution: image resolution in pixels
    :type image_resolution: int or tuple of ints

    :param resolution_3d: pixel 3d resolution in mm/pixel
    :type resolution_3d: float

    :param radius: Z coordinate for the rays origins
    :type resolution_3d: float

    :return:
        rays_screen_coords:     (W * H, 2): screen coordinates for the rays
        rays_origins:           (W * H, 3): world coordinates for rays origins
        ray_directions:         (W * H, 3): unit norm vectors pointing in directions of the rays
    """
    if isinstance(image_resolution, tuple):
        assert len(image_resolution) == 2
    else:
        image_resolution = (image_resolution, image_resolution)
    image_width, image_height = image_resolution

    # generate an array of screen coordinates for the rays
    # (rays are placed at locations [i, j] in the image)
    rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(
        2, image_height * image_width).T    # [h, w, 2]

    # place rays physically in locations determined by screen coordinates,
    # then shift so that camera origin is in the midpoint in the image,
    # and linearly stretch so each pixel takes exactly resolution_3d space
    screen_aspect_ratio = image_width / image_height
    rays_origins = (rays_screen_coords / np.array([[image_height, image_width]]))   # [h, w, 2], in [0, 1]
    factor = image_height / 2 * resolution_3d
    rays_origins[:, 0] = (-2 * rays_origins[:, 0] + 1) * factor  # to [-1, 1] + aspect transform
    rays_origins[:, 1] = (-2 * rays_origins[:, 1] + 1) * factor * screen_aspect_ratio
    rays_origins = np.concatenate([
        rays_origins,
        radius + np.zeros_like(rays_origins[:, [0]])
    ], axis=1)  # [h, w, 3]

    # ray directions are always facing towards Z axis
    ray_directions = np.tile(np.array([0, 0, -1]), (rays_origins.shape[0], 1))

    return rays_screen_coords, rays_origins, ray_directions


def ray_cast_mesh(mesh, rays_origins, ray_directions):
    intersector = RayMeshIntersector(mesh)
    index_triangles, index_ray, point_cloud = intersector.intersects_id(
        ray_origins=rays_origins, ray_directions=ray_directions,
        multiple_hits=False, return_locations=True)
    return index_triangles, index_ray, point_cloud


#
# def ray_cast_mesh(mesh, labels, ortho_scale=2.0, image_sz=(512, 512), num_angles=4, z_shift=-3):
#     # ortho_scale: controls how much zoom the
#
#
#     image_height, image_width = image_sz
#
#     angles = np.linspace(0, np.pi, num_angles)
#     mesh_grid = np.mgrid[0:image_height, 0:image_width].reshape(2, image_height * image_width).T  # screen coordinates
#
#     # to [0, 1]
#     aspect = image_width / image_height
#     rays_origins = (mesh_grid / np.array([[image_height, image_width]]))  # [h, w, 2]
#
#     # to [-1, 1] + aspect transform
#     rays_origins[:, 0] = (-2 * rays_origins[:, 0] + 1) * ortho_scale / 2
#     rays_origins[:, 1] = (-2 * rays_origins[:, 1] + 1) * ortho_scale / 2 * aspect
#
#     rays_origins = np.concatenate([
#         rays_origins,
#         np.zeros_like(rays_origins[:, [0]])
#     ], axis=1)
#     ray_directions = np.tile(np.array([0, 0, -1]), (rays_origins.shape[0], 1))
#
#
#     curves = resample_sharp_edges(mesh, labels)
#     poses = []
#     renders = []
#     distances = []
#     annotations = []
#     sharp_curves = []
#     pcs = []
#
#     i = 0
#     for (x, y, z) in itertools.combinations_with_replacement(angles, 3):
#         i += 1
#
#         transform = view.from_pose([x, y, z], [0, 0, 0])
#
#         # transform mesh and get normalized one
#         mesh_ = mesh.copy()
#         mesh_.apply_transform(transform)
#         mesh_.apply_translation([0, 0, z_shift])
#         mesh_normalized = mesh_.copy()
#         mean = mesh_normalized.vertices.mean()
#         mesh_normalized.vertices -= mean
#         scale = mesh_normalized.vertices.std()  # np.linalg.norm(mesh_normalized.vertices, axis=1).max()
#         mesh_normalized.vertices /= scale
#         mesh_normalized.vertices *= 10
#
#         # rotate sharp lines
#         curve_points = curves.reshape(-1, 3)
#         curve_points = curve_points.dot(transform.numpy().T[:3, :3]) + [0, 0, z_shift]
#         curve_points -= mean
#         curve_points /= scale
#         curve_points *= 10
#         curves_trans = curve_points.reshape(-1, 2, 3)
#
#         # get point cloud
#         ray_mesh = RayMeshIntersector(mesh_)
#         index_triangles, index_ray, point_cloud = ray_mesh.intersects_id(ray_origins=rays_origins,
#                                                                          ray_directions=ray_directions, \
#                                                                          multiple_hits=False, return_locations=True)
#         point_cloud -= mean
#         point_cloud /= scale
#         point_cloud *= 10
#
#         start_4 = time.clock()
#         annotation = annotate(curves_trans, mesh_normalized, labels, point_cloud, distance_upper_bound=3.0)
#         end_4 = time.clock()
#         print('annotation time', end_4 - start_4)
#
#         render = np.zeros((image_height, image_width))
#         render[
#             mesh_grid[index_ray][:, 0],
#             mesh_grid[index_ray][:, 1]
#         ] = point_cloud[:, 2]
#
#         dist = np.zeros((image_height, image_width))
#         dist[mesh_grid[index_ray][:, 0], mesh_grid[index_ray][:, 1]] = annotation
#
#         renders.append(render)
#         distances.append(dist)
#         annotations.append(annotation)
#         poses.append({'rotation': (x, y, z), 'location': (0, 0, z_shift)})
#         #         pcs.append(point_cloud)
#         #         sharp_curves.append(curves_trans)
#         del transform, mesh_, mesh_normalized
#
#     return renders, distances, annotations  # , pcs, sharp_curves
