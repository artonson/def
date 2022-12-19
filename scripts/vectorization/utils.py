import k3d
import randomcolor
import trimesh
import igl
# import Parallel, delayed

import numpy as np
from chainer import cuda
# from pyaabb import pyaabb
from functools import partial
from scipy.interpolate import splev
from scipy.spatial.ckdtree import cKDTree


def get_sharp_edge_endpoints_degen(positions, pairs):
    """For computing distances using
    https://libigl.github.io/libigl-python-bindings/igl_docs/#point_mesh_squared_distance."""

    sharp_face_indexes = np.hstack((np.array(pairs), np.atleast_2d(np.array(pairs)[:, 1]).T))

    edge_pseudo_mesh = trimesh.base.Trimesh(
        vertices=positions,
        faces=sharp_face_indexes,
        process=False,
        validate=False)
    return edge_pseudo_mesh


def pointset_edgeset_distances_projections(points, edges_mesh):
    """Compute the distances and projections using libigl's
    functionality available in the method `point_mesh_squared_distance`:
    triangle [1 2 2] is treated as a segment [1 2]
    (see docs at https://libigl.github.io/libigl-python-bindings/igl_docs/#point_mesh_squared_distance )."""

    distances, I, projections = igl.point_mesh_squared_distance(
        points,
        edges_mesh.vertices,
        edges_mesh.faces)
    return distances, I, projections


def dist_vector_proj(p, i, lines):
    """A helper function that computes distance from a point
    to a line segment using a vector projection."""
    v1, v2 = lines[i]

    # compute unit norm vector from v2 to v1 (linear subspace spanning line)
    line_tangent = v2 - v1
    line_tangent_norm = np.linalg.norm(line_tangent)
    line_tangent = line_tangent / line_tangent_norm

    # subtract tangential component, obtain point in L^T(line_tangent)
    projection_diff = p - v1 - np.dot(p - v1, line_tangent) * line_tangent
    projection_diff_norm = np.linalg.norm(projection_diff)

    # compute line parameter value for projection
    projection = p - projection_diff
    t = np.dot(projection - v1, line_tangent) / line_tangent_norm

#     assert (np.linalg.norm(
#         ((1 - t) * v1 + t * v2) - projection) < 1e-10
#             )

    # compare distances to projection, v1 and v2, choose minimum
    if 0 <= t <= 1:
        return projection_diff_norm, projection
    else:
        proj_v1, proj_v2 = np.linalg.norm(p - v1), np.linalg.norm(p - v2)
        if proj_v1 < proj_v2:
            return proj_v1, v1
        else:
            return proj_v2, v2


def l2_norm(x, y):
    """
    Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=np.int32,
                            distances_dtype=np.float32):
    """
    Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)
    return indices, distances


def parallel_nearest_point(positions, pairs, points):
    # match points to curve segments
#     aabb_solver = pyaabb.AABB()
#     aabb_solver.build(aabboxes)
#     distance_func = partial(dist_vector_proj, lines=sharp_edges)
    sharp_edges = get_sharp_edge_endpoints_degen(positions, pairs)
#     return Parallel(n_jobs=4)(delayed(pointset_edgeset_distances_projections)(p, sharp_edges) for for p in points.astype(np.float32))
    return [pointset_edgeset_distances_projections(p, sharp_edges)[1]
            for p in points.astype(np.float32)]


# def parallel_nearest_point(aabboxes, sharp_edges, points):
#     # match points to curve segments
#     aabb_solver = pyaabb.AABB()
#     aabb_solver.build(aabboxes)
#     distance_func = partial(dist_vector_proj, lines=sharp_edges)
#     return [aabb_solver.nearest_point(p, distance_func)[0]
#             for p in points.astype(np.float32)]


def create_aabboxes(lines):
    # create bounding boxes for sharp edges
    nodes = []
    eps = 1e-8
    for l in lines:
        minc = np.array([
            min(l[0][0], l[1][0]) - eps,
            min(l[0][1], l[1][1]) - eps,
            min(l[0][2], l[1][2]) - eps
        ])
        maxc = np.array([
            max(l[0][0], l[1][0]) + eps,
            max(l[0][1], l[1][1]) + eps,
            max(l[0][2], l[1][2]) + eps
        ])
        nodes.append([minc, maxc])
    return nodes


def recalculate(points, distances, matching, corner_pairs):
    # using aabb, create lists of lists for points and distances that correspond to each segment
    curve_data = []
    curve_distances = []
    for i in range(len(corner_pairs)):
        curve_data.append(points[np.array(matching) == i]) 
        curve_distances.append(distances[np.array(matching) == i]) 
    return curve_data, curve_distances


def draw(points, corner_positions, corner_pairs, path_to_save, filename, DISPLAY_RES, tcks=[], closed_tcks=[], straight_lines=[]):
#     print(corner_positions.shape)
    plot = k3d.plot()

    k3d_points = k3d.points(points, point_size=DISPLAY_RES, opacity=0.1, shader='3d', name='sharp_points')
    plot += k3d_points
    
    if np.logical_and(np.logical_and(len(tcks) == 0, len(straight_lines) == 0), len(closed_tcks) == 0):
        
        points_corner_centers = k3d.points(corner_positions,
                                           color=0xFF0000, point_size=DISPLAY_RES, shader='3d', name='polyline_nodes')
        plot += points_corner_centers

        for edge in corner_pairs:
            e = k3d.line(np.array(corner_positions)[edge], name='polyline_edge')
            plot+=e
            
        with open('{path_to_save}/{filename}__result.html'.format(path_to_save=path_to_save, filename=filename), 'w') as f:
            f.write(plot.get_snapshot())

    else:
        for i in range(len(straight_lines)):
            spline = k3d.line(straight_lines[i], color=0xff0000, width=DISPLAY_RES-0.015)
            plot += spline
        for i in range(len(tcks)):    
            spline = k3d.points(np.array(splev(np.linspace(0,1,2500), tcks[i])).T, color=0xff0000, point_size=DISPLAY_RES-0.015, shader='flat')
            plot += spline
        for i in range(len(closed_tcks)):    
            spline = k3d.points(np.array(splev(np.linspace(0,1,2500), closed_tcks[i])).T, color=0xff0000, point_size=DISPLAY_RES-0.015, shader='flat')
            plot += spline
        with open('{path_to_save}/{filename}__final_result.html'.format(path_to_save=path_to_save, filename=filename), 'w') as f:
            f.write(plot.get_snapshot())
