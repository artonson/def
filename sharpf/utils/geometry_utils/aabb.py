from functools import partial

import numpy as np

from sharpf.utils.geometry_utils.geometry import point_edgeset_squared_distance


def create_aabboxes(lines):
    """Creates a list of bounding boxes for a list of edge endpoints.
    The edge set must be given as a list of (XYZ_1, XYZ_2) pairs
    of edge endpoints. """
    corners = []
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
        corners.append([minc, maxc])
    return corners


def pointset_edgeset_distances_projections(points, edges):
    from pyaabb import pyaabb
    aabb_solver = pyaabb.AABB()
    aabboxes = create_aabboxes(edges)
    aabb_solver.build(aabboxes)
    distance_func = partial(point_edgeset_squared_distance, lines=edges)
    results = [aabb_solver.nearest_point(p, distance_func)
               for p in points.astype(np.float32)]

    projections = np.array([projection for idx, projection, distance in results])
    distances = np.array([distance for idx, projection, distance in results])
    return distances, projections
