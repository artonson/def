from sharpf.utils.geometry import dist_vector_proj
from pyaabb import pyaabb
aabb_solver = pyaabb.AABB()
import numpy as np


def create_aabboxes(lines):
    # create bounding boxes for sharp edges
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


lines = np.array([
      [[0, 0, 0], [1, 1, 1]],
      [[2, 2, 2], [3, 3, 3]]
  ])

aabboxes = create_aabboxes(lines)

aabb_solver.build(aabboxes)

from functools import partial
distance_func = partial(dist_vector_proj, lines=lines)

points = np.random.normal(size=(10,3)).astype(np.float32)

aabb_solver.nearest_point_array(points.tolist(), distance_func)
