import numpy as np


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

    assert (np.linalg.norm(
        ((1 - t) * v1 + t * v2) - projection) < 1e-10
            )

    # compare distances to projection, v1 and v2, choose minimum
    if 0 <= t <= 1:
        return projection_diff_norm, projection
    else:
        proj_v1, proj_v2 = np.linalg.norm(p - v1), np.linalg.norm(p - v2)
        if proj_v1 < proj_v2:
            return proj_v1, v1
        else:
            return proj_v2, v2