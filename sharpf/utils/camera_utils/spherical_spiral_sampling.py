import numpy as np

# The code in this section is inspired by the following paper:
# Hüttig, Christian, and Kai Stemmer. "The spiral grid: A new approach to
# discretize the sphere and its application to mantle convection."
# Geochemistry, Geophysics, Geosystems 9.2 (2008).
#
# However, instead of using the exact computations, it approximates the spiral
# by selecting points spaced approximately evenly.


def spherical_spiral_sampling(
    sphere_radius: float = 1.0,
    layer_radius: float = 0.1,
    resolution: float = 0.01,
    n_initial_samples: int = 1001,
    min_arc_length: float = 0.1,
):
    """Return a set of points evenly sampled on a spherical spiral.

    :param sphere_radius: radius of the sphere
    :param layer_radius: the larger, the closer two twists
            of the spiral are going to be (more twists, sparser points per twist)
    :param resolution: the larger, the farther two twists
            of the spiral are going to be (less twists, denser points per twist)
    :param n_initial_samples: how many points to sample along the spiral
    :param min_arc_length: how fine the spiral has to be in the end

    :return: points: [x, y, y] (n, 3) array of point locations.
    """
    a_max = (3 * np.pi ** 2 * layer_radius) / (2 * resolution)

    a = np.linspace(0, a_max, n_initial_samples)
    arg = np.pi / 2. * (- 1 + 2 * a / a_max)
    x = sphere_radius * np.cos(a) * np.cos(arg)
    y = sphere_radius * np.sin(a) * np.cos(arg)
    z = -sphere_radius * np.sin(arg)
    points = np.stack((x, y, z)).T

    # compute arc length (approximated by line segments) between
    # each consecutive pair of points
    arcs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    ind_max = np.argmax(arcs)

    # add points, starting from a position of longer arcs,
    # ensuring at least min_arc_length between each pair of points
    # (this is possible to ensure when max(arcs) << min_arc_length)
    points_arcs_1 = []
    i = ind_max
    while i < len(arcs):
        arc = 0
        while i < len(arcs) and arc < min_arc_length:
            arc += arcs[i]
            i += 1

        points_arcs_1.append(points[i])
        i += 1

    points_arcs_0 = []
    i = ind_max
    while i > 0:
        arc = 0
        while i > 0 and arc < min_arc_length:
            arc += arcs[i]
            i -= 1

        points_arcs_0.append(points[i])
        i -= 1

    points_arcs = np.concatenate((
        np.array(points_arcs_0[::-1]),
        np.array(points_arcs_1)))

    return points_arcs
