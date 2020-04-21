import numpy as np


def fibonacci_sphere_sampling(number_of_views=1, seed=None, radius=1.0, positive_z=False):
    # Returns [x,y,z] tuples of a fibonacci sphere sampling
    # http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/
    # commonly needed to evenly cover an sphere enclosing the object
    # for rendering from that points
    # (hypothetically this should be giving us most important projections of a 3D shape)
    if positive_z:
        number_of_views *= 2
    rnd = 1.
    if seed is not None:
        np.random.seed(seed)
        rnd = np.random.random() * number_of_views

    points = []
    offset = 2. / number_of_views
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(number_of_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % number_of_views) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        if positive_z:
            s = np.arcsin(z / radius) * 180.0 / np.pi
            if z > 0.0 and s > 30:
                points.append([radius * x, radius * y, radius * z])
        else:
            points.append([radius * x, radius * y, radius * z])

    return points
