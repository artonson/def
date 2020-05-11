import numpy as np


def read_file(path, read_points=True, read_distances=True):
    """Read txt file and return numpy arrays for points and distance values

    # Read points and normals
    >>> points, distances = read_file("distance_07102.txt")

    # Read points only:
    >>> points, _ = read_file("distance_07102.txt", read_distances=False)
    """

    assert read_points or read_distances, 'at least one of read_points or read_distances must be specified'

    data_target = np.loadtxt(path)

    if read_points and read_distances:
        data, target = data_target[:, :3], data_target[:, 3:]

    elif read_points:  # and not read_distances
        data, target = data_target[:, :3], None

    else:  # not read_points and read_distances
        data, target = None, data_target[:, 3:]

    return data, target


def mean_error_field(ref, est):
    """Calculates the mean errors between reference distance field and estimated distance field."""
    assert len(est.shape) == len(ref.shape) == 1, "1d arrays expected"

    # make sure that estimations live in [0, 1]
    est = np.maximum(0, np.minimum(1, est))

    mse = np.mean((ref - est) ** 2)

    return mse


def mean_error_(ref, est):
    """Calculates the mean errors between reference distance field and estimated distance field."""
    assert len(est.shape) == len(ref.shape) == 1, "1d arrays expected"

    # make sure that estimations live in [0, 1]
    est = np.maximum(0, np.minimum(1, est))

    mse = np.mean((ref - est) ** 2)

    return mse
