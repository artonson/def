from io import BytesIO
import tarfile

import numpy as np
from sklearn.metrics import balanced_accuracy_score, jaccard_score


def _get_requested(data_target, read_points=True, read_distances=True):
    if read_points and read_distances:
        data, target = data_target[:, :, :3], np.squeeze(data_target[:, :, 3:])

    elif read_points:  # and not read_distances
        data, target = data_target, None

    else:  # not read_points and read_distances
        data, target = None, data_target

    return data, target


def read_npy_file(path, read_points=True, read_distances=True):
    """Read numpy file and return numpy arrays
    for points (n_obj, n_points, 3) and distances (n_obj, n_points) values

    # Read points and normals
    >>> points, distances = read_npy_file("distance_07102.npy")

    # Read points only:
    >>> points, _ = read_npy_file("distance_07102_data.npy", read_distances=False)
    """
    assert read_points or read_distances, 'at least one of read_points or read_distances must be specified'
    data_target = np.load(path)
    return _get_requested(data_target, read_points=read_points, read_distances=read_distances)


def read_txt_file(path, read_points=True, read_distances=True):
    """Read txt file and return numpy arrays for points (n_points, 3) and distance (n_points, ) values

    # Read points and normals
    >>> points, distances = read_txt_file("distance_07102.txt")

    # Read points only:
    >>> points, _ = read_txt_file("distance_07102_data.txt", read_distances=False)
    """
    assert read_points or read_distances, 'at least one of read_points or read_distances must be specified'

    data_target = np.loadtxt(path)

    if read_points and read_distances:
        data, target = data_target[:, :3], np.squeeze(data_target[:, 3:])

    elif read_points:  # and not read_distances
        data, target = data_target, None

    else:  # not read_points and read_distances
        data, target = None, data_target

    return data, target


def read_targz_npy_file(path, read_points=True, read_distances=True):
    """Read tarzipped numpy file and return numpy arrays
    for points (n_obj, n_points, 3) and distances (n_obj, n_points) values

    # Read points and normals
    >>> points, distances = read_targz_npy_file("npy/high_res_train_0.tar.gz")

    # Read points only:
    >>> points, _ = read_targz_npy_file("npy/high_res_test_0_target.tar.gz", read_distances=False)
    """
    assert read_points or read_distances, 'at least one of read_points or read_distances must be specified'

    with tarfile.open(path, 'r:gz') as targz:
        member_name = next(name for name in targz.getnames() if name.endswith('npy'))
        file_obj = targz.extractfile(member_name)
        io = BytesIO(file_obj.read())
        data_target = np.load(io)

    return _get_requested(data_target, read_points=read_points, read_distances=read_distances)


def segmentation_balanced_accuracy_error(ref, est):
    """Calculates the balanced accuracy between reference segmentation and estimated segmentation."""
    assert len(est.shape) == len(ref.shape) == 1, '1d arrays expected'
    assert est.shape == ref.shape, 'inconsistent shapes'
    return balanced_accuracy_score(ref, est)


def segmentation_iou_error(ref, est):
    """Calculates the IoU between reference segmentation and estimated segmentation."""
    assert len(est.shape) == len(ref.shape) == 1, '1d arrays expected'
    assert est.shape == ref.shape, 'inconsistent shapes'
    return jaccard_score(ref, est)

