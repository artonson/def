import numpy as np
from scipy import interpolate
from scipy.interpolate import fitpack
from scipy.spatial import cKDTree
from tqdm import tqdm

from sharpf.utils.numpy_utils.masking import compress_mask


def bisplrep_interpolate(xs, ys, vs, xt, yt):
    x_min, x_max = np.amin(xs), np.amax(xs)
    y_min, y_max = np.amin(ys), np.amax(ys)

    out_of_bounds_x = (xt < x_min) | (xt > x_max)
    out_of_bounds_y = (yt < y_min) | (yt > y_max)

    any_out_of_bounds_x = np.any(out_of_bounds_x)
    any_out_of_bounds_y = np.any(out_of_bounds_y)

    if any_out_of_bounds_x or any_out_of_bounds_y:
        raise ValueError("Values out of range; x must be in %r, y in %r"
                         % ((x_min, x_max),
                            (y_min, y_max)))

    tck = fitpack.bisplrep(
        xs.squeeze(),
        ys.squeeze(),
        vs.squeeze(),
        kx=1,
        ky=1,
        s=len(xs.squeeze()))
    vt = fitpack.bisplev(xt, yt, tck)
    return vt


def interp2d_interpolate(xs, ys, vs, xt, yt):
    interpolator = interpolate.interp2d(
        xs.squeeze(),
        ys.squeeze(),
        vs.squeeze(),
        kind='linear',
        bounds_error=True)

    vt = interpolator(xt, yt)[0]
    return vt


def pointwise_interpolate_image(
        source_points: np.ndarray,
        source_signal: np.ndarray,
        target_points: np.ndarray,
        nn_set_size: int = 8,
        distance_interp_thr = 'auto',
        z_distance_threshold: int = 2.0,
        interpolator_function: str = 'bisplrep',
        verbose: bool = False,
):
    """Execute view-view interpolation point by point.

    :param source_points: source [n, 3] locations to interpolate from
    :param source_signal: source [n,] signal to interpolate from
    :param target_points: target [n, 3] locations to interpolate onto
    :param nn_set_size: number of nearest neighbors to use for interpolation
    :param distance_interp_thr: max distance that a nearest neighbor can be away from
    :param z_distance_threshold: max distance along depth that a nearest neighbor can be away from
    :param interpolator_function: type of interpolate function to use ('bisplrep' or 'interp2d')
    :param verbose: print something on output

    :return: target_signal, can_interpolate
    """
    if 'auto' == distance_interp_thr:
        if verbose:
            print('Determining distance_interp_thr... ')
        mean_nn_distance = cKDTree(source_points).query(source_points, k=2)[0][:, 1].mean()
        distance_interp_thr = mean_nn_distance * 4.

    if verbose:
        print('Finding nearest neighbors... ')
    source_image_tree = cKDTree(source_points)
    _, nn_indexes = source_image_tree.query(
        target_points[:, :2],
        k=nn_set_size,
        distance_upper_bound=distance_interp_thr)

    if verbose:
        print('Masking far away neighbors... ')
    xy_mask = np.all(nn_indexes != source_image_tree.n, axis=1)
    z_distances = np.abs(
        target_points[:, 2][xy_mask, np.newaxis] -
        source_points[:, 2][nn_indexes[xy_mask]])
    z_mask = np.all(z_distances < z_distance_threshold, axis=1)

    can_interpolate = compress_mask(xy_mask, z_mask)
    can_interpolate_indexes = np.where(can_interpolate)[0]
    target_signal = np.zeros(len(target_points), dtype=float)

    interp_fn = {
        'bisplrep': bisplrep_interpolate,
        'interp2d': interp2d_interpolate,
    }[interpolator_function]

    for idx in tqdm(can_interpolate_indexes, desc='Interpolating'):
        xt, yt, _ = target_points[idx]

        xs, ys, _ = np.split(
            source_points[nn_indexes[idx]],
            [1, 2],
            axis=1)
        vs = source_signal[nn_indexes[idx]]

        try:
            target_signal[idx] = interp_fn(xs, ys, vs, xt, yt)

        except ValueError as e:
            if verbose:
                print('Error while interpolating point {idx}:'
                      '{what}, skipping this point'.format(idx=idx, what=str(e)))
            can_interpolate[idx] = False

        except RuntimeWarning:
            break

    return target_signal, can_interpolate
