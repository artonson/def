import numpy as np
from scipy.interpolate import fitpack

from sharpf.utils.numpy_utils.masking import compress_mask


def bisplrep_interpolate():
    pass


def interp2d_interpolate():
    pass


def pointwise_interpolate_image(
        source_points,
        source_signal,
        target_points,
        source_image_tree: cKDTree = None,
        neighbours_to_interpolate: int = 6,
        distance_interpolation_threshold='auto',
        z_distance_threshold: int = 2.0,
        verbose: bool = False,
        interpolator_function: str = 'bisplrep',
):
    if 'auto' == distance_interpolation_threshold:
        mean_nn_distance = cKDTree(source_points).query(source_points, k=2)[0][:, 1].mean()
        distance_interpolation_threshold = mean_nn_distance * 4.

    n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
    _, nn_indexes = source_image_tree.query(
        target_points[:, :2],
        k=neighbours_to_interpolate,
        n_jobs=n_omp_threads,
        distance_upper_bound=distance_interpolation_threshold)

    xy_mask = np.all(nn_indexes != source_image_tree.n, axis=1)

    z_distances = np.abs(
        target_points[:, 2][xy_mask, np.newaxis] -
        source_points[:, 2][nn_indexes[xy_mask]])

    z_mask = np.all(z_distances < z_distance_threshold, axis=1)

    can_interpolate = compress_mask(xy_mask, z_mask)
    can_interpolate_indexes = np.where(can_interpolate)[0]
    #     print(target_points.shape, can_interpolate.shape, xy_mask.shape, nn_indexes.shape)

    target_signal = np.zeros(len(target_points), dtype=float)

    for idx in tqdm(can_interpolate_indexes):
        x, y, _ = target_points[idx]

        xs, ys, zs = np.split(
            source_points[nn_indexes[idx]],
            [1, 2],
            axis=1)
        ps = source_signal[nn_indexes[idx]]

        try:
            x_min, x_max = np.amin(xs), np.amax(xs)
            y_min, y_max = np.amin(ys), np.amax(ys)

            out_of_bounds_x = (x < x_min) | (x > x_max)
            out_of_bounds_y = (y < y_min) | (y > y_max)

            any_out_of_bounds_x = np.any(out_of_bounds_x)
            any_out_of_bounds_y = np.any(out_of_bounds_y)

            if any_out_of_bounds_x or any_out_of_bounds_y:
                raise ValueError("Values out of range; x must be in %r, y in %r"
                                 % ((x_min, x_max),
                                    (y_min, y_max)))

            tck = fitpack.bisplrep(
                xs.squeeze(), ys.squeeze(), ps.squeeze(),
                kx=1, ky=1, s=len(xs.squeeze()))
            target_signal[idx] = fitpack.bisplev(x, y, tck)

        #             interpolator = interpolate.interp2d(
        #                 xs.squeeze(), ys.squeeze(), ps.squeeze(),
        #                 kind='linear',
        #                 bounds_error=True)
        #             target_signal[idx] = interpolator(x, y)[0]
        except ValueError as e:
            #             eprint_t('Error while interpolating point {idx}: {what}, skipping this point'.format(
            #                 idx=idx, what=str(e)))
            can_interpolate[idx] = False

        except RuntimeWarning as w:
            break

    return target_signal, can_interpolate

