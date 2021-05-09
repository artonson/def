import numpy as np
from numpy.lib.stride_tricks import as_strided


# from https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a
def sliding_window_view(arr, window_shape, steps):
    """ Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping
        over the boundary.

        Parameters
        ----------
        arr : numpy.ndarray, shape=(...,[x, (...), z])
            The array to slide the window over.

        window_shape : Sequence[int]
            The shape of the window to raster: [Wx, (...), Wz],
            determines the length of [x, (...), z]

        steps : Sequence[int]
            The step size used when applying the window
            along the [x, (...), z] directions: [Sx, (...), Sz]

        Returns
        -------
        view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz])
            Where X = (x - Wx) // Sx + 1

        Notes
        -----
        In general, given
          `out` = sliding_window_view(arr,
                                      window_shape=[Wx, (...), Wz],
                                      steps=[Sx, (...), Sz])

           out[ix, (...), iz] = arr[..., ix*Sx:ix*Sx+Wx,  (...), iz*Sz:iz*Sz+Wz]

         Examples
         --------
         >>> import numpy as np
         >>> x = np.arange(9).reshape(3,3)
         >>> x
         array([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])

         >>> y = sliding_window_view(x, window_shape=(2, 2), steps=(1, 1))
         >>> y
         array([[[[0, 1],
                  [3, 4]],

                 [[1, 2],
                  [4, 5]]],


                [[[3, 4],
                  [6, 7]],

                 [[4, 5],
                  [7, 8]]]])
        >>> np.shares_memory(x, y)
         True

        # Performing a neural net style 2D conv (correlation)
        # placing a 4x4 filter with stride-1
        >>> data = np.random.rand(10, 3, 16, 16)  # (N, C, H, W)
        >>> filters = np.random.rand(5, 3, 4, 4)  # (F, C, Hf, Wf)
        >>> windowed_data = sliding_window_view(data,
        ...                                     window_shape=(4, 4),
        ...                                     steps=(1, 1))

        >>> conv_out = np.tensordot(filters,
        ...                         windowed_data,
        ...                         axes=[[1,2,3], [3,4,5]])

        # (F, H', W', N) -> (N, F, H', W')
        >>> conv_out = conv_out.transpose([3,0,1,2])
         """
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)


def crop_around_nonzero(image, return_bbox=False):
    """Returns sub-image of a 2D image which contains
    all values of the image."""
    yy, xx = np.where(image)
    top, bottom = np.min(yy), np.max(yy)
    left, right = np.min(xx), np.max(xx)

    crop = image[top:bottom, left:right]
    if return_bbox:
        return crop, [top, bottom, left, right]
    else:
        return crop


def sliding_window_with_offset(arr, window_shape, steps):
    arr_size_y, arr_size_x = arr.shape
    wy, wx = window_shape
    sy, sx = steps

    for y in range(0, arr_size_y - wy + 1, sy):
        for x in range(0, arr_size_x - wx + 1, sx):
            yield arr[y:y + wy, x:x + wx], (y, x)
