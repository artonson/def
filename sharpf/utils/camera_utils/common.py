
def check_is_image(depth, signal=None):
    # assume [n, 3] array for image, [n, d] array for signal
    assert len(depth.shape) == 2 and depth.shape[-1] == 3, \
        'image: expected shape [n, 3], got: {}'.format(depth.shape)
    if None is not signal:
        assert len(signal.shape) in [1, 2], \
            'signal: expected shape [n, d] or [n, ], got: {}'.format(signal.shape)
        assert depth.shape[0] == signal.shape[0], \
            'points/signal: points and signal have different shapes'


def check_is_points(depth, signal=None):
    # assume [n, 3] array for points, [n, d] array for signal
    assert len(depth.shape) == 2 and depth.shape[-1] == 3, \
        'points: expected shape [n, 3], got: {}'.format(depth.shape)
    if None is not signal:
        assert len(signal.shape) in [1, 2], \
            'signal: expected shape [n, d] or [n, ], got: {}'.format(signal.shape)
        assert depth.shape[0] == signal.shape[0], \
            'points/signal: points and signal have different shapes'


def check_is_pixels(depth, signal=None):
    # assume [h, w] array for points, [h, w, d] array for signal
    assert len(depth.shape) == 2, \
        'cannot project points: expected shape [h, w], got: {}'.format(depth.shape)
    if None is not signal:
        assert len(signal.shape) in [2, 3], \
            'signal: expected shape [h, w, d] or [h, w], got: {}'.format(signal.shape)
        assert depth.shape[[0, 1]] == signal.shape[[0, 1]], \
            'points/signal: points and signal have different shapes'
