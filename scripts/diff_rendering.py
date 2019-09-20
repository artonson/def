from typing import Tuple
import torch


def render(
        vector: torch.DoubleTensor,
        raster_res: Tuple[int, int],
        confidence_threshold=.5,
        sigmoid_rate=100
    ) -> torch.DoubleTensor:

    """Renders a line drawing in a differentiable way.

    The algorithm is based on a simple and a seemingly widespread idea
    that the rendering could be obtained as a thresholded distance
    transform of the pixel grid w.r.t. the line [1].

    To implement it, we compute distances between each point of the raster
    drawing and the line using vector projection [https://en.wikipedia.org/wiki/Vector_projection]
    Vector projection = Vector.DotProduct(A, B) / Vector.DotProduct(B, B) * B

    [1] Li, L., Zou, C., Zheng, Y., Su, Q., Fu, H., & Tai, C. L. (2018).
    Sketch-R2CNN: An Attentive Network for Vector Sketch Recognition.
    (see URL: https://arxiv.org/abs/1811.08170)

    Parameters
    ----------
    :param vector: [..., PT_LINE + 1 = 6] tensor describing the vector image
    :type vector: torch.DoubleTensor

    :param raster_res: resolution of desired raster image (in pixels)
    :type raster_res: Tuple[int, int]

    :param confidence_threshold: the confidence threshold (we render the line
                                 if its confidence exceeds the threshold)
    :type confidence_threshold: float

    :param sigmoid_rate: parameter regulating steepness of the sigmoid
                         (the larger the value, the more steep the change in sigmoid)
    :type sigmoid_rate: float

    Examples:

    >>> import torch
    >>> vector = torch.DoubleTensor([
    ...     [0, 0, 3, 3, 1, .99],
    ...     [0, 3, 3, 0, 1, .99]])
    >>> raster_res = (4, 4)
    >>> from vectran.train.diff_rendering import render
    >>> render(vector, raster_res)
    Out[6]:
    tensor([[[1.0000e+00, 2.0253e-09, 1.0000e+00, 5.0633e-10],
             [2.0253e-09, 1.0000e+00, 2.0253e-09, 3.9558e-40],
             [1.0000e+00, 2.0253e-09, 1.0000e+00, 5.0633e-10],
             [5.0633e-10, 3.9558e-40, 5.0633e-10, 1.9532e-31]]],
           dtype=torch.float64)
    """

    # We commonly work with batches of `n_batch` images, each having `n_lines` lines.
    # However, for the debugging and ergonomic reasons, we also accept as valid
    # single vector images and even single vector lines.
    vector_shape = vector.size()
    if len(vector_shape) == 1:  # single line only
        vector = vector.unsqueeze(0).unsqueeze(1)
    elif len(vector_shape) == 2:  # single vector image
        vector = vector.unsqueeze(0)

    # Define helper variables designating dimensions in vector.
    n_batch, n_lines, n_dim = vector.size()  # vector.size() = [b, n, (x1, y1, x2, y2, w, p)]
    assert n_dim == 6, 'we don\'t handle non-lines in diff-rendering as of now'


    # Define helper variables designating dimensions in pixel grid.
    batch_dim, line_dim, y_dim, x_dim, val_dim = range(5)

    # Compute coordinates for each pixel of the raster drawing
    # using a two-channel image (x, y) of
    # shape [n_batch, n_lines, raster_height, raster_width, 2]
    raster_width, raster_height = raster_res
    pixel_grid_coords = torch.stack(
        (torch.arange(1, raster_width + 1).repeat((raster_height, 1)).float() - .5,
         torch.arange(1, raster_height + 1).reshape((raster_height, 1)).repeat((1, raster_width)).float() - .5)) \
        .permute((1, 2, 0)) \
        .unsqueeze(batch_dim) \
        .unsqueeze(line_dim) \
        .repeat((n_batch, n_lines, 1, 1, 1)) \
        .to(vector.device)


    # Extract the endpoints, width, and the confidence values from the `vector` tensor.
    p1, p2, w, p = vector[..., :2], vector[..., 2:4], vector[..., 4], vector[..., 5]
    for t in [p1, p2, w, p]:
        t.unsqueeze_(y_dim).unsqueeze_(x_dim)  # make compatible to pixel grid shape


    # The main algorithm begins here.
    # Compute distances between each point of the raster drawing and the line
    # using vector projection [https://en.wikipedia.org/wiki/Vector_projection]
    #       Vector projection = Vector.DotProduct(A, B) / Vector.DotProduct(B, B) * B
    #
    # Here `A` would be every pixel in the pixel grid, defined by its coordinates.
    # `B` is the second endpoint in the line, defined by `p2`.
    #
    # We will also require the unit vector `e2`, which is the (unit) vector from p1 to p2.
    #
    # First, we offset coordinates of all pixel points by first endpoint.
    # Next, we compute projection onto vector determined by second endpoint.
    # We further offset back by adding the first endpoint, and compute distances.
    max_coord_along_e2 = torch.norm(
        p2 - p1, dim=val_dim, keepdim=True)  # maximum allowed value along e2 is the stroke length
                                             # keep the normed-out dimension for shape compatibility
    e2 = (p2 - p1) / (max_coord_along_e2 + 10e-6)

    coords_along_e2 = torch.sum(
        (pixel_grid_coords - p1) * e2,
        dim=val_dim, keepdim=True)           # compute coordinates along e2 vector
                                             # via batch- line-, and pixel-wise dot-product
                                             # (here: sum over last dim of elementwise tensor product)
                                             # between (x, y) coords of the pixel and those of e2;
                                             # keep the summed-out dimension for shape compatibility

    projections = coords_along_e2 * e2 + p1  # x_proj = (x, e2) e2

    distances = torch.norm(
        pixel_grid_coords - projections,
        dim=val_dim, keepdim=True)           # the final distances in the euclidean norm
                                             # keep the normed-out dimension for shape compatibility

    # Make linecaps square
    coords_along_e2 = coords_along_e2 + w[..., None] / 2
    max_coord_along_e2 = max_coord_along_e2 + w[..., None]

    # Compute the drawing as a product of 4 indicator functions:
    #  - indicator (distances < w / 2): the pixel is closer to the line skeleton than its half-width
    #  - indicator (coords_along_e2 > 0): pixel projection lies on the line (and not outside first endpoint)
    #  - indicator (coords_along_e2 < max_coord_along_e2): same as above, outside second endpoint
    #  - indicator (confidence > .5): the line must exist

    # To imitate such indicator functions, we use sigmoids and pass
    # the following quantities to them:
    distance_indicator = w.unsqueeze(val_dim) / 2. - distances
    endpoint1_indicator = coords_along_e2
    endpoint2_indicator = max_coord_along_e2 - coords_along_e2
    confidence_indicator = p.unsqueeze(val_dim) - confidence_threshold

    # We multiply the quantities with `sigmoid_rate` -- the larger the value,
    # the more steep the change in sigmoid
    if not (isinstance(sigmoid_rate,(float,int)) ):
        sigmoid_rate = sigmoid_rate[..., None, None, None, None]

    raster = torch.sigmoid(distance_indicator * sigmoid_rate) * \
             torch.sigmoid(endpoint1_indicator * sigmoid_rate) * \
             torch.sigmoid(endpoint2_indicator * sigmoid_rate) * \
             torch.sigmoid(confidence_indicator * sigmoid_rate)

    raster = raster.squeeze(val_dim)  # remove the last dimension (which was left for algebraic compatibility reasons)

    raster = raster.sum(dim=line_dim) \
        .clamp(0, 1)  # sum over lines

    return raster


def test_render():
    import torch
    from torch.autograd import gradcheck

    vector = torch.DoubleTensor([[0, 0, 3, 3, 1, .99], [0, 3, 3, 0, 1, .99]])
    raster_res = (4, 4)

    vector.requires_grad = True
    gradcheck(render, (vector, raster_res))
