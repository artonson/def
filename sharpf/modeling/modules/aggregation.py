import torch
import torch.nn as nn


class AggregationMax(nn.Module):
    """
    AggregationMax - class for max aggregation method for ParameterizedPointNet.
    Current class deals with point feature aggregation from neigbourhood.

    ...

    Parameters
    ----------
    keepdim: bool, whether to keep number of dimensions or not
    dim: int, over which dimension to take max

    Methods
    -------
    forward(x)
       Performs an aggregation operation

    """

    def __init__(self, keepdim, dim, **kwargs):
        super().__init__(**kwargs)
        self.keepdim = keepdim
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: x: batch of features for aggregation, shape = (B, N_in, C_in, M_in)
                  B - batch size,
                  N_in - number of points,
                  C_in - number of features,
                  M_in - number of points in the neighbourhood
        output: out: (B, N_out, C_out, M_out) tensor
        """
        out = x.max(dim=self.dim, keepdim=self.keepdim)[0]
        return out


class AggregationMaxPooling(nn.Module):
    """
    AggregationMaxPooling - class for max pooling aggregation method for ParameterizedPointNet.
    Current class deals with point feature aggregation from neigbourhood.

    ...

    Parameters
    ----------
    kernel_size: list, size of the window to take max over

    Methods
    -------
    forward(x)
       Performs an aggregation operation

    """

    def __init__(self, kernel_size, repeat_times=1, **kwargs):
        super().__init__(**kwargs)
        kernel_size = list(kernel_size)
        self.kernel_size = kernel_size
        self.repeat_times = repeat_times
        self.max_pool = nn.MaxPool2d(self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: x: batch of features for aggregation, shape = (B, N_in, C_in, M_in)
                  B - batch size,
                  N_in - number of points,
                  C_in - number of features,
                  M_in - number of points in the neighbourhood
        output: out: (B, N_out, C_out, M_out) tensor
        """
        x = x.transpose(2, 1).contiguous()
        out = self.max_pool(x)
        out = torch.repeat_interleave(out, self.repeat_times, 2)
        out = out.transpose(2, 1).contiguous()
        return out
