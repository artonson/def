
from .conv_modules import ConvBase, conv_module_by_kind
from .neighbour_modules import neighbour_module_by_kind
from .local_modules import local_module_by_kind
from .aggregation_modules import aggregation_module_by_kind
from .interpolation_modules import interpolation_module_by_kind
from .point_blocks import  point_block_by_kind


module_by_kind = {  # not Python 2.7 compatible!
    **conv_module_by_kind,
    **neighbour_module_by_kind,
    **local_module_by_kind,
    **aggregation_module_by_kind,
    **interpolation_module_by_kind,
    **point_block_by_kind
}
