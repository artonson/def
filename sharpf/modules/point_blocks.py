import torch
import torch.nn

from sharpf.modules import (
    neighbour_module_by_kind,
    local_module_by_kind,
    aggregation_module_by_kind,
    interpolation_module_by_kind,
    conv_module_by_kind
)

from sharpf.modules.base import load_with_spec, ParameterizedModule


class PointOpBlock(ParameterizedModule):
    def __init__(self, neighbours, local_transform, feature_extractor, aggregation,
                   interpolation, in_features, **kwargs):
        super(PointOpBlock, self).__init__(**kwargs)
        self.in_features = in_features
        self._op = torch.nn.Sequential(*(
            neighbours, local_transform, feature_extractor,
            aggregation, interpolation
        ))

    def forward(self, points):
        block_features = self._op.forward(points)
        return block_features

    @classmethod
    def from_spec(cls, spec):
        # TODO check if layer exists, otherwise get base
        in_features = spec.get("in_features", [])
        neighbours = load_with_spec(spec['neighbours'], neighbour_module_by_kind)
        local_transform = load_with_spec(spec['local_transform'], local_module_by_kind)
        feature_extractor = load_with_spec(spec['features'], conv_module_by_kind)
        aggregation = load_with_spec(spec['aggregation'], aggregation_module_by_kind)
        interpolation = load_with_spec(spec['interpolation'], interpolation_module_by_kind)
        return cls(neighbours, local_transform, feature_extractor, aggregation,
                   interpolation, in_features)


point_block_by_kind = {
    'point_block_nlfai': PointOpBlock
}
