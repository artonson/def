from sharpf.modules import module_by_kind
from sharpf.modules.base import ParameterizedModule, load_with_spec


class GenericPointBasedNet(ParameterizedModule):
    def __init__(self, encoder_blocks):
        super(GenericPointBasedNet, self).__init__()
        self.encoder_blocks = encoder_blocks

    def forward(self, points):
        activations = [points]
        features = points
        for block in self.encoder_blocks:
            features = block(features)
            activations.append(features)
        return features

    @classmethod
    def from_spec(cls, spec):
        blocks = []
        for block_spec in spec['encoder_blocks']:
            block = load_with_spec(block_spec, module_by_kind)
            blocks.append(block)
        return cls(blocks)

