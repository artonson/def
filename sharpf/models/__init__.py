import json

from .generic import GenericPointBasedNet
from sharpf.modules.base import load_with_spec


MODEL_BY_NAME = {
    'GenericPointBasedNet': GenericPointBasedNet,
    'DGCNN': DGCNN
}


def load_model(spec_filename, checkpoint=None):
    model_spec = None
    if None is not spec_filename:
        with open(spec_filename) as model_spec_file:
            model_spec = json.load(model_spec_file)
    model = load_with_spec(model_spec, MODEL_BY_NAME)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


__all__ = [
    'load_model'
]
