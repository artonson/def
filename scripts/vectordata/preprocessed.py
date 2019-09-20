from vectran.data.vectordata.prepatched import PrepatchedSVG
from vectran.data.preprocessing import PreprocessedBase


class PreprocessedSVG(PrepatchedSVG, PreprocessedBase):
    def __init__(self, **kwargs):
        PreprocessedBase.__init__(self, **kwargs)
        PrepatchedSVG.__init__(self, **kwargs)

    def __getitem__(self, idx):
        return self.preprocess_sample(PrepatchedSVG.__getitem__(self, idx))
