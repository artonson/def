import numpy as np

import vectran.data.graphics_primitives as graphics_primitives


class PreprocessedBase:
    def __init__(self, *, patch_size, **kwargs):
        patch_width, patch_height = patch_size
        assert patch_width == patch_height
        self.patch_width = patch_width

        self._xs = np.arange(1, patch_width+1, dtype=np.float32)[None].repeat(patch_height, 0) / patch_width
        self._ys = np.arange(1, patch_height+1, dtype=np.float32)[..., None].repeat(patch_width, 1) / patch_height
    
    
    def __getitem__(self, idx):
        r''' Child should do
            return self.preprocess_sample(DatasetClass.__getitem__(self, idx))
        '''
        raise NotImplementedError

    
    def preprocess_image(self, image):
        image = 1 - image # 0 -- background
        mask = (image > 0).astype(np.float32)
        return np.stack([image, self._xs * mask, self._ys * mask], axis=-3)

    
    def preprocess_primitives(self, primitives):
        primitives[..., :-1] /= self.patch_width
        return primitives


    def preprocess_sample(self, sample):
        return self.preprocess_image(sample['raster'].astype(np.float32)), \
               self.preprocess_primitives(sample['vector'][graphics_primitives.PrimitiveType.PT_LINE].astype(np.float32))
