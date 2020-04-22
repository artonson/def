from sharpf.data.annotation import ANNOTATOR_BY_TYPE
from sharpf.utils.abc_utils.abc.abc_data import ABCChunk
from sharpf.utils.abc_utils.unittest import ABCDownloadableTestCase, CHUNK_FILENAMES
from sharpf.utils.py_utils.config import load_func_from_config

class AnnotationTestCase(ABCDownloadableTestCase):
    # TODO implement this test
    def test_instantiate_annotation(self):
        _ = load_func_from_config(ANNOTATOR_BY_TYPE, {
            "type": "surface_based_aabb",
            "distance_upper_bound": 1.0,
            "validate_annotation": True
        })
        _ = load_func_from_config(ANNOTATOR_BY_TYPE, {
            "type": "global_aabb",
            "distance_upper_bound": 1.0,
            "validate_annotation": True,
            "closest_matching_distance_q": 0.95,
        })
        _ = load_func_from_config(ANNOTATOR_BY_TYPE, {
            "type": "resampling",
            "distance_upper_bound": 1.0,
            "validate_annotation": True,
            "sharp_discretization": 0.1,
            "max_empty_envelope_radius": 0.1,
        })

    def test_resampling_aabb_annotation_coincide(self):
        with ABCChunk(CHUNK_FILENAMES) as chunk:
            pass
