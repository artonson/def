from abc import ABC as AbstractBase, abstractmethod

import numpy as np


class ShapeFilter(AbstractBase):
    @abstractmethod
    def __call__(self, item, **kwargs):
        """

        :param item: ABCItem used for filtering
        :param kwargs: additional args
        :return is_ok: True if item passes the test, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, **kwargs):
        pass


class FaceAreaFilter(ShapeFilter):
    """Filtering based on the ratio of faces with areas X times
    smaller than an average face area: count these triangles

    Leave big enough triangles only (the rate of small triangles in the whole mesh is smaller than 5%).
    """
    def __init__(self, area_to_avg_area_ratio, bad_triangles_thr):
        self.area_to_avg_area_ratio = area_to_avg_area_ratio
        self.bad_triangles_thr = bad_triangles_thr

    def __call__(self, item, **kwargs):
        mesh = item.obj
        lower_face_area = mesh.area_faces.mean() / self.area_to_avg_area_ratio
        fraction_of_bad_triangles = (mesh.area_faces < lower_face_area).mean()
        is_ok = fraction_of_bad_triangles <= self.bad_triangles_thr
        return is_ok

    @classmethod
    def from_config(cls, area_to_avg_area_ratio, bad_triangles_thr, **kwargs):
        return cls(area_to_avg_area_ratio, bad_triangles_thr)


class FaceAspectRatioFilter(ShapeFilter):
    """Filtering based on the aspect ratio of faces:
    count the number of triangles with aspect ratio more than 5"""
    def __init__(self, aspect_ratio_thr, outer_inner_ratio_thr):
        self.aspect_ratio_thr = aspect_ratio_thr
        self.outer_inner_ratio_thr = outer_inner_ratio_thr

    def __call__(self, item, **kwargs):
        mesh = item.obj
        vertices_in_edge = mesh.vertices[mesh.edges_unique[mesh.faces_unique_edges]]
        edge_lens_by_coord_sq = (vertices_in_edge[:, :, 0] - vertices_in_edge[:, :, 1]) ** 2  # [n_faces, 3 = n_edges_per_face, 3 = n_coords_per_vertex]
        edge_lens = np.sqrt(np.sum(edge_lens_by_coord_sq, axis=-1))  # [n_faces, 3 = n_edges_per_face]

        outer_radius = np.prod(edge_lens, axis=-1) / 4 / mesh.area_faces  # [n_faces]
        half_perimeter = edge_lens.sum(axis=-1) / 2
        inner_radius = mesh.area_faces / half_perimeter  # [n_faces]

        outer_inner_ratio = (outer_radius / inner_radius >= self.aspect_ratio_thr).mean()
        is_ok = outer_inner_ratio <= self.outer_inner_ratio_thr

        return is_ok

    @classmethod
    def from_config(cls, aspect_ratio_thr, outer_inner_ratio_thr, **kwargs):
        return cls(aspect_ratio_thr, outer_inner_ratio_thr)


class AllFilter(ShapeFilter):
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, item, **kwargs):
        return all(f(item) for f in self.filters)

    @classmethod
    def from_config(cls, filters, **kwargs):
        filter_objs = []
        for filter_config in filters:
            filter_objs.append(
                load_from_options(filter_config)
            )
        return cls(filter_objs)


FILTER_DICT = {
    'face_area': FaceAreaFilter,
    'face_aspect_ratio': FaceAspectRatioFilter,
    'all': AllFilter,
}


def load_from_options(opts):
    name = opts['type']
    assert name in FILTER_DICT, 'unknown kind of filter: "{}"'.format(name)
    filter_cls = FILTER_DICT[name]
    params = opts['params']
    return filter_cls.from_config(**params)

