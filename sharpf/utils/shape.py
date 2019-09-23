from abc import ABC as AbstractBase, abstractmethod, abstractclassmethod

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

#
# # the function for patch generator: breadth-first search
#
# def find_and_add(sets, desired_number_of_points, adjacency_graph):
#     counter = len(sets)  # counter for number of vertices added to the patch;
#     # sets is the list of vertices included to the patch
#     for verts in sets:
#         for vs in adjacency_graph.neighbors(verts):
#             if vs not in sets:
#                 sets.append(vs)
#                 counter += 1
#         #                 print(counter)
#         if counter >= desired_number_of_points:
#             break # stop when the patch has more than 1024 vertices
#
#
#
# def generate_patches(addrs_very_good_triangles):
#     points = []  # for storing initial coordinates of points
#     points_normalized = []  # for storing normalized coordinates of points
#     labels = []  # for storing 0-1 labels for non-sharp/sharp points
#     normals = []
#     surface_rate = []  # for counting how many surfaces are there in the patch
#     sharp_rate = []  # for indicator whether the oatch containes sharp vertices at all
#     times = []  # for times (useless)
#     p_names = []  # for names of the patches in the format "initial_mesh_name_N", where N is the starting vertex index
#     desired_number_of_points = 1024
#
#     for addr in addrs_very_good_triangles:
#
#         t_0 = datetime.datetime.now()
#
#         if addr[:4] == '0005':
#             folder = 'yml_05/'
#         else:
#             folder = 'yml_06/'
#
#         yml = yaml.load(
#             open('/home/Albert.Matveev/sharp/abc_fine/' + folder + addr[:-11] + 'features' + addr[-4:] + '.yml', 'r'))
#
#         t_yml_read = datetime.datetime.now()
#
#         sharp_idx = []
#         short_idx = []
#         for i in yml['curves']:
#             if len(i['vert_indices']) < 5:  # this is for filtering based on short curves:
#                 # append all the vertices which are in the curves with less than 5 vertices
#                 short_idx.append(np.array(i['vert_indices']) - 1)  # you need to substract 1 from vertex index,
#                 # since it starts with 1
#             if ('sharp' in i.keys() and i['sharp'] == True):
#                 sharp_idx.append(np.array(i['vert_indices']) - 1)  # append all the vertices which are marked as sharp
#         if len(sharp_idx) > 0:
#             sharp_idx = np.unique(np.concatenate(sharp_idx))
#         if len(short_idx) > 0:
#             short_idx = np.unique(np.concatenate(short_idx))
#
#         t_curves_read = datetime.datetime.now()
#
#         surfaces = []
#         for i in yml['surfaces']:
#             if 'vert_indices' in i.keys():
#                 surfaces.append(np.array(i['face_indices']) - 1)
#
#         t_surfaces_read = datetime.datetime.now()
#
#         vertices = []
#         faces = []
#         if addr[:4] == '0005':
#             folder = 'obj_05/'
#         else:
#             folder = 'obj_06/'
#
#         for line in open("/home/Albert.Matveev/sharp/abc_fine/" + folder + addr + ".obj",
#                          "r"):  # read the mesh: since trimesh
#             # messes the indices,
#             # this has to be done manually
#             values = line.split()
#             if not values: continue
#             if values[0] == 'v':
#                 vertices.append(np.array(values[1:4], dtype='float'))
#             elif values[0] == 'f':
#                 faces.append(np.array([values[1].split('//')[0], values[2].split('//')[0], values[3].split('//')[0]],
#                                       dtype='int'))
#
#         t_obj_read = datetime.datetime.now()
#
#         vertices = np.array(vertices)
#         faces = np.array(faces) - 1
#         sharp_indicator = np.zeros((len(vertices),))
#         sharp_indicator[sharp_idx] = 1
#
#         mesh = trm.base.Trimesh(vertices=vertices, faces=faces, process=False)  # create a mesh from the vertices
#         # and faces read previously
#         adjacency_graph = mesh.vertex_adjacency_graph
#
#         t_mesh_read = datetime.datetime.now()
#
#         for j in np.linspace(0, len(vertices), 7, dtype='int')[:-1]:  # select starting vertices to grow patches from,
#             # while iterating over them use BFS to
#             # generate patches
#             #         for j in [529]:
#             set_of_verts = [j]
#             surfaces_numbers = []
#             find_and_add(sets=set_of_verts, desired_number_of_points=desired_number_of_points,
#                          adjacency_graph=adjacency_graph)  # BFS function
#             a = sharp_indicator[np.array(set_of_verts)[-100:]]
#             b = np.isin(np.array(set_of_verts)[-100:], np.array(set_of_verts)[-100:] - 1)
#             if (a[b].sum() > 3):
#                 #                 print('here! border!',j)
#                 continue
#             set_of_verts = np.unique(np.array(set_of_verts))  # the resulting list of vertices in the patch
#             if np.isin(set_of_verts, short_idx).any():  # discard a patch if there are short lines
#                 continue
#             patch_vertices = mesh.vertices[set_of_verts]
#             patch_sharp = sharp_indicator[set_of_verts]
#             patch_normals = mesh.vertex_normals[set_of_verts]
#
#             if patch_sharp.sum() != 0:
#                 sharp_rate.append(1)
#             else:
#                 sharp_rate.append(0)
#
#             if patch_vertices.shape[0] >= desired_number_of_points:
#                 # select those vertices, which are not sharp in order to use them for counting surfaces (sharp vertices
#                 # are counted twice, since they are on the border between two surfaces, hence they are discarded)
#                 appropriate_verts = set_of_verts[:desired_number_of_points][
#                     patch_sharp[:desired_number_of_points].astype(int) == 0]
#                 for surf_idx, surf_faces in enumerate(surfaces):
#                     surf_verts = np.unique(mesh.faces[surf_faces].ravel())
#                     if len(np.where(np.isin(appropriate_verts, surf_verts))[0]) > 0:
#                         surface_ratio = sharp_indicator[np.unique(np.array(surf_verts))].sum() / len(
#                             np.unique(np.array(surf_verts)))
#                         if (surface_ratio > 0.6):
#                             break
#                         surfaces_numbers.append(surf_idx)  # write indices of surfaces which are present in the patch
#                         continue
#                 if (surface_ratio > 0.6):
#                     continue
#                 surface_rate.append(np.unique(np.array(surfaces_numbers)))
#                 patch_vertices = patch_vertices[:desired_number_of_points]
#                 points.append(patch_vertices)
#                 patch_vertices_normalized = patch_vertices - patch_vertices.mean(axis=0)
#                 patch_vertices_normalized = patch_vertices_normalized / np.linalg.norm(patch_vertices_normalized,
#                                                                                        ord=2, axis=1).max()
#                 points_normalized.append(patch_vertices_normalized)
#                 patch_normals = patch_normals[:desired_number_of_points]
#                 normals.append(patch_normals)
#                 labels.append(patch_sharp[:desired_number_of_points])
#
#                 p_names.append('%s_%i' % (addr, j))
#
#         t_patches_ready = datetime.datetime.now()
#         times.append(np.array([(t_yml_read - t_0).microseconds, (t_curves_read - t_yml_read).microseconds,
#                                (t_surfaces_read - t_curves_read).microseconds,
#                                (t_obj_read - t_surfaces_read).microseconds,
#                                (t_mesh_read - t_obj_read).microseconds, (t_patches_ready - t_mesh_read).microseconds,
#                                (t_patches_ready - t_0).microseconds]))
#
#     times = np.array(times)
#     p_names = np.array(p_names)
#     points = np.array(points)
#     points_normalized = np.array(points_normalized)
#     labels = np.array(labels).reshape(-1, 1024, 1)
#     normals = np.array(normals)
#     sharp_rate = np.array(sharp_rate)
#     return times, p_names, points, points_normalized, labels, sharp_rate, surface_rate, normals
#
#
#
#
