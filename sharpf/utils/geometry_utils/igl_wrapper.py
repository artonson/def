import igl


def pointset_edgeset_distances_projections(points, edges_mesh):
    """Compute the distances and projections using libigl's
    functionality available in the method `point_mesh_squared_distance`:
    triangle [1 2 2] is treated as a segment [1 2]
    (see docs at https://libigl.github.io/libigl-python-bindings/igl_docs/#point_mesh_squared_distance )."""

    distances, _, projections = igl.point_mesh_squared_distance(
        points,
        edges_mesh.vertices,
        edges_mesh.faces)
    return distances, projections
