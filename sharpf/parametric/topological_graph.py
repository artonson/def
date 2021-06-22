import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from sharpf.parametric.optimization import subdivide_wireframe
import sharpf.parametric.utils as utils
from sharpf.data.patch_cropping import farthest_point_sampling
from sharpf.utils.geometry_utils.aabb import create_aabboxes
from sharpf.utils.py_utils.parallel import loky_parallel


def separate_points_to_subsets(
        points: np.ndarray,
        knn_radius: float,
        return_barycenters: bool = False,
        min_cluster_size_to_return: int = 0
):
    """Given a set of points in Euclidean space, returns
    a list of subsets of indexes into these points corresponding to
    a list of connected components of its kNN graph (a clustering method).
    Optionally, filter by number of points involved in each cluster
    and compute its barycenter.

    Create knn graph and separate graph nodes into connected components.
    Args:
        points (np.ndarray): 3D coordinates of points to create graph on
        knn_radius (float): radius to use when connecting points into a knn graph
        return_barycenters (bool): whether to compute connected component barycenters
        min_cluster_size_to_return (int): minimal number of nodes
            in connected component to pass through filtering
    Returns:
        clusters (list of lists): indices of points from points_to_tree that compose a connected component
        (optional) barycenters (list): indices for each cluster that is a barycenter
    """
    # create knn graph
    edges = cKDTree(points).query_pairs(knn_radius)
    graph = nx.Graph()
    graph.add_edges_from(edges)

    # divide it into connected components
    cc_graphs = [graph.subgraph(c).copy()
                 for c in tqdm(nx.connected_components(graph))]

    # compute barycenters of connected components; 
    # if there is more than one barycenter, add the first one
    if return_barycenters:
        clusters, centers = [], []
        for subgraph in cc_graphs:
            if len(subgraph) > min_cluster_size_to_return:
                # add connected component if number of nodes
                # is greater than filtering factor
                clusters.append(list(subgraph))
                centers.append(nx.barycenter(subgraph)[0])
        return clusters, centers
    else:
        clusters = []
        for subgraph in cc_graphs:
            if len(subgraph) > min_cluster_size_to_return:
                clusters.append(list(subgraph))
        return clusters
    
    
def get_explained_variance_ratio(X):
    pca = PCA(3)
    if X.shape[0] > 5:  # if size of neighbourhood is sufficient
        return pca.fit(X).explained_variance_ratio_
    else:
        return np.ndarray([1, 0, 0])


def detect_corners(
        points: np.ndarray,
        distances: np.ndarray,
        seeds_rate: float = 1.0,
        corner_detector_radius: float = 1.0,
        upper_variance_threshold: float = 0.2,
        lower_variance_threshold: float = 0.1,
        cornerness_threshold: float = 1.25,
        connected_components_radius: float = 0.2,
        box_margin: float = 0.2,
        quantile: float = 0.2,
        n_jobs: int = 0,
):
    """
    Detects corners from points annotated with predicted distances
    and extracts them into separate clusters
    Args:
        points (np.array): 3D coordinates of points
        distances (np.array): 1D array of distance-to-feature values
        seeds_rate (float): fraction of points to use as seeds for corner detection (all points by default)
        corner_detector_radius (float): radius of neighbourhood to detect corners
        corner_extractor_radius (float): radius of neighbourhood to extract corners
        upper_variance_threshold (float):
        lower_variance_threshold (float):
        variance_threshold (float): threshold to decide whether a neighbourhood contains corner
        connected_components_radius (float): radius to connect points into knn
        n_jobs (int): number of CPU jobs to use for extracting corners; if set to 0, all CPUs are used
    Returns:
        corners (list of lists): indices of points from points that compose a connected component
        corner_neighbours (list): indices of points that are close to corners
        corner_clusters (list of lists): indices of points[corner_neighbours] that compose a corner cluster
        corner_centers (list): indices from each cluster that indicate a barycenter
    """
    if seeds_rate < 1.0:
        num_seeds = np.ceil(len(points) * seeds_rate).astype(np.int)
        points_indexes, _ = farthest_point_sampling(points, num_seeds)
    elif seeds_rate == 1.0:
        points_indexes = np.arange(len(points))
    else:
        raise ValueError('seeds_rate > 1.0')

    tree = cKDTree(points)
    neighbours = tree.query_ball_point(
        points[points_indexes],
        r=corner_detector_radius)

    iterable = (points[n] for n in neighbours)
    variances = loky_parallel(
        get_explained_variance_ratio,
        iterable,
        n_jobs=n_jobs,
        verbose=20,
        batch_size=64)

    # smooth intersection: add more weight, if a point with small distance is present in corner-like neighbourhood
    # subtract more weight if a point with large distance is present in corner-like neighbourhood
    cornerness_score = np.zeros((len(points)))
    weights = (distances - distances.min()) / (distances.max() - distances.min())
    for i in np.where(np.array(variances)[:, 1] > upper_variance_threshold)[0]:
        cornerness_score[neighbours[i]] += 1 - weights[neighbours[i]]
    for i in np.where(np.array(variances)[:, 1] <= lower_variance_threshold)[0]:
        cornerness_score[neighbours[i]] -= weights[neighbours[i]]

    # separate high-cornerness clusters
    high_cornerness_indexes = np.where(cornerness_score > cornerness_threshold)
    high_cornerness_clusters = separate_points_to_subsets(
        points[high_cornerness_indexes],
        knn_radius=connected_components_radius)

    # inflate high-cornerness clusters by choosing a bounding box,
    # in order to reliably separate corners from curves
    boxes = []
    for cluster in high_cornerness_clusters:
        upper_b = points[high_cornerness_indexes][cluster].max(0) + box_margin
        lower_b = points[high_cornerness_indexes][cluster].min(0) - box_margin
        inside_box_mask = np.logical_and(
            (points < upper_b).all(-1),
            (points > lower_b).all(-1))
        boxes.append(np.where(inside_box_mask)[0])

    corner_centers = []
    corner_clusters = []
    corners = np.unique(np.concatenate(boxes))

    tmp_corner_clusters, tmp_corner_centers = separate_points_to_subsets(
        points[corners],
        knn_radius=connected_components_radius,
        return_barycenters=True)
    
    # for every corner box, sample two points as corners,
    # and store connections between them
    init_connections = []
    norms = []
    for i, cluster in enumerate(tmp_corner_clusters):
        tmp_p = points[corners][cluster]
        tmp_p -= tmp_p.mean(0)
        norms.append(np.linalg.norm(tmp_p, axis=1).max())

    for i, cluster in enumerate(tmp_corner_clusters):
        if norms[i] < np.quantile(norms, quantile):
            corner_clusters.append(cluster)
            corner_centers.append(tmp_corner_centers[i])
            continue
        else:
            gmm = GaussianMixture(2)
            res = gmm.fit_predict(points[corners][cluster])
            corner_clusters.append(np.array(cluster)[res == 0].tolist())
            corner_clusters.append(np.array(cluster)[res == 1].tolist())
            endpoints_query = cKDTree(points[corners]).query(gmm.means_, 1)
            ind = len(corner_centers)
            corner_centers.append(endpoints_query[1][0])
            corner_centers.append(endpoints_query[1][1])
            init_connections.append([ind, ind + 1])
            
    return corners, corner_clusters, corner_centers, init_connections


def connect_dangling_nodes(corner_pairs, corner_positions, connector_radius):
    G = nx.Graph()
    G.add_edges_from(corner_pairs)

    nodes = [k for k,v in dict(G.degree()).items() if v != 2]
    nodes_1 = [k for k,v in dict(G.degree()).items() if v == 1]

    nearest_corner_distances, nearest_corner_indices = cKDTree(np.array(corner_positions)[nodes]).query(
                np.array(corner_positions)[nodes_1], 2,
                distance_upper_bound=connector_radius)

    for i in range(len(nodes_1)):
        if np.logical_not(np.isinf(nearest_corner_distances[i][1])): 
            index = nearest_corner_indices[i][1]
            pair = [nodes_1[i], nodes[index]]
            if np.isin(corner_pairs, pair).all(-1).any():
                continue
            corner_pairs.append(pair)
    return corner_pairs


def initialize_topological_graph(points, distances, 
                                 not_corners, curves, 
                                 corners, corner_centers, init_connections, 
                                 endpoint_detector_radius, endpoint_threshold, 
                                 initial_split_threshold, corner_connector_radius):
    """
    Initialize topological graph and do initial fit to the curves
    Args:
        points (np.array): 3D coordinates of points
        distances (np.array): per-point sharpness distances
        not_corners (list): indices of points far from corners
        curves (list of lists): indices of points[not_corners] that compose a curve cluster
        corners (list): indices of points near corners
        corner_centers (list): indices from 'corners' that indicate a corner
        endpoint_detector_radius (float): radius of neighbourhood to detect endpoints
        endpoint_threshold (float): threshold to decide whether a neighbourhood contains endpoint
        initial_split_threshold (float): threshold for initial rough splits
        corner_connector_radius (float): radius to connect endpoints to corners
    Returns:
        corner_positions (list of lists): 3D coordinates of polyline nodes
        corner_pairs (list of lists): pairs of indices from corner_positions that compose a segment
    """
    corner_positions_all = []
    corner_pairs_all = []

    # add already detected corner positions
    corner_positions_all.append(points[corners][corner_centers].tolist())

    # for each curve
    for i in tqdm(range(len(curves))):
        # calculate index shift value for global indexing
        if len(corner_positions_all) > 0:
            global_index_shift = len(np.concatenate(corner_positions_all))
        else: global_index_shift = 0
            
        # if fps would produce not enough samples
        if points[not_corners][curves[i]].shape[0] // 10 < 2:
            fps_curve = farthest_point_sampling(points[not_corners][curves[i]], 2)
            endpoint_pairs = [[0, 1]]
            endpoint_positions = points[not_corners][curves[i]][fps_curve[0][0]].tolist()
            # identify which corners are close to the curve endpoints
            nearest_corner_distances, nearest_corner_indices = cKDTree(points[corners][corner_centers]).query(
                endpoint_positions[:2], 1,
                distance_upper_bound=corner_connector_radius)

            # move endpoint indices in correspondance to the global indexing
            endpoint_pairs = (np.array(endpoint_pairs) + global_index_shift).tolist()
            for i in range(2):
                # if for an endpoint the closest corner is not too far away,
                # add a segment to connect with initial corners
                if np.logical_not(np.isinf(nearest_corner_distances[i])): 
                    index = nearest_corner_indices[i]
                    index_shift = len(endpoint_positions)
                    pair = [i+global_index_shift, index]
                    endpoint_pairs.append(pair)
            corner_positions_all.append(endpoint_positions)
            corner_pairs_all.append((np.array(endpoint_pairs)).tolist())
            continue
    
        # sample fps on curves for endpoint detection
        fps_curve = farthest_point_sampling(points[not_corners][curves[i]], 
                                            points[not_corners][curves[i]].shape[0] // 10)

        # create neighbourhoods for endpoint detection
        neighbours = cKDTree(points[not_corners][curves[i]]).query_ball_point(
            points[not_corners][curves[i]][fps_curve[0][0]], r=endpoint_detector_radius)
        endpoint_candidate_indicators = []
        endpoint_candidate_indicators_mean = []
        for n in range(len(neighbours)):
            pca = PCA(1)
            if len(neighbours[n]) > 3:
                fitted_pca = pca.fit(points[not_corners][curves[i]][neighbours[n]])
                
                # calculate linear point embedding inside neighbourhood
                point_linear_embeddings = fitted_pca.transform(points[not_corners][curves[i]][neighbours[n]]).flatten() - fitted_pca.transform(points[not_corners][curves[i]][fps_curve[0][0]][n][None,:]).flatten()
                
                # if neighbourhood center embedding is greater or smaller than embeddings of other points, it is 
                # suspected to be an endpoint
                endpoint_candidate_indicators_mean.append(np.abs(np.sign(point_linear_embeddings).mean()))
                endpoint_candidate_indicators.append(np.abs(np.sign(point_linear_embeddings).mean()) > endpoint_threshold)
                
        if np.sum(endpoint_candidate_indicators) > 0:
#             if np.sum(endpoint_candidate_indicators) < 2:
                # if there is only one detected endpoint, take second largest value for the second endpoint
                # and add them to per-curve list of endpoints
            argsort = np.argsort(endpoint_candidate_indicators_mean)
            endpoint_positions = points[not_corners][curves[i]][fps_curve[0][0]][argsort[-2:]].tolist()
#             else:
#                 # if there is two or more endpoints, take two most distant ones
#                 # and add them to per-curve list of endpoints
#                 endpoints_query = cKDTree(
#                     points[not_corners][curves[i]][fps_curve[0][0]][endpoint_candidate_indicators]).query(
#                     points[not_corners][curves[i]][fps_curve[0][0]][endpoint_candidate_indicators], 
#                     np.sum(endpoint_candidate_indicators))
#                 endpoints = [endpoints_query[0].argmax() // np.sum(endpoint_candidate_indicators), 
#                              endpoints_query[1][endpoints_query[0].argmax() // np.sum(endpoint_candidate_indicators), 
#                                                 endpoints_query[0].argmax() % np.sum(endpoint_candidate_indicators)]]
#                 endpoint_positions = points[not_corners][curves[i]][fps_curve[0][0]][endpoint_candidate_indicators][endpoints].tolist()

            # initialize per-curve list of endpoint pairs
            endpoint_pairs = [[0,1]]
            
            # select per-curve point coordinates for polyline splitting
            points_ref = points[not_corners][curves[i]]
            distances_ref = distances[not_corners][curves[i]]

            previous_len = 0
            # while splits are available, do splits
            while len(endpoint_positions) != previous_len:
                previous_len = len(endpoint_positions)
                aabboxes = create_aabboxes(np.array(endpoint_positions)[np.array(endpoint_pairs)])
                matching = parallel_nearest_point(aabboxes, np.array(endpoint_positions)[np.array(endpoint_pairs)], points_ref)
                curve_data, curve_distances = utils.recalculate(points_ref, distances_ref, matching, endpoint_pairs)

                endpoint_positions, endpoint_pairs = subdivide_wireframe(endpoint_positions, endpoint_pairs, 
                                                                         curve_data, curve_distances, 
                                                                         split_threshold=initial_split_threshold)

            # identify which corners are close to the curve endpoints
            nearest_corner_distances, nearest_corner_indices = cKDTree(points[corners][corner_centers]).query(
                endpoint_positions[:2], 1,
                distance_upper_bound=corner_connector_radius)

            # move endpoint indices in correspondance to the global indexing
            endpoint_pairs = (np.array(endpoint_pairs) + global_index_shift).tolist()
            for i in range(2):
                # if for an endpoint the closest corner is not too far away,
                # add a segment to connect with initial corners
                if np.logical_not(np.isinf(nearest_corner_distances[i])): 
                    index = nearest_corner_indices[i]
                    index_shift = len(endpoint_positions)
                    pair = [i+global_index_shift, index]
                    endpoint_pairs.append(pair)

        # if no endpoints were detected, assume it's a closed curve
        elif np.sum(endpoint_candidate_indicators) == 0:
            if len(points[not_corners][curves[i]][fps_curve[0][0]][:3].tolist()) < 3:
                continue
            # sample a triangle
            endpoint_positions = points[not_corners][curves[i]][fps_curve[0][0]][:3].tolist()
            endpoint_pairs = [[0,1],
                             [1,2],
                             [2,0]]
            
            # select per-curve point coordinates for polyline splitting
            points_ref = points[not_corners][curves[i]]
            distances_ref = distances[not_corners][curves[i]]

            previous_len = 0
            # while splits are available, do splits
            while len(endpoint_positions) != previous_len:
                previous_len = len(endpoint_positions)
                aabboxes = create_aabboxes(np.array(endpoint_positions)[np.array(endpoint_pairs)])
                matching = parallel_nearest_point(aabboxes, np.array(endpoint_positions)[np.array(endpoint_pairs)], points_ref)
                curve_data, curve_distances = utils.recalculate(points_ref, distances_ref, matching, endpoint_pairs)

                endpoint_positions, endpoint_pairs = subdivide_wireframe(endpoint_positions, endpoint_pairs, 
                                                                         curve_data, curve_distances, 
                                                                         split_threshold=initial_split_threshold)
            endpoint_pairs = (np.array(endpoint_pairs) + global_index_shift).tolist()

        # add per-curve polyline nodes and segment pairs into global array
        corner_positions_all.append(endpoint_positions)
        corner_pairs_all.append((np.array(endpoint_pairs)).tolist())
    
    corner_positions, corner_pairs = np.concatenate(corner_positions_all).tolist(), np.concatenate(corner_pairs_all).tolist()
    
    # add initial connection between detected cornerpoints
    for pair in init_connections:
        corner_pairs.append(pair)
#     corner_pairs = connect_dangling_nodes(corner_pairs, corner_positions, 5*0.02)  
    return corner_positions, corner_pairs


# def initialize_topological_graph(points, distances, 
#                                  not_corners, curves, 
#                                  corners, corner_centers,
#                                  init_connections,
#                                  endpoint_detector_radius, endpoint_threshold, 
#                                  initial_split_threshold, corner_connector_radius):
#     """
#     Initialize topological graph and do initial fit to the curves
#     Args:
#         points (np.array): 3D coordinates of points
#         distances (np.array): per-point sharpness distances
#         not_corners (list): indices of points far from corners
#         curves (list of lists): indices of points[not_corners] that compose a curve cluster
#         corners (list): indices of points near corners
#         corner_centers (list): indices from 'corners' that indicate a corner
#         endpoint_detector_radius (float): radius of neighbourhood to detect endpoints
#         endpoint_threshold (float): threshold to decide whether a neighbourhood contains endpoint
#         initial_split_threshold (float): threshold for initial rough splits
#         corner_connector_radius (float): radius to connect endpoints to corners
#     Returns:
#         corner_positions (list of lists): 3D coordinates of polyline nodes
#         corner_pairs (list of lists): pairs of indices from corner_positions that compose a segment
#     """
#     corner_positions_all = []
#     corner_pairs_all = []

#     # add already detected corner positions
#     corner_positions_all.append(points[corners][corner_centers].tolist())

#     # for each curve
#     for i in tqdm(range(len(curves))):
#         # calculate index shift value for global indexing
#         if len(corner_positions_all) > 0:
#             global_index_shift = len(np.concatenate(corner_positions_all))
#         else: global_index_shift = 0
            
#         # if fps would produce not enough samples
#         if points[not_corners][curves[i]].shape[0] // 10 < 2:
#             continue
    
#         # sample fps on curves for endpoint detection
#         fps_curve = farthest_point_sampling(points[not_corners][curves[i]], 
#                                             points[not_corners][curves[i]].shape[0] // 10)

#         # create neighbourhoods for endpoint detection
#         neighbours = cKDTree(points[not_corners][curves[i]]).query_ball_point(
#             points[not_corners][curves[i]][fps_curve[0][0]], r=endpoint_detector_radius)
#         endpoint_candidate_indicators = []
#         endpoint_candidate_indicators_mean = []
#         for n in range(len(neighbours)):
#             pca = PCA(1)
#             if len(neighbours[n]) > 3:
#                 fitted_pca = pca.fit(points[not_corners][curves[i]][neighbours[n]])
                
#                 # calculate linear point embedding inside neighbourhood
#                 point_linear_embeddings = fitted_pca.transform(points[not_corners][curves[i]][neighbours[n]]).flatten() - fitted_pca.transform(points[not_corners][curves[i]][fps_curve[0][0]][n][None,:]).flatten()
                
#                 # if neighbourhood center embedding is greater or smaller than embeddings of other points, it is 
#                 # suspected to be an endpoint
#                 endpoint_candidate_indicators_mean.append(np.abs(np.sign(point_linear_embeddings).mean()))
#                 endpoint_candidate_indicators.append(np.abs(np.sign(point_linear_embeddings).mean()) > endpoint_threshold)
                
#         if np.sum(endpoint_candidate_indicators) > 0:
#             if np.sum(endpoint_candidate_indicators) < 2:
#                 # if there is only one detected endpoint, take second largest value for the second endpoint
#                 # and add them to per-curve list of endpoints
#                 argsort = np.argsort(endpoint_candidate_indicators_mean)
#                 endpoint_positions = points[not_corners][curves[i]][fps_curve[0][0]][argsort[-2:]].tolist()
#             else:
#                 # if there is two or more endpoints, take two most distant ones
#                 # and add them to per-curve list of endpoints
#                 endpoints_query = cKDTree(
#                     points[not_corners][curves[i]][fps_curve[0][0]][endpoint_candidate_indicators]).query(
#                     points[not_corners][curves[i]][fps_curve[0][0]][endpoint_candidate_indicators], 
#                     np.sum(endpoint_candidate_indicators))
#                 endpoints = [endpoints_query[0].argmax() // np.sum(endpoint_candidate_indicators), 
#                              endpoints_query[1][endpoints_query[0].argmax() // np.sum(endpoint_candidate_indicators), 
#                                                 endpoints_query[0].argmax() % np.sum(endpoint_candidate_indicators)]]
#                 endpoint_positions = points[not_corners][curves[i]][fps_curve[0][0]][endpoint_candidate_indicators][endpoints].tolist()

#             # initialize per-curve list of endpoint pairs
#             endpoint_pairs = [[0,1]]
            
#             # select per-curve point coordinates for polyline splitting
#             points_ref = points[not_corners][curves[i]]
#             distances_ref = distances[not_corners][curves[i]]

#             previous_len = 0
#             # while splits are available, do splits
#             while len(endpoint_positions) != previous_len:
#                 previous_len = len(endpoint_positions)
#                 aabboxes = create_aabboxes(np.array(endpoint_positions)[np.array(endpoint_pairs)])
#                 matching = parallel_nearest_point(aabboxes, np.array(endpoint_positions)[np.array(endpoint_pairs)], points_ref)
#                 curve_data, curve_distances = recalculate(points_ref, distances_ref, matching, endpoint_pairs)

#                 endpoint_positions, endpoint_pairs = subdivide_wireframe(endpoint_positions, endpoint_pairs, 
#                                                                          curve_data, curve_distances, 
#                                                                          split_threshold=initial_split_threshold)

#             # identify which corners are close to the curve endpoints
#             nearest_corner_distances, nearest_corner_indices = cKDTree(points[corners][corner_centers]).query(
#                 endpoint_positions[:2], 1,
#                 distance_upper_bound=corner_connector_radius)

#             # move endpoint indices in correspondance to the global indexing
#             endpoint_pairs = (np.array(endpoint_pairs) + global_index_shift).tolist()
#             for i in range(2):
#                 # if for an endpoint the closest corner is not too far away,
#                 # add a segment to connect with initial corners
#                 if np.logical_not(np.isinf(nearest_corner_distances[i])): 
#                     index = nearest_corner_indices[i]
#                     index_shift = len(endpoint_positions)
#                     pair = [i+global_index_shift, index]
#                     endpoint_pairs.append(pair)

#         # if no endpoints were detected, assume it's a closed curve
#         elif np.sum(endpoint_candidate_indicators) == 0:
#             # sample a triangle
#             endpoint_positions = points[not_corners][curves[i]][fps_curve[0][0]][:3].tolist()
#             endpoint_pairs = [[0,1],
#                              [1,2],
#                              [2,0]]
            
#             # select per-curve point coordinates for polyline splitting
#             points_ref = points[not_corners][curves[i]]
#             distances_ref = distances[not_corners][curves[i]]

#             previous_len = 0
#             # while splits are available, do splits
#             while len(endpoint_positions) != previous_len:
#                 previous_len = len(endpoint_positions)
#                 aabboxes = create_aabboxes(np.array(endpoint_positions)[np.array(endpoint_pairs)])
#                 matching = parallel_nearest_point(aabboxes, np.array(endpoint_positions)[np.array(endpoint_pairs)], points_ref)
#                 curve_data, curve_distances = recalculate(points_ref, distances_ref, matching, endpoint_pairs)

#                 endpoint_positions, endpoint_pairs = subdivide_wireframe(endpoint_positions, endpoint_pairs, 
#                                                                          curve_data, curve_distances, 
#                                                                          split_threshold=initial_split_threshold)
#             endpoint_pairs = (np.array(endpoint_pairs) + global_index_shift).tolist()

#         # add per-curve polyline nodes and segment pairs into global array
#         corner_positions_all.append(endpoint_positions)
#         corner_pairs_all.append((np.array(endpoint_pairs)).tolist())
        
#     # add initial connection between detected cornerpoints
#     for pair in init_connections:
#         corner_pairs.append(pair)
#     corner_positions, corner_pairs = np.concatenate(corner_positions_all).tolist(), np.concatenate(corner_pairs_all).tolist()
#     return corner_positions, corner_pairs
