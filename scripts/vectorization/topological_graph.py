import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from optimization import subdivide_wireframe
from utils import *


def separate_graph_connected_components(points_to_tree, radius, compute_barycenters=False, 
                                        filtering_mode=False, filtering_factor=10):
    """
    Create knn graph and separate graph nodes into connected components.
    Args:
        points_to_tree (np.array): 3D coordinates of points to create graph on
        radius (float): radius to connect points into knn
        compute_barycenters (bool): whether to compute connected component barycenters
        filtering_mode (bool): whether to use filetring mode
        filtering_factor (int): minimal number of nodes in connected component to pass through filtering
    Returns:
        clusters (list of lists): indices of points from points_to_tree that compose a connected component
        (optional) centers (list): indices for each cluster that is a barycenter
    """
    # create knn graph
    tree = cKDTree(points_to_tree)
    out = tree.query_pairs(radius)
    G = nx.Graph()
    G.add_edges_from(out)
    # divide it into connected components
    S = [G.subgraph(c).copy() for c in tqdm(nx.connected_components(G))]
    clusters = []
    if filtering_mode:
        for subgraph in S:
            subgraph = list(subgraph)
            
            # add connected component if number of nodes is greater than filtering factor
            if len(subgraph) > filtering_factor:
                clusters.append(subgraph)
        return clusters   
    # compute barycenters of connected components; 
    # if there is more than one barycenter, add the first one
    if compute_barycenters:
        centers = []
        for subgraph in S:
            clusters.append(list(subgraph))
            centers.append(nx.barycenter(subgraph)[0])
        return clusters, centers
    else:
        for subgraph in S:
            clusters.append(list(subgraph))
        return clusters


# def identify_corners(points, distances, fps, corner_detector_radius, upper_variance_threshold, lower_variance_threshold, cornerness_threshold, connected_components_radius, box_margin, quantile):
#     """
#     Detect corners and extract them into separate clusters
#     Args:
#         points (np.array): 3D coordinates of points
#         fps (list): indices of farthest sampled points
#         corner_detector_radius (float): radius of neighbourhood to detect corners
#         corner_extractor_radius (float): radius of neighbourhood to extract corners
#         variance_threshold (float): threshold to decide whether a neighbourhood contains corner
#         connected_components_radius (float): radius to connect points into knn
#     Returns:
#         clusters (list of lists): indices of points from points_to_tree that compose a connected component
#         corner_neighbours (list): indices of points that are close to corners
#         corner_clusters (list of lists): indices of points[corner_neighbours] that compose a corner cluster
#         corner_centers (list): indices from each cluster that indicate a barycenter
#     """
#     tree = cKDTree(points)
#     neighbours = tree.query_ball_point(points[fps], r=corner_detector_radius)
    
#     variances = []
#     for n in tqdm(neighbours):
#         pca = PCA(3)
#         # if size of neighbourhood is sufficient
#         if len(n) > 5:
#             pca.fit(points[n])
#             variances.append(pca.explained_variance_ratio_)
#         else: variances.append([1,0,0])
    
#     # smooth intersection: add more weight, if a point with small distance is present in corner-like neighbourhood
#     # subtract more wight if a point with large distance is present in corner-like neighbourhood
#     cornerness_counter = np.zeros((len(points)))
#     weights = (distances - distances.min()) / (distances.max() - distances.min())
#     for i in np.where(np.array(variances)[:,1] > upper_variance_threshold)[0]:
#         cornerness_counter[neighbours[i]] += 1 - weights[neighbours[i]]
#     for i in np.where(np.array(variances)[:0] > lower_variance_threshold)[0]:
#         cornerness_counter[neighbours[i]] -= weights[neighbours[i]]
       
    # corners = np.where(cornerness_counter > cornerness_threshold)
    # separate high-cornerness clusters
#     crnr_clst = separate_graph_connected_components(points[], radius=connected_components_radius)
    
    # inflate high-cornerness clusters by choosing a bounding box, in order to reliably separate corners from curves
#     boxes = []
#     for c in crnr_clst:
#         upper_b = points[np.where(cornerness_counter > cornerness_threshold)][c].max(0) + box_margin
#         lower_b = points[np.where(cornerness_counter > cornerness_threshold)][c].min(0) - box_margin
#         boxes.append(np.where(np.logical_and((points < upper_b).all(-1), (points > lower_b).all(-1)))[0])
    
#     corner_centers = []
#     corner_clusters = []
#     corners = np.unique(np.concatenate(boxes))
    
#     tmp_corner_clusters = separate_graph_connected_components(points[corners], 
#                                                                               radius=connected_components_radius,
#                                                                               compute_barycenters=False)
#     tmp_corner_centers = [np.median(points[corners][clust], axis=0).tolist() for clust in tmp_corner_clusters] 
    
    # for every corner box, sample two points as corners, and store connections between them
    
#     init_connections = []
#     norms = []
#     for i, cluster in enumerate(tmp_corner_clusters):
#         tmp_p = points[corners][cluster]
#         tmp_p -= tmp_p.mean(0)
#         norms.append(np.linalg.norm(tmp_p, axis=1).max())
#     for i, cluster in enumerate(tmp_corner_clusters):
#         if norms[i] < np.quantile(norms, quantile):
#             corner_clusters.append(cluster)
#             query = cKDTree(points[corners]).query(tmp_corner_centers[i], 1)
#             corner_centers.append(query[1])
#             continue
#         else:
#             gmm = GaussianMixture(2)
#             res = gmm.fit_predict(points[corners][cluster])
#             corner_clusters.append(np.array(cluster)[res == 0].tolist())
#             corner_clusters.append(np.array(cluster)[res == 1].tolist())
#             endpoints_query = cKDTree(points[corners]).query(gmm.means_, 1)
#             ind = len(corner_centers)
#             corner_centers.append(endpoints_query[1][0])
#             corner_centers.append(endpoints_query[1][1])
#             init_connections.append([ind, ind+1])    
    
#     return cornerness_counter


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


def knn_wireframe(endpoint_positions, points_ref, fps_curve):
    tree = cKDTree(points_ref[fps_curve[0][0]])
    radius = tree.query(points_ref[fps_curve[0][0]], 2)[0][:,1].mean()

    out = tree.query_pairs(radius*2)
    knn_G = nx.Graph()
    knn_G.add_edges_from(out)

    detected_endpoint_indxs = tree.query(endpoint_positions, 1)[1]

    path = nx.shortest_path(knn_G, detected_endpoint_indxs[0], detected_endpoint_indxs[1])

    path = path[::10]
    if path[-1] != detected_endpoint_indxs[1]:
        path.append(detected_endpoint_indxs[1])

    path_edges = [list(a) for a in zip(path[:-1], path[1:])]
    # path_positions = points_ref[fps_curve[0][0]][np.unique(path_edges)]

    dict_ = {}
    a = np.unique(path_edges)
    dict_[detected_endpoint_indxs[0]] = 0
    dict_[detected_endpoint_indxs[1]] = 1
    i=2
    for it in a:
        if it == detected_endpoint_indxs[0] or it == detected_endpoint_indxs[1]:
            continue
        dict_[it] = i
        i += 1

    b = []
    for it in a:
        b.append(dict_[it])

    for edge in path_edges:
        edge[0] = dict_[edge[0]]
        edge[1] = dict_[edge[1]]

    path_positions = points_ref[fps_curve[0][0]][list(dict_.keys())]
    
    return path_positions, path_edges
    

def initialize_topological_graph(points, distances, 
                                 not_corners, curves, 
                                 corners, corner_centers, init_connections, 
                                 endpoint_detector_radius, endpoint_threshold, 
                                 initial_split_threshold, corner_connector_radius, curve_extraction_mode):
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
    if len(corners)!=0:
        corner_positions_all.append(points[corners][corner_centers].tolist())

    # for each curve
    for i in tqdm(range(len(curves))):
        # calculate index shift value for global indexing
        if len(corner_positions_all) > 0:
            global_index_shift = len(np.concatenate(corner_positions_all))
        else: global_index_shift = 0
            
        # if fps would produce not enough samples
        if points[not_corners][curves[i]].shape[0] < 10:
            continue
        if points[not_corners][curves[i]].shape[0] // 10 < 2 or np.linalg.norm(points[not_corners][curves[i]] - points[not_corners][curves[i]].mean(0), 1).max() < endpoint_detector_radius:
            fps_curve = farthest_point_sampling(points[not_corners][curves[i]], 2)
            endpoint_pairs = [[0, 1]]
            endpoint_positions = points[not_corners][curves[i]][fps_curve[0][0]].tolist()
            # identify which corners are close to the curve endpoints
            nearest_corner_distances, nearest_corner_indices = cKDTree(points[corners][corner_centers]).query(
                points[not_corners][curves[i]], 1,
                distance_upper_bound=corner_connector_radius)
            if np.logical_not(np.isinf(nearest_corner_distances)).all():
                if len(np.unique(nearest_corner_indices)) == 2:
                    endpoint_pairs = [[np.unique(nearest_corner_indices)[0], np.unique(nearest_corner_indices)[1]]]
                    corner_pairs_all.append((np.array(endpoint_pairs)).tolist())
                    continue
                elif len(np.unique(nearest_corner_indices)) == 1:
                    endpoint_pairs = [[np.unique(nearest_corner_indices)[0], global_index_shift]]
                    corner_pairs_all.append((np.array(endpoint_pairs)).tolist())
                    corner_positions_all.append([endpoint_positions[0]])
                    continue
            elif np.logical_not(np.isinf(nearest_corner_distances)).any():
                index = nearest_corner_indices[np.logical_not(np.isinf(nearest_corner_distances))]
                if (len(index)>0):
                    endpoint_pairs = [[index[0], global_index_shift]]
                else:
                     endpoint_pairs = [[index, global_index_shift]]
                corner_pairs_all.append((np.array(endpoint_pairs)).tolist())
                corner_positions_all.append([endpoint_positions[0]])
                continue
            else:
                corner_pairs_all.append((np.array(endpoint_pairs)).tolist())
                corner_positions_all.append(endpoint_positions)
                continue
    
        # sample fps on curves for endpoint detection
        fps_curve = farthest_point_sampling(points[not_corners][curves[i]], 
                                            points[not_corners][curves[i]].shape[0] // 10)

        # create neighbourhoods for endpoint detection
        neighbours = cKDTree(points[not_corners][curves[i]]).query_ball_point(
            points[not_corners][curves[i]][fps_curve[0][0]], r=endpoint_detector_radius)
        endpoint_candidate_indicators = []
        endpoint_candidate_indicators_mean = []
#         print(len(neighbours))
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
            else:
                endpoint_candidate_indicators_mean.append(0)
                endpoint_candidate_indicators.append(False)

                
        if np.sum(endpoint_candidate_indicators) > 0:
            if np.sum(endpoint_candidate_indicators) < 2:
                # if there is only one detected endpoint, take second largest value for the second endpoint
                # and add them to per-curve list of endpoints
                argsort = np.argsort(endpoint_candidate_indicators_mean)
                endpoint_positions = points[not_corners][curves[i]][fps_curve[0][0]][argsort[-2:]].tolist()
            else:
#                 # if there is two or more endpoints, take two most distant ones
#                 # and add them to per-curve list of endpoints
                endpoints_query = cKDTree(
                    points[not_corners][curves[i]][fps_curve[0][0]][endpoint_candidate_indicators]).query(
                    points[not_corners][curves[i]][fps_curve[0][0]][endpoint_candidate_indicators], 
                    np.sum(endpoint_candidate_indicators))
                endpoints = [endpoints_query[0].argmax() // np.sum(endpoint_candidate_indicators), 
                             endpoints_query[1][endpoints_query[0].argmax() // np.sum(endpoint_candidate_indicators), 
                                                endpoints_query[0].argmax() % np.sum(endpoint_candidate_indicators)]]
                endpoint_positions = points[not_corners][curves[i]][fps_curve[0][0]][endpoint_candidate_indicators][endpoints].tolist()

            # initialize per-curve list of endpoint pairs
            endpoint_pairs = [[0,1]]
            
            # select per-curve point coordinates for polyline splitting
            points_ref = points[not_corners][curves[i]]
            distances_ref = distances[not_corners][curves[i]]
            
            nearest_corner_distances, nearest_corner_indices = cKDTree(points[corners][corner_centers]).query(
                endpoint_positions[:2], 1,
                distance_upper_bound=corner_connector_radius)

            # move endpoint indices in correspondance to the global indexing
            for j in range(2):
                # if for an endpoint the closest corner is not too far away,
                # add a segment to connect with initial corners
                if np.logical_not(np.isinf(nearest_corner_distances[j])):
                    endpoint_positions[j] = points[corners][corner_centers][nearest_corner_indices[j]]

            if curve_extraction_mode == 'subdivision':
                previous_len = 0
                # while splits are available, do splits
                while len(endpoint_positions) != previous_len:
                    previous_len = len(endpoint_positions)
    #                 print('start aabboxing')
                    matching = parallel_nearest_point(np.array(endpoint_positions), np.array(endpoint_pairs), points_ref)
                    curve_data, curve_distances = recalculate(points_ref, distances_ref, matching, endpoint_pairs)

    #                 print('finish aabboxing')
                    endpoint_positions, endpoint_pairs = subdivide_wireframe(endpoint_positions, endpoint_pairs, 
                                                                             curve_data, curve_distances, 
                                                                             split_threshold=initial_split_threshold)
                    if len(endpoint_positions) > 15:
                        break
            elif curve_extraction_mode == 'knn':
                try:
                    endpoint_positions, endpoint_pairs = knn_wireframe(endpoint_positions, points_ref, fps_curve)
                except:
                    continue
            else:
                raise Exception("No other option for polyline initialization") 

#             # move endpoint indices in correspondance to the global indexing
            endpoint_pairs = (np.array(endpoint_pairs) + global_index_shift).tolist()
            try:
                first_end = endpoint_pairs[0][0]
                second_end = endpoint_pairs[-1][-1]
            except:
                continue

            if np.logical_not(np.isinf(nearest_corner_distances[0])):
                endpoint_pairs = (np.array(endpoint_pairs) - 1).tolist()
                endpoint_pairs[0][0] = nearest_corner_indices[0]
                endpoint_positions = endpoint_positions[1:]
            if len(np.unique(nearest_corner_indices)) == 2:
                if np.logical_not(np.isinf(nearest_corner_distances[1])):
                    endpoint_pairs = (np.array(endpoint_pairs) - 1).tolist()
                    if np.logical_not(np.isinf(nearest_corner_distances[0])):
                        endpoint_pairs[0][0] = nearest_corner_indices[0]
                    else:
                        endpoint_pairs[0][0] = first_end
                    endpoint_pairs[-1][-1] = nearest_corner_indices[1]
                    if len(endpoint_positions) > 1:
                        if np.logical_not(np.isinf(nearest_corner_distances[0])):
                            endpoint_positions = endpoint_positions[1:]
                        else:
                            endpoint_positions = np.delete(np.array(endpoint_positions), 1, axis=0)
                    else:
                        endpoint_positions = None
#             elif len(np.unique(nearest_corner_indices)) == 1:

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
#                 aabboxes = create_aabboxes(np.array(endpoint_positions)[np.array(endpoint_pairs)])
                matching = parallel_nearest_point(np.array(endpoint_positions), np.array(endpoint_pairs), points_ref)
                curve_data, curve_distances = recalculate(points_ref, distances_ref, matching, endpoint_pairs)

                endpoint_positions, endpoint_pairs = subdivide_wireframe(endpoint_positions, endpoint_pairs, 
                                                                         curve_data, curve_distances, 
                                                                         split_threshold=initial_split_threshold)
                if len(endpoint_positions) == 15:
                    break
            endpoint_pairs = (np.array(endpoint_pairs) + global_index_shift).tolist()

        # add per-curve polyline nodes and segment pairs into global array
#         print(endpoint_pairs)
        if endpoint_positions is not None:
            corner_positions_all.append(endpoint_positions)
        corner_pairs_all.append((np.array(endpoint_pairs)).tolist())
    corner_positions, corner_pairs = np.concatenate(corner_positions_all).tolist(), np.concatenate(corner_pairs_all).tolist()
    
    # add initial connection between detected cornerpoints
    for pair in init_connections:
        corner_pairs.append(pair)
    return corner_positions, corner_pairs
