import numpy as np
import networkx as nx
import torch
from torch import optim
from tqdm import tqdm, trange

import sharpf.parametric.utils as utils


def subdivide_wireframe(corner_positions, corner_pairs, curve_data, curve_distances, split_threshold):
    """
    Make a split in polyline segment if max projections onto the segment is greater than threshold
    Args:
        corner_positions (list): 3D coordinates of polyline nodes
        corner_pairs (list of lists): pairs of indices from corner_positions that compose a segment
        curve_data (list of lists): 3D coordiantes of points that correspond to each of the segments 
        curve_distances (list of lists): per-point sharpness distances that correspond to each of the segments
        split_threshold (float): threshold for splits
    Returns:
        new_corners (list of lists): updated 3D coordinates of polyline nodes
        new_pairs (list of lists): updated pairs of indices from corner_positions that compose a segment
    """
    new_pairs = []
    new_corners = corner_positions.copy()
    assert len(corner_pairs) == len(curve_data)
    assert len(curve_distances) == len(curve_data)
    
    # for each segment
    for i in range(len(curve_data)):
        edge = corner_pairs[i]
        endpoints = np.array(corner_positions)[edge]
        data = curve_data[i]
        distances = curve_distances[i]
        if len(data) == 0:
            continue
            
        # compute prejections of points onto segments
        cosines = np.dot(data - endpoints[0], endpoints[1] - endpoints[0]) / (np.dot(endpoints[1] - endpoints[0], endpoints[1] - endpoints[0]) + 1e-8)
        projections = endpoints[0] + cosines[:,None] * (endpoints[1] - endpoints[0])
        proj_distances = np.linalg.norm(projections - data, axis=1)
        where = np.logical_and(cosines >= 0, cosines <= 1)
        if where.sum() == 0:
            new_pairs.append(edge)
            continue
        residual = np.abs(proj_distances - distances)[where]
        
        # check condition
        if residual.max() < split_threshold:
            new_pairs.append(edge)
            continue
        # do split
        else:   
            argmax = np.arange(len(cosines))[where][residual.argmax()]
            midpoint = data[argmax].tolist()
#             side_1 = [endpoints[0], midpoint] 
#             side_2 = [midpoint, endpoints[1]] 
#             prjs = []
#             for endpoints in [side_1, side_2]:
#                 # check whether split makes fit better
#                 cosines = np.dot(data - endpoints[0], endpoints[1] - endpoints[0]) / np.dot(endpoints[1] - endpoints[0], endpoints[1] - endpoints[0])
#                 projections = endpoints[0] + cosines[:,None] * (endpoints[1] - endpoints[0])
#                 proj_distances = np.linalg.norm(projections - data, axis = 1)
#                 residual_tmp = np.abs(proj_distances - distances)
#                 prjs.append(residual_tmp)
#             # if it does, do split
# #             if allow_arbitrary_splits:
# #                 condition = True
# #             else:
# #                 condition = np.array(prjs).min(0).mean() < residual.mean()
#             if np.array(prjs).min(0).mean() < residual.mean():
            new_corners.append(midpoint)
            if len(new_pairs) == 0:
                midpoint_ind = np.max(np.concatenate(corner_pairs)) + 1
            else:
                midpoint_ind = np.max(np.concatenate([new_pairs, np.array(corner_pairs).reshape(-1,2)])) + 1
            new_pairs.append([edge[0], midpoint_ind])
            new_pairs.append([midpoint_ind, edge[1]])
#             else:
#                 new_pairs.append(edge)
#                 continue
    return new_corners, new_pairs


def corner_loss(init_corners, corners_improved, pairs, data, distances, triplets=[], dangling=[], alpha_fid=0, alpha_fit=1, alpha_ang=0):
    """
    Compute a projection loss
    Args:
        init_corners (Tensor): 3D coordinates of polyline nodes
        corners_improved (Tensor): placeholder for updated 3D coordinates of polyline nodes
        pairs (list of lists): pairs of indices from corner_positions that compose a segment
        data (list of lists): 3D coordiantes of points that correspond to each of the segments 
        distances (list of lists): per-point sharpness distances that correspond to each of the segments
        alpha_fid (float): fidelity term weight
        alpha_fit (float): fit term weight
    Returns:
        loss (Tensor): loss value
    """
    fidelity_term = torch.sum((init_corners - corners_improved) ** 2)
    fit_term = 0
    # for each segment
    for i in range(len(data)):
        if len(data[i]) == 0:
            continue
        endpoints = corners_improved[pairs[i]]
        distances_curve = torch.Tensor(distances[i])
        data_curve = torch.Tensor(data[i])
        # compute projections
        cosines = torch.mm(torch.Tensor(data[i]) - endpoints[0][None,:], endpoints[1][:, None] - endpoints[0][:, None])/ (torch.dot(endpoints[1] - endpoints[0], endpoints[1] - endpoints[0]) + 1e-8)
        projections = endpoints[0] + cosines * (endpoints[1] - endpoints[0])
#         projections[cosines.squeeze() < 0] = endpoints[0]
#         projections[cosines.squeeze() > 1] = endpoints[1]
        proj_distances = torch.norm(projections - data_curve, dim=1)
        residual = torch.sum((proj_distances - distances_curve) ** 2)
#         if len(dangling) > 0:
#             for d in dangling:
#                 if (np.array(pairs[i]) == d).any():
#                     where = np.array([0,1])[np.array(pairs[i]) == d]
#                     for w in where:
#                         if w == 0:
#                             residual += 10*torch.sum(torch.norm(endpoints[0] - data_curve[cosines.squeeze().argmin()]))
#                         if w == 1:
#                             residual += 10*torch.sum(torch.norm(endpoints[1] - data_curve[cosines.squeeze().argmax()]))
        fit_term += residual
    angles = 0
    for triplet in triplets:
        trp = corners_improved[triplet]
        angles += torch.clamp(torch.dot((trp[1] - trp[0])/torch.norm(trp[1] - trp[0]),(trp[2] - trp[1])/torch.norm(trp[2] - trp[1])), -1, 1)
    return alpha_fit * fit_term + alpha_fid * fidelity_term - alpha_ang * angles


def move_corners(corner_positions, corner_pairs, curve_data, curve_distances, triplets=[], dangling=[], alpha_fid=0, alpha_fit=1, alpha_ang=0): 
    """
    Do corner positions optimization based on corner loss
    Args:
        corner_positions (list): 3D coordinates of polyline nodes
        corner_pairs (list of lists): pairs of indices from corner_positions that compose a segment
        curve_data (list of lists): 3D coordiantes of points that correspond to each of the segments 
        curve_distances (list of lists): per-point sharpness distances that correspond to each of the segments
        alpha_fid (float): fidelity term weight
        alpha_fit (float): fit term weight
    Returns:
        corners_improved (list of lists): updated 3D coordinates of polyline nodes
    """
    init_corners = torch.Tensor(corner_positions)
    corners_improved = torch.Tensor(corner_positions)
    corners_improved.requires_grad_()
    
    # create optimizer
    optimizer = optim.SGD([corners_improved], lr=0.001, momentum=0.9)
    t = trange(150, desc='Optimization', leave=True)
    for i in t:
        optimizer.zero_grad()
        loss = corner_loss(init_corners, corners_improved, corner_pairs, curve_data, curve_distances, triplets, 
                           dangling, alpha_fid, alpha_fit, alpha_ang)
        loss.backward()
        optimizer.step()
        s = 'Optimization: step #{0:}, loss: {1:3.1f}'.format(i, loss.item())
        t.set_description(s)
        t.refresh()
    return corners_improved.detach().numpy().tolist()


def get_paths_and_corners(corner_pairs, corner_positions):
    G = nx.Graph()
    G.add_edges_from(corner_pairs)
    nodes = [k for k,v in dict(G.degree()).items() if v != 2]
    nodes_1 = [k for k,v in dict(G.degree()).items() if v == 1]
    corners = [k for k,v in dict(G.degree()).items() if v > 2]
    paths = []
    triplets = []
    for i in nodes:
        for j in nodes:
            try:
                path = nx.shortest_path(G, i, j)
            except:
                continue
#             for path in all_path:
            if len(path) == 1:
                continue
            if (len(path) - len(np.setdiff1d(path, nodes))) > 2:
                continue
            if path[::-1] in paths:
                continue
            if path in paths:
                continue
            paths.append(path)
            triplet = [[path[k], path[k+1], path[k+2]] for k in [0,len(path)-3] if k+2 < len(path)]
            if len(triplet) > 0:
                triplets.append(triplet)
   
    closed_nodes = np.setdiff1d(np.arange(len(corner_positions)), np.unique(np.concatenate(paths)))
    closed_paths = []
    path = []
    for i in closed_nodes:
        if np.isin(i, path):
            continue
        try:
            path = nx.algorithms.cycles.find_cycle(G, i)
        except:
            continue
        if np.isin([item for sublist in paths for item in sublist], path).any():
            continue
        closed_paths.append(np.append(np.array(path)[:,0], np.array(path)[0,0]).tolist()) 
        
        
    what_is_left = np.setdiff1d(closed_nodes, np.unique([item for sublist in closed_paths for item in sublist]))
    what_is_left = np.append(what_is_left, nodes)
    pairs_left = np.array(corner_pairs)[np.isin(np.array(corner_pairs), what_is_left).all(-1)]
    
    
    G = nx.Graph()
    G.add_edges_from(pairs_left)
    nodes = [k for k,v in dict(G.degree()).items() if v != 2]
    for i in nodes:
        for j in nodes:
            try:
                path = nx.shortest_path(G, i, j)
            except:
                continue
#             for path in all_path:
            if len(path) == 1:
                continue
            if (len(path) - len(np.setdiff1d(path, nodes))) > 2:
                continue
            if path[::-1] in paths:
                continue
            if path in paths:
                continue
            paths.append(path)
            triplet = [[path[k], path[k+1], path[k+2]] for k in [0,len(path)-3] if k+2 < len(path)]
            if len(triplet) > 0:
                triplets.append(triplet)
        
    triplets = np.concatenate(triplets).tolist()
#     dangling = []
#     for i in nodes_1:
#         where = (np.array(corner_pairs) == i).any(-1)
#         dangling.append(np.array(corner_pairs)[where].reshape(-1,2))
#     dangling = np.concatenate(dangling).tolist()
    return np.array(corner_positions)[corners], paths, triplets, closed_paths, nodes_1


def optimize_topological_graph(corner_positions, corner_pairs, points, distances, split_threshold, alpha_ang):
    """
    Do final optimization of the topological graph.
    Args:
        corner_positions (list): 3D coordinates of polyline nodes
        corner_pairs (list of lists): pairs of indices from corner_positions that compose a segment
        points (np.array): 3D coordiantes of points
        distances (np.array): per-point sharpness distances
        alpha_fid (float): fidelity term weight
        alpha_fit (float): fit term weight
        split_threshold (float): threshold for splits
    Returns:
        new_corners (list of lists): updated 3D coordinates of polyline nodes
        new_pairs (list of lists): updated pairs of indices from corner_positions that compose a segment
    """
    # match points to segments using aabb
    aabboxes = create_aabboxes(np.array(corner_positions)[np.array(corner_pairs)])
    matching = parallel_nearest_point(aabboxes, np.array(corner_positions)[np.array(corner_pairs)], points)
    curve_data, curve_distances = recalculate(points, distances, matching, corner_pairs)

    # optimize corner positions
    _, _, triplets, _, _ = get_paths_and_corners(corner_pairs, corner_positions)
    
    corner_positions = move_corners(corner_positions, corner_pairs, curve_data, curve_distances, triplets, [], 
                                    alpha_fit=1, alpha_fid=0, alpha_ang=alpha_ang / len(triplets))

    # rematch points to segments
    aabboxes = create_aabboxes(np.array(corner_positions)[np.array(corner_pairs)])
    matching = parallel_nearest_point(aabboxes, np.array(corner_positions)[np.array(corner_pairs)], points)
    curve_data, curve_distances = recalculate(points, distances, matching, corner_pairs)

    # while splits are available, do splits
#     previous_len = 0
#     while len(corner_positions) != previous_len:
#         previous_len = len(corner_positions)
#         corner_positions, corner_pairs = subdivide_wireframe(corner_positions, corner_pairs, 
#                                                              curve_data, curve_distances,
#                                                              split_threshold)
#         aabboxes = create_aabboxes(np.array(corner_positions)[np.array(corner_pairs)])
#         matching = parallel_nearest_point(aabboxes, np.array(corner_positions)[np.array(corner_pairs)], points)
#         curve_data, curve_distances = recalculate(points, distances, matching, corner_pairs)
    

    # final corner positions optimization
#     _, _, _, _, dangling = get_paths_and_corners(corner_pairs, corner_positions)

    corner_positions = move_corners(corner_positions, corner_pairs, curve_data, curve_distances, [], [],
                                    alpha_fit=1, alpha_fid=0)
    
    return corner_positions, corner_pairs
